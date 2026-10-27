"""GPU training and fine-tuning engine for Causal TCN-AE on NVIDIA RTX 5050.

Implements rigorous self-supervised normal regime learning:
1. Filters out ground truth anomaly intervals from all 58 NAB series to create pure normal corpora.
2. 80/20 train/validation split with Cosine Annealing LR scheduling.
3. Checkpoints the model with lowest validation loss.
4. Audits reconstruction separation ratio (Anomaly Reconstruction Error vs Normal Reconstruction Error).
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data.loader import NABDataLoader
from .detectors.tcn_autoencoder import TCNAEModule


class TimeSeriesWindowDataset(Dataset):
    """Slices 1D time series into sliding windows for autoencoder reconstruction."""

    def __init__(self, windows: list[np.ndarray]):
        self.data = np.stack(windows) if windows else np.empty((0, 32), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Returns [1, window_size]
        return torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)


def extract_normal_and_anomalous_windows(
    loader: NABDataLoader, window_size: int = 32, stride: int = 4
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Extracts strictly clean normal windows and separate ground truth anomaly windows."""
    df = loader.load_df()
    gt_windows = loader.get_ground_truth_windows_by_series_id()

    normal_windows: list[np.ndarray] = []
    anomaly_windows: list[np.ndarray] = []

    for sid, grp in df.groupby("series_id", sort=False):
        sid_str = str(sid)
        anom_intervals = []
        for w in gt_windows.get(sid_str, []):
            try:
                s_ts = pd.to_datetime(w[0])
                e_ts = pd.to_datetime(w[1])
                anom_intervals.append((s_ts, e_ts))
            except Exception:
                pass

        times = pd.to_datetime(grp["timestamp"].values)
        vals = grp["value"].values.astype(np.float32)
        n = len(vals)

        if n < window_size:
            continue

        for i in range(0, n - window_size + 1, stride):
            w_vals = vals[i : i + window_size]
            w_start = times[i]
            w_end = times[i + window_size - 1]

            # Check overlap with any true anomaly window
            is_anom = False
            for a_start, a_end in anom_intervals:
                if not (w_end < a_start or w_start > a_end):
                    is_anom = True
                    break

            # Normalize window
            mu = float(np.mean(w_vals))
            std = float(max(np.std(w_vals), 1e-4))
            norm_w = ((w_vals - mu) / std).astype(np.float32)

            if is_anom:
                anomaly_windows.append(norm_w)
            else:
                normal_windows.append(norm_w)

    return normal_windows, anomaly_windows


def train_tcn_autoencoder(
    data_path: Optional[str] = None,
    output_checkpoint: str = "models/sentinel_tcn_best.pt",
    window_size: int = 32,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 1.5e-3,
    device: str = "cuda",
) -> dict[str, float]:
    """Train and fine-tune Causal TCN-AE with validation tracking and anomaly discrimination audit."""
    os.makedirs(os.path.dirname(os.path.abspath(output_checkpoint)), exist_ok=True)

    if device == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
    else:
        dev = torch.device("cpu")
        device_name = "CPU"

    print(f"\n=======================================================")
    print(f"[*] Sentinel GPU Fine-Tuning on {device_name}")
    print(f"=======================================================")

    loader = NABDataLoader(data_path=data_path)
    print("[1/4] Partitioning NAB benchmark into clean normal and anomalous sets...")
    normal_windows, anomaly_windows = extract_normal_and_anomalous_windows(loader, window_size=window_size, stride=4)
    print(f"       Clean Normal Windows    : {len(normal_windows)}")
    print(f"       Ground Truth Anomaly Windows: {len(anomaly_windows)}")

    # 80/20 train/validation split on normal regimes
    np.random.seed(42)
    indices = np.random.permutation(len(normal_windows))
    split = int(0.8 * len(normal_windows))
    train_indices = indices[:split]
    val_indices = indices[split:]

    train_set = [normal_windows[i] for i in train_indices]
    val_set = [normal_windows[i] for i in val_indices]

    train_loader = DataLoader(TimeSeriesWindowDataset(train_set), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TimeSeriesWindowDataset(val_set), batch_size=batch_size, shuffle=False)
    anom_loader = DataLoader(TimeSeriesWindowDataset(anomaly_windows), batch_size=batch_size, shuffle=False)

    print(f"[2/4] Dataset Splits: {len(train_set)} train, {len(val_set)} validation, {len(anomaly_windows)} test anomaly.")

    model = TCNAEModule(input_dim=1, hidden_dim=32, latent_dim=16).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    start_time = time.time()

    print(f"[3/4] Executing {epochs} Training Epochs on {dev}...")
    for epoch in range(1, epochs + 1):
        # Training loop
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            batch = batch.to(dev)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_train += 1

        scheduler.step()
        avg_train = train_loss / max(n_train, 1)

        # Validation loop
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(dev)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item()
                n_val += 1

        avg_val = val_loss / max(n_val, 1)

        saved = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), output_checkpoint)
            saved = "[Checkpoint Saved]"

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch [{epoch:02d}/{epochs:02d}] | Train MSE: {avg_train:.6f} | Val MSE: {avg_val:.6f} | LR: {curr_lr:.6f} | {saved}"
        )

    elapsed = time.time() - start_time
    print(f"\n[4/4] Auditing Anomaly Discrimination Ratio on test windows...")

    # Load best checkpoint
    model.load_state_dict(torch.load(output_checkpoint, map_location=dev))
    model.eval()

    anom_loss = 0.0
    n_anom = 0
    with torch.no_grad():
        for batch in anom_loader:
            batch = batch.to(dev)
            recon = model(batch)
            loss = criterion(recon, batch)
            anom_loss += loss.item()
            n_anom += 1

    avg_anom_loss = anom_loss / max(n_anom, 1)
    separation_ratio = avg_anom_loss / max(best_val_loss, 1e-6)

    print("-------------------------------------------------------")
    print(f"  Best Normal Validation MSE : {best_val_loss:.6f}")
    print(f"  Test Anomaly Window MSE     : {avg_anom_loss:.6f}")
    print(f"  Anomaly Separation Ratio   : {separation_ratio:.2f}x higher error on anomalies!")
    print(f"  Training Time              : {elapsed:.2f} seconds")
    print(f"  Model Saved To             : {output_checkpoint}")
    print("-------------------------------------------------------\n")

    return {
        "best_val_loss": float(best_val_loss),
        "test_anom_loss": float(avg_anom_loss),
        "separation_ratio": float(separation_ratio),
        "epochs": epochs,
        "elapsed_seconds": float(elapsed),
        "device": str(dev),
    }


if __name__ == "__main__":
    train_tcn_autoencoder(epochs=15)
