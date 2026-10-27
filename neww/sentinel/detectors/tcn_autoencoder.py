"""Deep Neural Candidate B3: Causal Temporal Convolutional Autoencoder (TCN-AE).

Optimized for NVIDIA RTX 5050 GPU (CUDA) acceleration with causal 1D dilated convolutions,
latent compression, and reconstruction error anomaly scoring.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from ..contracts import Point, RawDetectorOutput
from .base import BaseDetector


class CausalConv1d(nn.Module):
    """1D convolution with left-padding to guarantee causality (no future leakage)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove future padding from the end of sequence
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNAEModule(nn.Module):
    """Causal TCN Autoencoder architecture."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        # Encoder: Dilated causal convs
        self.enc1 = CausalConv1d(input_dim, hidden_dim, kernel_size=3, dilation=1)
        self.act1 = nn.LeakyReLU(0.1)
        self.enc2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2)
        self.act2 = nn.LeakyReLU(0.1)
        self.bottleneck = nn.Conv1d(hidden_dim, latent_dim, kernel_size=1)

        # Decoder: Latent expansion and causal reconstruction
        self.dec1 = nn.Conv1d(latent_dim, hidden_dim, kernel_size=1)
        self.dec_act1 = nn.LeakyReLU(0.1)
        self.dec2 = CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=1)
        self.dec_act2 = nn.LeakyReLU(0.1)
        self.out = nn.Conv1d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, channels, seq_len]
        h = self.act1(self.enc1(x))
        h = self.act2(self.enc2(h))
        z = self.bottleneck(h)

        h_dec = self.dec_act1(self.dec1(z))
        h_dec = self.dec_act2(self.dec2(h_dec))
        recon = self.out(h_dec)
        return recon


class CausalTCNAEDetector(BaseDetector):
    """Streaming anomaly detector using Causal TCN-AE reconstruction error."""

    def __init__(
        self,
        window_size: int = 32,
        warmup_steps: int = 25,
        hidden_dim: int = 32,
        latent_dim: int = 16,
        scale_floor: float = 1e-4,
        device: str = "cuda",
        version: str = "tcn_ae-0.1.0",
    ) -> None:
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.scale_floor = scale_floor
        self._version = version

        # Resolve device (GPU if available)
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.net = TCNAEModule(
            input_dim=1,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        ).to(self.device)
        self.net.eval()

        self._buffers: dict[str, deque[float]] = {}
        self._residual_history: dict[str, deque[float]] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_buffer(self, series_id: str) -> deque[float]:
        buf = self._buffers.get(series_id)
        if buf is None:
            buf = self._buffers[series_id] = deque(maxlen=self.window_size)
        return buf

    def _get_res_history(self, series_id: str) -> deque[float]:
        res_buf = self._residual_history.get(series_id)
        if res_buf is None:
            res_buf = self._residual_history[series_id] = deque(maxlen=100)
        return res_buf

    def score_one(self, point: Point) -> RawDetectorOutput:
        buf = self._get_buffer(point.series_id)
        n = len(buf)

        if n < self.window_size:
            # Warm-up phase
            val = float(point.value)
            med = float(np.median(list(buf))) if n > 0 else val
            return RawDetectorOutput(
                raw_score=0.0,
                observed=val,
                baseline=med,
                residual=val - med,
                context={"is_warmup": True, "device": str(self.device), "window_size": n},
            )

        # Formulate past sequence + current point
        seq = list(buf)[-(self.window_size - 1):] + [point.value]
        seq_arr = np.array(seq, dtype=np.float32)

        # Local window normalization for numerical stability
        loc_mean = float(np.mean(seq_arr[:-1]))
        loc_std = float(max(np.std(seq_arr[:-1]), self.scale_floor))
        norm_seq = (seq_arr - loc_mean) / loc_std

        tensor_x = torch.tensor(norm_seq, dtype=torch.float32, device=self.device).view(1, 1, -1)

        with torch.no_grad():
            recon = self.net(tensor_x)
            recon_point_norm = recon[0, 0, -1].item()

        # Invert normalization to find baseline reconstructed value
        recon_point = (recon_point_norm * loc_std) + loc_mean
        residual = point.value - recon_point
        abs_err = abs(residual)

        # Standardize score against recent residual history
        res_buf = self._get_res_history(point.series_id)
        if len(res_buf) >= 10:
            res_arr = np.array(res_buf, dtype=np.float32)
            res_med = float(np.median(res_arr))
            res_mad = float(max(np.median(np.abs(res_arr - res_med)), self.scale_floor))
            score = 0.6745 * abs(abs_err - res_med) / res_mad
        else:
            score = abs_err / loc_std

        if not math.isfinite(score):
            score = 1e6

        return RawDetectorOutput(
            raw_score=float(score),
            observed=float(point.value),
            baseline=float(recon_point),
            residual=float(residual),
            context={
                "reconstruction_error": abs_err,
                "device": str(self.device),
                "is_warmup": False,
                "contributors": ["tcn_reconstruction_loss", "residual_magnitude"],
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        buf = self._get_buffer(point.series_id)
        if not is_anomalous or len(buf) < self.warmup_steps:
            if math.isfinite(point.value):
                buf.append(point.value)

        # Update residual tracking for score calibration
        if not is_anomalous:
            res_buf = self._get_res_history(point.series_id)
            if len(buf) >= self.window_size:
                med = float(np.median(list(buf)))
                res_buf.append(abs(point.value - med))

    def reset_state(self, series_id: Optional[str] = None) -> None:
        if series_id:
            self._buffers.pop(series_id, None)
            self._residual_history.pop(series_id, None)
        else:
            self._buffers.clear()
            self._residual_history.clear()
