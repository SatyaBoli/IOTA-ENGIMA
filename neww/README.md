# NEWW detector integration

This folder contains the NEWW streaming detector source and candidate configuration,
bundled under the MIT license in `LICENSE`. The app's default detector is the NEWW
hybrid ensemble, which combines the repository's causal Rolling MAD, EWMA, and
trained TCN autoencoder using the weights configured by NEWW.

The trained TCN checkpoint is `../models/sentinel_tcn_best.pt`. The app resolves
that path relative to its own source tree, verifies that it exists, and fails
explicitly if PyTorch cannot load the checkpoint. It does not silently substitute
an untrained network. Its SHA-256 is pinned to
`6bd10611237536852794d46c70c02bd341891053273d5184a6e36001f1d31717`.
The bundled code and checkpoint match upstream commit
`c83580cb5aa52d2c494fb2d150f39be2dca6603c`.

Install the app dependencies from the project directory:

```powershell
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

The included candidate configuration uses the repository's likelihood calibration,
hysteresis, drift detection, and guarded baseline updates. The dashboard reports
local replay measurements only; official benchmark metrics still require the
NEWW benchmark data and evaluator.

Upstream project: <https://github.com/SatyaBoli/NEWW>
