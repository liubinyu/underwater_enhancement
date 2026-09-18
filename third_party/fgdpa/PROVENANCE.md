# FGDPA provenance

Source: https://github.com/LethyZhang/FGDPA

Pinned commit: `530c692feb99ecf6205bb929118a9167e2809d4d` (retrieved 2026-09-10).

`uie.py`, `utils.py`, `upstream_config.yaml`, and `LICENSE` are unmodified
copies of `model/uie.py`, `model/utils.py`, `config/uie.yaml`, and `LICENSE`.
The upstream class name is FGDRAUIENetS despite the repository name FGDPA.
Local adapter and CLI live outside this directory.

Paper (as identified by the authors): Zhang, Leshen; Li, Ao; Zhu, Ce.
Real-Time Underwater Image Enhancement via Frequency-Guided Dual-Path Attention.
ICME 2026. See the upstream README for citation and experimental protocol.

Released checkpoint: `experiments/pretrain/models/model_best_slim.pkl`.
SHA256: `7f278f37eca028e91a11a53b178ed3cb0b5d966fc48ccc7e7082bf8c4c1a8d3c`.
Use `python scripts/download_fgdpa.py` to restore the ignored local checkpoint.
UIEB was used for training; local UIEB examples are demonstrations and cannot
establish held-out performance without verifying the original split.
