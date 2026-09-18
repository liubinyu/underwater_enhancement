"""Full-resolution FGDPA inference using the authors' released slim weights."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch

from third_party.fgdpa.uie import FGDRAUIENetS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints/fgdpa/model_best_slim.pkl"
CHECKPOINT_SHA256 = "7f278f37eca028e91a11a53b178ed3cb0b5d966fc48ccc7e7082bf8c4c1a8d3c"
UPSTREAM_COMMIT = "530c692feb99ecf6205bb929118a9167e2809d4d"


class FGDPAEnhancer:
    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 device: str = "auto") -> None:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Missing FGDPA checkpoint: {path}. Run python scripts/download_fgdpa.py")
        self.checkpoint_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if self.checkpoint_sha256 != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint SHA256 mismatch: expected the pinned official slim weights")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.model = FGDRAUIENetS(channels=12, fft_size=32)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def enhance(self, image: Image.Image) -> Image.Image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        array = np.array(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0).to(self.device)
        # Whole-image FFT/global attention: tiling would change the method.
        result = self.model(tensor)
        if result.shape != tensor.shape or not torch.isfinite(result).all():
            raise RuntimeError("FGDPA returned invalid image values or dimensions")
        pixels = result[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(pixels)
