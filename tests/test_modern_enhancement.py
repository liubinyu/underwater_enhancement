"""Exercise real checkpoint compatibility and image boundary conditions."""
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

torch = pytest.importorskip("torch")
from aqua_align.modern_enhancement import FGDPAEnhancer, DEFAULT_CHECKPOINT


def test_missing_and_corrupted_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_fgdpa"):
        FGDPAEnhancer(tmp_path / "missing.pkl")
    invalid = tmp_path / "invalid.pkl"
    invalid.write_bytes(b"not a valid checkpoint")
    with pytest.raises(ValueError, match="SHA256"):
        FGDPAEnhancer(invalid)


@pytest.mark.skipif(not DEFAULT_CHECKPOINT.exists(), reason="Official checkpoint not downloaded")
def test_pretrained_inference_matches_upstream_and_preserves_dimensions():
    torch.set_num_threads(2)
    enhancer = FGDPAEnhancer(device="cpu")
    rng = np.random.default_rng(42)
    for height, width in [(1, 1), (17, 25), (65, 97)]:
        array = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        actual = np.asarray(enhancer.enhance(Image.fromarray(array)))
        x = torch.from_numpy(array.astype(np.float32).transpose(2, 0, 1).copy() / 255).unsqueeze(0)
        with torch.inference_mode():
            expected = enhancer.model(x)[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).numpy()
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == array.shape
        assert actual.dtype == np.uint8
    assert enhancer.enhance(Image.new("L", (29, 31))).mode == "RGB"
