from .baseline_net import BaselineEnhancementNet
from .physics_guided_net import PhysicsGuidedEnhancementNet


def build_model(name: str, cfg: dict | None = None):
    cfg = cfg or {}
    name = name.lower()
    if name in {"baseline", "baseline_net"}:
        return BaselineEnhancementNet(**cfg.get("model_args", {}))
    if name in {"physics_guided", "physics_guided_model", "physics"}:
        return PhysicsGuidedEnhancementNet(**cfg.get("model_args", {}))
    raise ValueError(f"Unknown model name: {name}")

