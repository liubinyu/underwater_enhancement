from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch import optim
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets import create_dataloaders
from losses import EnhancementLoss
from models import build_model
from utils.image_ops import ensure_dir, make_comparison, tensor_to_image
from utils.logger import save_json, setup_logger
from utils.vis import plot_loss_curve, save_background_light, save_tensor_image, save_transmission_map


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, best: float, cfg: dict) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "best_val": best,
            "config": cfg,
        },
        path,
    )


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool, sample_dir: Path, epoch: int) -> float:
    model.eval()
    total = 0.0
    n = 0
    first_saved = False
    for batch in loader:
        src = batch["input"].to(device, non_blocking=True)
        tgt = batch.get("target")
        tgt = tgt.to(device, non_blocking=True) if tgt is not None else None
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(src)
            loss, _ = criterion(out, src, tgt)
        total += float(loss.detach().cpu())
        n += 1
        if not first_saved:
            comp = make_comparison(
                [tensor_to_image(src), tensor_to_image(out["final_enhanced"])],
                ["input", "enhanced"],
            )
            ensure_dir(sample_dir)
            comp.save(sample_dir / f"epoch_{epoch:03d}_comparison.jpg")
            if "coarse_restored" in out:
                save_tensor_image(out["coarse_restored"], sample_dir / f"epoch_{epoch:03d}_coarse.jpg")
            if "estimated_t" in out:
                save_transmission_map(out["estimated_t"], sample_dir / f"epoch_{epoch:03d}_transmission.png")
            if "estimated_A" in out:
                save_background_light(out["estimated_A"], sample_dir / f"epoch_{epoch:03d}_A.png")
            first_saved = True
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "physics_guided.yaml"))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--model", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"] = args.model
    run_name = cfg.get("run_name", cfg.get("model", "run"))
    ckpt_dir = ensure_dir(ROOT / cfg.get("checkpoint_dir", "checkpoints") / run_name)
    result_dir = ensure_dir(ROOT / cfg.get("result_dir", "results") / run_name)
    logger = setup_logger(result_dir / "train.log")
    save_json(cfg, result_dir / "config_used.json")

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    train_loader, val_loader = create_dataloaders(cfg)
    model = build_model(cfg["model"], cfg).to(device)
    criterion = EnhancementLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=float(cfg["train"].get("lr", 2e-4)), weight_decay=float(cfg["train"].get("weight_decay", 1e-4)))
    amp = bool(cfg["train"].get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")
    start_epoch, best_val = 1, float("inf")

    resume_path = args.resume or cfg["train"].get("resume", "")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") and scaler:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        logger.info(f"Resumed from {resume_path} at epoch {start_epoch}")

    history: list[dict[str, float]] = []
    epochs = int(cfg["train"].get("epochs", 50))
    log_interval = int(cfg["train"].get("log_interval", 20))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_total = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for step, batch in enumerate(pbar, 1):
            src = batch["input"].to(device, non_blocking=True)
            tgt = batch.get("target")
            tgt = tgt.to(device, non_blocking=True) if tgt is not None else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(src)
                loss, logs = criterion(out, src, tgt)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach().cpu())
            if step % log_interval == 0:
                pbar.set_postfix(loss=f"{logs['total']:.4f}", color=f"{logs.get('color', 0):.3f}", exp=f"{logs.get('exposure', 0):.3f}")

        train_avg = train_total / max(len(train_loader), 1)
        val_avg = validate(model, val_loader, criterion, device, amp, result_dir / "samples", epoch)
        history.append({"epoch": epoch, "train_total": train_avg, "val_total": val_avg})
        logger.info(f"epoch={epoch} train={train_avg:.5f} val={val_avg:.5f}")
        save_json(history, result_dir / "loss_history.json")
        plot_loss_curve(history, result_dir / "loss_curve.png")

        save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, scaler, epoch, best_val, cfg)
        if val_avg < best_val:
            best_val = val_avg
            save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scaler, epoch, best_val, cfg)
            logger.info(f"Saved best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    main()

