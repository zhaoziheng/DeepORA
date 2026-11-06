"""Single-device pretraining, imaging fine-tuning and risk-factor fusion."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, RandomSampler
from deepora.data import VolumeDataset, TaskSampler, read_manifest
from deepora.loss import MultiTaskLoss
from deepora.risk import compute_risk_feature_stats
from deepora.runtime import (
    build_model,
    load_weights,
    forward,
    predict,
    validation_metrics,
    save_checkpoint,
)
from deepora.scheduler import cosine_lr


def train(
    config,
    train_manifest,
    output,
    valid_manifest=None,
    init=None,
    resume=None,
    device="cuda",
):
    """Run a configured stage on user-supplied arrays; save last and valid-best."""
    if init and resume:
        raise ValueError("Choose either initialization or resumption.")
    c = dict(config)
    device = torch.device(device)
    if c["stage"] not in ("pretrain", "imaging", "fusion") or c["modality"] not in (
        "ct",
        "mr",
    ):
        raise ValueError("Choose a supported training stage and imaging modality.")
    if not (len(c["steps"]) == len(c["warmup"]) == len(c["lr"])):
        raise ValueError("steps, warmup and lr must have equal lengths.")
    if any(n <= 0 or not 0 <= warm < n for n, warm in zip(c["steps"], c["warmup"])):
        raise ValueError("Each schedule needs positive steps and 0 <= warmup < steps.")
    random.seed(c["seed"])
    np.random.seed(c["seed"])
    torch.manual_seed(c["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    records = read_manifest(train_manifest)
    if c["stage"] == "pretrain" and not any(
        (c.get("report") and r.get("report"))
        or (c.get("brain_seg") and r.get("brain_seg"))
        or (c.get("vessel_seg") and r.get("vessel_seg"))
        for r in records
    ):
        raise ValueError(
            "Pretraining requires an enabled report or segmentation target."
        )
    if c["stage"] != "pretrain" and valid_manifest is None:
        raise ValueError(
            "Classification training requires a separate validation manifest."
        )
    if valid_manifest:
        val = read_manifest(valid_manifest)
        if {r["subject_id"] for r in records} & {r["subject_id"] for r in val}:
            raise ValueError("Training and validation subjects overlap.")
    stats = None
    if c["stage"] == "fusion":
        mean, std = compute_risk_feature_stats(
            [{"cvd_features": r.get("risk", {})} for r in records]
        )
        stats = {"mean": mean, "std": std}
    checkpoint = (
        torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    )
    if checkpoint:
        if checkpoint["config"] != c:
            raise ValueError("Resumption requires the exact saved configuration.")
        if checkpoint.get("stats") != stats:
            raise ValueError(
                "Training risk statistics changed since checkpoint creation."
            )
        stats = checkpoint["stats"]
    model = build_model(c).to(device)
    if init:
        load_weights(
            model,
            torch.load(init, map_location="cpu", weights_only=False),
            transfer=True,
        )
    if checkpoint:
        load_weights(model, checkpoint)
    if c["stage"] == "fusion":
        if not init and not resume:
            raise ValueError("Fusion requires a trained imaging checkpoint.")
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(("factor_embedder.", "cls_task_heads."))
    data = VolumeDataset(train_manifest, c, stats, training=True)
    count = sum(c["steps"]) * c["batch_size"]
    sampler = (
        TaskSampler(records, count, c["sampling"])
        if c["sampling"] != "random"
        else RandomSampler(data, replacement=True, num_samples=count)
    )
    loader = DataLoader(
        data, batch_size=c["batch_size"], sampler=sampler, num_workers=0
    )
    valid_loader = (
        DataLoader(VolumeDataset(valid_manifest, c, stats), batch_size=c["batch_size"])
        if valid_manifest
        else None
    )
    loss_fn = MultiTaskLoss(
        device=device,
        do_pretrain=c["stage"] == "pretrain",
        target_labels=model.target_labels,
        use_focal=True,
        focal_gamma=c.get("focal_gamma", 2.0),
        use_masked_loss=True,
        use_brainstructure_seg_loss=c.get("brain_seg", False),
        brainstructure_seg_loss_weight=c.get("brain_weight", 0),
        use_vessel_seg_loss=c.get("vessel_seg", False),
        vessel_seg_loss_weight=c.get("vessel_weight", 0),
        use_clip_loss=c.get("report", False),
        clip_loss_weight=c.get("report_weight", 0),
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=c["lr"][0],
        betas=(0.9, 0.9999),
        eps=1e-8,
        weight_decay=0.0,
    )
    schedule = cosine_lr(optimizer, c["lr"], c["warmup"], c["steps"])
    amp = device.type == "cuda" and c.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best = 0, -float("inf")
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start, best = checkpoint["step"], checkpoint["best_validation_mean_auc"]
        random.setstate(checkpoint["random_state"])
        np.random.set_state(checkpoint["numpy_state"])
        torch.set_rng_state(checkpoint["torch_state"])
        if device.type == "cuda" and checkpoint.get("cuda_state") is not None:
            torch.cuda.set_rng_state(checkpoint["cuda_state"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(c, indent=2))
    if stats:
        (output / "risk_statistics.json").write_text(json.dumps(stats, indent=2))
    iterator = iter(loader)
    accum = c.get("accumulation", 1)
    if sum(c["steps"]) % accum or c["save_every"] % accum or start % accum:
        raise ValueError(
            "Training and checkpoint boundaries must align with gradient accumulation."
        )
    optimizer.zero_grad(set_to_none=True)
    for step in range(start + 1, sum(c["steps"]) + 1):
        # A step means one minibatch, as in the research training loop.
        model.train()
        lr = schedule(step)
        batch = next(iterator)
        with torch.autocast(device_type=device.type, enabled=amp):
            result, kw = forward(model, batch, device)
            loss, _ = loss_fn(result, **kw)
        if not torch.isfinite(loss):
            raise ValueError(f"Nonfinite loss at step {step}")
        scaler.scale(loss / accum).backward()
        if step % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"step": step, "loss": loss.item(), "lr": lr}), flush=True)
        if step % c["save_every"] == 0 or step == sum(c["steps"]):
            metrics = None
            if valid_loader:
                score, aucs, thresholds = validation_metrics(
                    predict(model, valid_loader, device)
                )
                metrics = {"mean_auc": score, "aucs": aucs, "thresholds": thresholds}
            improved = metrics is not None and metrics["mean_auc"] > best
            if improved:
                best = metrics["mean_auc"]
            state = {
                "config": c,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "stats": stats,
                "best_validation_mean_auc": best,
                "validation": metrics,
                "random_state": random.getstate(),
                "numpy_state": np.random.get_state(),
                "torch_state": torch.get_rng_state(),
                "cuda_state": torch.cuda.get_rng_state()
                if device.type == "cuda"
                else None,
            }
            save_checkpoint(output / "last.pt", state)
            if improved:
                save_checkpoint(output / "best.pt", state)
            if metrics:
                print(json.dumps({"validation": metrics}), flush=True)
    return output / "last.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--valid")
    p.add_argument("--output", required=True)
    p.add_argument("--init")
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    c = yaml.safe_load(Path(a.config).read_text())
    train(c, a.train, a.output, a.valid, a.init, a.resume, a.device)


if __name__ == "__main__":
    main()
