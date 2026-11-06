"""Inference from a package checkpoint; writes probabilities for two endpoints."""

import argparse
import csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from deepora.data import VolumeDataset, LABELS
from deepora.runtime import build_model, load_weights, predict


def infer(checkpoint, manifest, output, device="cuda", batch_size=1):
    """Use stored normalization and validation thresholds without refitting."""
    checkpoint = torch.load(checkpoint, map_location="cpu", weights_only=False)
    c = dict(checkpoint["config"])
    if c["stage"] == "pretrain":
        raise ValueError("A classification checkpoint is required.")
    # Segmentation/report branches are inactive in evaluation. Keep their weights for strict loading.
    model = build_model(c).to(device)
    load_weights(model, checkpoint)
    data = VolumeDataset(manifest, c, checkpoint.get("stats"))
    rows = predict(model, DataLoader(data, batch_size=batch_size), device)
    thresholds = (checkpoint.get("validation") or {}).get("thresholds", {})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        ["id"]
        + [k + "_probability" for k in LABELS[:2]]
        + [k + "_positive" for k in LABELS[:2] if k in thresholds]
    )
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {"id": r["id"]}
            for i, k in enumerate(LABELS[:2]):
                row[k + "_probability"] = r["probabilities"][i]
                if k in thresholds:
                    row[k + "_positive"] = int(r["probabilities"][i] >= thresholds[k])
            writer.writerow(row)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=1)
    a = p.parse_args()
    infer(a.checkpoint, a.manifest, a.output, a.device, a.batch_size)


if __name__ == "__main__":
    main()
