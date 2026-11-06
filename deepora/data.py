"""Portable JSONL/NumPy input boundary; no hospital records are bundled."""

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .risk import encode_risk_factors

LABELS = [
    "intracranial_severe_stenosis",
    "coronary_severe_stenosis",
    "vertebral_origin_severe_stenosis",
    "cervical_carotid_severe_stenosis",
]


def read_manifest(path):
    """Read one record per examination with explicit subject identifiers."""
    records = [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Examination ids must be unique within a manifest.")
    if any(not r.get("subject_id") for r in records):
        raise ValueError("Every record requires subject_id for split checks.")
    return records


class VolumeDataset(Dataset):
    """Load already resampled/cropped arrays and normalize each channel.

    JSONL paths resolve relative to the manifest. Missing labels are null.
    MRI channels: T1, T2, T2-FLAIR, DWI; training can zero one sequence before augmentation.
    Segmentation maps use background 0, foreground 1..16 (CT) or 1..95 (MRI).
    """

    def __init__(self, manifest, config, stats=None, training=False):
        self.records = read_manifest(manifest)
        self.base = Path(manifest).resolve().parent
        self.config, self.stats, self.training = config, stats, training
        self.modality = config["modality"]
        self.channels = 1 if self.modality == "ct" else 4
        self.shape = tuple(config["image_size"])
        self.augment = None
        if training and config.get("augmentation"):
            from .augmentation import get_3D_image_augmentator

            aug = json.loads(Path(config["augmentation"]).read_text())
            self.augment = get_3D_image_augmentator(aug, with_seg=True)

    def __len__(self):
        return len(self.records)

    def _array(self, name, channels, integer=False, upper=None):
        x = np.load(self.base / name, allow_pickle=False)
        if channels == 1 and x.shape == self.shape:
            x = x[None]
        if x.shape != (channels, *self.shape) or not np.isfinite(x).all():
            raise ValueError(f"Invalid array shape or values: {name}; got {x.shape}")
        if integer and (np.any(x != np.rint(x)) or x.min() < 0 or x.max() > upper):
            raise ValueError(f"Segmentation values must be integers in [0,{upper}].")
        return torch.from_numpy(np.array(x, dtype=np.float32, copy=True))

    def __getitem__(self, idx):
        r = self.records[idx]
        image = self._array(r["image"], self.channels)
        if self.modality == "ct":
            image.clamp_(-10, 90)
        lo = image.amin(dim=(1, 2, 3), keepdim=True)
        hi = image.amax(dim=(1, 2, 3), keepdim=True)
        image = (image - lo) / (hi - lo + 1e-8)
        brain_classes = 16 if self.modality == "ct" else 95
        brain = (
            self._array(r["brain_seg"], 1, True, brain_classes)
            if r.get("brain_seg")
            else torch.zeros(1, *self.shape)
        )
        vessel = (
            self._array(r["vessel_seg"], 1, True, 1)
            if r.get("vessel_seg")
            else torch.zeros(1, *self.shape)
        )
        if (
            self.training
            and self.modality == "mr"
            and random.random() < self.config.get("mr_dropout", 0.0)
        ):
            # Research pipeline: with this probability drop exactly one sequence.
            image[random.randint(0, 3)] = 0.0
        if self.augment:
            a = self.augment({"image": image, "mask": torch.cat([brain, vessel])})
            image, brain, vessel = a["image"], a["mask"][:1], a["mask"][1:]
        raw_labels = r.get("labels", [None] * len(LABELS))
        if len(raw_labels) != len(LABELS) or any(
            v not in (None, 0, 1) for v in raw_labels
        ):
            raise ValueError("labels must contain four entries: 0, 1 or null.")
        risk = encode_risk_factors(
            r.get("risk", {}),
            r.get("history"),
            r.get("history") is not None,
            self.stats["mean"] if self.stats else None,
            self.stats["std"] if self.stats else None,
        )
        if not torch.isfinite(risk).all():
            raise ValueError("Risk factors must be finite numbers or null.")
        return {
            "id": r["id"],
            "image": image.as_tensor() if hasattr(image, "as_tensor") else image,
            "labels": torch.tensor([v or 0 for v in raw_labels], dtype=torch.float32),
            "label_mask": torch.tensor([v is not None for v in raw_labels]),
            "brain": brain,
            "vessel": vessel,
            "brain_mask": bool(r.get("brain_seg")),
            "vessel_mask": bool(r.get("vessel_seg")),
            "report": r.get("report") or "",
            "report_mask": bool(r.get("report")),
            "risk": risk,
        }


class TaskSampler(Sampler):
    """Choose task, polarity and examination using the research sampling rule."""

    def __init__(self, records, count, mode="sqrt"):
        self.count = count
        self.queues = []
        for i in range(len(LABELS)):
            neg = [
                j for j, r in enumerate(records) if r.get("labels", [None] * 4)[i] == 0
            ]
            pos = [
                j for j, r in enumerate(records) if r.get("labels", [None] * 4)[i] == 1
            ]
            if pos or neg:
                p = (
                    0.5
                    if mode == "balanced"
                    else math.sqrt(len(pos))
                    / (math.sqrt(len(pos)) + math.sqrt(len(neg)))
                )
                self.queues.append((neg, pos, p))
        if not self.queues:
            raise ValueError("Task sampling requires at least one observed label.")

    def __iter__(self):
        for _ in range(self.count):
            neg, pos, p = random.choices(
                self.queues,
                weights=[math.sqrt(len(n) + len(p)) for n, p, _ in self.queues],
            )[0]
            pool = pos if random.random() < p else neg
            yield random.choice(pool or pos or neg)

    def __len__(self):
        return self.count
