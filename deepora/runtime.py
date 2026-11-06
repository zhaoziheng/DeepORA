"""Shared model construction, checkpoint transfer and validation selection."""

import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from .model import DeepORA
from .data import LABELS


def build_model(config):
    """Construct one CT or MRI model for the requested training substage."""
    stage = config["stage"]
    return DeepORA(
        use_ct=config["modality"] == "ct",
        use_mr=config["modality"] == "mr",
        do_pretrain=stage == "pretrain",
        add_brainstructure_seg=config.get("brain_seg", False),
        add_vessel_seg=config.get("vessel_seg", False),
        add_report=config.get("report", False),
        target_labels=LABELS,
        image_size=config["image_size"],
        num_transformer_layers=2,
        use_risk_factors=stage == "fusion",
        risk_encoder_dropout=0.2,
        clinical_feature_dropout_rate=config.get("feature_dropout", 0.0),
        text_encoder=config.get("text_encoder", "FremyCompany/BioLORD-2023-C"),
    )


def load_weights(model, checkpoint, transfer=False):
    """Load strictly, or transfer a declared preceding training substage.

    Transfer is limited to auxiliary heads and the image-to-fusion classifier
    transition. Missing encoder/attention parameters are always an error.
    Only load checkpoints from trusted sources.
    """
    state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    state = {k.removeprefix("module."): v for k, v in state.items()}
    if not transfer:
        model.load_state_dict(state, strict=True)
        return
    own = model.state_dict()
    if "visual_head.0.weight" in own and "cls_head.0.weight" in state:
        state["visual_head.0.weight"] = state.pop("cls_head.0.weight")
        state["visual_head.0.bias"] = state.pop("cls_head.0.bias")
    shared = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    missing = set(own) - set(shared)
    allowed = (
        "ct_seg_decoder.",
        "mr_seg_decoder.",
        "report_",
        "logit_scale",
        "cls_token",
        "transformer_decoder.",
        "cls_head.",
        "cls_task_heads.",
        "factor_embedder.",
    )
    invalid = [k for k in missing if not k.startswith(allowed)]
    # Attention may be newly created only when transferring from pure pretraining.
    if "cls_token" in state:
        invalid += [
            k for k in missing if k.startswith(("transformer_decoder.", "cls_token"))
        ]
    if invalid:
        raise ValueError(f"Checkpoint misses required parameters: {invalid[:5]}")
    model.load_state_dict(shared, strict=False)
    print(
        json.dumps(
            {"transferred_parameters": len(shared), "new_parameters": sorted(missing)}
        )
    )


def forward(model, batch, device):
    """Return model outputs and the loss keyword arguments for one batch."""
    m = "ct" if model.use_ct else "mr"
    image = batch["image"].to(device)
    present = torch.ones(image.shape[0], device=device, dtype=torch.bool)
    result = model(
        **{m + "_input": image, m + "_mask": present},
        risk_input=batch["risk"].to(device) if model.use_risk_factors else None,
        report=batch["report"] if model.add_report else None,
    )
    kw = {
        "labels": {
            label: batch["labels"][:, i].to(device) for i, label in enumerate(LABELS)
        },
        "label_mask": batch["label_mask"].to(device),
        m + "_mask": present,
        m + "_segs": batch["brain"].to(device),
        m + "_seg_mask": batch["brain_mask"].to(device),
        m + "_vessel_segs": batch["vessel"].to(device),
        m + "_vessel_seg_mask": batch["vessel_mask"].to(device),
        "report_mask": batch["report_mask"].to(device),
    }
    return result, kw


@torch.inference_mode()
def predict(model, loader, device):
    """Predict probabilities without labels or reports as model inputs."""
    model.eval()
    rows = []
    for b in loader:
        result, _ = forward(model, b, device)
        probabilities = (
            torch.stack([result["predictions"][k] for k in LABELS], dim=1)
            .sigmoid()
            .cpu()
        )
        for i, identifier in enumerate(b["id"]):
            rows.append(
                {
                    "id": identifier,
                    "probabilities": probabilities[i].tolist(),
                    "labels": b["labels"][i].tolist(),
                    "label_mask": b["label_mask"][i].tolist(),
                }
            )
    return rows


def validation_metrics(rows):
    """Select by mean validation AUROC over the two principal endpoints."""
    aucs, thresholds = {}, {}
    for i, label in enumerate(LABELS[:2]):
        pairs = [
            (r["labels"][i], r["probabilities"][i]) for r in rows if r["label_mask"][i]
        ]
        if not pairs or len({p[0] for p in pairs}) != 2:
            raise ValueError(f"Validation needs both classes for {label}.")
        y, s = map(np.asarray, zip(*pairs))
        aucs[label] = float(roc_auc_score(y, s))
        # >= threshold is positive; include an all-negative candidate.
        candidates = np.r_[np.nextafter(s.max(), np.inf), np.unique(s)]
        acceptable = [t for t in candidates if np.mean(s[y == 0] < t) >= 0.9]
        t = min(acceptable, key=lambda t: (-np.mean(s[y == 1] >= t), -t))
        thresholds[label] = float(t)
    return float(np.mean(list(aucs.values()))), aucs, thresholds


def save_checkpoint(path, contents):
    """Atomically replace one resumable checkpoint."""
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(contents, temp)
    temp.replace(path)
