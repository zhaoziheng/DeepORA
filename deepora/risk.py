"""Interleaved value/missing-indicator encoding of 29 clinical risk factors."""

import numpy as np
import torch


def parse_sex(sex_value) -> int | None:
    """Map male/female codes to 0/1; return None for unknown values."""
    if sex_value is None:
        return None
    sex_text = str(sex_value).strip()
    if sex_text == "女":
        return 1
    if sex_text == "男":
        return 0
    if sex_text in {"1", "1.0"}:
        return 1
    if sex_text in {"0", "0.0"}:
        return 0
    return None


RISK_CONTINUOUS_FEATURES = [
    "age",
    "BMI",
    "SBP",
    "DBP",
    "HR",
    "TC",
    "TG",
    "LDLC",
    "HDLC",
    "FPG",
    "HbA1c",
    "Scr",
    "eGFR",
    "UA",
    "Hcy",
    "CRP",
    "Lp(a)",
]
RISK_DISEASE_FEATURES = [
    "hypertension",
    "diabetes",
    "hyperlipidemia",
    "chd",
    "stroke_tia",
    "af",
    "ckd",
    "copd",
    "malignancy",
    "chronic_inflam",
    "gout_hyperuric",
]
RISK_FEATURE_DIM = (
    len(RISK_CONTINUOUS_FEATURES) * 2 + 2 + len(RISK_DISEASE_FEATURES) * 2
)


def _extract_risk_value(cvd_features, key):
    """Read a numeric scalar or a dictionary containing a value field."""
    if not cvd_features:
        return None
    v = cvd_features.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        val = v.get("value")
    else:
        val = v
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def encode_risk_factors(
    cvd_features,
    diseases_before=None,
    has_any_diag=False,
    risk_mean=None,
    risk_std=None,
) -> torch.Tensor:
    """Encode 29 interleaved value/missing pairs using training statistics."""
    values = []
    for feat in RISK_CONTINUOUS_FEATURES:
        val = _extract_risk_value(cvd_features, feat)
        if val is None:
            values.extend([0.0, 1.0])
        else:
            if risk_mean is not None and risk_std is not None:
                mean = risk_mean.get(feat, 0.0)
                std = risk_std.get(feat, 1.0)
                values.append((val - mean) / std if std > 0 else 0.0)
            else:
                values.append(val)
            values.append(0.0)
    sex = parse_sex(cvd_features.get("sex") if cvd_features else None)
    values.append(0.0 if sex is None else float(sex))
    values.append(1.0 if sex is None else 0.0)
    for dis in RISK_DISEASE_FEATURES:
        v = (
            diseases_before.get(dis)
            if diseases_before is not None and has_any_diag
            else None
        )
        if v is None:
            values.extend([0.0, 1.0])
        else:
            values.append(1.0 if v > 0 else -1.0)
            values.append(0.0)
    assert len(values) == RISK_FEATURE_DIM, (
        f"风险因子维度错误: {len(values)} != {RISK_FEATURE_DIM}"
    )
    return torch.tensor(values, dtype=torch.float32)


def compute_risk_feature_stats(samples):
    """Estimate continuous means/population standard deviations on training rows."""
    vals = {f: [] for f in RISK_CONTINUOUS_FEATURES}
    for s in samples:
        cvd = s.get("cvd_features")
        if not cvd:
            continue
        for f in RISK_CONTINUOUS_FEATURES:
            v = _extract_risk_value(cvd, f)
            if v is not None:
                vals[f].append(v)
    (mean, std) = ({}, {})
    for f in RISK_CONTINUOUS_FEATURES:
        arr = vals[f]
        mean[f] = float(np.mean(arr)) if len(arr) else 0.0
        std[f] = float(np.std(arr)) if len(arr) > 1 else 1.0
        if std[f] == 0:
            std[f] = 1.0
    return (mean, std)
