"""VSI-Bench scoring: exact-match accuracy for MC, Mean Relative Accuracy for numerical."""
import re

import numpy as np

from .data import NUMERICAL_TYPES


def parse_letter(text, options):
    text = text.strip()
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1)
    # match against option bodies
    low = text.lower()
    for opt in options:
        letter, _, body = opt.partition(". ")
        if body and body.lower() in low:
            return letter
    return None


def parse_number(text):
    m = re.search(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(m.group()) if m else None


def score_row(row, pred_text):
    qt = row["question_type"]
    if qt in NUMERICAL_TYPES:
        gt = float(row["ground_truth"])
        pred = parse_number(pred_text)
        if pred is None:
            return 0.0
        rel = abs(pred - gt) / max(abs(gt), 1e-8)
        thetas = np.arange(0.5, 1.0, 0.05)
        return float(np.mean(rel < (1.0 - thetas)))
    else:
        letter = parse_letter(pred_text, row["options"] or [])
        return float(letter == row["ground_truth"].strip())


def aggregate(rows, scores):
    by_type = {}
    for r, s in zip(rows, scores):
        by_type.setdefault(r["question_type"], []).append(s)
    out = {k: float(np.mean(v)) for k, v in sorted(by_type.items())}
    out["OVERALL_mean_of_types"] = float(np.mean(list(out.values())))
    out["OVERALL_micro"] = float(np.mean(scores))
    out["n"] = len(scores)
    return out
