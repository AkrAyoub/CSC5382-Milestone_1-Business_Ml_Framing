from __future__ import annotations

import math

import pandas as pd


def confusion_counts(y_true: list[int], y_pred: list[int]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for actual, pred in zip(y_true, y_pred):
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def safe_div(num: float, denom: float) -> float:
    return 0.0 if denom == 0 else num / denom


def roc_auc_score_from_ranks(y_true: list[int], y_score: list[float]) -> float:
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Rank-based AUC keeps the dependency surface small while remaining stable
    # for offline comparison runs.
    ranked = pd.Series(y_score).rank(method="average")
    rank_sum_pos = float(ranked[pd.Series(y_true) == 1].sum())
    return (rank_sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)


def log_loss(y_true: list[int], y_score: list[float], eps: float = 1e-12) -> float:
    total = 0.0
    for actual, score in zip(y_true, y_score):
        clipped = min(max(score, eps), 1.0 - eps)
        total += -(actual * math.log(clipped) + (1 - actual) * math.log(1.0 - clipped))
    return total / max(1, len(y_true))


def binary_classification_report(
    y_true: list[int],
    y_score: list[float],
    threshold: float,
) -> dict[str, float | int]:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    counts = confusion_counts(y_true, y_pred)

    precision = safe_div(counts["tp"], counts["tp"] + counts["fp"])
    recall = safe_div(counts["tp"], counts["tp"] + counts["fn"])
    accuracy = safe_div(counts["tp"] + counts["tn"], len(y_true))
    f1 = safe_div(2.0 * precision * recall, precision + recall)

    return {
        **counts,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score_from_ranks(y_true, y_score),
        "log_loss": log_loss(y_true, y_score),
        "support": len(y_true),
        "positive_rate": safe_div(sum(y_true), len(y_true)),
        "threshold": threshold,
    }
