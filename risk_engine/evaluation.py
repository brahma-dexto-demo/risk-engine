"""Evaluate AUC, probability bounds and reproducible inference."""

import math

from sklearn.metrics import roc_auc_score

from risk_engine.scoring import probabilities


def evaluate(rows, artifact, thresholds):
    scores = probabilities(rows, artifact)
    metrics = {
        "auc": float(roc_auc_score([row["churned"] for row in rows], scores)),
        "score_range_check": all(math.isfinite(p) and 0 <= p <= 1 for p in scores),
        "determinism_check": scores == probabilities(rows, artifact),
    }
    metrics["passed"] = (
        metrics["auc"] >= thresholds["min_auc"]
        and metrics["score_range_check"]
        and metrics["determinism_check"]
    )
    return metrics
