"""Load the trusted, repository-owned churn model."""

import math

import joblib

from risk_engine.features import features
from risk_engine.train import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def probabilities(accounts, artifact):
    if not accounts:
        return []
    return artifact["model"].predict_proba(features(accounts))[:, 1].tolist()


def risk_score(probability):
    """Convert a probability to an integer percentage, rounding halves upward."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("probability must be finite and between 0 and 1")
    return max(0, min(100, math.floor(probability * 100 + 0.5)))


def score_accounts(accounts, artifact):
    return [
        {"id": row["id"], "probability": probability, "risk_score": risk_score(probability)}
        for row, probability in zip(accounts, probabilities(accounts, artifact), strict=True)
    ]
