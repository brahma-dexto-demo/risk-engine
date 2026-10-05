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


def risk_score(probability: float) -> int:
    """Convert a finite probability to a clamped, half-up integer percentage."""
    if not math.isfinite(probability):
        raise ValueError("Probability must be finite")
    return math.floor(min(1.0, max(0.0, probability)) * 100 + 0.5)


def score_accounts(accounts, artifact):
    return [
        {
            "id": row["id"],
            "probability": probability,
            "risk_score": risk_score(probability),
        }
        for row, probability in zip(accounts, probabilities(accounts, artifact), strict=True)
    ]
