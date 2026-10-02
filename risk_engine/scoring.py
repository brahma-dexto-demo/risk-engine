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
    """Convert a finite churn probability to a half-up integer percentage."""
    if (
        isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(probability)
        or not 0 <= probability <= 1
    ):
        raise ValueError("Churn probability must be a finite number in [0, 1]")
    return math.floor(probability * 100 + 0.5)


def score_accounts(accounts, artifact):
    return [
        {"id": row["id"], "probability": probability, "risk_score": risk_score(probability)}
        for row, probability in zip(accounts, probabilities(accounts, artifact), strict=True)
    ]
