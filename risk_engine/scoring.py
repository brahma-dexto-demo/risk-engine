"""Load the trusted, repository-owned churn model."""

import joblib

from risk_engine.features import features
from risk_engine.train import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def probabilities(accounts, artifact):
    if not accounts:
        return []
    return artifact["model"].predict_proba(features(accounts))[:, 1].tolist()


def score_accounts(accounts, artifact):
    return [
        {"id": row["id"], "probability": probability}
        for row, probability in zip(accounts, probabilities(accounts, artifact), strict=True)
    ]
