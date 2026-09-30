"""Train a small churn baseline and generate a separate held-out evaluation set."""

import json
import random
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from risk_engine.features import features

ROOT = Path(__file__).resolve().parents[1]
MODEL_VERSION = "churn-logreg-v1"
MODEL_PATH = ROOT / "models/churn.joblib"


def labelled(seed, count):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        days = rng.randint(0, 100)
        tickets = rng.randint(0, 14)
        spend = round(rng.uniform(100, 25000), 2)
        # Synthetic labels encode inactivity and support load with mild noise.
        latent = days / 20 + tickets / 5 - spend / 15000 + rng.gauss(0, 0.35)
        rows.append(
            {
                "id": f"label_{seed}_{i:04d}",
                "days_since_last_login": days,
                "open_tickets": tickets,
                "monthly_spend_usd": spend,
                "churned": int(latent > 3.2),
            }
        )
    return rows


def train():
    rows = labelled(858, 1000)
    model = make_pipeline(StandardScaler(), LogisticRegression(random_state=858, max_iter=1000))
    model.fit(features(rows), [row["churned"] for row in rows])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "version": MODEL_VERSION}, MODEL_PATH)
    evaluation = ROOT / "eval/labelled.json"
    evaluation.write_text(json.dumps(labelled(859, 100), indent=2) + "\n")
    print(json.dumps({"model_version": MODEL_VERSION, "training_rows": len(rows)}))


if __name__ == "__main__":
    train()
