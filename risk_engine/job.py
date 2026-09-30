"""Batch entry points: score customer accounts or evaluate the baseline."""

import argparse
import json
from datetime import datetime, timezone

from risk_engine.evaluation import evaluate
from risk_engine.scoring import load_model, score_accounts
from risk_engine.storage import JsonStore
from risk_engine.train import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["score", "eval"])
    args = parser.parse_args()
    artifact = load_model()
    if args.command == "score":
        store = JsonStore()
        accounts = store.read("accounts/accounts.json")
        result = {
            "model_version": artifact["version"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "scores": score_accounts(accounts, artifact),
        }
        store.write("scores/latest.json", result)
        print(json.dumps({"accounts_scored": len(accounts), "model_version": artifact["version"]}))
        return 0
    rows = json.loads((ROOT / "eval/labelled.json").read_text())
    thresholds = json.loads((ROOT / "eval/thresholds.json").read_text())
    metrics = evaluate(rows, artifact, thresholds)
    print(json.dumps(metrics))
    return 0 if metrics["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
