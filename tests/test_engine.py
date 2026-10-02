import json
import subprocess
import sys

import numpy as np

from risk_engine.evaluation import evaluate
from risk_engine.features import features
from risk_engine.scoring import load_model, probabilities, score_accounts
from risk_engine.train import ROOT, labelled


def test_feature_order():
    np.testing.assert_array_equal(
        features([{"days_since_last_login": 30, "open_tickets": 2, "monthly_spend_usd": 400}]),
        [[30, 2, 400]],
    )


def test_scoring_is_deterministic_and_bounded():
    rows = labelled(859, 100)
    artifact = load_model()
    scores = score_accounts(rows, artifact)
    assert scores == score_accounts(rows, artifact)
    assert [s["id"] for s in scores] == [r["id"] for r in rows]
    assert all(0 <= s["probability"] <= 1 for s in scores)
    assert probabilities([], artifact) == []
    low, high = probabilities(
        [
            {"days_since_last_login": 0, "open_tickets": 0, "monthly_spend_usd": 20000},
            {"days_since_last_login": 100, "open_tickets": 14, "monthly_spend_usd": 100},
        ],
        artifact,
    )
    assert low < high


def test_evaluation_pass_and_fail():
    rows = json.loads((ROOT / "eval/labelled.json").read_text())
    artifact = load_model()
    assert evaluate(rows, artifact, {"min_auc": 0.80})["passed"]
    assert not evaluate(rows, artifact, {"min_auc": 1.01})["passed"]
    reversed_rows = [{**row, "churned": 1 - row["churned"]} for row in rows]
    assert not evaluate(reversed_rows, artifact, {"min_auc": 0.80})["passed"]


def test_job_score_and_eval(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    (tmp_path / "accounts").mkdir()
    rows = labelled(860, 3)
    (tmp_path / "accounts/accounts.json").write_text(json.dumps(rows))
    subprocess.run([sys.executable, "-m", "risk_engine.job", "score"], check=True)
    output = json.loads((tmp_path / "scores/latest.json").read_text())
    assert len(output["scores"]) == 3
    assert output["generated_at"]
    assert output["model_version"] == "churn-logreg-v2"
    result = subprocess.run(
        [sys.executable, "-m", "risk_engine.job", "eval"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["passed"]


def test_eval_cli_exits_nonzero_on_threshold_failure(tmp_path, monkeypatch, capsys):
    from risk_engine import job

    (tmp_path / "eval").mkdir()
    (tmp_path / "eval/labelled.json").write_text((ROOT / "eval/labelled.json").read_text())
    (tmp_path / "eval/thresholds.json").write_text('{"min_auc": 1.01}')
    monkeypatch.setattr(job, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["job", "eval"])
    assert job.main() == 1
    assert not json.loads(capsys.readouterr().out)["passed"]


def test_demo_accounts_get_a_spread_of_probabilities():
    accounts = json.loads((ROOT / "data/accounts/accounts.json").read_text())
    scores = probabilities(accounts, load_model())
    high = sum(p >= 0.7 for p in scores) / len(scores)
    saturated = sum(p >= 0.995 or p <= 0.005 for p in scores) / len(scores)
    assert 0.15 <= high <= 0.35
    assert saturated <= 0.05
