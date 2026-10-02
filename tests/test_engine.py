import json
import subprocess
import sys
from unittest.mock import Mock

import numpy as np
import pytest

from risk_engine.evaluation import evaluate
from risk_engine.features import features
from risk_engine.scoring import load_model, probabilities, risk_score, score_accounts
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
    assert all(s["risk_score"] == risk_score(s["probability"]) for s in scores)
    assert all(type(s["risk_score"]) is int and 0 <= s["risk_score"] <= 100 for s in scores)
    assert score_accounts([], artifact) == []
    assert probabilities([], artifact) == []
    low, high = probabilities(
        [
            {"days_since_last_login": 0, "open_tickets": 0, "monthly_spend_usd": 20000},
            {"days_since_last_login": 100, "open_tickets": 14, "monthly_spend_usd": 100},
        ],
        artifact,
    )
    assert low < high


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0, 0), (0.3949, 39), (0.395, 40), (0.6949, 69), (0.695, 70),
     (0.705, 71), (0.734, 73), (0.995, 100), (1, 100)],
)
def test_risk_score_rounds_half_up_at_boundaries(probability, expected):
    assert risk_score(probability) == expected


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_risk_score_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError):
        risk_score(probability)


def test_score_accounts_preserves_probability_and_order():
    model = Mock()
    model.predict_proba.return_value = np.array([[0.266, 0.734], [0.295, 0.705]])
    rows = [
        {"id": "a", "days_since_last_login": 1, "open_tickets": 2,
         "monthly_spend_usd": 100},
        {"id": "b", "days_since_last_login": 3, "open_tickets": 4,
         "monthly_spend_usd": 200},
    ]
    expected = [
        {"id": "a", "probability": 0.734, "risk_score": 73},
        {"id": "b", "probability": 0.705, "risk_score": 71},
    ]
    assert score_accounts(rows, {"model": model}) == expected
    assert score_accounts(rows, {"model": model}) == expected


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
    assert output["model_version"] == "churn-logreg-v1"
    assert [score["id"] for score in output["scores"]] == [row["id"] for row in rows]
    assert all(set(score) == {"id", "probability", "risk_score"} for score in output["scores"])
    assert all(score["risk_score"] == risk_score(score["probability"])
               for score in output["scores"])
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
