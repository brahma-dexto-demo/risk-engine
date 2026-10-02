import json
import math
import subprocess
import sys

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
    assert all(type(s["risk_score"]) is int and 0 <= s["risk_score"] <= 100 for s in scores)
    assert all(s["risk_score"] == math.floor(s["probability"] * 100 + 0.5) for s in scores)
    assert [s["probability"] for s in scores] == probabilities(rows, artifact)
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
    assert output["scores"] == score_accounts(rows, load_model())
    assert all(set(s) == {"id", "probability", "risk_score"} for s in output["scores"])
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


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0, 0), (0.0049, 0), (0.005, 1), (0.385, 39), (0.3949, 39),
     (0.395, 40), (0.6849, 68), (0.685, 69), (0.6949, 69), (0.695, 70),
     (0.734, 73), (0.985, 99), (0.9949, 99), (0.995, 100), (1, 100)],
)
def test_risk_score_half_up_boundaries(probability, expected):
    assert risk_score(probability) == expected
    assert type(risk_score(probability)) is int


@pytest.mark.parametrize(
    "probability",
    [-0.001, 1.001, float("nan"), float("inf"), -float("inf"), None, "0.5", True, False],
)
def test_risk_score_rejects_invalid_probabilities(probability):
    with pytest.raises(ValueError, match="finite number"):
        risk_score(probability)


def test_scoring_contract_example(monkeypatch):
    monkeypatch.setattr("risk_engine.scoring.probabilities", lambda rows, artifact: [0.734])
    assert score_accounts([{"id": "acc-001"}], {}) == [
        {"id": "acc-001", "probability": 0.734, "risk_score": 73}
    ]


def test_invalid_model_probability_does_not_overwrite_scores(tmp_path, monkeypatch):
    from risk_engine import job

    monkeypatch.delenv("DATA_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    (tmp_path / "accounts").mkdir()
    (tmp_path / "accounts/accounts.json").write_text('[{"id": "acc-001"}]')
    (tmp_path / "scores").mkdir()
    output = tmp_path / "scores/latest.json"
    output.write_text('{"previous": true}')
    monkeypatch.setattr(job, "load_model", lambda: {"version": "test"})
    monkeypatch.setattr("risk_engine.scoring.probabilities", lambda rows, artifact: [float("nan")])
    monkeypatch.setattr(sys, "argv", ["job", "score"])
    with pytest.raises(ValueError, match="finite number"):
        job.main()
    assert output.read_text() == '{"previous": true}'
