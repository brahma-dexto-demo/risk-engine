import io
import json
import math
import sys
from datetime import datetime
from unittest.mock import Mock

import pytest

from risk_engine import job, scoring
from risk_engine.scoring import risk_score


@pytest.mark.parametrize(
    ("probability", "expected"),
    [
        (-1e308, 0), (-1, 0), (-0.0, 0), (0, 0),
        (0.0049, 0), (0.005, 1), (0.125, 13), (0.395, 40),
        (0.6949, 69), (0.695, 70), (0.734, 73), (0.745, 75),
        (0.995, 100), (1, 100), (2, 100), (1e308, 100),
    ],
)
def test_clamping_and_rounding_examples(probability, expected):
    assert risk_score(probability) == expected
    assert type(risk_score(probability)) is int


@pytest.mark.parametrize("integer", range(101))
def test_every_integer_and_interior_of_each_bucket(integer):
    assert risk_score(integer / 100) == integer
    assert risk_score((integer - 0.49) / 100) == integer
    assert risk_score((integer + 0.49) / 100) == integer


@pytest.mark.parametrize("integer", range(100))
def test_every_rounding_boundary(integer):
    midpoint = (integer + 0.5) / 100
    assert risk_score(midpoint - 1e-12) == integer
    assert risk_score(midpoint + 1e-12) == integer + 1
    # Follow the specified floating-point expression even at inexact decimal ties.
    assert risk_score(midpoint) == math.floor(midpoint * 100 + 0.5)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_probability_is_rejected(invalid):
    with pytest.raises(ValueError, match="finite"):
        risk_score(invalid)


def test_dense_grid_is_bounded_monotone_and_deterministic():
    inputs = [i / 10000 for i in range(-10000, 20001)]
    scores = [risk_score(p) for p in inputs]
    assert scores == [risk_score(p) for p in inputs]
    assert scores == sorted(scores)
    assert set(scores) == set(range(101))
    assert all(type(value) is int for value in scores)


def test_conversion_retains_id_probability_and_order(monkeypatch):
    monkeypatch.setattr(scoring, "probabilities", lambda rows, artifact: [0.734, -0.1, 1.1])
    assert scoring.score_accounts([{"id": "acct-001"}, {"id": "b"}, {"id": "c"}], {}) == [
        {"id": "acct-001", "probability": 0.734, "risk_score": 73},
        {"id": "b", "probability": -0.1, "risk_score": 0},
        {"id": "c", "probability": 1.1, "risk_score": 100},
    ]


def test_batch_s3_output_contract(monkeypatch):
    monkeypatch.setenv("DATA_BUCKET", "batch-test")
    client = Mock()
    rows = [{"id": "acct-001", "days_since_last_login": 30,
             "open_tickets": 2, "monthly_spend_usd": 400}]
    client.get_object.return_value = {"Body": io.BytesIO(json.dumps(rows).encode())}
    monkeypatch.setattr("risk_engine.storage.boto3.client", lambda service: client)
    monkeypatch.setattr(sys, "argv", ["job", "score"])
    artifact = scoring.load_model()
    assert job.main() == 0
    client.get_object.assert_called_once_with(Bucket="batch-test", Key="accounts/accounts.json")
    request = client.put_object.call_args.kwargs
    assert request["Bucket"] == "batch-test"
    assert request["Key"] == "scores/latest.json"
    assert request["ContentType"] == "application/json"
    output = json.loads(request["Body"])
    assert set(output) == {"model_version", "generated_at", "scores"}
    assert output["model_version"] == artifact["version"]
    assert datetime.fromisoformat(output["generated_at"]).utcoffset().total_seconds() == 0
    assert output["scores"] == scoring.score_accounts(rows, artifact)
    assert type(output["scores"][0]["risk_score"]) is int


def test_invalid_probability_does_not_overwrite_latest(tmp_path, monkeypatch):
    monkeypatch.delenv("DATA_BUCKET", raising=False)
    monkeypatch.setenv("LOCAL_DATA_DIR", str(tmp_path))
    (tmp_path / "accounts").mkdir()
    (tmp_path / "accounts/accounts.json").write_text('[{"id":"acct-001"}]')
    (tmp_path / "scores").mkdir()
    latest = tmp_path / "scores/latest.json"
    latest.write_text('{"previous":"valid output"}')
    monkeypatch.setattr(scoring, "probabilities", lambda rows, artifact: [math.nan])
    monkeypatch.setattr(sys, "argv", ["job", "score"])
    with pytest.raises(ValueError, match="finite"):
        job.main()
    assert latest.read_text() == '{"previous":"valid output"}'
