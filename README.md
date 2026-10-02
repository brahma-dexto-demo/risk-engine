# Risk Engine

Batch scoring engine for customer accounts. A logistic regression model scores each account's churn probability (0–1) and writes the results to S3.

This is a demo repository with synthetic, fictional accounts; it contains no real customer data.

## Local development

```sh
uv sync --locked
uv run python -m risk_engine.train
uv run python -m risk_engine.job score
uv run python -m risk_engine.job eval
uv run pytest
uv run ruff check .
python scripts/generate_accounts.py
```

Run from the repository root with Python 3.12. Training uses 1,000 seeded synthetic
labels (seed 858) and generates 100 independent evaluation rows (seed 859).
`models/churn.joblib` is a trusted, committed artifact; only load artifacts you
control. Feature ordering is fixed in `risk_engine/features.py`. Retrain and
commit the model with the lockfile if the scikit-learn version changes.

`LOCAL_DATA_DIR` defaults to `./data`. `DATA_BUCKET` takes precedence and selects
S3 with the default AWS credential chain. Scoring reads `accounts/accounts.json`
and writes `scores/latest.json` containing `model_version`, UTC `generated_at`,
and `scores` entries with `id`, `probability` in [0, 1], and integer
`risk_score` in [0, 100]. Probabilities are
reproducible; the run timestamp changes. Output files are not committed.
Evaluation always uses bundled `eval/labelled.json` and `eval/thresholds.json`,
prints AUC, bounds, determinism and pass status as JSON, and exits 1 on failure.
The synthetic AUC threshold is 0.80; this is a demo gate, not real model validation.
The container runs score followed by eval. See [DEPLOY.md](DEPLOY.md) for staging.


## Shared risk score contract

The risk engine owns the calculation: `risk_score = floor(probability * 100 + 0.5)`
(round half up, not Python's ties-to-even `round`). Each entry retains the original
churn probability, for example:

```json
{"id": "acc-001", "probability": 0.734, "risk_score": 73}
```

`model_version` and UTC `generated_at` remain at the artifact root. Scores are
integers (including endpoints 0 and 100), deterministic for the same model/input;
timestamps are not. Non-numeric, non-finite, or out-of-range probabilities fail
scoring before writing the artifact; they are never clamped or defaulted to zero.
The committed model and its feature ordering are unchanged.

Consumer contract (implemented separately in accounts-api and ops-console):
- API joins scores by account `id`, exposing `risk_score: integer | null` on
  `GET /accounts` and `GET /accounts/{id}`. Missing score file/id means null;
  malformed artifacts and non-missing storage failures return 503, not zero/null.
- `GET /accounts?high_risk=true` filters `risk_score >= 70` before pagination and
  total calculation, combined with existing `q` and `industry`; default is false.
- Console maps the field to nullable Integer. Badges include score and band:
  Low green (0–39), Medium amber (40–69), High red (70–100), Unknown neutral (null).
  The high-risk filter uses API `high_risk` and persists across search, industry,
  and pagination changes.

Roll out/merge **producer → API → console**. Fields are additive; existing
probability consumers remain compatible. Run the producer to refresh
`scores/latest.json` before enabling consumers. No retraining is required.
