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

## Score compatibility contract

The engine owns conversion: `risk_score = floor(clamp(probability, 0, 1) * 100 + 0.5)`.
This is half-up rounding, not Python's ties-to-even `round`. Non-finite values
fail the scoring job before it writes output. Conversion never replaces or rounds
the original probability. Example entry:

```json
{"id": "acct-001", "probability": 0.734, "risk_score": 73}
```

The new field is additive: `id`, `probability`, `model_version`, `generated_at`,
the `scores` array and `scores/latest.json` key retain their existing meanings.
No model retraining or Batch definition change is needed. Consumers join by
account id, never position. Deploy/merge in order: **engine → API → console**;
run scoring to refresh the artifact before enabling dependent consumers.

The API exposes `risk_score` as integer or null. Null means the scoring file is
absent or the account has no entry; it is never a zero fallback. Invalid/corrupt
scores or permission/network failures must surface as errors, not absence.
`high_risk=true` selects non-null scores >=70 before pagination, combined with
`q` and `industry`. Console bands are Low 0–39, Medium 40–69, High 70–100;
null is Not scored. API and console implementation live in their own repositories.
