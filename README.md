# Risk Engine

Batch scoring engine for customer accounts. A logistic regression model scores each account's churn probability (0–1) and writes the results to S3.

This is a demo repository with synthetic, fictional accounts; it contains no real customer data.

## Local development

```sh
uv sync
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
and `scores` entries with `id` and `probability` in [0, 1]. Probabilities are
reproducible; the run timestamp changes. Output files are not committed.
Evaluation always uses bundled `eval/labelled.json` and `eval/thresholds.json`,
prints AUC, bounds, determinism and pass status as JSON, and exits 1 on failure.
The synthetic AUC threshold is 0.80; this is a demo gate, not real model validation.
The container runs score followed by eval. See [DEPLOY.md](DEPLOY.md) for staging.
