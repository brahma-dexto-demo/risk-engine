FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY risk_engine/ ./risk_engine/
COPY data/ ./data/
COPY models/ ./models/
COPY eval/ ./eval/
RUN uv sync --frozen --no-dev && useradd --uid 10001 --create-home app && chown -R app:app /app
USER app
CMD ["sh", "-c", "python -m risk_engine.job score && python -m risk_engine.job eval"]
