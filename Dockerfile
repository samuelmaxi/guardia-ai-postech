FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed \
    PORT=8501

RUN useradd --create-home appuser
WORKDIR /app

# dependências primeiro (camada em cache enquanto pyproject/uv.lock não mudarem)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

# modelo e índice gerados no build: o container sobe pronto
RUN python -m guardia.ml.train && python -W ignore -m guardia.rag.ingest

# só logs/ precisa ser gravável em runtime; chown -R em /app duplicaria a camada inteira (~3 GB)
RUN mkdir -p /app/logs && chown appuser /app/logs
USER appuser

EXPOSE 8501
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
