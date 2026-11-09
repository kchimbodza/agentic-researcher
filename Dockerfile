# Agentic Researcher container. Documents are mounted at /data (never baked into the image).
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    AR_DATA_DIR=/data HOME=/home/app

RUN useradd --create-home --uid 1000 app
WORKDIR /srv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY models ./models
RUN mkdir -p /srv/logs /data/datasheets /data/papers && chown -R app /srv/logs /data

USER app
EXPOSE 8770
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8770/healthz')" || exit 1
CMD ["/srv/.venv/bin/uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8770"]
