# Guardrails: FastAPI + torch CPU (torch ставим с CPU-индекса для лёгкого образа)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models_cache/hf

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
# torch с CPU-индекса (без CUDA, ~200 MB вместо 2 GB)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Модель качается при старте в volume (см. docker-compose.yml):
# gbv/mdeberta-ru-prompt-injection (~600 MB)
ENV GUARDRAILS_MODEL_DIR="" \
    GUARDRAILS_BLOCK_THRESHOLD=0.7 \
    GUARDRAILS_REVIEW_THRESHOLD=0.4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
