# Сервис анонимизации ПДн: FastAPI + GLiNER + Natasha + regex
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Системные зависимости для torch (минимум) и сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Сначала requirements — для кэширования слоя
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY app/ ./app/

# Модели монтируются volume-ом в /models (см. docker-compose.yml).
# GLiNER и Natasha загружаются из этих путей:
ENV GLINER_MODEL_DIR=/models/gliner_ru_finetuned_v2 \
    NATASHA_CACHE=/models/natasha_cache

# Natasha качает эмбеддинги при первом запуске — направим кэш в volume
ENV HOME=/tmp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
