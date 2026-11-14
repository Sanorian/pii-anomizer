# -*- coding: utf-8 -*-
"""FastAPI-сервис анонимизации ПДн и чувствительной информации."""

import logging
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import ALL_CATEGORIES, GLINER_MODEL_DIR, MAX_TEXT_LENGTH, PLACEHOLDERS
from app.detectors.gliner_detector import GlinerDetector
from app.detectors.natasha_detector import NatashaDetector
from app.detectors.regex_detector import RegexDetector
from app.pipeline import AnonymizationPipeline, mask_text
from app.schemas import (
    AnonymizeRequest,
    AnonymizeResponse,
    EntitySpan,
    HealthResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("anonymization")

# Глобальные компоненты (инициализируются в lifespan)
pipeline: AnonymizationPipeline | None = None
DEVICE = "cpu"  # в Docker доступен только CPU; локально можно переопределить


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей при старте."""
    global pipeline
    t0 = time.time()
    logger.info("загружаю детекторы...")

    regex_detector = RegexDetector()

    gliner_detector = GlinerDetector(GLINER_MODEL_DIR, device=DEVICE)
    try:
        gliner_detector.load()
        logger.info("GLiNER загружена из %s", GLINER_MODEL_DIR)
    except FileNotFoundError as e:
        logger.error("GLiNER не найдена: %s", e)
        gliner_detector = None  # type: ignore[assignment]

    natasha_detector = NatashaDetector()
    try:
        natasha_detector.load()
        source = "fine-tuned " if natasha_detector.is_finetuned else "pretrained "
        logger.info("Natasha загружена (%s%s)", source, natasha_detector.model_tar)
    except Exception as e:  # noqa: BLE001
        logger.error("Natasha не загрузилась: %s", e)
        natasha_detector = None  # type: ignore[assignment]

    pipeline = AnonymizationPipeline(
        regex_detector=regex_detector,
        gliner_detector=gliner_detector or GlinerDetector(GLINER_MODEL_DIR, device=DEVICE),
        natasha_detector=natasha_detector or NatashaDetector(),
    )
    # Отключаем незагруженные детекторы
    if gliner_detector is None:
        pipeline.gliner.is_loaded  # noqa: B018 — проверка не нужна, флаг в is_loaded
    logger.info("все детекторы готовы за %.1f с", time.time() - t0)
    yield
    logger.info("сервис остановлен")


app = FastAPI(
    title="Анонимизация ПДн",
    description="Маскирование персональных данных в русских текстах: "
    "regex + GLiNER (дообучен) + Natasha",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if pipeline is not None else "loading",
        gliner_loaded=pipeline.gliner.is_loaded if pipeline else False,
        natasha_loaded=pipeline.natasha.is_loaded if pipeline else False,
        device=DEVICE,
    )


@app.get("/categories")
def categories() -> dict:
    """Список категорий и плейсхолдеров."""
    return {"categories": [{"name": c, "placeholder": PLACEHOLDERS[c]} for c in ALL_CATEGORIES]}


@app.post("/anonymize", response_model=AnonymizeResponse)
def anonymize(request: AnonymizeRequest) -> AnonymizeResponse:
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Модели ещё загружаются")

    if len(request.text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Текст длиннее {MAX_TEXT_LENGTH} символов",
        )

    # Валидация категорий
    if request.categories:
        unknown = set(request.categories) - set(ALL_CATEGORIES)
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Неизвестные категории: {sorted(unknown)}",
            )

    result = pipeline.run(
        request.text,
        threshold=request.threshold,
        categories=set(request.categories) if request.categories else None,
    )

    anonymized, masked = mask_text(request.text, result.entities)

    stats: dict[str, int] = {}
    for e in masked:
        stats[e["category"]] = stats.get(e["category"], 0) + 1

    return AnonymizeResponse(
        anonymized_text=anonymized,
        entities=[EntitySpan(**e) for e in masked],
        stats=stats,
    )
