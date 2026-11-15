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
from app.guardrails_client import GUARDRAILS_ENABLED, GUARDRAILS_MODE, scan_text
from app.pipeline import AnonymizationPipeline
from app.schemas import (
    AnonymizeRequest,
    AnonymizeResponse,
    DeanonymizeRequest,
    DeanonymizeResponse,
    EntitySpan,
    GuardrailsResult,
    HealthResponse,
)
from app.vault import (
    VAULT_TTL_HOURS,
    PostgresVault,
    Vault,
    deanonymize_text,
    get_vault,
    is_valid_vault_key,
    new_vault_key,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("anonymization")

# Глобальные компоненты (инициализируются в lifespan)
pipeline: AnonymizationPipeline | None = None
vault: Vault | None = None
DEVICE = "cpu"  # в Docker доступен только CPU; локально можно переопределить


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей при старте."""
    global pipeline, vault
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

    # Vault: memory или postgres (см. VAULT_BACKEND)
    vault = get_vault()
    if isinstance(vault, PostgresVault):
        vault.init_schema()
        vault.cleanup()
        logger.info("vault: postgres (%s)", "да" if vault.exists("00000000-0000-0000-0000-000000000000") else "проверено")
    else:
        logger.info("vault: memory (TTL %s ч)", VAULT_TTL_HOURS)
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
    if vault is None:
        raise HTTPException(status_code=503, detail="Vault ещё инициализируется")

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

    # --- Guardrails: проверка на промпт-инъекции (до анонимизации) ---
    gr_result: GuardrailsResult | None = None
    if request.check_injection:
        scan = scan_text(request.text)
        if scan is None:
            gr_result = GuardrailsResult(
                status="disabled" if not GUARDRAILS_ENABLED else "unavailable"
            )
        else:
            gr_result = GuardrailsResult(
                status="ok",
                decision=scan["decision"],
                score=scan["score"],
                matched_rules=scan["matched_rules"],
            )
            if GUARDRAILS_MODE == "block" and scan["decision"] == "block":
                return AnonymizeResponse(
                    anonymized_text="",
                    entities=[],
                    stats={},
                    blocked=True,
                    guardrails=gr_result,
                )

    # --- Vault: ключ сессии ---
    # Без vault_key создаём новую сессию; с существующим — проверяем.
    # Просроченный/неизвестный ключ -> 404 (клиент начинает новую сессию).
    if request.vault_key is None:
        vault_key = new_vault_key()
    else:
        if not is_valid_vault_key(request.vault_key):
            raise HTTPException(status_code=422, detail="vault_key должен быть UUID")
        if not vault.exists(request.vault_key):
            raise HTTPException(status_code=404, detail="vault-сессия не найдена или истекла")
        vault_key = request.vault_key

    result = pipeline.run(
        request.text,
        threshold=request.threshold,
        categories=set(request.categories) if request.categories else None,
    )

    # --- Номерованные плейсхолдеры через vault ---
    # Стабильные ID: та же сущность в сессии -> тот же [CATEGORY_N].
    # Плейсхолдеры из текста запроса (предыдущий ответ LLM) деанонимизируем,
    # чтобы «назови его [PERSON_1] снова» не создало сущность "[PERSON_1]".
    anonymized = request.text
    masked: list[dict] = []
    stats: dict[str, int] = {}
    for e in sorted(result.entities, key=lambda x: x["start"]):
        original = request.text[e["start"] : e["end"]]
        placeholder = vault.get_or_assign(vault_key, e["category"], original)
        masked.append({
            **e,
            "text": original,
            "placeholder": placeholder,
            "key": vault_key,
        })

    # Замена по тексту справа налево, чтобы оффсеты не съезжали
    for e in reversed(masked):
        anonymized = anonymized[: e["start"]] + e["placeholder"] + anonymized[e["end"] :]

    for e in masked:
        stats[e["category"]] = stats.get(e["category"], 0) + 1

    return AnonymizeResponse(
        anonymized_text=anonymized,
        entities=[EntitySpan(**e) for e in masked],
        stats=stats,
        blocked=False,
        guardrails=gr_result,
        vault_key=vault_key,
    )


@app.post("/deanonymize", response_model=DeanonymizeResponse)
def deanonymize(request: DeanonymizeRequest) -> DeanonymizeResponse:
    """Деанонимизация: плейсхолдеры -> значения по ключу vault-сессии.

    Неизвестные плейсхолдеры остаются как есть (текст мог не содержать
    ПДн или LLM придумал плейсхолдер).
    """
    if vault is None:
        raise HTTPException(status_code=503, detail="Vault ещё инициализируется")
    if not is_valid_vault_key(request.vault_key):
        raise HTTPException(status_code=422, detail="vault_key должен быть UUID")
    if not vault.exists(request.vault_key):
        raise HTTPException(status_code=404, detail="vault-сессия не найдена или истекла")

    text, replaced = deanonymize_text(request.text, lambda ph: vault.resolve(request.vault_key, ph))
    return DeanonymizeResponse(text=text, replaced=replaced)
