# -*- coding: utf-8 -*-
"""Pydantic-схемы API."""

from pydantic import BaseModel, Field


class AnonymizeRequest(BaseModel):
    """Запрос на анонимизацию."""

    text: str = Field(..., min_length=1, description="Исходный текст")
    threshold: float = Field(
        0.45, ge=0.1, le=0.95, description="Порог уверенности GLiNER"
    )
    categories: list[str] | None = Field(
        None,
        description="Категории для маскирования (None = все). См. GET /categories",
    )
    check_injection: bool = Field(
        True,
        description="Проверка на промпт-инъекции через guardrails-сервис",
    )
    vault_key: str | None = Field(
        None,
        description="Ключ vault-сессии (UUID). None = новая сессия, "
        "сервис вернёт новый vault_key в ответе",
    )


class GuardrailsResult(BaseModel):
    """Результат проверки guardrails (промпт-инъекции)."""

    status: str  # ok | unavailable | disabled
    decision: str | None = None  # allow | review | block
    score: float | None = None
    matched_rules: list[dict] = Field(default_factory=list)


class EntitySpan(BaseModel):
    """Найденная сущность."""

    start: int
    end: int
    text: str
    category: str
    placeholder: str
    source: str  # regex | gliner | natasha
    score: float | None = None


class AnonymizeResponse(BaseModel):
    """Ответ анонимизации."""

    anonymized_text: str
    entities: list[EntitySpan]
    stats: dict[str, int]  # категория -> количество
    blocked: bool = False  # True = текст не обработан (инъекция, mode=block)
    guardrails: GuardrailsResult | None = None
    vault_key: str | None = None  # ключ vault-сессии для /deanonymize


class DeanonymizeRequest(BaseModel):
    """Запрос деанонимизации (плейсхолдеры -> значения)."""

    text: str = Field(..., min_length=1, max_length=40_000)
    vault_key: str = Field(..., description="Ключ vault-сессии из /anonymize")


class DeanonymizeResponse(BaseModel):
    """Ответ деанонимизации."""

    text: str
    replaced: int  # сколько плейсхолдеров заменено


class HealthResponse(BaseModel):
    status: str
    gliner_loaded: bool
    natasha_loaded: bool
    device: str
