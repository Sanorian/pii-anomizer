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


class HealthResponse(BaseModel):
    status: str
    gliner_loaded: bool
    natasha_loaded: bool
    device: str
