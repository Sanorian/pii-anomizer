# -*- coding: utf-8 -*-
"""Пайплайн анонимизации: 3 детектора -> слияние -> маскирование."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import PLACEHOLDERS
from app.detectors.gliner_detector import GlinerDetector
from app.detectors.natasha_detector import NatashaDetector
from app.detectors.regex_detector import RegexDetector

# Приоритет источников при конфликте пересечений
SOURCE_PRIORITY = {"regex": 3, "gliner": 2, "natasha": 1}

# Вес категорий внутри одного источника: специфичный контекст (ВУ/полис)
# переигрывает дефолтный паспортный формат
CATEGORY_WEIGHT = {
    "doc_number": 1.1,  # найден по контексту документа — специфичнее паспорта
    "oms": 1.1,         # найден по контексту "полис/ОМС" — специфичнее карты
}


@dataclass
class PipelineResult:
    """Результат пайплайна до маскирования."""

    entities: list[dict] = field(default_factory=list)


class AnonymizationPipeline:
    """Оркестрирует детекторы и сливает их результаты (union-стратегия)."""

    def __init__(
        self,
        regex_detector: RegexDetector,
        gliner_detector: GlinerDetector,
        natasha_detector: NatashaDetector,
    ) -> None:
        self.regex = regex_detector
        self.gliner = gliner_detector
        self.natasha = natasha_detector

    def run(
        self,
        text: str,
        threshold: float = 0.45,
        categories: set[str] | None = None,
    ) -> PipelineResult:
        """Прогоняет текст через все детекторы и сливает сущности.

        categories: None = все категории; иначе только указанные.
        """
        merged: list[dict] = []

        # 1. Regex — всегда
        for e in self.regex.detect(text):
            merged.append({**e, "source": "regex"})

        # 2. GLiNER
        if self.gliner.is_loaded:
            for e in self.gliner.detect(text, threshold=threshold):
                merged.append({**e, "source": "gliner"})

        # 3. Natasha
        if self.natasha.is_loaded:
            for e in self.natasha.detect(text):
                merged.append({**e, "source": "natasha"})

        # Фильтр по категориям
        if categories is not None:
            merged = [e for e in merged if e["category"] in categories]

        # Слияние пересечений
        merged = self._resolve_overlaps(merged)
        return PipelineResult(entities=merged)

    @staticmethod
    def _resolve_overlaps(entities: list[dict]) -> list[dict]:
        """Union-слияние: убираем пересечения по правилам приоритета.

        Правила:
        1. Более высокий приоритет источника (regex > gliner > natasha).
        2. При равном приоритете — более длинный спан.
        3. При прочих равных — более высокая уверенность.
        Пересекающиеся сущности более низкого ранга отбрасываются
        (не маскируем дважды, не оставляем кусочков).
        """
        ents = sorted(
            entities,
            key=lambda e: (
                e["start"],
                -SOURCE_PRIORITY.get(e.get("source", "gliner"), 0),
                -(e["end"] - e["start"]),
                -(e.get("score") or 0.0),
            ),
        )
        kept: list[dict] = []
        for e in sorted(entities, key=lambda x: (x["start"], -(x["end"] - x["start"]))):
            overlaps = [k for k in kept if e["start"] < k["end"] and k["start"] < e["end"]]
            if not overlaps:
                kept.append(e)
                continue
            # приоритет источника, затем вес (контекстный > категорийный),
            # затем длина, затем score
            def _rank(x: dict) -> tuple:
                w = x.get("weight")
                if w is None:
                    w = CATEGORY_WEIGHT.get(x.get("category"), 1.0)
                return (
                    SOURCE_PRIORITY.get(x.get("source", "gliner"), 0),
                    w,
                    x["end"] - x["start"],
                    x.get("score") or 0.0,
                )
            my_rank = _rank(e)
            best_overlap_rank = max(_rank(k) for k in overlaps)
            if my_rank > best_overlap_rank:
                # заменяем все пересекающие на текущую
                for k in overlaps:
                    kept.remove(k)
                kept.append(e)
        return sorted(kept, key=lambda e: e["start"])


def mask_text(text: str, entities: list[dict]) -> tuple[str, list[dict]]:
    """Заменяет сущности плейсхолдерами.

    Возвращает (замаскированный текст, список замаскированных сущностей).
    Сущности должны быть отсортированы и не пересекаться.
    """
    parts = []
    cursor = 0
    masked_entities = []
    for e in sorted(entities, key=lambda x: x["start"]):
        placeholder = PLACEHOLDERS.get(e["category"], "[SENSITIVE]")
        parts.append(text[cursor : e["start"]])
        parts.append(placeholder)
        masked_entities.append({**e, "text": text[e["start"] : e["end"]], "placeholder": placeholder})
        cursor = e["end"]
    parts.append(text[cursor:])
    return "".join(parts), masked_entities
