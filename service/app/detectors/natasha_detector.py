# -*- coding: utf-8 -*-
"""Natasha-детектор: Slovnet NER, дообученный на данных ПДн (меток: 10).

Загружает кастомный tar (env NATASHA_MODEL_TAR, по умолчанию
models/slovnet_ner_ru_ft_v1.tar). Метки берутся из тег-вокаба модели
динамически (B-person -> person). При отсутствии tar — fallback на
pretrained slovnet_ner_news_v1 (PER/LOC/ORG).
"""

import gzip
import json
import os
import re
import tarfile
from pathlib import Path

from natasha import Doc, MorphVocab, NewsEmbedding, NewsNERTagger, Segmenter
from slovnet import NER as SlovnetNER

# Fallback-маппинг pretrained-меток -> категории сервиса
NATASHA_TO_CATEGORY = {
    "PER": "person",
    "LOC": "location",
    "ORG": "organization",
}

# Категории, для которых latin-only спан — почти наверняка шум дистилляции
# (editbox30, setText, "Visa", IPv6): требуем хотя бы одну кириллическую букву
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
CYRILLIC_ONLY_CATEGORIES = {"person", "location", "organization"}

DEFAULT_MODEL_TAR = str(
    Path(__file__).resolve().parents[3] / "models" / "slovnet_ner_ru_ft_v1.tar"
)


def load_tag_vocab(tar_path: str) -> dict[str, str] | None:
    """B/I-метки из тег-вокаба tar -> категории (B-person -> person)."""
    try:
        with tarfile.open(tar_path) as tar:
            tags = gzip.decompress(tar.extractfile("vocabs/tag.gz").read()).decode().splitlines()
    except (FileNotFoundError, KeyError, tarfile.TarError):
        return None
    mapping: dict[str, str] = {}
    for tag in tags:
        if tag.startswith(("B-", "I-")):
            mapping[tag.split("-", 1)[1]] = tag.split("-", 1)[1]
    return mapping or None


# Категории, которые FT-Natasha не отдаёт в union: url полностью покрыт
# regex+GLiNER (натasha-версии шумят на латинице: «Рис», «п . 5 . 1 . 7»),
# person/phone — тоже (GLiNER+regex сильнее, natasha даёт FP)
FINETUNE_DROP_CATEGORIES = {"phone", "person", "url"}


class NatashaDetector:
    """Обёртка над Slovnet NER (fine-tuned на ПДн или pretrained)."""

    name = "natasha"

    def __init__(self, model_tar: str | None = None) -> None:
        self.model_tar = model_tar or os.environ.get("NATASHA_MODEL_TAR") or DEFAULT_MODEL_TAR
        self._segmenter: Segmenter | None = None
        self._ner: NewsNERTagger | None = None
        self._morph: MorphVocab | None = None
        self._label_map: dict[str, str] = NATASHA_TO_CATEGORY
        self.is_finetuned = False

    def load(self) -> "NatashaDetector":
        self._segmenter = Segmenter()
        emb = NewsEmbedding()
        if os.path.exists(self.model_tar):
            self._label_map = load_tag_vocab(self.model_tar) or NATASHA_TO_CATEGORY
            if self._label_map != NATASHA_TO_CATEGORY:  # FT-вокаб (10 меток, не PER/LOC/ORG)
                self._label_map = {
                    k: v for k, v in self._label_map.items() if v not in FINETUNE_DROP_CATEGORIES
                }
            self._ner = NewsNERTagger(emb, self.model_tar)
            self.is_finetuned = True
        else:
            # pretrained: скачивается автоматически при первом NewsNERTagger(emb)
            self._label_map = NATASHA_TO_CATEGORY
            self._ner = NewsNERTagger(emb)
        self._morph = MorphVocab()
        return self

    @property
    def is_loaded(self) -> bool:
        return self._ner is not None

    def detect(self, text: str) -> list[dict]:
        if self._ner is None or self._segmenter is None:
            raise RuntimeError("Natasha не загружена")
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._ner)
        results = []
        for span in doc.spans:
            category = self._label_map.get(span.type)
            if category is None:
                continue
            # обрезаем пунктуацию по краям (Natasha иногда прихватывает)
            span_text = text[span.start : span.stop]
            stripped = span_text.strip("«»\"'.,!?;:() ")
            if not stripped:
                continue
            # latin-only спан «имя/место/организация» — шум на коде/URL/IPv6
            if category in CYRILLIC_ONLY_CATEGORIES and not CYRILLIC_RE.search(span_text):
                continue
            lead = len(span_text) - len(span_text.lstrip("«»\"'.,!?;:() "))
            trail = len(span_text) - len(stripped) - lead
            results.append({
                "start": span.start + lead,
                "end": span.stop - trail,
                "category": category,
                "score": 1.0,
            })
        return results
