# -*- coding: utf-8 -*-
"""Конфигурация сервиса анонимизации."""

import os
from pathlib import Path

# Корень проекта anonimization (на 2 уровня выше service/app).
# В Docker путь переопределяется через GLINER_MODEL_DIR.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Модели
GLINER_MODEL_DIR = Path(
    os.environ.get("GLINER_MODEL_DIR", PROJECT_ROOT / "models" / "gliner_ru_finetuned")
)
GLINER_THRESHOLD_DEFAULT = 0.45

# Категории сущностей и их плейсхолдеры.
# Порядок = приоритет при слиянии пересечений (раньше в списке = выше приоритет).
PLACEHOLDERS: dict[str, str] = {
    # Финансы и документы (regex в первую очередь)
    "card": "[CARD]",
    "account": "[ACCOUNT]",
    "snils": "[SNILS]",
    "inn": "[INN]",
    "passport": "[PASSPORT]",
    "oms": "[OMS]",
    "doc_number": "[DOC_NUMBER]",
    "money": "[MONEY]",
    # Контакты
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "url": "[URL]",
    # Персональные
    "person": "[PERSON]",
    "organization": "[ORG]",
    "location": "[LOCATION]",
    "profession": "[PROFESSION]",
    "age": "[AGE]",
    "date": "[DATE]",
    # Учебный домен
    "group_code": "[GROUP_CODE]",
}

# Категории, которые можно включать/выключать в запросе (все по умолчанию)
ALL_CATEGORIES = sorted(PLACEHOLDERS.keys())

# Максимальная длина текста на запрос (символы)
MAX_TEXT_LENGTH = 16_000

# Batch инференса GLiNER
GLINER_BATCH_SIZE = 8
