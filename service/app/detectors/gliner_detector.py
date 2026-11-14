# -*- coding: utf-8 -*-
"""GLiNER-детектор: дообученная модель для русского языка.

Находит person, organization, location, profession, age, date, money,
group_code — то, что не покрывается регэкспами.
"""

from pathlib import Path

import re

from gliner import GLiNER

# Метки, которые передаём в модель (в единственном числе, англ. —
# так модель обучалась)
GLINER_LABELS = [
    "person",
    "organization",
    "location",
    "profession",
    "age",
    "date",
    "money",
    "group code",
    "phone",
    "email",
    "url",
    "passport",
    "doc number",
]

# Маппинг меток модели -> категории сервиса
LABEL_TO_CATEGORY = {
    "person": "person",
    "organization": "organization",
    "location": "location",
    "profession": "profession",
    "age": "age",
    "date": "date",
    "money": "money",
    "group code": "group_code",
    "phone": "phone",
    "email": "email",
    "url": "url",
    "passport": "passport",
    "doc number": "doc_number",
}

# --- Валидация структурированных категорий (GLiNER галлюцинирует на них) ---
# Спан должен содержать хотя бы одну цифру/структурный знак, иначе это текст
# («СНИЛС», «ИНН», «ул. Тверская» без номера)
_DIGIT_RE = re.compile(r"\d")
# url: доменное имя (две метки через точку) или схема/www/@ник
_URL_SHAPE_RE = re.compile(
    r"(?:https?://|www\.|t\.me/|@[A-Za-z0-9_]{4,})"
    r"|^(?:[A-Za-z0-9\-]+\.)+[A-Za-z]{2,}(?:/[\w\-./?%&=#]*)?$"
)
# email: локальная часть @ домен
_EMAIL_SHAPE_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")
# passport/doc_number: числовая часть минимум 6 цифр (отсекает «номер заказа
# 321», «заявка 123» и прочие короткие числа, которые модель зовёт doc_number)
_DOC_DIGITS_RE = re.compile(r"\d{6,}")
# НЕ телефон: IPv4, UUID, длинные числа 12+ цифр (СНИЛС/ИНН слитно, VIN)
_IPV4_RE = re.compile(r"(?:\d{1,3}\s*[.]\s*){3}\d{1,3}")
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_PHONE_TOTAL_DIGITS_RE = re.compile(r"^(?:\+?\d[\d\s\-()]{8,16})$")


def _is_phone_span(s: str) -> bool:
    """Телефон: 10-11 цифр в «телефонной» форме; не IPv4/UUID/длинные номера."""
    digits = sum(c.isdigit() for c in s)
    if digits < 10 or digits > 11:
        return False
    # только цифры без разделителей: 11 цифр с 8/7 в начале — телефон,
    # прочие 10-11-значные «голые» числа — СНИЛС/ИНН/номер документа
    stripped = s.strip()
    if re.fullmatch(r"\d{10,11}", stripped):
        return bool(re.fullmatch(r"[78]\d{10}", stripped))
    if _UUID_RE.search(s) or re.search(r"\bvin\b", s, re.IGNORECASE):
        return False
    if _IPV4_RE.search(s) and digits == sum(
        c.isdigit() for c in _IPV4_RE.search(s).group(0)
    ):
        return False  # все цифры спана — это IPv4
    # телефонные разделители: скобки/дефисы/пробелы/ведущий +; слова вида
    # «телефон 89251234567» — разделители обязательны только внутри числа
    core = re.sub(r"(?:телефон|тел\.?|номер)\s*:?\s*", "", s, flags=re.IGNORECASE).strip()
    if not re.match(r"^\+?[\d\s\-()]+$", core):
        return False
    return True


_STRUCTURAL_VALIDATORS = {
    "url": lambda s: bool(_URL_SHAPE_RE.search(s.strip())),
    "email": lambda s: bool(_EMAIL_SHAPE_RE.match(s.strip())),
    "phone": _is_phone_span,
    "passport": lambda s: bool(_DOC_DIGITS_RE.search(s)),
    "doc_number": lambda s: bool(_DOC_DIGITS_RE.search(s)),
}


# Категории, для которых смежные предсказания склеиваются в один спан.
# pii_train размечает ФИО тремя отдельными тегами (FIRST/LAST/MIDDLE), из-за
# чего модель иногда отдаёт person куски («Miller» + «Amy Демьяновна»).
# Для анонимизации склейка безвредна (оба куска маскируются одинаково),
# а границы совпадают с gold (слитное ФИО).
MERGE_ADJACENT_CATEGORIES = {"person", "location", "organization"}


class GlinerDetector:
    """Обёртка над дообученным GLiNER."""

    name = "gliner"

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.model: GLiNER | None = None

    def load(self) -> "GlinerDetector":
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Каталог модели GLiNER не найден: {self.model_dir}")
        self.model = GLiNER.from_pretrained(str(self.model_dir)).to(self.device)
        self.model.eval()
        return self

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def detect(self, text: str, threshold: float = 0.45) -> list[dict]:
        if self.model is None:
            raise RuntimeError("GLiNER не загружена")
        entities = self.model.predict_entities(text, GLINER_LABELS, threshold=threshold)
        results = []
        for e in entities:
            label = e["label"].strip().lower()
            category = LABEL_TO_CATEGORY.get(label)
            if category is None:
                continue
            span_text = e["text"]
            # GLiNER галлюцинирует на структурированных категориях
            # («СНИЛС» как url, «PDF» как email, «ул. Тверская» как phone) —
            # проверяем форму спана
            validator = _STRUCTURAL_VALIDATORS.get(category)
            if validator is not None and not validator(span_text):
                continue
            results.append({
                "start": e["start"],
                "end": e["end"],
                "category": category,
                "score": round(float(e["score"]), 3),
            })
        return self._merge_adjacent(results)

    @staticmethod
    def _merge_adjacent(entities: list[dict]) -> list[dict]:
        """Склейка смежных спанов одной категории: «Miller» + «Amy Демьяновна»
        -> «Miller Amy Демьяновна». Гэп — ровно один пробел (без пунктуации
        между кусками: «Иванов, Пётр» не склеиваем — это разные gold-спаны).
        score merged = min (консервативно)."""
        if not entities:
            return entities
        entities = sorted(entities, key=lambda e: (e["start"], e["end"]))
        out = [dict(entities[0])]
        for e in entities[1:]:
            prev = out[-1]
            same_cat = e["category"] == prev["category"]
            mergeable = e["category"] in MERGE_ADJACENT_CATEGORIES
            gap_is_space = (
                e["start"] > prev["end"]
                and e["start"] - prev["end"] == 1
            )
            if same_cat and mergeable and gap_is_space:
                prev["end"] = e["end"]
                prev["score"] = min(prev["score"], e["score"])
            else:
                out.append(dict(e))
        return out
