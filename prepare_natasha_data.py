# -*- coding: utf-8 -*-
"""Подготовка данных для дообучения Natasha (Slovnet NER).

Источник: data_ft/train_merged.json (формат GLiNER: tokenized_text + ner-спаны
в индексах токенов). Конвертация в BIO над токенами razdel — как ожидает
Slovnet при инференсе.

Выход: data_natasha/train.jsonl, dev.jsonl — по строке:
    {"words": [...], "tags": ["O", "B-person", ...]}
"""

import json
import random
from pathlib import Path

from razdel import tokenize as razdel_tokenize

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data_ft" / "train_merged_v2.json"  # round 2: + синтетика passport/doc_number/url
OUT_DIR = ROOT / "data_natasha"

# Метки, которые учим (как в GLiNER-детекторе сервиса, без passport — его
# покрывает regex). "group code" в данных -> group_code.
LABEL_MAP = {
    "person": "person",
    "organization": "organization",
    "location": "location",
    "date": "date",
    "age": "age",
    "phone": "phone",
    "email": "email",
    "url": "url",
    "money": "money",
    "group code": "group_code",
    "group_code": "group_code",
}


def rebuild_char_offsets(tokenized_text: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Склеивает токены пробелами и возвращает (текст, [(start, end), ...])."""
    parts = []
    offsets = []
    pos = 0
    for tok in tokenized_text:
        if parts:
            pos += 1  # пробел-разделитель
        offsets.append((pos, pos + len(tok)))
        parts.append(tok)
        pos += len(tok)
    return " ".join(parts), offsets


def convert_example(example: dict) -> dict | None:
    """GLiNER-формат -> BIO над razdel-токенами."""
    text, tok_offsets = rebuild_char_offsets(example["tokenized_text"])

    # Спаны в символьных координатах
    char_spans = []
    for start_idx, end_idx, label in example["ner"]:
        label = LABEL_MAP.get(label.strip().lower())
        if label is None:
            continue
        if start_idx >= len(tok_offsets) or end_idx >= len(tok_offsets):
            continue
        s = tok_offsets[start_idx][0]
        e = tok_offsets[end_idx][1]
        if e <= s:
            continue
        char_spans.append((s, e, label))
    if not char_spans:
        return None
    char_spans.sort()

    # razdel-токены -> BIO
    words = []
    tags = []
    span_i = 0
    for tok in razdel_tokenize(text):
        t_start, t_stop = tok.start, tok.stop
        # пропускаем пробелы неявно — razdel их не даёт
        while span_i < len(char_spans) and char_spans[span_i][1] <= t_start:
            span_i += 1
        tag = "O"
        if span_i < len(char_spans):
            s, e, label = char_spans[span_i]
            if t_start < e and s < t_stop:  # пересечение
                tag = f"B-{label}" if t_start == s else f"I-{label}"
                if t_stop >= e:
                    span_i += 1
        words.append(tok.text)
        tags.append(tag)

    if not any(t != "O" for t in tags):
        return None
    return {"words": words, "tags": tags}


def bio_validity_check(tags: list[str]) -> bool:
    """BIO-последовательность корректна: I- после B-/I- той же метки."""
    prev_type = None
    for tag in tags:
        if tag == "O":
            prev_type = None
            continue
        part, etype = tag.split("-", 1)
        if part == "I" and prev_type != etype:
            return False
        prev_type = etype if part in ("B", "I") else None
    return True


def main() -> None:
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Источник: {len(data)} примеров")

    converted, skipped = [], 0
    for ex in data:
        c = convert_example(ex)
        if c is None or not bio_validity_check(c["tags"]):
            skipped += 1
            continue
        converted.append(c)
    print(f"Конвертировано: {len(converted)}, пропущено: {skipped}")

    random.seed(42)
    random.shuffle(converted)
    n_dev = max(50, int(len(converted) * 0.05))
    dev, train = converted[:n_dev], converted[n_dev:]

    OUT_DIR.mkdir(exist_ok=True)
    for name, items in [("train.jsonl", train), ("dev.jsonl", dev)]:
        with open(OUT_DIR / name, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"{name}: {len(items)}")

    # Статистика меток
    from collections import Counter
    cnt = Counter(t.split("-", 1)[1] for ex in train for t in ex["tags"] if t != "O")
    print("\nМетки в train:", dict(cnt.most_common()))


if __name__ == "__main__":
    main()
