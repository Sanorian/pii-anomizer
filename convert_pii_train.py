# -*- coding: utf-8 -*-
"""Конвертация redmadrobot-rnd/pii_train (BIO CSV) в GLiNER span-формат.

Вход:  benchmark/pii_train/train.csv (text, tokens, ner_tags)
Выход: data_ft/pii_train_gliner.json — [{"tokenized_text": [...], "ner": [[s, e, label], ...]}]

Маппинг 21 типа pii_train -> метки сервиса (см. план раунда 5):
- granular ФИО -> person, гранулярные адреса -> location;
- документы: DRIVER_LICENSE/MILITARY_ID/BIRTH_CERTIFICATE -> doc number;
- IP_ADDRESS -> ip address (новая метка).

Плюс дедуп-проверка против pii_benchmark (нормализованный текст):
совпадающие примеры удаляются — контаминация train/eval недопустима.

Запуск: ./venv/bin/python convert_pii_train.py
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent
SRC = ROOT / "benchmark" / "pii_train" / "train.csv"
BENCH = ROOT / "benchmark" / "pii_benchmark_raw.jsonl"
OUT = ROOT / "data_ft" / "pii_train_gliner.json"

# 21 тип pii_train -> 11 меток GLiNER (как в сервисе + ip address)
LABEL_MAP = {
    "FIRST_NAME": "person",
    "LAST_NAME": "person",
    "MIDDLE_NAME": "person",
    "COUNTRY": "location",
    "REGION": "location",
    "DISTRICT": "location",
    "CITY": "location",
    "STREET": "location",
    "HOUSE": "location",
    "EMAIL": "email",
    "PHONE": "phone",
    "URL": "url",
    "IP_ADDRESS": "ip address",
    "PASSPORT": "passport",
    "INN": "inn",
    "SNILS": "snils",
    "OMS": "oms",
    "CREDIT_CARD": "card",
    "DRIVER_LICENSE": "doc number",
    "MILITARY_ID": "doc number",
    "BIRTH_CERTIFICATE": "doc number",
}

# Нормализация для дедупа: только кириллица/латиница/цифры в нижнем регистре
def norm_text(s: str) -> str:
    return re.sub(r"[^a-zа-яё0-9]", "", s.lower())


def bio_to_spans(tags):
    """BIO-теги -> [(start_tok, end_tok, type)]. Вложенность вpii_train плоская."""
    spans = []
    cur_type, cur_start = None, None
    for i, tag in enumerate(tags + ["O"]):  # O-страж в конце
        if tag.startswith("B-"):
            if cur_type is not None:
                spans.append((cur_start, i - 1, cur_type))
            cur_type, cur_start = tag[2:], i
        elif tag.startswith("I-") and cur_type is not None and tag[2:] == cur_type:
            pass  # продолжение
        else:
            if cur_type is not None:
                spans.append((cur_start, i - 1, cur_type))
                cur_type, cur_start = None, None
    return spans


def merge_adjacent(spans):
    """Склейка смежных спанов одной метки.

    pii_train размечает ФИО тремя отдельными тегами (FIRST/LAST/MIDDLE вплотную),
    адрес — STREET и HOUSE вплотную. После маппинга в общие метки (person,
    location) смежные спаны одной метки склеиваем: 'макаров Родион Евгеньевич'
    = ОДИН person-спан, консистентно с бенчмарком (person = полное ФИО) и
    нашей синтетикой. Спаны через запятую/пробел-токен ('ул. Пушкина , д. 176')
    НЕ склеиваем — это разные сущности и в gold бенчмарка.
    """
    if not spans:
        return spans
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e, lbl in spans[1:]:
        if lbl == merged[-1][2] and s <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e, lbl])
    return merged


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    print(f"строк в CSV: {len(rows)}")

    # gold-множество бенчмарка для дедупа
    bench_norms = set()
    with BENCH.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            bench_norms.add(norm_text(r["text"]))
    print(f"бенчмарк: {len(bench_norms)} уникальных текстов")

    out, stats, dropped_dup, dropped_bench = [], Counter(), 0, 0
    seen = set()
    for r in rows:
        tokens = json.loads(r["tokens"])
        tags = json.loads(r["ner_tags"])
        if len(tokens) != len(tags):
            stats["_skip_len_mismatch"] += 1
            continue
        spans = [
            [s, e, LABEL_MAP[t]]
            for s, e, t in bio_to_spans(tags)
            if t in LABEL_MAP
        ]
        spans = merge_adjacent(spans)
        if not spans:
            continue  # примеры без сущностей не несут пользы для NER-FT
        nt = norm_text(r["text"])
        if nt in bench_norms:
            dropped_bench += 1
            continue
        if nt in seen:
            dropped_dup += 1
            continue
        seen.add(nt)
        out.append({"tokenized_text": tokens, "ner": spans})
        for _, _, lbl in spans:
            stats[lbl] += 1

    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"\nвыгружено: {len(out)} примеров -> {OUT.name}")
    print(f"удалено: контаминация с бенчмарком={dropped_bench}, дубликатов={dropped_dup}")
    print("распределение меток:")
    for lbl, n in stats.most_common():
        if not lbl.startswith("_"):
            print(f"  {lbl:<12}{n}")
    if stats.get("_skip_len_mismatch"):
        print(f"  пропущено (len mismatch): {stats['_skip_len_mismatch']}")


if __name__ == "__main__":
    main()
