# -*- coding: utf-8 -*-
"""Разбор ошибок бенчмарка по слабым категориям: passport, doc_number, url, email, phone.

Выгружает примеры FN (пропущено) и FP (лишнее) с контекстом, классифицирует
типы ошибок. Запуск: ./venv/bin/python error_analysis.py [--n 250]
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent

WEAK = ["passport", "doc_number", "url", "email", "phone"]

TAG_TO_CATEGORY = {
    "PHONE": "phone", "EMAIL": "email", "URL": "url", "CREDIT_CARD": "card",
    "PASSPORT": "passport", "INN": "inn", "SNILS": "snils", "OMS": "oms",
    "DRIVER_LICENSE": "doc_number", "BIRTH_CERTIFICATE": "doc_number",
    "MILITARY_ID": "doc_number",
    "FIRST_NAME": "person", "MIDDLE_NAME": "person", "LAST_NAME": "person",
    "STREET": "location", "HOUSE": "location", "CITY": "location",
    "REGION": "location", "DISTRICT": "location", "COUNTRY": "location",
}
NON_BENCH_CATEGORIES = {"organization", "profession", "age", "date", "money", "group_code"}

CYR = re.compile(r"[а-яёА-ЯЁ]")


def cyr_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CYR.match(c)) / len(letters)


def text_gap_has_letters(text: str, start: int, end: int) -> bool:
    return any(c.isalnum() for c in text[start:end])


def merge_adjacent(spans: list[dict], text: str, max_gap: int = 2) -> list[dict]:
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged = [dict(spans[0])]
    for s in spans[1:]:
        prev = merged[-1]
        if (
            s["category"] == prev["category"]
            and s["start"] - prev["end"] <= max_gap
            and not text_gap_has_letters(text, prev["end"], s["start"])
        ):
            prev["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged


def bio_to_spans(tokens: list[str], tags: list[str], text: str) -> list[dict]:
    spans = []
    cursor = 0
    current_type, current_start = None, None
    cursor_prev = 0

    def find_token(token: str, start: int) -> int:
        idx = text.find(token, start)
        if idx == -1:
            idx = text.find(token.strip(), start)
        return idx

    for token, tag in zip(tokens, tags):
        idx = find_token(token, cursor)
        if idx == -1:
            if current_type is not None:
                spans.append({"start": current_start, "end": cursor_prev, "type": current_type})
                current_type = None
            continue
        tok_start, tok_end = idx, idx + len(token)
        cursor = tok_end
        if tag.startswith("B-"):
            if current_type is not None:
                spans.append({"start": current_start, "end": cursor_prev, "type": current_type})
            current_type, current_start = tag[2:], tok_start
        elif tag.startswith("I-") and current_type == tag[2:]:
            pass
        else:
            if current_type is not None:
                spans.append({"start": current_start, "end": cursor_prev, "type": current_type})
                current_type = None
        cursor_prev = tok_end
    if current_type is not None:
        spans.append({"start": current_start, "end": cursor_prev, "type": current_type})
    return spans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=250, help="примеров на категорию")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "service"))

    from app.detectors.gliner_detector import GlinerDetector
    from app.detectors.natasha_detector import NatashaDetector
    from app.detectors.regex_detector import RegexDetector
    from app.pipeline import AnonymizationPipeline

    gliner = GlinerDetector(ROOT / "models" / "gliner_ru_finetuned_v2", device="cpu").load()
    pipe = AnonymizationPipeline(RegexDetector(), gliner, NatashaDetector().load())

    rows = []
    with open(ROOT / "benchmark" / "pii_benchmark_raw.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["tokens"] = json.loads(r["tokens"])
            r["ner_tags"] = json.loads(r["ner_tags"])
            rows.append(r)

    samples = []
    for r in rows:
        if cyr_ratio(r["text"]) < 0.15:
            continue
        spans = bio_to_spans(r["tokens"], r["ner_tags"], r["text"])
        gold = [
            {"start": s["start"], "end": s["end"], "category": TAG_TO_CATEGORY.get(s["type"])}
            for s in spans if s["type"] in TAG_TO_CATEGORY
        ]
        gold = [g for g in gold if g["end"] > g["start"]]
        gold = merge_adjacent(gold, r["text"])
        samples.append({"text": r["text"], "gold": gold})

    fn = defaultdict(list)   # category -> [(text, gold_span)]
    fp = defaultdict(list)   # category -> [(text, pred_span)]
    for s in samples:
        try:
            res = pipe.run(s["text"], threshold=0.45)
        except Exception:
            continue
        pred = [{"start": e["start"], "end": e["end"], "category": e["category"]}
                for e in res.entities]
        used = set()
        for g in s["gold"]:
            ms = next((j for j, p in enumerate(pred)
                       if j not in used and p["category"] == g["category"]
                       and p["start"] == g["start"] and p["end"] == g["end"]), None)
            if ms is None:
                ms = next((j for j, p in enumerate(pred)
                           if j not in used and p["category"] == g["category"]
                           and p["start"] < g["end"] and g["start"] < p["end"]), None)
            if ms is not None:
                used.add(ms)
            elif g["category"] in WEAK:
                fn[g["category"]].append((s["text"], g))
        for j, p in enumerate(pred):
            if j not in used and p["category"] in WEAK:
                fp[p["category"]].append((s["text"], p))

    random.seed(args.seed)
    out = {}
    for cat in WEAK:
        print(f"\n{'=' * 70}\n### {cat}: FN={len(fn[cat])}, FP={len(fp[cat])}\n{'=' * 70}")
        out[cat] = {"fn": [], "fp": []}
        for kind, bucket in [("FN", fn[cat]), ("FP", fp[cat])]:
            shown = random.sample(bucket, min(args.n, len(bucket)))
            for text, span in shown:
                ctx_s = max(0, span["start"] - 40)
                ctx_e = min(len(text), span["end"] + 40)
                rec = {
                    "context": text[ctx_s:ctx_e].replace("\n", " "),
                    "span": text[span["start"]:span["end"]],
                }
                out[cat][{"FN": "fn", "FP": "fp"}[kind]].append(rec)
            print(f"\n-- {kind} ({len(shown)} из {len(bucket)}):")
            for rec in out[cat][{"FN": "fn", "FP": "fp"}[kind]][:40]:
                print(f"   «{rec['span']}»   ...{rec['context']}...")

    with open(ROOT / "benchmark" / "error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nСохранено: benchmark/error_analysis.json")


if __name__ == "__main__":
    main()
