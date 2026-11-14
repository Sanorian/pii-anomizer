# -*- coding: utf-8 -*-
"""Дебаг ошибок бенчмарка: показать расхождения по категориям."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT))

from eval_benchmark import (  # noqa: E402
    TAG_TO_CATEGORY, bio_to_spans, cyr_ratio, merge_adjacent, overlaps, exact,
)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SHOW_CAT = sys.argv[2].split(",") if len(sys.argv) > 2 else ["card", "snils", "email", "passport", "phone"]

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
        {"start": s["start"], "end": s["end"], "category": TAG_TO_CATEGORY[s["type"]], "type": s["type"]}
        for s in spans if s["type"] in TAG_TO_CATEGORY
    ]
    gold = [g for g in gold if g["end"] > g["start"]]
    text = r["text"]
    gold = merge_adjacent(gold)
    samples.append({"text": text, "gold": gold})
    if len(samples) >= N:
        break

from app.detectors.gliner_detector import GlinerDetector
from app.detectors.natasha_detector import NatashaDetector
from app.detectors.regex_detector import RegexDetector
from app.pipeline import AnonymizationPipeline

gliner = GlinerDetector(ROOT / "models" / "gliner_ru_finetuned", device="cpu").load()
pipe = AnonymizationPipeline(RegexDetector(), gliner, NatashaDetector().load())

for i, s in enumerate(samples):
    res = pipe.run(s["text"], threshold=0.45)
    pred = [{"start": e["start"], "end": e["end"], "category": e["category"]} for e in res.entities]

    for g in s["gold"]:
        if g["category"] not in SHOW_CAT:
            continue
        matched = any(
            p["category"] == g["category"] and (exact(g, p) or overlaps(g, p)) for p in pred
        )
        if not matched:
            print(f"[{i}] MISS {g['category']} ({g.get('type', '?')}): «{s['text'][g['start']:g['end']]}»")
            print(f"     текст: …{s['text'][max(0, g['start'] - 40):g['end'] + 40]}…")

    for p in pred:
        if p["category"] not in SHOW_CAT:
            continue
        matched = any(
            g["category"] == p["category"] and (exact(g, p) or overlaps(g, p)) for g in s["gold"]
        )
        if not matched:
            print(f"[{i}] FALSE {p['category']}: «{s['text'][p['start']:p['end']]}»")
