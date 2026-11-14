# -*- coding: utf-8 -*-
"""Оценка пайплайна анонимизации на бенчмарке redmadrobot-rnd/pii_benchmark.

Формат: {"text", "tokens": [...], "ner_tags": [...]} — BIO-разметка.
Сравнение на уровне символьных спанов:
  - strict: точное совпадение границ
  - overlap: любое пересечение (что важно для анонимизации)

Запуск:  ../venv/bin/python eval_benchmark.py [--limit N]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "service"))

# Маппинг меток бенчмарка -> наши категории (None = не оцениваем)
TAG_TO_CATEGORY = {
    "PHONE": "phone",
    "EMAIL": "email",
    "URL": "url",
    "CREDIT_CARD": "card",
    "PASSPORT": "passport",
    "INN": "inn",
    "SNILS": "snils",
    "OMS": "oms",
    "DRIVER_LICENSE": "doc_number",
    "BIRTH_CERTIFICATE": "doc_number",
    "MILITARY_ID": "doc_number",
    "FIRST_NAME": "person",
    "MIDDLE_NAME": "person",
    "LAST_NAME": "person",
    "STREET": "location",
    "HOUSE": "location",
    "CITY": "location",
    "REGION": "location",
    "DISTRICT": "location",
    "COUNTRY": "location",
    # Не маппим (нет прямого аналога в сервисе):
    # IP_ADDRESS — в бенчмарке это IPv4/IPv6; наш URL_RE частично ловит IP,
    # но категория другая; оставляем вне оценки
}

CYR = re.compile(r"[а-яёА-ЯЁ]")

gold_text = ""  # текущий текст примера (для merge_adjacent)

# Категории, которых нет в разметке бенчмарка (их предсказания не считаем FP,
# но и оценивать по ним нельзя) — считаем отдельно как "extra detections"
NON_BENCH_CATEGORIES = {"organization", "profession", "age", "date", "money", "group_code"}

def merge_adjacent(spans: list[dict], max_gap: int = 2) -> list[dict]:
    """Сливает смежные спаны одной категории (gap из пробелов/пунктуации).

    Нужно, т.к. в бенчмарке "Иванов Иван Иванович" = 3 отдельных спана
    (LAST_NAME, FIRST_NAME, MIDDLE_NAME), а для анонимизации это одна сущность.
    """
    if not spans:
        return spans
    spans = sorted(spans, key=lambda s: (s["start"], s["end"]))
    merged = [dict(spans[0])]
    for s in spans[1:]:
        prev = merged[-1]
        if (
            s["category"] == prev["category"]
            and s["start"] - prev["end"] <= max_gap
            and not text_gap_has_letters(prev["end"], s["start"])
        ):
            prev["end"] = s["end"]
        else:
            merged.append(dict(s))
    return merged


def text_gap_has_letters(start: int, end: int) -> bool:
    """Проверяет, что текст между start и end не содержит букв (только разделители)."""
    return any(c.isalnum() for c in gold_text[start:end])


def bio_to_spans(tokens: list[str], tags: list[str], text: str) -> list[dict]:
    """BIO-разметка -> символьные спаны. Токены ищутся в тексте последовательно."""
    spans = []
    cursor = 0
    current_type, current_start = None, None

    def find_token(token: str, start: int) -> int:
        """Ищет токен в тексте с позиции start (токены могут опускать пробелы)."""
        idx = text.find(token, start)
        if idx == -1:
            # пробуем без лишних пробелов
            idx = text.find(token.strip(), start)
        return idx

    for token, tag in zip(tokens, tags):
        idx = find_token(token, cursor)
        if idx == -1:
            # не нашли — сбрасываем текущую сущность
            if current_type is not None:
                spans.append({"start": current_start, "end": cursor, "type": current_type})
                current_type = None
            continue
        tok_start, tok_end = idx, idx + len(token)
        cursor = tok_end

        if tag.startswith("B-"):
            if current_type is not None:
                spans.append({"start": current_start, "end": cursor_prev, "type": current_type})
            current_type, current_start = tag[2:], tok_start
        elif tag.startswith("I-") and current_type == tag[2:]:
            pass  # продолжение
        else:  # O или несовпадение
            if current_type is not None:
                spans.append({"start": current_start, "end": cursor_prev, "type": current_type})
                current_type = None
        cursor_prev = tok_end

    if current_type is not None:
        spans.append({"start": current_start, "end": cursor_prev, "type": current_type})

    return spans


def cyr_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if CYR.match(c)) / len(letters)


def overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] and b["start"] < a["end"]


def exact(a: dict, b: dict) -> bool:
    return a["start"] == b["start"] and a["end"] == b["end"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gliner", default="models/gliner_ru_finetuned_v2",
                        help="путь до GLiNER-модели")
    parser.add_argument("--out", default="benchmark/eval_results.json")
    args = parser.parse_args()

    # --- Загрузка данных ---
    raw_path = ROOT / "benchmark" / "pii_benchmark_raw.jsonl"
    rows = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            r["tokens"] = json.loads(r["tokens"])
            r["ner_tags"] = json.loads(r["ner_tags"])
            rows.append(r)

    # --- Конвертация в спаны + фильтр шума ---
    samples = []
    skipped_noise = 0
    for r in rows:
        # фильтр: слишком мало кириллицы = англ. шум (python-код, ASCII и т.п.)
        if cyr_ratio(r["text"]) < 0.15:
            skipped_noise += 1
            continue
        spans = bio_to_spans(r["tokens"], r["ner_tags"], r["text"])
        gold = [
            {"start": s["start"], "end": s["end"], "category": TAG_TO_CATEGORY.get(s["type"])}
            for s in spans
            if s["type"] in TAG_TO_CATEGORY
        ]
        gold = [g for g in gold if g["end"] > g["start"]]
        globals()["gold_text"] = r["text"]
        gold = merge_adjacent(gold)
        samples.append({"text": r["text"], "gold": gold})

    if args.limit:
        samples = samples[: args.limit]

    total_gold = sum(len(s["gold"]) for s in samples)
    print(f"Примеров: {len(samples)} (отфильтровано шума: {skipped_noise})")
    print(f"Gold-сущностей (маппимых): {total_gold}")

    # --- Прогон пайплайна ---
    from app.detectors.gliner_detector import GlinerDetector
    from app.detectors.natasha_detector import NatashaDetector
    from app.detectors.regex_detector import RegexDetector
    from app.pipeline import AnonymizationPipeline

    gliner = GlinerDetector(ROOT / args.gliner, device="cpu").load()
    pipe = AnonymizationPipeline(RegexDetector(), gliner, NatashaDetector().load())

    # Статистика по категориям:
    #   tp_s      — строгие совпадения (точные границы)
    #   tp_o_only — совпадения только по пересечению (границы неточны)
    #   fn        — gold без совпадения
    #   fp        — pred без совпадения (категории бенчмарка)
    stats = defaultdict(lambda: {"tp_s": 0, "tp_o_only": 0, "fp": 0, "fn": 0})

    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(samples)}")
        try:
            res = pipe.run(s["text"], threshold=args.threshold)
            pred = [
                {"start": e["start"], "end": e["end"], "category": e["category"]}
                for e in res.entities
            ]
        except Exception as e:  # noqa: BLE001
            print(f"  ОШИБКА на примере {i}: {e}")
            continue

        gold = s["gold"]
        used_pred = set()
        used_gold = set()

        # Multi-coverage (двусторонний): gold покрыт, если его пересекает
        # хотя бы один pred той же категории; pred задействован, если покрыл
        # хотя бы один gold. Бенчмарк режет «серия + номер» на 2 gold-спана,
        # мы маскируем одной конструкцией: оба gold → tp, pred → 1 TP.
        # Строгие матчи (1:1) считаются первыми.
        for g_idx, g in enumerate(gold):
            strict_j = next(
                (j for j, p in enumerate(pred)
                 if j not in used_pred and p["category"] == g["category"]
                 and exact(g, p)),
                None,
            )
            if strict_j is not None:
                stats[g["category"]]["tp_s"] += 1
                used_pred.add(strict_j)
                used_gold.add(g_idx)

        for g_idx, g in enumerate(gold):
            if g_idx in used_gold:
                continue
            ov_js = [
                j for j, p in enumerate(pred)
                if p["category"] == g["category"] and overlaps(g, p)
            ]
            if ov_js:
                stats[g["category"]]["tp_o_only"] += 1
                used_gold.add(g_idx)
                used_pred.update(ov_js)
            else:
                stats[g["category"]]["fn"] += 1

        # Незадействованные pred = FP (кроме категорий вне бенчмарка)
        for j, p in enumerate(pred):
            if j not in used_pred and p["category"] not in NON_BENCH_CATEGORIES:
                stats[p["category"]]["fp"] += 1

    # --- Отчёт ---
    # strict:      P = tp_s/(tp_s+fp),      R = tp_s/(tp_s+tp_o_only+fn)
    # overlap:     P = (tp_s+tp_o_only)/(tp_s+tp_o_only+fp), R = то же по gold
    print("\n=== РЕЗУЛЬТАТЫ (strict = точные границы / overlap = любое пересечение) ===")
    print(f"{'категория':<12} {'gold':>6} {'pred':>6} {'P_s':>6} {'R_s':>6} {'F1_s':>6} {'P_o':>6} {'R_o':>6} {'F1_o':>6}")

    totals = {"tp_s": 0, "tp_o_only": 0, "fp": 0, "fn": 0}
    report = {}

    def _prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f1

    for cat in sorted(stats.keys()):
        st = stats[cat]
        for k in totals:
            totals[k] += st[k]
        gold_n = st["tp_s"] + st["tp_o_only"] + st["fn"]
        pred_n = st["tp_s"] + st["tp_o_only"] + st["fp"]
        p_s, r_s, f1_s = _prf(st["tp_s"], st["fp"], st["tp_o_only"] + st["fn"])
        p_o, r_o, f1_o = _prf(st["tp_s"] + st["tp_o_only"], st["fp"], st["fn"])
        report[cat] = {
            "gold": gold_n, "pred": pred_n,
            "strict": {"p": round(p_s, 3), "r": round(r_s, 3), "f1": round(f1_s, 3)},
            "overlap": {"p": round(p_o, 3), "r": round(r_o, 3), "f1": round(f1_o, 3)},
            **st,
        }
        print(
            f"{cat:<12} {gold_n:>6} {pred_n:>6} "
            f"{p_s:>6.3f} {r_s:>6.3f} {f1_s:>6.3f} "
            f"{p_o:>6.3f} {r_o:>6.3f} {f1_o:>6.3f}"
        )

    gold_n = totals["tp_s"] + totals["tp_o_only"] + totals["fn"]
    pred_n = totals["tp_s"] + totals["tp_o_only"] + totals["fp"]
    p_s, r_s, f1_s = _prf(totals["tp_s"], totals["fp"], totals["tp_o_only"] + totals["fn"])
    p_o, r_o, f1_o = _prf(
        totals["tp_s"] + totals["tp_o_only"], totals["fp"], totals["fn"]
    )
    print("-" * 78)
    print(
        f"{'MICRO':<12} {gold_n:>6} {pred_n:>6} "
        f"{p_s:>6.3f} {r_s:>6.3f} {f1_s:>6.3f} "
        f"{p_o:>6.3f} {r_o:>6.3f} {f1_o:>6.3f}"
    )

    out = {
        "benchmark": "redmadrobot-rnd/pii_benchmark",
        "n_samples": len(samples),
        "skipped_noise": skipped_noise,
        "threshold": args.threshold,
        "micro": {
            "gold": gold_n, "pred": pred_n,
            "strict": {"p": round(p_s, 3), "r": round(r_s, 3), "f1": round(f1_s, 3)},
            "overlap": {"p": round(p_o, 3), "r": round(r_o, 3), "f1": round(f1_o, 3)},
        },
        "per_category": report,
    }
    with open(ROOT / args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {args.out}")


if __name__ == "__main__":
    main()
