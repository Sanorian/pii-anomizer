# -*- coding: utf-8 -*-
"""Оценка GLiNER на русском: NER на ручном наборе.

Прогоняет обе модели через одинаковый набор кейсов, считает
precision/recall/F1 (строгий и пересечение), печатает разбор ошибок.

Запуск: ./venv/bin/python eval_gliner_ru.py
"""

import json
import time
from pathlib import Path

import torch
from gliner import GLiNER

from eval_data import (
    ALL_CASES,
    DOMAIN_SPAM_EXAMPLES,
    LABEL_ALIASES,
    MODEL_LABELS,
)

MODELS = {
    "gliner_multi": "models/gliner_multi",
    "multitask_large": "models/gliner_multitask_large",
    "ru_finetuned": "models/ft_output/checkpoint-3000",
    "ru_finetuned_v2": "models/ft_output_v2/checkpoint-2000",
}

THRESHOLD = 0.4
ROOT = Path(__file__).parent

CANON_LABELS = {
    "person", "organization", "location", "phone", "email", "url", "money", "group_code",
}


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def canon(label: str) -> str:
    """Метка модели -> каноническая метка набора."""
    return LABEL_ALIASES.get(label.strip().lower(), label.strip().lower())


def norm_text(s: str) -> str:
    """Нормализация текста span'а: регистр и ё/е (частые расхождения)."""
    return s.strip().lower().replace("ё", "е")


def match_type(pred_span: str, gold_span: str) -> str:
    """Тип совпадения: exact / partial / none."""
    p, g = norm_text(pred_span), norm_text(gold_span)
    if p == g:
        return "exact"
    if p in g or g in p:
        return "partial"
    # пересечение по токенам
    pt, gt = set(p.split()), set(g.split())
    if pt & gt:
        return "partial"
    return "none"


def evaluate(model, name: str) -> dict:
    print(f"\n{'=' * 70}\nМодель: {name}\n{'=' * 70}")

    results = []  # (case_idx, pred_list, matches)
    tp_exact = tp_partial = fp = fn = 0
    per_label: dict[str, dict[str, int]] = {}
    t0 = time.time()

    for idx, (text, gold) in enumerate(ALL_CASES):
        try:
            raw = model.predict_entities(text, MODEL_LABELS, threshold=THRESHOLD)
        except Exception as e:
            print(f"  [ОШИБКА] кейс {idx}: {e}")
            raw = []

        preds = [
            {"span": e["text"], "label": canon(e["label"]), "score": round(e["score"], 3)}
            for e in raw
        ]
        preds = [p for p in preds if p["label"] in {  # фильтр нераспознанных меток
            "person", "organization", "location", "phone", "email", "url", "money", "group_code"
        }]

        matched_gold = set()
        matched_pred = set()
        exact = 0
        detail = []
        for gi, (g_sub, g_label) in enumerate(gold):
            best = "none"
            for pi, p in enumerate(preds):
                if pi in matched_pred:
                    continue
                if p["label"] != g_label:
                    continue
                m = match_type(p["span"], g_sub)
                if m != "none":
                    best = m
                    matched_pred.add(pi)
                    matched_gold.add(gi)
                    if m == "exact":
                        exact += 1
                        break  # exact лучше partial, берём его
            detail.append({"gold": g_sub, "label": g_label, "match": best})

        fn += len(gold) - len(matched_gold)
        fp += len(preds) - len(matched_pred)
        tp_exact += exact
        # partial = сколько ещё gold-сущностей нашли (совпадение не exact)
        tp_partial += len(matched_gold) - exact

        for gi, (g_sub, g_label) in enumerate(gold):
            lbl = g_label
            per_label.setdefault(lbl, {"gold": 0, "found": 0, "exact": 0})
            per_label[lbl]["gold"] += 1
            if gi in matched_gold:
                per_label[lbl]["found"] += 1
                if detail[gi]["match"] == "exact":
                    per_label[lbl]["exact"] += 1
        results.append({"idx": idx, "preds": preds, "detail": detail})

    dt = time.time() - t0
    total_gold = tp_exact + tp_partial + fn
    strict_p = tp_exact / (tp_exact + fp) if tp_exact + fp else 0.0
    strict_r = tp_exact / total_gold if total_gold else 0.0
    lenient_tp = tp_exact + tp_partial
    lenient_p = lenient_tp / (lenient_tp + fp) if lenient_tp + fp else 0.0
    lenient_r = lenient_tp / total_gold if total_gold else 0.0
    f1 = lambda p, r: 2 * p * r / (p + r) if p + r else 0.0

    print(f"\nВремя инференса: {dt:.1f} c ({dt / len(ALL_CASES):.2f} c/кейс)")
    print(f"\n{'Метрика':<22} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"{'строгий (exact)':<22} {strict_p:>10.3f} {strict_r:>10.3f} {f1(strict_p, strict_r):>10.3f}")
    print(f"{'мягкий (+partial)':<22} {lenient_p:>10.3f} {lenient_r:>10.3f} {f1(lenient_p, lenient_r):>10.3f}")

    print(f"\nПо меткам (gold / found(exact+partial) / exact):")
    for lbl, st in sorted(per_label.items()):
        print(f"  {lbl:<14} gold={st['gold']:<3} found={st['found']:<3} exact={st['exact']}")

    return {
        "model": name,
        "strict": {"p": strict_p, "r": strict_r, "f1": f1(strict_p, strict_r)},
        "lenient": {"p": lenient_p, "r": lenient_r, "f1": f1(lenient_p, lenient_r)},
        "per_label": per_label,
        "results": results,
        "time": dt,
    }


def main():
    device = resolve_device()
    print(f"Устройство: {device}, threshold={THRESHOLD}")
    out = {"device": device, "threshold": THRESHOLD, "models": {}}

    for name, path in MODELS.items():
        model = GLiNER.from_pretrained(str(ROOT / path)).to(device)
        model.eval()
        res = evaluate(model, name)
        out["models"][name] = res
        del model
        if device != "cpu":
            torch.mps.empty_cache()

    # --- Разбор ошибок для второй (лучшей) модели ---
    print(f"\n{'=' * 70}\nРАЗБОР ПРЕДСКАЗАНИЙ (только ru_finetuned_v2)\n{'=' * 70}")
    for name in ["ru_finetuned_v2"]:
        print(f"\n----- {name} -----")
        for r in out["models"][name]["results"]:
            idx = r["idx"]
            text, gold = ALL_CASES[idx]
            print(f"\n[{idx}] {text[:90]}{'...' if len(text) > 90 else ''}")
            if not r["preds"] and not gold:
                print("    (пусто с обеих сторон — ок)")
                continue
            for d in r["detail"]:
                mark = {"exact": "OK ", "partial": "~  ", "none": "X  "}[d["match"]]
                print(f"    {mark} gold[{d['label']}]: «{d['gold']}»")
            for p in r["preds"]:
                print(f"      pred[{p['label']}|{p['score']}]: «{p['span']}»")

    # --- Доменные спам-примеры: просто посмотреть глазами ---
    print(f"\n{'=' * 70}\nДОМЕННЫЕ СПАМ-ПРИМЕРЫ (только ru_finetuned_v2)\n{'=' * 70}")
    model = GLiNER.from_pretrained(str(ROOT / MODELS["ru_finetuned_v2"])).to(device)
    model.eval()
    for text in DOMAIN_SPAM_EXAMPLES:
        ents = model.predict_entities(text, MODEL_LABELS, threshold=THRESHOLD)
        print(f"\nТекст: {text[:100]}...")
        if not ents:
            print("    (сущностей не найдено)")
        for e in ents:
            print(f"    [{canon(e['label'])}|{e['score']:.2f}] «{e['text']}»")

    # --- Сохранение результатов ---
    out_path = ROOT / "results_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "device": out["device"],
                "threshold": out["threshold"],
                "models": {
                    n: {k: v for k, v in m.items() if k != "results"}
                    for n, m in out["models"].items()
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nМетрики сохранены в {out_path.name}")


if __name__ == "__main__":
    main()
