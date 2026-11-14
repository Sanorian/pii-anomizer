# -*- coding: utf-8 -*-
"""Свип порога для multitask_large: F1 при threshold 0.3..0.7."""

import torch
from gliner import GLiNER

from eval_data import ALL_CASES, LABEL_ALIASES, MODEL_LABELS
from eval_gliner_ru import ROOT, canon, match_type


def quick_eval(model, threshold: float) -> tuple[float, float, float, float, float]:
    tp_exact = tp_found = fp = total_gold = 0
    for text, gold in ALL_CASES:
        raw = model.predict_entities(text, MODEL_LABELS, threshold=threshold)
        preds = [
            {"span": e["text"], "label": canon(e["label"])}
            for e in raw
            if canon(e["label"]) in {
                "person", "organization", "location", "phone",
                "email", "url", "money", "group_code",
            }
        ]
        matched_pred = set()
        for g_sub, g_label in gold:
            total_gold += 1
            for pi, p in enumerate(preds):
                if pi in matched_pred or p["label"] != g_label:
                    continue
                if match_type(p["span"], g_sub) != "none":
                    matched_pred.add(pi)
                    if match_type(p["span"], g_sub) == "exact":
                        tp_exact += 1
                    break
        fp += len(preds) - len(matched_pred)
        tp_exact += sum(
            1
            for g_sub, g_label in gold
            for pi, p in enumerate(preds)
            if pi not in matched_pred
            and p["label"] == g_label
            and match_type(p["span"], g_sub) == "exact"
        )
    # found считаем заново простым проходом
    found = 0
    for text, gold in ALL_CASES:
        raw = model.predict_entities(text, MODEL_LABELS, threshold=threshold)
        preds = [
            {"span": e["text"], "label": canon(e["label"])}
            for e in raw
            if canon(e["label"]) in {
                "person", "organization", "location", "phone",
                "email", "url", "money", "group_code",
            }
        ]
        used = set()
        for g_sub, g_label in gold:
            for pi, p in enumerate(preds):
                if pi in used or p["label"] != g_label:
                    continue
                if match_type(p["span"], g_sub) != "none":
                    used.add(pi)
                    found += 1
                    break
    prec_found = found / (found + fp) if found + fp else 0.0
    rec_found = found / total_gold if total_gold else 0.0
    prec_exact = tp_exact / (tp_exact + fp) if tp_exact + fp else 0.0
    rec_exact = tp_exact / total_gold if total_gold else 0.0
    f1_len = 2 * prec_found * rec_found / (prec_found + rec_found) if prec_found + rec_found else 0.0
    f1_str = 2 * prec_exact * rec_exact / (prec_exact + rec_exact) if prec_exact + rec_exact else 0.0
    return prec_exact, rec_exact, f1_str, f1_len, fp


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GLiNER.from_pretrained(str(ROOT / "models/gliner_multitask_large")).to(device)
    model.eval()
    print(f"multitask_large, свип порога (device={device})")
    print(f"{'thr':<6} {'P(exact)':>9} {'R(exact)':>9} {'F1(exact)':>10} {'F1(мягк.)':>10} {'FP':>4}")
    for thr in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]:
        p_e, r_e, f1_e, f1_l, fp = quick_eval(model, thr)
        print(f"{thr:<6} {p_e:>9.3f} {r_e:>9.3f} {f1_e:>10.3f} {f1_l:>10.3f} {fp:>4}")


if __name__ == "__main__":
    main()
