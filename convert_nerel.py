# -*- coding: utf-8 -*-
"""Конвертация NEREL-v1.1 в GLiNER-формат обучения.

NEREL: ann-файлы с символьными спанами, текст в txt.
Выход: JSON со списком {"tokenized_text": [...], "ner": [[start, end, label], ...]}
где start/end — индексы ТОКЕНОВ.

Нюансы:
- некоторые аннотации имеют несколько спанов через ';' — берём только первый
  сегмент (плоские сущности), вложенность отбрасываем фильтром перекрытий;
- 29 типов маппим в 16 компактных меток;
- документ режем на чанки ~max_words слов по границам предложений.
"""

import json
import random
import re
from pathlib import Path

SRC = Path(__file__).parent / "data_nerel" / "NEREL-v1.1"
OUT = Path(__file__).parent / "data_ft"

# Маппинг типов NEREL -> наши метки (компактный набор)
LABEL_MAP = {
    "PERSON": "person",
    "PROFESSION": "profession",
    "ORGANIZATION": "organization",
    "EVENT": "event",
    "DATE": "date",
    "COUNTRY": "location",
    "CITY": "location",
    "STATE_OR_PROVINCE": "location",
    "DISTRICT": "location",
    "LOCATION": "location",
    "FACILITY": "facility",
    "NUMBER": "number",
    "ORDINAL": "number",
    "AGE": "age",
    "AWARD": "award",
    "NATIONALITY": "nationality",
    "LAW": "law",
    "WORK_OF_ART": "work of art",
    "CRIME": "crime",
    "DISEASE": "disease",
    "IDEOLOGY": "ideology",
    "PRODUCT": "product",
    "MONEY": "money",
    "PENALTY": "money",
    "TIME": "time",
    "PERCENT": "percent",
    "LANGUAGE": "language",
    "RELIGION": "religion",
    "FAMILY": "person",
}

# Для сэмплирования меток в батчах
ALL_LABELS = sorted(set(LABEL_MAP.values()))

MAX_WORDS = 192  # чанк ~ до 192 токенов (GLiNER обучают на 384 с парами)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

random.seed(42)


def parse_ann(path: Path):
    """Возвращает [(start, end, nerel_type, text), ...]."""
    ents = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("T"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        tid, middle, text = parts[0], parts[1], parts[2]
        fields = middle.split()
        etype = fields[0]
        spans_part = middle[len(etype):].strip()
        # берём первый сегмент (до ';')
        first = spans_part.split(";")[0].strip()
        try:
            start, end = (int(x) for x in first.split())
        except ValueError:
            continue
        ents.append((start, end, etype, text))
    return ents


def resolve_overlaps(ents):
    """Убираем вложенные/перекрывающиеся сущности: сортируем по длине, оставляем первые."""
    ents_sorted = sorted(ents, key=lambda e: (e[0], -(e[1] - e[0])))
    kept = []
    taken = set()
    for start, end, etype, text in ents_sorted:
        if any(pos in taken for pos in range(start, end)):
            continue  # вложенная или пересекающаяся — пропускаем
        kept.append((start, end, etype, text))
        taken.update(range(start, end))
    return sorted(kept)


def tokenize_with_spans(text: str):
    """Токенизация слов + пунктуации, возвращает [(token, start, end)]."""
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE)]


def chunk_text(text: str):
    """Режем текст на чанки по границам предложений (~MAX_WORDS)."""
    sentences = SENT_SPLIT.split(text.strip())
    chunks, cur, cur_len = [], [], 0
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        n = len(sent.split())
        if cur_len + n > MAX_WORDS and cur:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(sent)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_text_with_offsets(text: str):
    """Режем текст на чанки по границам предложений, возвращаем [(chunk, offset)]."""
    sentences = SENT_SPLIT.split(text)
    chunks = []
    cur, cur_start, cur_len = [], None, 0
    pos = 0
    # вычисляем оффсеты предложений вручную
    sent_spans = []
    search_from = 0
    for sent in sentences:
        if not sent.strip():
            pos += len(sent)
            continue
        idx = text.find(sent, search_from)
        if idx < 0:
            continue
        sent_spans.append((idx, idx + len(sent)))
        search_from = idx + len(sent)

    for s_start, s_end in sent_spans:
        sent = text[s_start:s_end]
        n = len(sent.split())
        if cur_len + n > MAX_WORDS and cur:
            chunk_text_str = text[cur_start:cur[-1][1]]
            chunks.append((chunk_text_str, cur_start))
            cur, cur_start, cur_len = [], None, 0
        if cur_start is None:
            cur_start = s_start
        cur.append((s_start, s_end))
        cur_len += n
    if cur:
        chunk_text_str = text[cur_start:cur[-1][1]]
        chunks.append((chunk_text_str, cur_start))
    return chunks


def convert_split(split: str):
    """Конвертирует один сплит NEREL в GLiNER-примеры."""
    src_dir = SRC / split
    examples = []
    stats = {}
    skipped_long = 0
    for ann_path in sorted(src_dir.glob("*.ann")):
        txt_path = ann_path.with_suffix(".txt")
        text = txt_path.read_text(encoding="utf-8")
        ents = parse_ann(ann_path)
        ents = resolve_overlaps(ents)

        # Маппим и фильтруем
        mapped = []
        for start, end, etype, ent_text in ents:
            label = LABEL_MAP.get(etype)
            if label is None:
                continue
            # подрезаем по краям пробелы/кавычки из разметки
            while start < end and text[start] in " \n\t«»\"'":
                start += 1
            while end > start and text[end - 1] in " \n\t«»\"'":
                end -= 1
            if start >= end:
                continue
            mapped.append((start, end, label))

        for chunk, offset in chunk_text_with_offsets(text):
            chunk_ents = []
            chunk_tokenized = tokenize_with_spans(chunk)
            for start, end, label in mapped:
                # сущность должна целиком попадать в чанк
                if start < offset or end > offset + len(chunk):
                    continue
                w_start = None
                w_end = None
                for wi, (tok, ts, te) in enumerate(chunk_tokenized):
                    tok_off = ts + offset
                    if tok_off <= start < tok_off + len(tok) or (start <= tok_off and tok_off + len(tok) <= end):
                        if w_start is None:
                            w_start = wi
                        w_end = wi
                if w_start is None:
                    continue
                chunk_ents.append((w_start, w_end, label))
            if not chunk_ents:
                continue
            examples.append({
                "tokenized_text": [t for t, _, _ in chunk_tokenized],
                "ner": chunk_ents,
                "_chunk_text": chunk,
            })

        # статистика
        for _, _, label in mapped:
            stats[label] = stats.get(label, 0) + 1

    return examples, stats


def main():
    OUT.mkdir(exist_ok=True)
    random.seed(42)

    train_ex, train_stats = convert_split("train")
    val_ex, _ = convert_split("dev")
    test_ex, _ = convert_split("test")

    # Немного негативных примеров из train (чанки без сущностей)
    # GLiNER negatives-механизм сам справится, добавим 60 пустых чанков
    neg_added = 0
    for ann_path in sorted((SRC / "train").glob("*.ann"))[:60]:
        txt_path = ann_path.with_suffix(".txt")
        text = txt_path.read_text(encoding="utf-8")
        for chunk, _offset in chunk_text_with_offsets(text)[:1]:
            neg_added += 1
            train_ex.append({
                "tokenized_text": [t for t, _, _ in tokenize_with_spans(chunk)],
                "ner": [],
                "_chunk_text": chunk,
            })

    random.shuffle(train_ex)

    def strip_meta(examples):
        return [{"tokenized_text": e["tokenized_text"], "ner": e["ner"]} for e in examples]

    (OUT / "train.json").write_text(
        json.dumps(strip_meta(train_ex), ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "val.json").write_text(
        json.dumps(strip_meta(val_ex), ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "test.json").write_text(
        json.dumps(strip_meta(test_ex), ensure_ascii=False), encoding="utf-8"
    )

    print(f"train: {len(train_ex)} примеров (вкл. {neg_added} негативных)")
    print(f"val:   {len(val_ex)}")
    print(f"test:  {len(test_ex)}")
    print("\nРаспределение меток (train):")
    for lbl, cnt in sorted(train_stats.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<15} {cnt}")
    print(f"\nВсего уникальных меток: {len(ALL_LABELS)}: {ALL_LABELS}")


if __name__ == "__main__":
    main()
