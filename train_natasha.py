# -*- coding: utf-8 -*-
"""Дообучение Slovnet NER (Natasha) на данных ПДн.

Стратегия:
- Эмбеддинги Navec frozen (PQ-коды без градиента) — как в pretrained.
- CNN-энкодер и CRF инициализируем из pretrained-модели (тёплый старт),
  голову (proj) — заново под новый набор меток.
- Loss: CRF (negative log-likelihood) с маской паддинга.
- Экспорт: tar-пакет формата slovnet (meta/model/arrays/vocabs),
  совместимый с natasha.NewsNERTagger.

Запуск: ../venv/bin/python train_natasha.py
"""

import gzip
import json
import tarfile
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import torch
from torch import nn

from slovnet.model.tag import NER as TrainNER, NERHead, TagEncoder, TagEmbedding
from slovnet.model.emb import NavecEmbedding
from slovnet.model.crf import CRF
from slovnet.model.cnn import CNNEncoder
from slovnet.vocab import Vocab
from slovnet.shape import SHAPES as SHAPE_ITEMS, word_shape
from slovnet.const import UNK, PAD
from slovnet.model.exec import ExecVisitor

from navec import Navec

ROOT = Path(__file__).resolve().parent
PRETRAINED_TAR = ROOT / "venv/lib/python3.13/site-packages/natasha/data/model/slovnet_ner_news_v1.tar"
NAVEC_TAR = ROOT / "venv/lib/python3.13/site-packages/natasha/data/emb/navec_news_v1_1B_250K_300d_100q.tar"
DATA_DIR = ROOT / "data_natasha"
OUT_TAR = ROOT / "models" / "slovnet_ner_ru_ft_v1.tar"

# --- Гиперпараметры ---
BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-3
MAX_LEN = 200  # обрезка длинных последовательностей
SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_bio_data(path: Path) -> list[tuple[list[str], list[str]]]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            items.append((d["words"], d["tags"]))
    return items


def extract_pretrained(tar_path: Path) -> dict:
    """Достаёт vocab слов, shape-элементы и веса CNN из pretrained tar."""
    with tarfile.open(tar_path) as tar:
        words = gzip.decompress(tar.extractfile("vocabs/word.gz").read()).decode().splitlines()
        model_json = json.load(tar.extractfile("model.json"))
        weights = model_json["encoder"]["layers"]
        head = model_json["head"]
        # arrays
        arrays = {}
        for m in tar.getmembers():
            if m.name.startswith("arrays/"):
                arrays[m.name] = tar.extractfile(m).read()
    return {"words": words, "encoder_json": weights, "head_json": head, "arrays": arrays}


def load_navec(tar_path: Path) -> Navec:
    return Navec.load(str(tar_path))


# Shape vocab: pretrained = [<pad>] + SHAPES (66). Наш: [<pad>, <unk>] + SHAPES (67),
# т.е. наш id = pretrained id + 1. Держим свой vocab и грузим веса со сдвигом.
SHAPE_DIM = 30
SHAPE_VOCAB = Vocab(["<pad>", "<unk>"] + list(SHAPE_ITEMS))


def build_model(tags: Vocab, pretrained: dict) -> TrainNER:
    """Собирает train-модель: Navec frozen + shape + CNN (pretrained) + новая голова."""
    navec = load_navec(NAVEC_TAR)
    emb_word = NavecEmbedding(navec)
    emb_shape = nn.Embedding(len(SHAPE_VOCAB), SHAPE_DIM)
    emb_word.dim = emb_word.dim  # PQEmbedding определяет dim
    emb_shape.dim = SHAPE_DIM  # slovnet-стиль: атрибут dim у эмбеддингов
    emb = TagEmbedding(emb_word, emb_shape)

    encoder = CNNEncoder(emb.dim, [256, 128, 64], 3)
    head = NERHead(64, len(tags))

    # Тёплый старт: веса CNN + shape-эмбеддинг из pretrained (layout: id = индексу)
    import numpy as np
    arrays = pretrained["arrays"]

    def load_arr(arr_id: int, tensor: torch.Tensor) -> None:
        data = arrays.get(f"arrays/{arr_id}.bin")
        if data is not None:
            arr = np.frombuffer(data, dtype="float32")
            expected = int(np.prod(tensor.shape))
            if arr.size != expected:
                return  # размер не сошёлся — пропускаем (не тот массив)
            tensor.data.copy_(torch.from_numpy(arr.reshape(tensor.shape)))

    # shape-эмбеддинг pretrained: 66 строк = [<pad>] + 65 SHAPES.
    # Наш vocab: [<pad>, <unk>] + 65 SHAPES (id = pretrained id + 1).
    pretrained_shapes = np.frombuffer(arrays["arrays/0.bin"], dtype="float32").reshape(66, SHAPE_DIM)
    with torch.no_grad():
        emb_shape.weight[0] = 0.0                       # <pad>
        emb_shape.weight[2:] = torch.from_numpy(pretrained_shapes[1:])  # SHAPES
    # <unk> (строка 1) остаётся случайной — встречается редко
    for li, layer in enumerate(encoder.layers):
        base = li * 7 + 1
        load_arr(base, layer.conv.weight)
        load_arr(base + 1, layer.conv.bias)
        load_arr(base + 2, layer.norm.weight)
        load_arr(base + 3, layer.norm.bias)
        load_arr(base + 4, layer.norm.running_mean)
        load_arr(base + 5, layer.norm.running_var)

    model = TrainNER(emb, encoder, head)
    return model


def words_to_ids(words: list[str], vocab: Vocab, max_len: int) -> tuple[list[int], list[int]]:
    unk_word = vocab.unk_id if vocab.unk_id is not None else 0
    unk_shape = SHAPE_VOCAB.unk_id if SHAPE_VOCAB.unk_id is not None else 0
    ids, shapes = [], []
    for w in words[:max_len]:
        wid = vocab.encode(w.lower())
        ids.append(unk_word if wid is None else wid)
        sid = SHAPE_VOCAB.encode(word_shape(w))
        shapes.append(unk_shape if sid is None else sid)
    return ids, shapes


def _shape(word: str) -> str:
    from slovnet.shape import word_shape
    return word_shape(word)


def pad_batch(batch: list[tuple[list[int], list[int]]], pad_word: int, pad_shape: int):
    max_len = max(len(ids) for ids, _ in batch)
    word_ids, shape_ids, mask = [], [], []
    for ids, shapes in batch:
        n = len(ids)
        word_ids.append(ids + [pad_word] * (max_len - n))
        shape_ids.append(shapes + [pad_shape] * (max_len - n))
        mask.append([1] * n + [0] * (max_len - n))
    # UNK вместо pad- слов (pad не должен встречаться во входе до маски)
    return (
        torch.tensor(word_ids, dtype=torch.long),
        torch.tensor(shape_ids, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )


def crf_nll(crf: CRF, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CRF loss (средний NLL по батчу). emissions: batch x seq x tags."""
    emissions = emissions.transpose(1, 0)  # seq x batch x tags
    tags = tags.transpose(1, 0)
    mask = mask.transpose(1, 0)
    log_likelihood = crf.score(emissions, tags, mask) - crf.normalization(emissions, mask)
    return -log_likelihood.mean()


def viterbi_decode(crf: CRF, emissions: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
    """Жадный viterbi: emissions batch x seq x tags, mask batch x seq."""
    transitions = crf.transitions.detach()
    paths = []
    em = emissions.detach()
    m = mask
    for b in range(em.shape[0]):
        seq_len = int(m[b].sum())
        score = em[b, 0]
        history = []
        for t in range(1, seq_len):
            # score[t] = max_prev (score + trans) + emission
            cand = score.unsqueeze(1) + transitions  # tags x tags
            best, best_idx = cand.max(dim=0)
            score = best + em[b, t]
            history.append(best_idx)
        # backtrace
        path = [0] * seq_len
        path[seq_len - 1] = int(score.argmax())
        for t in range(seq_len - 2, -1, -1):
            path[t] = int(history[t][path[t + 1]])
        paths.append(path)
    return paths


def span_f1(pred_spans: list[set], gold_spans: list[set]) -> tuple[float, float, float]:
    tp = sum(len(p & g) for p, g in zip(pred_spans, gold_spans))
    n_pred = sum(len(p) for p in pred_spans)
    n_gold = sum(len(g) for g in gold_spans)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def spans_from_tags(tags: list[str]) -> set[tuple[int, int, str]]:
    result = set()
    start, etype = None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag == "O":
            if etype:
                result.add((start, i, etype))
            start, etype = None, None
        else:
            part, t = tag.split("-", 1)
            if part == "B":
                if etype:
                    result.add((start, i, etype))
                start, etype = i, t
            elif etype != t:
                if etype:
                    result.add((start, i, etype))
                start, etype = i, t
    return result


def evaluate(model, crf, data, words_vocab, batch_size=32) -> tuple[float, float, float]:
    model.eval()
    pad_w, pad_s = words_vocab.pad_id, SHAPE_VOCAB.pad_id
    pred_spans, gold_spans = [], []
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            chunk = data[i : i + batch_size]
            batch = [words_to_ids(w, words_vocab, MAX_LEN) for w, _ in chunk]
            word_ids, shape_ids, mask = pad_batch(batch, pad_w, pad_s)
            if DEVICE != "cpu":
                word_ids, shape_ids, mask = word_ids.to(DEVICE), shape_ids.to(DEVICE), mask.to(DEVICE)
            em = model(word_ids, shape_ids, ~mask)
            paths = viterbi_decode(model.head.crf, em, mask)
            for (words, tags), path in zip(chunk, paths):
                pred_tags = [TAG_VOCAB.decode(idx) for idx in path]
                pred_spans.append(spans_from_tags(pred_tags))
                gold_spans.append(spans_from_tags(tags[:MAX_LEN]))
    return span_f1(pred_spans, gold_spans)


def export_tar(model, tags_vocab: Vocab, words_vocab: Vocab, out_path: Path) -> None:
    """Экспорт train-модели в slovnet tar (exec-формат, совместим с natasha.NERTagger).

    Цепочка как в оригинальном пакете slovnet:
    ExecVisitor -> strip_navec (natasha инжектит navec при загрузке)
    -> separate_arrays (numpy -> id) -> DumpPack (tar).
    """
    import numpy as np
    from slovnet.exec.pack import DumpPack, Meta
    from slovnet.const import WORD, SHAPE, TAG

    model.eval()
    exec_model = ExecVisitor().visit(model.to("cpu"))

    # natasha.NERTagger.__init__ вызывает self.navec(emb), который делает
    # inject_navec — поэтому в tar navec-массивы кладём пустыми (id-ссылки нет).
    exec_model = exec_model.strip_navec()

    arrays, scheme = exec_model.separate_arrays()

    out_path.parent.mkdir(exist_ok=True)
    with DumpPack(str(out_path)) as pack:
        pack.dump_meta(Meta("slovnet_ner_ru_ft_v1", 1))
        pack.dump_model(scheme)
        pack.dump_arrays(arrays)
        pack.dump_vocab(words_vocab, WORD)
        pack.dump_vocab(SHAPE_VOCAB, SHAPE)
        pack.dump_vocab(tags_vocab, TAG)
    print(f"Экспортировано: {out_path} ({out_path.stat().st_size / 1e6:.1f} МБ, массивов: {len(arrays)})")


def main() -> None:
    torch.manual_seed(SEED)

    train = load_bio_data(DATA_DIR / "train.jsonl")
    dev = load_bio_data(DATA_DIR / "dev.jsonl")
    print(f"train: {len(train)}, dev: {len(dev)}, device: {DEVICE}")

    # Теги
    tag_items = ["<pad>", "O"]
    for t in sorted({t.split("-", 1)[1] for _, tags in train for t in tags if t != "O"}):
        tag_items += [f"B-{t}", f"I-{t}"]
    tags_vocab = Vocab(tag_items)
    globals()["TAG_VOCAB"] = tags_vocab
    print("Теги:", tag_items)

    # Vocab слов от pretrained (+ новые слова как <unk> — допустимо)
    pretrained = extract_pretrained(PRETRAINED_TAR)
    words_vocab = Vocab(pretrained["words"])
    unk_count = sum(1 for w, _ in train for x in w if words_vocab.encode(x.lower()) == words_vocab.unk_id)
    total = sum(len(w) for w, _ in train)
    print(f"Словарь pretrained: {len(words_vocab)}, UNK-доля в train: {unk_count / total:.1%}")

    # CRF-веса pretrained (8 меток) не подходят под новые метки — голова новая.
    # CRF transitions инициализируем заново (NERHead делает это сам).

    model = build_model(tags_vocab, pretrained).to(DEVICE)
    print(f"Обучаемых параметров: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    crf = model.head.crf
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR
    )

    pad_w, pad_s = words_vocab.pad_id, SHAPE_VOCAB.pad_id
    n_batches = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
    best_f1, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        total_loss, t0 = 0.0, time.time()
        for bi in range(n_batches):
            chunk = [train[i] for i in order[bi * BATCH_SIZE : (bi + 1) * BATCH_SIZE]]
            batch = [words_to_ids(w, words_vocab, MAX_LEN) for w, _ in chunk]
            word_ids, shape_ids, mask = pad_batch(batch, pad_w, pad_s)
            if DEVICE != "cpu":
                word_ids, shape_ids, mask = word_ids.to(DEVICE), shape_ids.to(DEVICE), mask.to(DEVICE)
            # Теги в матрицу batch x seq (pad=0), ширина = ширине word_ids,
            # маска отсечёт лишнее в CRF
            width = word_ids.shape[1]
            tag_rows = []
            for _, tags in chunk:
                enc = [tags_vocab.encode(t) for t in tags[:width]]
                tag_rows.append(enc + [0] * (width - len(enc)))
            tag_ids = torch.tensor(tag_rows, dtype=torch.long)
            if DEVICE != "cpu":
                tag_ids = tag_ids.to(DEVICE)

            em = model(word_ids, shape_ids, ~mask)
            loss = crf_nll(crf, em, tag_ids, mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        p, r, f1 = evaluate(model, crf, dev, words_vocab)
        marker = ""
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = "  <- best"
        print(
            f"epoch {epoch}: loss={total_loss / n_batches:.3f} "
            f"dev P={p:.3f} R={r:.3f} F1={f1:.3f} ({time.time() - t0:.0f}с){marker}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    export_tar(model, tags_vocab, words_vocab, OUT_TAR)
    model.to(DEVICE)  # export_tar переводит модель на CPU для numpy-экспорта

    # Финальная оценка
    p, r, f1 = evaluate(model, crf, dev, words_vocab)
    print(f"\nФинал dev: P={p:.3f} R={r:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()
