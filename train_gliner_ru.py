# -*- coding: utf-8 -*-
"""Обучение GLiNER на русских данных (NEREL).

Дообучаем gliner_multi (mdeberta-v3-base) с сохранением zero-shot способности:
- negatives + рандомизация меток в батчах (штатный механизм GLiNER через collator);
- низкий LR на энкодер, выше на остальные слои;
- промежуточные чекпоинты.

Запуск: ./venv/bin/python train_gliner_ru.py
"""

from pathlib import Path

import torch
from gliner import GLiNER
from transformers import EarlyStoppingCallback
from gliner.training import Trainer

ROOT = Path(__file__).parent
DATA = ROOT / "data_ft"
# round 5: старт с базового gliner_multi (данные pii_train сильно отличаются от round-1/2)
BASE_MODEL = ROOT / "models" / "gliner_ru_finetuned_v2"  # round 5b: тёплый старт от v2
OUT_DIR = ROOT / "models" / "ft_output_v5_pii_ws"

# Гиперпараметры (подобраны под MPS/Mac, 16 ГБ unified memory)
MAX_STEPS = 4000        # лимит задаёт early stopping, не шаги
LR_ENCODER = 5e-6
LR_OTHERS = 2e-5
BATCH_SIZE = 1          # MPS: OOM при 4 на длинных чанках
MAX_LEN = 384
NEGATIVES = 1.0
EVAL_STEPS = 250
SAVE_STEPS = 250
SAVE_LIMIT = 3
LOGGING_STEPS = 50
EARLY_STOP_PATIENCE = 4  # остановка после 4 оценок (1000 шагов) без улучшения


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Устройство: {device}")

    train_data = json.load(open(DATA / "train_v3.json", encoding="utf-8"))
    val_data = json.load(open(DATA / "dev_v3.json", encoding="utf-8"))
    all_labels = sorted({lbl for ex in train_data for _, _, lbl in ex["ner"]})
    print(f"Метки ({len(all_labels)}): {all_labels}")
    print(f"train={len(train_data)}, val={len(val_data)}")

    model = GLiNER.from_pretrained(str(BASE_MODEL))

    # Тело model.train_model с ручным созданием Trainer — чтобы передать
    # callbacks (EarlyStoppingCallback). gliner-Trainer не имеет
    # compute_metrics: eval-метрика только loss => metric_for_best_model=loss.
    # Итоговый выбор лучшего чекпоинта делаем по F1 (model.evaluate) после
    # обучения — см. eval_checkpoints.
    training_args = model.create_training_args(
        output_dir=str(OUT_DIR),
        learning_rate=LR_ENCODER,
        others_lr=LR_OTHERS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        max_grad_norm=1.0,
        negatives=NEGATIVES,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        max_steps=MAX_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        save_total_limit=SAVE_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        report_to="none",
        use_cpu=(device == "cpu"),
        bf16=False,  # MPS не дружит с bf16 в этой сборке
        dataloader_num_workers=0,  # spawn-процессы ломают пайпы на macOS
    )
    data_collator = model._create_data_collator()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PATIENCE)
        ],
    )
    trainer.train()
    print("\nОбучение завершено (best model по eval_loss загружен).")
    # сохранить best-модель (load_best_model_at_end уже подгрузил её в model)
    model.save_pretrained(str(OUT_DIR / "best"))
    print(f"Best сохранён: {OUT_DIR / 'best'}")


if __name__ == "__main__":
    import json

    main()
