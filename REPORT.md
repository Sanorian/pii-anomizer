# Результаты: GLiNER на русском языке (NER)

Тест: 29 ручных кейсов (37 gold-сущностей) — классический NER, спам-сущности
(телефоны, e-mail, URL, деньги, номера групп), русские ловушки (падежи, строчный
регистр, кириллический домен) и 3 negative-кейса. Плюс 4 реальных доменных
спам-сообщения для ручного просмотра.

## Модели

| ID | Что это |
|---|---|
| `gliner_multi` | базовая `urchade/gliner_multi` (mdeberta-v3-base) |
| `multitask_large` | `knowledgator/gliner-multitask-large-v0.5` (XLM-R large) |
| `ru_finetuned` | gliner_multi + NEREL (3000 шагов) |
| `ru_finetuned_v2` | gliner_multi + NEREL + доменная синтетика (2000 шагов) |

Устройство: Apple MPS, threshold=0.4.

## Итоговые метрики (ручной набор)

| Модель | P (exact) | R (exact) | F1 (exact) | F1 (мягкий) |
|---|---|---|---|---|
| gliner_multi | 0.462 | 0.649 | 0.539 | 0.587 |
| multitask_large | 0.649 | 0.649 | 0.649 | 0.780 |
| ru_finetuned | 0.561 | 0.622 | 0.590 | 0.642 |
| **ru_finetuned_v2** | **0.698** | **0.811** | **0.750** | **0.780** |

## NEREL test (160 документов, exact match)

| Модель | P | R | F1 |
|---|---|---|---|
| gliner_multi (base) | 0.556 | 0.428 | 0.484 |
| **ru_finetuned_v2** | **0.769** | **0.792** | **0.780** |

Дообучение дало +0.30 F1 на «родных» данных NEREL.

## Zero-shot проверка (метки, которых НЕ было в обучении)

6 тестов: `sport team`, `scientific theory`, `century`, `painting`, `museum`,
`medical condition`, `medical treatment`, `position` на новых предложениях.

| Модель | Найдено |
|---|---|
| gliner_multi (base) | 14/14 |
| ru_finetuned (NEREL) | 14/14 |
| **ru_finetuned_v2** | **14/14 (100%)** |

**Zero-shot способность полностью сохранена после дообучения.**

## По меткам (ru_finetuned_v2, ручной набор)

| Метка | gold | found | exact | Комментарий |
|---|---|---|---|---|
| person | 10 | 10 | 9 | отличная работа, включая падежи |
| location | 5 | 5 | 5 | идеально |
| phone | 3 | 3 | 3 | идеально (было 0/3 у base) |
| group_code | 3 | 3 | 3 | идеально (было 0/3 у base) |
| email | 2 | 2 | 2 | идеально |
| url | 5 | 4 | 4 | кириллический URL теперь целиком |
| money | 5 | 2 | 2 | карты путает с phone |
| organization | 4 | 3 | 2 | рвёт «ПАО «Сбербанк»» |

## Что осталось сложным

1. **Номера карт/счетов** — классифицирует как `phone` (4-группы цифр).
   Лечится постфильтром: 16 цифр = карта, 20 = счёт, 11 = телефон.
2. **Границы с кавычками** («ПАО «Сбербанк»») — рвёт на части.
3. Скайп-ники помечает как `url`/`email` — не критично.
4. `@diplom_helper_bot` — email вместо url.

## Скорость

| Модель | c/кейс (MPS) |
|---|---|
| gliner_multi | 0.04 |
| multitask_large | 0.05 |
| ru_finetuned_v2 | **0.03** |

## Выводы

1. Дообучение на NEREL + доменная синтетика подняло F1 на ручном наборе с
   **0.539 → 0.750** (+0.21) и на NEREL test с **0.484 → 0.780** (+0.30).
2. Zero-shot способность не пострадала: 100% на новых метках после обучения.
3. ru_finetuned_v2 — лучшая по F1, при этом маленькая и самая быстрая.
   multitask_large остаётся хорошей альтернативой «без затрат на обучение».
4. Для production-анонимизации: ru_finetuned_v2 + регэкспы на карты/счета
   (модель находит их стабильно, но метит как phone — метку легко поправить
   постфильтром по длине цифровой последовательности).

## Как воспроизвести

```bash
./venv/bin/python convert_nerel.py       # NEREL -> GLiNER-формат
./venv/bin/python gen_synth_domain.py    # доменная синтетика
# смержить train.json + synth_domain.json (x3) -> train_merged.json
./venv/bin/python train_gliner_ru.py     # ~19 мин на MPS
./venv/bin/python eval_gliner_ru.py      # оценка
```

