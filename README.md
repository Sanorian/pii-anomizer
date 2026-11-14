# PII Anomizer — сервис анонимизации русских текстов

Маскирование персональных данных (ПДн) и коммерчески чувствительной
информации в русскоязычных текстах. Гибрид трёх детекторов:

- **regex** — детерминированные форматы: телефоны, e-mail, URL, карты
  (с проверкой Luhn), СНИЛС, ИНН, паспорт, ОМС, номера документов, деньги;
- **GLiNER** — контекстные сущности: person, organization, location,
  profession, age, date, money, group code. Дообучен на NEREL и
  синтетике с форматами российских ПДн
  ([Sanorian/gliner-ru-pii](https://huggingface.co/Sanorian/gliner-ru-pii));
- **Natasha Slovnet NER** — второй независимый источник для person, org,
  location ([Sanorian/slovnet-ner-ru-pii](https://huggingface.co/Sanorian/slovnet-ner-ru-pii)).

Слияние — union: сущность маскируется, если нашёл хотя бы один детектор;
пересечения разрешаются по приоритету regex > GLiNER > Natasha.

Качество на открытом бенчмарке
[redmadrobot-rnd/pii_benchmark](https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark)
(2703 текста): **MICRO overlap F1 = 0.881**.

## Быстрый старт

### Docker (рекомендуется)

```bash
# 1. Скачать модель GLiNER (~1.1 GB)
huggingface-cli download Sanorian/gliner-ru-pii \
    --local-dir models/gliner_ru_finetuned_v2

# 2. Скачать Natasha-модель (~2.4 MB)
huggingface-cli download Sanorian/slovnet-ner-ru-pii \
    --local-dir models

# 3. Поднять сервис (порт 8001 -> 8000)
cd service
docker compose up --build
```

Natasha-модель монтируется как tar: `slovnet_ner_ru_ft_v1.tar` лежит
в `models/`, эмбеддинги Navec качаются при первом старте в named volume.

### Локально

```bash
python -m venv venv && venv/bin/pip install -r service/requirements.txt
venv/bin/pip install gliner natasha

huggingface-cli download Sanorian/gliner-ru-pii --local-dir models/gliner_ru_finetuned_v2
huggingface-cli download Sanorian/slovnet-ner-ru-pii --local-dir models

cd service
../venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /anonymize`

```bash
curl -X POST http://localhost:8001/anonymize \
  -H "Content-Type: application/json" \
  -d '{"text": "Здравствуйте! Я ваша одногруппница, напишите в вацап 8 950 784 55 53. ТГУ РОСДИСТАНТ"}'
```

Ответ:

```json
{
  "anonymized_text": "Здравствуйте! Я ваша одногруппница, напишите в вацап [PHONE]. [ORG]",
  "entities": [
    {"start": 54, "end": 67, "text": "8 950 784 55 53", "category": "phone",
     "placeholder": "[PHONE]", "source": "regex", "score": 1.0},
    {"start": 69, "end": 84, "text": "ТГУ РОСДИСТАНТ", "category": "organization",
     "placeholder": "[ORG]", "source": "gliner", "score": 0.99}
  ]
}
```

Параметры: `text` (до 16 000 символов), `threshold` (порог GLiNER,
по умолчанию 0.45), `categories` (список категорий для маскирования).

### `GET /categories`, `GET /health`

Список категорий и статус моделей.

## Категории

| Категория | Плейсхолдер | Детектор |
|---|---|---|
| phone | `[PHONE]` | regex |
| email | `[EMAIL]` | regex |
| url | `[URL]` | regex (http, www, t.me, @ники) |
| card | `[CARD]` | regex + Luhn |
| account | `[ACCOUNT]` | regex (20 цифр) |
| snils | `[SNILS]` | regex |
| inn | `[INN]` | regex |
| passport | `[PASSPORT]` | regex |
| oms | `[OMS]` | regex (16 цифр + контекст «полис/ОМС») |
| doc_number | `[DOC_NUMBER]` | regex (ВУ, свидетельство, военный билет) |
| money | `[MONEY]` | regex + GLiNER |
| person | `[PERSON]` | GLiNER + Natasha |
| organization | `[ORG]` | GLiNER + Natasha |
| location | `[LOCATION]` | GLiNER + Natasha |
| profession | `[PROFESSION]` | GLiNER |
| age | `[AGE]` | GLiNER |
| date | `[DATE]` | GLiNER |
| group_code | `[GROUP_CODE]` | GLiNER |

## Архитектура

```
POST /anonymize
   │
   ▼
AnonymizationPipeline
   ├─ RegexDetector      ── телефоны, e-mail, URL, карты (Luhn), счета,
   │                        СНИЛС, ИНН, паспорт, деньги
   ├─ GlinerDetector     ── person, org, location, profession, age, date,
   │                        money, group code (+ склейка смежных спанов)
   └─ NatashaDetector    ── PER, LOC, ORG (Slovnet)
   │
   ▼
_resolve_overlaps()  ── union-слияние: приоритет источника, длина, score
   │
   ▼
mask_text()          ── замена плейсхолдерами, оффсеты оригинала сохранены
```

## Структура репозитория

```
service/               FastAPI-сервис (app/, tests/, Dockerfile, docker-compose.yml)
eval_benchmark.py      eval на pii_benchmark (strict/overlap, per-category, --gliner/--threshold)
convert_nerel.py       конвертация корпуса NEREL в GLiNER span-формат
convert_pii_train.py   конвертация redmadrobot-rnd/pii_train (BIO -> span, дедуп)
prepare_natasha_data.py  подготовка BIO-данных для Slovnet (NEREL + синтетика)
train_gliner_ru.py     дообучение GLiNER (early stopping, MPS/CPU)
train_natasha.py       дообучение Slovnet NER
gen_synth_domain.py    генерация доменной синтетики (адреса, документы, ОМС…)
gen_synth_weak_v2.py   генерация weak-supervision синтетики
benchmark/             eval-скрипты логов, eval_results*.json по версиям моделей
data_ft/               конвертированные обучающие данные (см. «Данные»)
```

## Данные

Сторонние датасеты не хранятся в репо — скачиваются:

- `redmadrobot-rnd/pii_benchmark` — eval
  (`benchmark/pii_benchmark_raw.jsonl`, 3.2 MB);
- `redmadrobot-rnd/pii_train` — опционально, для воспроизведения
  эксперимента с дообучением на нём (`convert_pii_train.py`);
- [NEREL](https://github.com/natasha/nerel) — корпус новостей с BIO-разметкой
  (`data_nerel/`, использовался в ранних версиях обучения).

Обучающие JSON в `data_ft/` генерируются конвертерами из этих датасетов
(`convert_*`), либо берутся из нашей синтетики (`gen_synth_*.py`).

Наши синтетические данные (`synth_domain.json`, `synth_weak_v2.json`)
в репо есть: сгенерированы скриптами `gen_synth_*.py`, реальных ПДн
не содержат.

## Тесты

```bash
cd service
../venv/bin/python -m pytest tests/ -v
```

35 тестов: Luhn, форматы телефонов/документов, слияние пересечений,
склейка смежных GLiNER-спанов, фильтр категорий, корректность оффсетов,
негативные кейсы.

## Модели

| Модель | HF | Что делает |
|---|---|---|
| GLiNER ru | [Sanorian/gliner-ru-pii](https://huggingface.co/Sanorian/gliner-ru-pii) | 13 меток: person, org, location, profession, age, date, money, group code, phone, email, url, passport, doc number |
| Slovnet NER ru | [Sanorian/slovnet-ner-ru-pii](https://huggingface.co/Sanorian/slovnet-ner-ru-pii) | 11 типов, BIO |

Обе модели MIT, обучены без реальных персональных данных — детали в
model cards.

## Лицензия

MIT
