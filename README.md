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

# 3. Поднять сервис (порт 8001 -> 8000), из корня репозитория
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
  -d '{"text": "Здравствуйте! Меня зовут Иванов, напишите в вацап 8 950 784 55 53. Компания ТехноСофт"}'
```

Ответ:

```json
{
  "anonymized_text": "Здравствуйте! Меня зовут [PERSON], напишите в вацап [PHONE]. Компания [ORG]",
  "entities": [
    {"start": 50, "end": 65, "text": "8 950 784 55 53", "category": "phone",
     "placeholder": "[PHONE]", "source": "regex", "score": 1.0},
    {"start": 76, "end": 85, "text": "ТехноСофт", "category": "organization",
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
docker-compose.yml     запуск сервиса из корня (порт 8001 -> 8000)
service/               FastAPI-сервис (app/, tests/, Dockerfile, requirements)
```

Модели дообучались на синтетике и открытых корпусах — скрипты обучения
и подготовки данных в репо не входят, только продовый сервис.

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

## Автор

**Крайнов Александр Витальевич** — Fullstack/ML-разработчик
([Hugging Face](https://huggingface.co/Sanorian),
[GitHub](https://github.com/Sanorian))

## Лицензия

MIT
