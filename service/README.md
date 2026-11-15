# Сервис анонимизации ПДн

Маскирование персональных и коммерчески чувствительных данных в русских текстах.
Гибрид из трёх детекторов: **регулярные выражения** (детерминированные форматы),
**GLiNER** (дообучен на NEREL + доменная синтетика) и **Natasha Slovnet NER**
(pretrained). Слияние union: сущность маскируется, если нашёл хотя бы один
детектор; пересечения разрешаются по приоритету regex > GLiNER > Natasha.

## Запуск

### Локально

```bash
cd service
../venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Модели ищутся в `../models/gliner_ru_finetuned` (переопределяется переменной
`GLINER_MODEL_DIR`). Natasha скачивает эмбеддинги при первом запуске.

### Docker

```bash
# из корня репозитория (docker-compose.yml в корне, собирает из service/)
docker compose up --build
```

Модели монтируются из `./models` (read-only). Сервис на CPU: GLiNER ~0.3 c на
сообщение, Natasha ~5 мс, regex мгновенно. Вместе с сервисом поднимается
контейнер **guardrails** (порт 8002): проверка входных текстов на
промпт-инъекции (правила RU+EN + ML-модель) — см. корневой README,
раздел «Guardrails». Проверка включена по умолчанию, отключается
полем запроса `"check_injection": false`; при недоступности guardrails
анонимизация продолжается (fail-open).

## API

### `POST /anonymize`

```bash
curl -X POST http://localhost:8000/anonymize \
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
  ],
  "stats": {"phone": 1, "organization": 1}
}
```

Параметры:
- `text` — исходный текст (до 16 000 символов);
- `threshold` — порог уверенности GLiNER, 0.1–0.95 (по умолчанию 0.45);
- `categories` — список категорий для маскирования (None = все).

### `GET /categories`

Список всех категорий и плейсхолдеров.

### `GET /health`

Статус загрузки моделей.

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
| doc_number | `[DOC_NUMBER]` | regex (ВУ, свидетельство о рождении, военный билет) |
| money | `[MONEY]` | regex + GLiNER |
| person | `[PERSON]` | GLiNER + Natasha |
| organization | `[ORG]` | GLiNER + Natasha |
| location | `[LOCATION]` | GLiNER + Natasha |
| profession | `[PROFESSION]` | GLiNER |
| age | `[AGE]` | GLiNER |
| date | `[DATE]` | GLiNER |
| group_code | `[GROUP_CODE]` | GLiNER |

## Тесты

```bash
cd service
../venv/bin/python -m pytest tests/ -v
```

26 тестов: Luhn, все форматы телефонов/документов, слияние пересечений,
фильтр категорий, корректность оффсетов, негативные кейсы (ham не маскируется).

## Архитектура

```
POST /anonymize
   │
   ▼
guardrails /scan    ── промпт-инъекции: правила RU+EN + mdeberta-ru
   │                   (report: флаг в ответе; block: отказ; fail-open)
   ▼
AnonymizationPipeline
   ├─ RegexDetector      ── телефоны, e-mail, URL, карты (Luhn), счета,
   │                        СНИЛС, ИНН, паспорт, деньги
   ├─ GlinerDetector     ── person, org, location, profession, age, date,
   │                        money, group code (дообученный GLiNER)
   └─ NatashaDetector    ── PER, LOC, ORG (Slovnet, pretrained)
   │
   ▼
_resolve_overlaps()  ── union-слияние: приоритет источника, длина, score
   │
   ▼
mask_text()          ── замена плейсхолдерами, оффсеты оригинала сохранены
```

## Автор

**Крайнов Александр Витальевич** — Fullstack/ML-разработчик
([Hugging Face](https://huggingface.co/Sanorian),
[GitHub](https://github.com/Sanorian))

## Лицензия

MIT
