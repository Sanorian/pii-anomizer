---
language: ru
license: mit
base_model: urchade/gliner_multi
tags:
- ner
- pii
- anonymization
- russian
- token-classification
library_name: gliner
pipeline_tag: token-classification
---

# GLiNER Russian PII

Модель детекции персональных данных (PII) в русскоязычных текстах на
архитектуре GLiNER (zero-shot NER). Дообучена из [urchade/gliner_multi](https://huggingface.co/urchade/gliner_multi)
(mdeberta-v3-base).

## На чём обучалась

Обучение выполнено исключительно на синтетических данных: ~1900
сгенерированных текстов с примерами типичных форматов российских ПДн —
ФИО, адреса (улица, дом, квартира, офис), телефоны, СНИЛС, ИНН, полисы
ОМС, номера водительских удостоверений, военных билетов и свидетельств
о рождении, URL и email. Синтетика покрывает реальные вариации
написания: разные разделители, опечатки в названиях типов документов,
«разболтанная» пунктуация в URL/email.

Реальные персональные данные не использовались.

## Метки

`person`, `organization`, `location`, `profession`, `age`, `date`,
`money`, `group code`, `phone`, `email`, `url`, `passport`,
`doc number`

## Качество

В составе пайплайна анонимизации (regex + GLiNER + Natasha) на открытом
бенчмарке [redmadrobot-rnd/pii_benchmark](https://huggingface.co/datasets/redmadrobot-rnd/pii_benchmark)
(2703 текста):

| категория | overlap F1 |
|---|---|
| email | 0.99 |
| url | 0.98 |
| oms | 0.93 |
| inn | 0.86 |
| doc_number | 0.88 |
| card | 0.88 |
| snils | 0.86 |
| **MICRO (весь пайплайн)** | **0.88** |

Сильные стороны: гранулярные категории закрыты regex-валидаторами
пайплайна, модель отвечает за контекстные сущности (person, location,
organization, profession, date, age, money).

## Использование

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("Sanorian/gliner-ru-pii")
labels = [
    "person", "organization", "location", "profession", "age", "date",
    "money", "group code", "phone", "email", "url", "passport", "doc number",
]
ents = model.predict_entities(
    "Иванов Иван, снилс 123-456-789 00", labels, threshold=0.45,
)
for e in ents:
    print(e["text"], e["label"], round(e["score"], 2))
```

Для структурированных категорий (phone/snils/inn и т.п.) рекомендуется
валидация формы спанов — модель галлюцинирует на структурированных
данных («СНИЛС» как url, «PDF» как email).

## Лицензия

MIT. Базовая модель [urchade/gliner_multi](https://huggingface.co/urchade/gliner_multi) — MIT.
