---
language: ru
license: mit
base_model: natasha/slovnet_ner_news_v1
tags:
- ner
- pii
- russian
- token-classification
library_name: natasha
---

# Slovnet NER Russian PII

Дообученная [Slovnet NER](https://github.com/natasha/slovnet) (Natasha)
для извлечения персональных данных из русскоязычных текстов. Тёплый
старт из [natasha/slovnet_ner_news_v1](https://huggingface.co/natasha/slovnet_ner_news_v1)
(Navec-эмбеддинги заморожены), 22 B/I-тега для 11 типов: person,
organization, location, date, age, money, url, phone, email,
profession, group code.

## На чём обучалась

Датасет: 2238 предложений с BIO-разметкой — конвертация открытого
новостного корпуса NEREL и синтетических текстов с форматами
российских ПДн (адреса, телефоны, документы, деньги).

Реальные персональные данные не использовались.

## Качество

Dev F1 = 0.79. В пайплайне анонимизации с regex и GLiNER даёт второй
независимый источник сигнала для date/age/money/organization.

## Использование

Модель распространяется как tar-архив в штатном формате
`SlovnetNER.load`:

```python
import tarfile
from slovnet import SlovnetNER

with tarfile.open("slovnet_ner_ru_ft_v1.tar", "rb") as f:
    f.extractall("model_dir")

ner = SlovnetNER.load("model_dir/model.pkg")
ner.navec(navec)
markup = ner("Макаров Родион Евгеньевич живёт в Воронеже")
```

В docker-сервисе архив подключается переменной окружения
`NATASHA_MODEL_TAR`.

## Лицензия

MIT. Базовая модель
[natasha/slovnet_ner_news_v1](https://huggingface.co/natasha/slovnet_ner_news_v1) — MIT.
