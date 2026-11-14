# -*- coding: utf-8 -*-
"""Синтетика раунда 2: слабые категории v6 — passport, doc_number, url.

Форматы взяты из FN/FP-анализа бенчмарка (benchmark/error_analysis.json):
- паспорт: "45 03 N 123456", "45.12.123456", "45_12_123456", "7703 - 556677",
  "серия 2223, номер 777777", "1122 334455", опечатки («паспард», «папорт»)
- doc_number: ВУ/свидетельства/военники — "II - КЗ 123456", "АВ 1234567",
  "серия IV - ОМ, номер 987654", "77 19 и номер 123456", "44 / 33 987654",
  "ВУ 7705 и номер 987654", серии без номера ("Серия моего билета — РА")
- url без схемы: "bitbucket.org/page", "sberbank.ru/login",
  "docs.python.org", "schedule.learn-online.org"

Выход: data_ft/synth_weak_v2.json (GLiNER-формат).
Запуск: ./venv/bin/python gen_synth_weak_v2.py
"""

import json
import random
import re
from pathlib import Path

OUT = Path(__file__).parent / "data_ft" / "synth_weak_v2.json"

random.seed(2026)

# --- Словари ---
FIRST_NAMES = ["Алексей", "Мария", "Дмитрий", "Ольга", "Сергей", "Анна", "Иван",
               "Екатерина", "Николай", "Светлана", "Павел", "Юлия", "Андрей"]
LAST_NAMES = ["Смирнов", "Иванова", "Кузнецов", "Петрова", "Соколов", "Морозова",
              "Волков", "Козлова", "Николаевна", "Валерьевна"]

ROMAN = ["I", "II", "III", "IV", "VII", "IX", "X", "XIII"]
CYR_SERIES = ["АВ", "КЗ", "ОМ", "МЮ", "АГ", "ЛД", "ЕЖ", "МЗ", "РА", "ВГ", "ЫЭ", "АБ", "КК", "УХ"]


def digits(n: int) -> str:
    return "".join(random.choice("0123456789") for _ in range(n))


def gen_passport_series() -> str:
    """4 цифры серии: 4512 / 45 12 / 45-12."""
    d = digits(4)
    form = random.random()
    if form < 0.5:
        return d
    if form < 0.75:
        return f"{d[:2]} {d[2:]}"
    return f"{d[:2]}-{d[2:]}"


def gen_passport_number() -> str:
    """6 цифр номера: 123456 / 123 456 / 123-456."""
    d = digits(6)
    form = random.random()
    if form < 0.6:
        return d
    if form < 0.8:
        return f"{d[:3]} {d[3:]}"
    return f"{d[:3]}-{d[3:]}"


def passport_str() -> str:
    """Полный паспорт в случайном формате."""
    form = random.random()
    ser, num = gen_passport_series(), gen_passport_number()
    if form < 0.15:
        return f"{ser} N {num}" if random.random() < 0.5 else f"{ser} № {num}"
    if form < 0.3:  # точечные/прочерковые разделители
        sep = random.choice([".", "_", "~", "="])
        return f"{digits(2)}{sep}{digits(2)}{sep}{digits(6)}"
    if form < 0.5:  # через дефис: 4719-596586 / 7703 - 556677
        return f"{digits(4)}{' - ' if random.random() < 0.4 else '-'}{digits(6)}"
    if form < 0.75:  # словесная конструкция
        return f"серия {ser}, номер {num}"
    return f"{ser} {num}"  # "45 10 123456" / "4512 123456"


PASSPORT_WORDS = ["паспорт", "паспорта", "паспорту", "паспарт", "папорт", "паспортом",
                  "российский паспорт", "внутренний паспорт", "гражданский паспорт"]

DOC_TYPES = [
    ("водительское удостоверение", "ВУ"),
    ("водительские права", "прав"),
    ("права", "права"),
    ("свидетельство о рождении", "св-во"),
    ("свидетельство", "св-во"),
    ("военный билет", "военник"),
    ("военник", "военник"),
    ("удостоверение", "удостов."),
]


def gen_doc_series() -> str:
    """Серия документа: римские+буквы, 2 буквы, 2+2 цифры."""
    form = random.random()
    if form < 0.4:
        return f"{random.choice(ROMAN)}{' - ' if random.random() < 0.5 else '-'}{random.choice(CYR_SERIES)}"
    if form < 0.7:
        return random.choice(CYR_SERIES)
    return f"{digits(2)} {digits(2)}"


def doc_number_str() -> str:
    """Полная конструкция номер-документа."""
    ser = gen_doc_series()
    num_len = random.choice([6, 6, 6, 7])
    num = digits(num_len)
    form = random.random()
    if form < 0.35:
        return f"серия {ser}, номер {num}"
    if form < 0.5:
        return f"{ser} {num}"
    if form < 0.6:
        return f"{ser} № {num}"
    if form < 0.7:
        return f"серию {ser} и номер {num}"
    if form < 0.8:
        return f"{ser} - {num}" if random.random() < 0.5 else f"{ser}/{num}"
    return f"{ser} и номер {num}"


URL_DOMAINS = [
    "bitbucket.org", "sberbank.ru", "medium.com", "docs.python.org",
    "confluent.io", "rabota.ru", "schedule.learn-online.org",
    "certs.security-portal.ru", "issues.workflow-hub.com", "elibraty.ru",
    "minzdrav.gov.ru", "unlockbody.ru", "student-helper.ru", "rosdistants-help.ru",
]
URL_PATHS = ["", "/page", "/login", "/faq", "/api/v1/users", "/products",
             "/contribute.html", "/item.asp?id=49831292", "/first-massage.html",
             "/paleznyeresursy/natsproektzdravoohranenie"]


def gen_url() -> tuple[str, str]:
    """URL в разных представлениях: возвращает (url, label)."""
    dom = random.choice(URL_DOMAINS)
    path = random.choice(URL_PATHS)
    core = dom + path
    form = random.random()
    if form < 0.35:
        return f"https://{core}", "url"
    if form < 0.5:
        return f"http://{core}", "url"
    if form < 0.65:
        return f"www.{core}", "url"
    return core, "url"  # без схемы


def annotate(text: str, entities: list[tuple[str, str]]):
    tokens = []
    tok_spans = []
    for m in re.finditer(r"\w+|[^\w\s]", text, re.UNICODE):
        tokens.append(m.group(0))
        tok_spans.append((m.start(), m.end()))
    ner = []
    used_chars = set()
    for sub, label in entities:
        idx = text.find(sub)
        if idx < 0:
            continue
        s, e = idx, idx + len(sub)
        char_range = set(range(s, e))
        if char_range & used_chars:
            continue
        used_chars |= char_range
        w_ids = [wi for wi, (ts, te) in enumerate(tok_spans) if ts < e and te > s]
        if not w_ids:
            continue
        ner.append([min(w_ids), max(w_ids), label])
    ner.sort()
    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner}


def gen_passport_doc(i: int) -> dict | None:
    """Тексты с паспортами в разных форматах."""
    name = f"{random.choice(LAST_NAMES)} {random.choice(FIRST_NAMES)[0]}. {random.choice(LAST_NAMES)[0]}."
    ps = passport_str()
    w = random.choice(PASSPORT_WORDS)
    inn = digits(10)
    snils = f"{digits(3)}-{digits(3)}-{digits(3)} {digits(2)}"
    templates = [
        (f"Прошу подтвердить личность по {w} {ps}. Также прикладываю ИНН {inn}.",
         [(ps, "passport"), (inn, "inn")]),
        (f"{name}, серия и номер паспорта: {ps}. Данные проверены.",
         [(name, "person"), (ps, "passport")]),
        (f"Копия {w} ({ps}) предоставлена в отдел кадров. СНИЛС: {snils}.",
         [(ps, "passport"), (snils, "snils")]),
        (f"Для оформления заявки нужны данные {w}: {ps}.",
         [(ps, "passport")]),
        (f"Мой {w} {ps}, дата выдачи 12.03.2015. Требую разобраться в ситуации.",
         [(ps, "passport")]),
        (f"Продиктуйте {w}: {ps}, а также ваш ИНН {inn} для проверки.",
         [(ps, "passport"), (inn, "inn")]),
        (f"В выписке {w} {ps} вбили с опечаткой. Прошу исправить.",
         [(ps, "passport")]),
        (f"Нотариус требует написать текстом {w} {ps}.",
         [(ps, "passport")]),
    ]
    return annotate(*random.choice(templates))


def gen_doc_number_doc(i: int) -> dict | None:
    """ВУ/свидетельства/военники."""
    name = f"{random.choice(LAST_NAMES)} {random.choice(FIRST_NAMES)[0]}."
    doc_type, _ = random.choice(DOC_TYPES)
    ds = doc_number_str()
    passport = passport_str()
    templates = [
        (f"{name} показал {doc_type} {ds} и паспорт {passport}.",
         [(name, "person"), (ds, "doc number"), (passport, "passport")]),
        (f"Также требуется {doc_type}: {ds}. Проверьте в базе.",
         [(ds, "doc number")]),
        (f"Серия моего {doc_type} — {gen_doc_series()}. Номер: {digits(6, )}.",
         []),  # заполним ниже
        (f"Не могу найти {doc_type}, помню только что {ds}.",
         [(ds, "doc number")]),
        (f"Скопировал {doc_type} {ds} в буфер обмена.",
         [(ds, "doc number")]),
        (f"Госуслуги: введите данные {doc_type} {ds}.",
         [(ds, "doc number")]),
        (f"На медкомиссии захватите {doc_type} {ds}.",
         [(ds, "doc number")]),
        (f"Реквизиты: {doc_type} {ds}. Остальные документы в порядке.",
         [(ds, "doc number")]),
    ]
    tpl = random.choice(templates)
    if tpl[0].startswith("Серия моего"):
        # отдельная логика: серия отдельно от номера
        ser = gen_doc_series()
        num = digits(random.choice([6, 7]))
        sep = random.choice([", ", " ", " / ", " - "])
        text = f"Серия моего {doc_type} — {ser}.{sep}Номер {num}. Прошу восстановить."
        return annotate(text, [(f"{ser}", "doc number"), (num, "doc number")])
    return annotate(*tpl)


def gen_url_doc(i: int) -> dict | None:
    """URL в контекстах — в основном без схемы."""
    url, _ = gen_url()
    email = f"{random.choice(['help', 'info', 'support', 'ivanov'])}@{random.choice(['mail.ru', 'ya.ru', 'gmail.com', 'internet.ru'])}"
    templates = [
        (f"Сайт {url} работает?",
         [(url, "url")]),
        (f"Информация размещена на {url}.",
         [(url, "url")]),
        (f"Подача документов через {url}.",
         [(url, "url")]),
        (f"Подробнее: {url} — там все инструкции.",
         [(url, "url")]),
        (f"Контакты: сайт {url}, почта {email}.",
         [(url, "url"), (email, "email")]),
        (f"Расписание вебинаров: {url}",
         [(url, "url")]),
        (f"Трекер задач переехал на {url}",
         [(url, "url")]),
        (f"Сертификаты загружены на {url}",
         [(url, "url")]),
    ]
    return annotate(*random.choice(templates))


def gen_mixed_doc(i: int) -> dict | None:
    """Смешанные тексты: паспорт + doc_number + телефоны/снилс (контекстный разделитель)."""
    ps = passport_str()
    ds = doc_number_str()
    doc_type, _ = random.choice(DOC_TYPES)
    phone = f"8 {digits(3)} {digits(3)} {digits(2)} {digits(2)}"
    templates = [
        (f"Указаны следующие данные: паспорт {ps}, и {doc_type} {ds}.",
         [(ps, "passport"), (ds, "doc number")]),
        (f"Проверка документов: паспорт {ps}, {doc_type} отсутстствует. Звоните: {phone}",
         [(ps, "passport"), (phone, "phone")]),
        (f"Серия и номер паспорта: {ps}. {doc_type}: {ds}.",
         [(ps, "passport"), (ds, "doc number")]),
    ]
    return annotate(*random.choice(templates))


def main():
    examples = []
    for i in range(600):
        ex = gen_passport_doc(i)
        if ex:
            examples.append(ex)
    for i in range(700):
        ex = gen_doc_number_doc(i)
        if ex:
            examples.append(ex)
    for i in range(350):
        ex = gen_url_doc(i)
        if ex:
            examples.append(ex)
    for i in range(250):
        ex = gen_mixed_doc(i)
        if ex:
            examples.append(ex)

    random.shuffle(examples)
    OUT.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    print(f"Сгенерировано {len(examples)} примеров -> {OUT.name}")

    # Статистика меток
    from collections import Counter
    cnt = Counter(lbl for ex in examples for _, _, lbl in ex["ner"])
    for lbl, c in cnt.most_common():
        print(f"  {lbl}: {c}")


if __name__ == "__main__":
    main()
