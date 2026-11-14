# -*- coding: utf-8 -*-
"""Синтетика домена: спам/ham сообщения с телефонами, группами, ссылками.

Генерирует примеры в GLiNER-формате (tokenized_text + ner) для дообучения.
Метки: phone, group code, url, email, money, person, organization, location.
"""

import json
import random
import re
from pathlib import Path

OUT = Path(__file__).parent / "data_ft" / "synth_domain.json"

FIRST_NAMES = [
    "Алексей", "Мария", "Дмитрий", "Ольга", "Сергей", "Анна", "Иван",
    "Екатерина", "Николай", "Светлана", "Павел", "Юлия", "Андрей", "Валерия",
]
LAST_NAMES = [
    "Смирнов", "Иванова", "Кузнецов", "Петрова", "Соколов", "Морозова",
    "Волков", "Козлова", "Сыротюк", "Петросян", "Валерьевна", "Николаевна",
]
GROUP_PREFIX = ["СТРбвп", "СТРбвд", "ИСТбзс", "ТБбп", "ПИбд", "ЭУбзс"]
CITIES = ["Тольятти", "Самаре", "Москве", "Казани", "Курске", "Железногорске"]
UNIVERSITIES = ["ТГУ", "РОСДИСТАНТ", "Тольяттинский государственный университет", "МГУ"]
MESSENGERS = ["вацап", "ватсап", "WhatsApp", "телеграм", "Telegram", "телеграмм"]

random.seed(1234)


def gen_phone() -> tuple[str, str]:
    """Возвращает (телефон, формат-тип). Простые пробельные форматы."""
    forms = [
        lambda: f"8 9{random.randint(10, 99)} {random.randint(100,999)} {random.randint(10,99)} {random.randint(10,99)}",
        lambda: f"+7 9{random.randint(10, 99)} {random.randint(100,999)} {random.randint(10,99)} {random.randint(10,99)}",
        lambda: f"8{random.randint(900, 999)}{random.randint(1000000, 9999999)}",
        lambda: f"8-{random.randint(900, 999)}-{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(10,99)}",
    ]
    return random.choice(forms)(), ""


def gen_group() -> str:
    return f"{random.choice(GROUP_PREFIX)}-{random.randint(2000, 2400)}{random.choice('абвгд')}"


def gen_email() -> str:
    n = random.choice(FIRST_NAMES).lower()
    l = random.choice(["smirnov", "ivanov", "petrov", "helper", "diplom", "study"])
    dom = random.choice(["gmail.com", "mail.ru", "yandex.ru", "bk.ru"])
    return f"{n}.{l}{random.randint(1, 999)}@{dom}"


def gen_url() -> str:
    slug = random.choice(["kupit-diplom", "help-student", "chat-2024", "pomosh-sessiya", "otvety"])
    dom = random.choice(["rosdistants-help.ru", "student-helper.ru", "t.me", "diplom-bez-problem.ru"])
    if dom == "t.me":
        return f"t.me/{slug}_{random.randint(1, 999)}"
    return f"https://{dom}/{slug}"


def annotate(text: str, entities: list[tuple[str, str]]):
    """Разбивает текст на токены и находит спаны для entities.

    entities: [(подстрока, метка)]
    Возвращает GLiNER-пример (не найденные подстроки пропускаются),
    None если ни одна сущность не нашлась.
    """
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
            continue  # пропускаем ненайденную сущность, остальное оставляем
        s, e = idx, idx + len(sub)
        char_range = set(range(s, e))
        if char_range & used_chars:
            continue  # пересечение — пропускаем
        used_chars |= char_range
        # индексы токенов, пересекающих спан
        w_ids = [wi for wi, (ts, te) in enumerate(tok_spans) if ts < e and te > s]
        if not w_ids:
            continue
        ner.append([min(w_ids), max(w_ids), label])
    ner.sort()
    if not ner:
        return None
    return {"tokenized_text": tokens, "ner": ner}


def gen_spam(i: int) -> dict | None:
    name = random.choice(FIRST_NAMES)
    phone, _ = gen_phone()
    group = gen_group()
    email = gen_email()
    url = gen_url()
    univ = random.choice(UNIVERSITIES)
    messenger = random.choice(MESSENGERS)
    templates = [
        (
            f"Здравствуйте. Я ваша одногруппница. Напишите в {messenger} на номер {phone}. "
            f"Вас добавят в беседу группы. Там решаем организационные и учебные вопросы. {univ}",
            [(phone, "phone"), (univ, "organization")],
        ),
        (
            f"Здравствуйте, я ваш одногруппник по учебе, из {univ}. "
            f"Напишите мне в {messenger}, я добавлю вас в беседу нашей группы "
            f"{group}: там мы обсуждаем задания. Напишите на номер {phone}.",
            [(phone, "phone"), (group, "group code"), (univ, "organization")],
        ),
        (
            f"Помогу с сессией под ключ. Цена вопроса {random.randint(5, 30)} {random.choice(['000 руб', '000 рублей'])}. "
            f"Группа {group}. Пишите {messenger}: {phone}",
            [(phone, "phone"), (group, "group code")],
        ),
        (
            f"Меня зовут {name}, я помогаю студентам {univ} в выполнении заданий. "
            f"Работаю без посредников. Пишите в {messenger} {phone} или на почту {email}",
            [(name, "person"), (phone, "phone"), (email, "email")],
        ),
        (
            f"Наш чат группы {group}: {url}. Вход бесплатный, все ответы на тесты там.",
            [(group, "group code"), (url, "url")],
        ),
        (
            f"Все материалы на сайте {url}. Оплата {random.randint(1, 20)}000 рублей на карту.",
            [(url, "url")],
        ),
        (
            f"Студент группы {group}, {random.choice(CITIES)}. Передайте преподавателю {name} {random.choice(LAST_NAMES)}, "
            f"что я заболел. Контакты: {phone}",
            [(group, "group code"), (phone, "phone")],
        ),
        (
            f"Срочно! Группа {group}, пишите в {messenger} {phone} или на {email}. "
            f"Скидка 10% первым 5 студентам {univ}!",
            [(group, "group code"), (phone, "phone"), (email, "email"), (univ, "organization")],
        ),
    ]
    tpl = random.choice(templates)
    return annotate(tpl[0], tpl[1])


def gen_ham(i: int) -> dict | None:
    name = random.choice(FIRST_NAMES)
    templates = [
        f"Добрый день! Отправил курсовую работу на оценку. Проверьте пожалуйста. Спасибо!",
        f"Уважаемый преподаватель, прошу открыть доступ к работам в статусе черновик.",
        f"Здравствуйте, {name}! Большое спасибо за подробную консультацию по заданию, всё понятно.",
        f"Подскажите, пожалуйста, когда будет известна оценка за реферат по дисциплине?",
        f"Здравствуйте! Приношу извинения за опоздание с отправкой задания по болезни.",
        f"Благодарим за активное участие в вебинаре. Запись и презентация доступны в курсе.",
    ]
    text = random.choice(templates)
    return annotate(text, [])


def main():
    examples = []
    for i in range(400):
        ex = gen_spam(i)
        if ex:
            examples.append(ex)
    for i in range(120):
        ex = gen_ham(i)
        if ex:
            examples.append(ex)

    # Фильтр "плохих" телефонов: скобки и дефисы между всеми группами токенизатор
    # разрывает — модель на них плохо учится. Оставляем пробельные форматы.
    good = []
    for ex in examples:
        toks = ex["tokenized_text"]
        ok = True
        for s, e, lbl in ex["ner"]:
            span = " ".join(toks[s:e + 1])
            if lbl == "phone" and ("(" in span or span.count("-") > 2):
                ok = False
                break
        if ok:
            good.append(ex)
    dropped = len(examples) - len(good)
    examples = good
    print(f"Отфильтровано примеров со сложными телефонами: {dropped}")

    random.shuffle(examples)
    OUT.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")

    # Статистика
    from collections import Counter
    cnt = Counter(lbl for ex in examples for _, _, lbl in ex["ner"])
    print(f"Сгенерировано {len(examples)} примеров -> {OUT.name}")
    for lbl, n in cnt.most_common():
        print(f"  {lbl:<15} {n}")

    # Валидация: спаны в границах
    bad = sum(
        1
        for ex in examples
        for s, e, _ in ex["ner"]
        if s > e or e >= len(ex["tokenized_text"])
    )
    print(f"Невалидных спанов: {bad}")
    # Показать 2 примера
    for ex in examples[:2]:
        toks = ex["tokenized_text"]
        print("\nTEXT:", " ".join(toks)[:100])
        for s, e, lbl in ex["ner"]:
            print(f"  [{lbl}] «{' '.join(toks[s:e+1])}»")


if __name__ == "__main__":
    main()
