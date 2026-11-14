# -*- coding: utf-8 -*-
"""Regex-детектор: детерминированные паттерны ПДн и финансовых данных.

Находит телефоны, e-mail, URL, банковские карты (с Luhn-проверкой),
банковские счета, СНИЛС, ИНН, паспорта РФ, суммы денег.
"""

import re

# --- Паттерны -----------------------------------------------------------------

# Телефоны РФ: +7..., 8..., различные разделители (пробел после "+" допускается:
# "+ 7 (962) 321 - 07 - 02" из токенизированных текстов)
PHONE_RE = re.compile(
    r"(?<![\d\-_])\+\s?(?:7|7[\s\(]*)[\s\-]*\(?\d{3}\)?[\s\-]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}(?![\d\-_])"
)
# Телефоны слитно: 89507845553
PHONE_COMPACT_RE = re.compile(r"(?<![\d\-_])[78]\d{10}(?![\d\-_])")
# Городские телефоны без префикса: 861-296-11-12, (861) 296-11-12
PHONE_CITY_RE = re.compile(
    r"(?<![\d\-_])(?:\(\d{3}\)|\d{3})[\s\-]\d{3}[\s\-]\d{2}[\s\-]\d{2}(?![\d\-_])"
)

# E-mail
EMAIL_RE = re.compile(
    r"(?<![\w.\-])[A-Za-zА-Яа-яЁё0-9._%+\-]+@[A-Za-zА-Яа-яЁё0-9.\-]+\.[A-Za-zА-Яа-яЁё]{2,}(?![\w.\-])"
)

# URL: http(s)://..., www..., t.me/..., домены .ru/.com в тексте
URL_RE = re.compile(
    r"(?<![\w@.\-])"
    r"(?:https?://[^\s<>\"']+|www\.[^\s<>\"']+|t\.me/[A-Za-z0-9_]+|@[A-Za-z0-9_]{4,})"
    r"(?![\w@.\-])"
)

# Банковские карты: 16 цифр группами (проверка Luhn отдельно)
CARD_RE = re.compile(r"(?<!\d)(?:\d{4}[\s\-]?){3}\d{4}(?!\d)")

# Банковские счета: 20 цифр (группами или слитно)
ACCOUNT_RE = re.compile(r"(?<!\d)(?:\d{4,5}[\s\-]?){4}\d{1,5}(?!\d)")

# СНИЛС: XXX-XXX-XXX XX (пробелы, дефисы, точки; разделители любой длины)
SNILS_RE = re.compile(r"(?<!\d)\d{3}(?:[\s\-.]+\d{3}){2}[\s\-.]*\d{2}(?!\d)")
# СНИЛС слитно: 11223344556 (не 7/8 в начале — это телефон)
SNILS_COMPACT_RE = re.compile(r"(?<![\d\-_.])[0-69]\d{10}(?![\d\-_.]\d)")
# СНИЛС группами без разделителей в первой части: "111222333 44", "123456789 01"
SNILS_GROUPED_RE = re.compile(r"(?<!\d)\d{3}[\s\-.]*\d{3}[\s\-.]*\d{3}[\s\-.]+\d{2}(?!\d)")

# ИНН: 10 или 12 цифр
INN_RE = re.compile(r"(?<!\d)(?:\d{10}|\d{12})(?!\d)")

# Паспорт РФ: серия 4 цифры + номер 6 цифр (с пробелом или без)
PASSPORT_RE = re.compile(r"(?<!\d)\d{2}[\s\-]?\d{2}[\s]+\d{6}(?!\d)")
# Паспорт с N/№: "45 03 N 123456", "4503 № 123456"
PASSPORT_N_RE = re.compile(r"(?<!\d)\d{2}[\s\-]?\d{2}[\s]*[N№][\s]*\d{6}(?!\d)")
# Паспорт с точечным/прочерковым разделителем: "45.12.123456", "45_12_123456",
# "45~12~123456", "45=12=123456" (опечатки/маскировка символов)
PASSPORT_DOT_RE = re.compile(r"(?<![\d.\-_~=+=])[45]\d[.\-_~=+=][01]\d[.\-_~=+=]\d{6}(?![\d.\-_~=+=])")
# Паспорт со словами: "серия 7518, номер 492137", "Серия - 98, номер - 765432",
# "серию 87 12 и номер 987 654", "номер 123 - 456"
PASSPORT_WORDS_RE = re.compile(
    r"сери\w{0,2}[\s:№-]*\d{2,4}(?:[\s\-]+\d{2})?[\s,]*\w{0,2}[\s]*[-–—]?[\s]*(?:и[\s]+)?"
    r"номер[\s:№-]*(?:\d{3}[\s\-]+\d{3}|\d{6})",
    re.IGNORECASE,
)
# Паспорт одним числом с дефисом от слова: "паспорт № 4719-596586",
# "паспорт 7703 - 556677" (4 цифры - 6 цифр, слитно или с пробелами)
PASSPORT_HYPHEN_RE = re.compile(
    r"(?:паспорт\w*|папорт\w*)[^\d\n]{0,10}\d{4}[\s]*[-–—][\s]*\d{6}(?!\d)",
    re.IGNORECASE,
)
# "Номер документа 5550666", "Номер документа: 098765"
DOC_NUMBER_ONLY_RE = re.compile(
    r"(?:номер|№)\s*(?:документа|паспорта|удостоверения|билета|свидетельства)"
    r"[^\d\n]{0,10}\d{6,7}(?!\d)",
    re.IGNORECASE,
)
# Номер, продиктованный словами: "восемь три восемь пять ..." (6-10 слов-чисел)
NUMBER_WORDS = (
    "ноль|один|два|три|четыре|пять|шесть|семь|восемь|девять"
)
NUMBER_WORDS_SEQ_RE = re.compile(
    rf"(?<![а-яё])(?:{NUMBER_WORDS})[\s]+(?:{NUMBER_WORDS})(?:[\s]+(?:{NUMBER_WORDS})){{4,11}}(?![а-яё])",
    re.IGNORECASE,
)

# Полис ОМС: 16 цифр (с пробелами, "="-разделителями или слитно)
OMS_RE = re.compile(r"(?<![\d=])\d{4}(?:[\s=]?\d{4}){3}(?![\d=])")

# Водительское удостоверение: серия 2+2 цифры + номер 6 цифр ("77 14 345678")
# и свидетельство о рождении: римские цифры/буквы + серия + номер ("II-КЗ 123456", "АВ 123456")
DRIVER_LICENSE_RE = re.compile(
    r"(?<![\dА-Яа-яЁёA-Za-z])(?:\d{2}[\s\-]\d{2}[\s\-]?\d{6}|\d{4}[\s]?\d{6})(?![\dА-Яа-яЁёA-Za-z])"
)
# Свидетельство о рождении: серия с римскими цифрами и буквами "II - КЗ",
# "III - АГ" (с пробелами вокруг дефиса) — рядом словосочетание "серия/номер"
DOC_SERIES_ROMAN_RE = re.compile(
    r"[IVXLCDM]{1,5}[\s\-–+]+[А-ЯЁ]{2,4}(?![А-ЯЁA-Za-z])", re.IGNORECASE
)
# Словесная связка "серия X и номером Y", "с серией X и номером Y"
DOC_SERIES_AND_NUM_RE = re.compile(
    r"сери\w{0,2}[\s:№-]*(?:[IVXLCDM]{1,5}[\s\-–+]*[А-ЯЁ]{2,4}|[А-ЯЁ]{2}|[\dIVX][\d\s\-IVX]{0,7})"
    r"[^.\d]{0,15}номер\w{0,2}[\s:№-]*\d{6,7}",
    re.IGNORECASE,
)
# Конструкция "удостоверения: 77 18, номер 123456" (без слова "серия")
DOC_NUM_ONLY_RE = re.compile(
    r"(?:удостоверени\w+|прав\w+|ВУ)[^.\d]{0,30}(\d{2}[\s\-]\d{2})[^.\d]{0,10}номер\w{0,2}[\s:№-]*(\d{6})",
    re.IGNORECASE,
)
# Маркеры документов (в window перед/после конструкции серия-номер)
# "прав" без \w+: иначе матчит "правильно", "правда" (ложный контекст ВУ)
DOC_MARKERS_RE = re.compile(
    r"(?:водительск\w+|удостоверени\w+|прав\w|ВУ|свидетельств\w+|рождени\w+|"
    r"военн\w+|военник|билет\w*|воинский)",
    re.IGNORECASE,
)
# Контекст doc_number. Явные словоформы + частые опечатки из реальных текстов
# («вадительское», «удостверение»), но НЕ "правильно/правда/право":
# только точные формы "прав/права/правах" (разговорное ВУ)
DOC_CONTEXT_RE = re.compile(
    r"(?:водительск\w+|вадительск\w+|вдительск\w+|удостов[еи]рени\w+|удоств[еи]рени\w+|"
    r"\bправ(?:а|ах)?\b|ВУ|свидетельств\w+|военн\w+|билет\w*|воинский)",
    re.IGNORECASE,
)
# Паспортный контекст (для выбора passport vs doc_number): паспорт + опечатки
PASSPORT_CONTEXT_RE = re.compile(
    r"(?:паспорт\w*|пасспорт\w*|папорт\w*|паспотр\w*)",
    re.IGNORECASE,
)
ROMAN = r"(?:[IVXLCDM]{1,5})"
DOC_RU_RE = re.compile(
    # II-КЗ 123456 / II + АВ + 123456 / АВ 123456 / ЕЖ4567890 (военный билет)
    r"(?<![А-ЯЁA-Z])"
    r"(?:"
    r"(?:[IVXLCDM]{1,5}[\s\-–+]*[А-ЯЁ]{2,4}[\s\-–+]*\d{6})"  # римские + серия(буквы) + номер
    r"|(?:[А-ЯЁ]{2}[\s\-–+]*\d{6,7})"                          # буквы + 6-7 цифр
    r")"
    r"(?![А-ЯЁA-Z\d])",
)
# Военный билет: 2 заглавные буквы + 7 цифр слитно или раздельно
MILITARY_ID_RE = re.compile(r"(?<![А-ЯЁA-Za-z])[А-ЯЁ]{2}[\s\-]?\d{7}(?![А-ЯЁA-Za-z\d])")
# Военный билет с маскировочными/опечаточными разделителями внутри:
# "АВ_12_34567", "АВ(12)34567", "АВ?12?34567", "АВхх12хх34567"
MILITARY_ID_MASKED_RE = re.compile(
    r"(?<![А-ЯЁA-Za-z])[А-ЯЁ]{2}[^\dА-ЯЁA-Za-z\n]{1,3}\d{2}[^\dА-ЯЁA-Za-z\n]{1,3}\d{5}(?![А-ЯЁA-Za-z\d])"
)
# URL без схемы: домен с путём или без ("bitbucket.org/page", "docs.python.org")
# Финального lookbehind/lookahead НЕТ: хвостовые знаки препинания (".", ",")
# обрезаются пост-обработкой спана в detect() — lookbehind после жадного
# матча смотрит назад и отсекает легитимные пути, кончающиеся "/" или "\w".
# Схемные и www-варианты ловятся URL_RE.
URL_BARE_RE = re.compile(
    r"(?<![\w@.\-])(?:[A-Za-z0-9\-]+\.)+(?:ru|com|org|net|io|gov|edu|so)(?:/[\w\-./?%&=#]*)?"
)
# «Разболтанные» URL/email: токенизатор вставил пробелы вокруг точек и @
# ("sberbank. ru/login", "https: / / doi. org / 10. 1098", "help @ ya . ru").
# Ловим через «склеенный» вид: убираем пробелы и проверяем форму.
URL_SPACED_RE = re.compile(
    r"(?:(?:https?|ftp)\s*[:]\s*/\s*/[\w@.\s\-/?:%&=#]+"
    r"|www\s*\.\s*[\w@.\s\-/?:%&=#]+"
    r"|(?:[A-Za-z0-9\-]+\s*\.)+\s*(?:ru|com|org|net|io|gov|edu|so)(?:\s*/[\w\-./?%&=#\s]*)?)"
)
# Аналогично email: "help @ unlockbody . ru", "ulockbody @ ya. ru"
EMAIL_SPACED_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+(?:\s*@\s*)[A-Za-z0-9.\-]+(?:\s*\.\s*)[A-Za-z]{2,}"
)
# Явное слово "сайт/домен" перед доменом
URL_CONTEXT_RE = re.compile(r"(?:сайт\w*|домен\w*|ресурс\w*)", re.IGNORECASE)
# Адреса: улицы. Форма 1 (маркер сначала): "ул. Центральная", "УЛ. РОКОССОВСКОГО",
# "наб Новоданиловская". Форма 2 (маркер в конце, имя с заглавной — отсекает
# "любимый бульвар"): "Тверской бульвар", "Старокоптевский пер.", "Промышленная ул."
# ВАЖНО: без флага IGNORECASE на весь паттерн (иначе [А-ЯЁA-Z] становится
# регистронезависимым и матчит "план на день", "школа рядом"); регистр
# маркера — через inline-группу (?i:...), а после маркера обязательна
# граница слова (отсекает "план", "школа", "перезвоню")
_STREET_KIND_FIRST = (
    r"(?i:ул|улиц[аы]\w*|переулок|пер|проезд|проспект|пр-кт|пр-т|"
    r"бульвар|б-р|набережн\w*|наб|шоссе|ш|площад[ья]|пл)(?:\.|\b)"
)
STREET_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z])"
    rf"{_STREET_KIND_FIRST}\s*[А-ЯЁA-Z][\w\-]+(?:\s+[А-ЯЁA-Z][\w\-]+){{0,2}}"
)
STREET_TAIL_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z])"
    # имя может начинаться с цифры ("2-Й ЛИХАЧЁВСКИЙ ПЕРЕУЛОК"); имя строго
    # с заглавной/капсом, маркер — любой регистр через (?i:)
    r"[А-ЯЁA-Z\d][\w\-]+(?:\s+[А-ЯЁA-Z][\w\-]+){0,1}\s+"
    r"(?i:бульвар|переулок|проезд|проспект|набережн\w+|шоссе|тупик|ул|пер|б-р|наб)(?:\.|\b)"
)
# Дом/квартира/офис: "дом № 3", "дом 107, кв. 274", "10с1", "9А", "офис 15".
# Голые числа (16, 10) и короткие формы — только при адресном контексте
# рядом (улица/дом-маркер до или после), иначе поймаем все числа текста
HOUSE_RE = re.compile(
    r"(?<!\d)"
    r"(?:"
    r"(?:дом|офис)\s*№?\s*\.?\s*\d{1,3}[А-ЯЁа-я]?(?:\s*,?\s*(?:кв|стр|оф)\.?\s*\d{1,4}[А-Яа-я]?)?"
    r"|(?<=[:,]\s)\d{1,3}(?:[А-ЯЁа-я]\d?)?"
    r"|(?<=[А-ЯЁа-яё]\s)\d{1,3}(?:[А-ЯЁа-я]\d?)?(?![\dА-ЯЁа-я])"
    r")"
)
# Маркер улицы/дома поблизости (для голых чисел "16, 10" и "д. 1").
# Границы слов слева: иначе "ул" матчит "резулЬтат", "пер" — "перезвоню",
# "наб" — "набор", "пл" — "план". "стр\."/«д\.»/«кв\.» с точкой
HOUSE_CONTEXT_RE = re.compile(
    r"(?i:(?<![а-яёa-z])(?:ул|пер|наб|пл|б-р)\.?(?![а-яёa-z])|улиц[аы]\w*|"
    r"переулок|проезд|проспект|бульвар|шоссе|(?<![а-яёa-z])дом\w*|д\.|кв\.|"
    r"стр\.|офис)",
)
# Однобуквенные маркеры ("д. 1", "кв. 274", "стр. 7") — отдельный паттерн
# с обязательной точкой (иначе "д 1" поймает лишнее)
HOUSE_SHORT_RE = re.compile(
    r"(?<![А-ЯЁа-яёA-Za-z.\d])(?:д|кв|стр|оф)\.\s*\d{1,3}[А-ЯЁа-я]?(?![\dА-ЯЁа-я])"
)
# Карточный номер с раздельными дефисами: 6271 - 8855 - 1122 - 3344
CARD_SPACED_RE = re.compile(r"(?<!\d)\d{4}(?:\s-\s\d{4}){3}(?!\d)")

# Контекст "карта/карточка" рядом с 16-значным числом
CARD_CONTEXT_RE = re.compile(r"карт\w*", re.IGNORECASE)
# Полис ОМС: 16 цифр (пробелы/дефисы/"="/"|" любой длины или слитно) + контекст
OMS_NUM_RE = re.compile(
    r"(?<![\d=|])(?:\d{16}|\d{4}(?:[|\s=]+\d{4,8}){2,3})(?![\d=|])"
)
# ОМС со экзотическими разделителями/произвольной группировкой:
# "1234?5678?90123456", "1234тире5678тире90123456", "№6203-57 97-2798 55 20",
# "7744. 1122. 3344. 5566". ТОЛЬКО при явном контексте полиса и сумме цифр 16
# (иначе поймаем карты/телефоны/даты).
OMS_LOOSE_RE = re.compile(r"(?<!\d)\d{2,4}(?:[^\d\n]{1,5}\d{2,8}){2,5}(?!\d)")
OMS_CONTEXT_RE = re.compile(
    r"(?:по+[оl]лис\w*|полис\w*|полес\w*|полюс\w*|ОМС|омс|осм|о\s?м\s?с|"
    r"оэмэс|амс|страхован\w+ номер\w*)",
    re.IGNORECASE,
)
MONEY_RE = re.compile(
    r"(?<![\d.,])(?:\d{1,3}(?:[ \u00A0,.]\d{3})*|\d+)(?:[.,]\d{1,2})?\s*"
    r"(?:руб(?:лей|\.|)?|₽|тыс\.?\s*(?:руб|₽)|долларов|\$|€|евро)(?![\w])",
    re.IGNORECASE,
)


def luhn_ok(digits: str) -> bool:
    """Проверка номера карты алгоритмом Луна."""
    digits = digits.replace(" ", "").replace("-", "")
    if len(digits) != 16:
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _nearest_context(
    text: str, start: int, end: int, ctx_re: re.Pattern, max_look: int = 60
) -> int | None:
    """Расстояние до ближайшего контекстного маркера слева/справа от спана.

    Возвращает min(расстояние) или None, если маркера нет. Ищем в обоих
    направлениях: «паспорт серия 7810 номер 112233» — маркер слева,
    «серия 77 14 — это мои права» — справа.
    """
    left_from = max(0, start - max_look)
    best: int | None = None
    for m in ctx_re.finditer(text, left_from, min(len(text), end + max_look)):
        # расстояние = зазор между маркером и спаном
        gap = start - m.end() if m.end() <= start else m.start() - end
        if gap < 0:
            gap = 0
        if best is None or gap < best:
            best = gap
    return best


def _pick_doc_category(text: str, start: int, end: int) -> tuple[str, float]:
    """Категория для «серия X номер Y»: по ближайшему маркеру документа.

    «паспорт серия 7810 номер 112233» → passport (маркер ближе),
    «права: серия 77 14, номер 123456» → doc_number. При равенстве — doc_number
    (числовая серия 2+2 бывает и у ВУ). Возвращает (категория, бонус к весу).
    """
    d_pass = _nearest_context(text, start, end, PASSPORT_CONTEXT_RE)
    d_doc = _nearest_context(text, start, end, DOC_CONTEXT_RE)
    if d_pass is not None and (d_doc is None or d_pass <= d_doc):
        return "passport", 1.05
    if d_doc is not None:
        return "doc_number", 1.1
    return "passport", 1.0


def _dedupe_and_sort(entries: list[dict]) -> list[dict]:
    """Убирает дубли и вложенные пересечения (оставляет более приоритетный).

    Приоритет: вес категории (специфичный контекст), затем более длинный спан.
    """
    from app.pipeline import CATEGORY_WEIGHT  # локальный импорт (циклическая зависимость)

    entries.sort(
        key=lambda e: (
            e["start"],
            -CATEGORY_WEIGHT.get(e.get("category", ""), 1.0),
            -(e["end"] - e["start"]),
        )
    )
    result = []
    for e in entries:
        if any(e["start"] < r["end"] and r["start"] < e["end"] for r in result):
            continue
        result.append(e)
    return sorted(result, key=lambda e: e["start"])


class RegexDetector:
    """Детерминированный детектор структурированных ПДн."""

    name = "regex"

    def detect(self, text: str) -> list[dict]:
        raw: list[dict] = []

        # Телефоны (сначала с разделителями, потом слитные, потом городские)
        for m in PHONE_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "phone", "score": 1.0})
        for m in PHONE_COMPACT_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "phone", "score": 1.0})
        for m in PHONE_CITY_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "phone", "score": 0.9})

        # E-mail
        for m in EMAIL_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "email", "score": 1.0})

        # URL (после e-mail, чтобы не забирать домены из адресов)
        for m in URL_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "url", "score": 1.0})
        # URL без схемы: "bitbucket.org/page", "docs.python.org" — но только
        # с двухуровневым TLD (student-helper.ru) или словом сайт/домен рядом,
        # иначе наловим фамилий с инициалами ("Иванов.П") и файлов ("backup.sql")
        for m in URL_BARE_RE.finditer(text):
            frag = m.group(0)
            # пост-обработка: обрезаем хвостовые знаки препинания
            # ("sberbank.ru/login.", "medium.com/faq,") — они не часть URL
            while frag and frag[-1] in ".,;:)]}":
                frag = frag[:-1]
            if not frag or "." not in frag:
                continue
            start, end = m.start(), m.start() + len(frag)
            window = text[max(0, start - 30) : end + 10]
            labels = frag.split(".")
            has_path = "/" in frag
            multi_tld = len(labels) >= 3 and len(labels[1]) >= 2
            if has_path or multi_tld or URL_CONTEXT_RE.search(window):
                raw.append({"start": start, "end": end, "category": "url", "score": 0.9})

        # «Разболтанные» URL/email (пробелы внутри из-за токенизации).
        # Валидируем склеенным видом; спан обрезаем по хвостовому мусору.
        for m in URL_SPACED_RE.finditer(text):
            frag = m.group(0)
            while frag and frag[-1] in ".,;:)]}?! \t":
                frag = frag[:-1]
            squeezed = re.sub(r"\s+", "", frag)
            # должен выглядеть как URL после склейки
            if not re.match(r"(?:https?|ftp)://", squeezed) and not re.match(
                r"www\.", squeezed
            ) and not re.search(r"\.(?:ru|com|org|net|io|gov|edu|so)(?:/|$)", squeezed):
                continue
            if squeezed.count(".") + squeezed.count("/") < 2:
                continue
            raw.append({"start": m.start(), "end": m.start() + len(frag), "category": "url", "score": 0.85})
        for m in EMAIL_SPACED_RE.finditer(text):
            frag = m.group(0)
            squeezed = re.sub(r"\s+", "", frag)
            if not re.match(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", squeezed):
                continue
            raw.append({"start": m.start(), "end": m.end(), "category": "email", "score": 0.85})

        # Банковские карты с раздельными дефисами "6271 - 8855 - ..." (до OMS/CARD_RE):
        # при контексте "полис/ОМС" это полис, иначе карта
        for m in CARD_SPACED_RE.finditer(text):
            window = text[max(0, m.start() - 40) : m.end() + 20]
            if OMS_CONTEXT_RE.search(window):
                raw.append({"start": m.start(), "end": m.end(), "category": "oms", "score": 0.95})
            else:
                score = 1.0 if luhn_ok(m.group(0)) else 0.8
                raw.append({"start": m.start(), "end": m.end(), "category": "card", "score": score})

        # Полис ОМС vs карта: 16 цифр. При контексте "полис/ОМС" — oms,
        # иначе card (по умолчанию 16 цифр считаем картой).
        for m in OMS_NUM_RE.finditer(text):
            frag = m.group(0)
            window = text[max(0, m.start() - 40) : m.end() + 20]
            if OMS_CONTEXT_RE.search(window):
                raw.append({"start": m.start(), "end": m.end(), "category": "oms", "score": 0.95})
            elif luhn_ok(frag):
                raw.append({"start": m.start(), "end": m.end(), "category": "card", "score": 1.0})
            else:
                raw.append({"start": m.start(), "end": m.end(), "category": "card", "score": 0.8})
        # ОМС со экзотическими разделителями/группировкой — только при
        # контексте полиса и ровно 16 цифрах
        for m in OMS_LOOSE_RE.finditer(text):
            frag = m.group(0)
            if sum(c.isdigit() for c in frag) != 16:
                continue
            window = text[max(0, m.start() - 40) : m.end() + 20]
            if OMS_CONTEXT_RE.search(window):
                raw.append({"start": m.start(), "end": m.end(), "category": "oms", "score": 0.9})

        # Банковские счета: 20 цифр (слитно или группами)
        for m in re.finditer(r"(?<!\d)(?:\d{5}[\s\-]?){4}(?!\d)", text):
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) == 20:
                raw.append({"start": m.start(), "end": m.end(), "category": "account", "score": 1.0})

        # СНИЛС
        for m in SNILS_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "snils", "score": 1.0})
        for m in SNILS_GROUPED_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "snils", "score": 0.95})
        for m in SNILS_COMPACT_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "snils", "score": 0.9})

        # "серия XXXX номер XXXXXX" и "серия II - КЗ номер 765432":
        # числовая серия = паспорт, если рядом нет маркера документа;
        # буквенная/римская серия (II - КЗ, МЗ) = doc_number всегда
        for m in DOC_SERIES_AND_NUM_RE.finditer(text):
            frag = m.group(0)
            letters_series = re.match(
                r"сери\w{0,2}[\s:№-]*(?:[IVXLCDM]{1,5}[\s\-–+]*[А-ЯЁ]{2,4}|[А-ЯЁ]{2})[\s,]*",
                frag, re.IGNORECASE,
            )
            if letters_series:
                raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.95})
                continue
            window = text[max(0, m.start() - 80) : m.end() + 20]
            if DOC_MARKERS_RE.search(window):
                # числовая серия: категория по ближайшему маркеру
                # («паспорт серия 7810...» → passport, «права: серия 77 14...» → doc_number)
                cat, w = _pick_doc_category(text, m.start(), m.end())
                raw.append({"start": m.start(), "end": m.end(), "category": cat, "score": 0.95, "weight": w})
        # "удостоверения: 77 18, номер 123456" — конструкция без слова "серия"
        for m in DOC_NUM_ONLY_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.95})
        # Серия свидетельства "II - КZ" отдельным спаном (когда номер далеко)
        for m in DOC_SERIES_ROMAN_RE.finditer(text):
            window = text[max(0, m.start() - 80) : m.end() + 40]
            if DOC_MARKERS_RE.search(window):
                raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.9})
        for m in PASSPORT_WORDS_RE.finditer(text):
            window = text[max(0, m.start() - 50) : m.end()]
            if m.group(0) in {r["text"] for r in raw if r.get("text")}:
                continue
            cat, w = _pick_doc_category(text, m.start(), m.end())
            raw.append({"start": m.start(), "end": m.end(), "category": cat, "score": 1.0, "weight": w})
        # Паспортный формат "45 10 123456": категория по ближайшему маркеру
        # (паспорт/ВУ/свидетельство рядом); 10 цифр слитно без контекста не
        # трогаем (может быть ИНН/телефон/счет)
        for m in PASSPORT_RE.finditer(text):
            window = text[max(0, m.start() - 80) : m.end() + 20]
            # при явной конструкции "серия ... номер ..." уже обработано выше
            if DOC_MARKERS_RE.search(window) and "сери" in window.lower():
                continue
            cat, w = _pick_doc_category(text, m.start(), m.end())
            raw.append({"start": m.start(), "end": m.end(), "category": cat, "score": 1.0, "weight": w})
        # "45 03 N 123456", "4503 № 123456" — категория по ближайшему маркеру
        # (у ВУ формат тот же: "ВУ 50 19 N 012345")
        for m in PASSPORT_N_RE.finditer(text):
            cat, w = _pick_doc_category(text, m.start(), m.end())
            raw.append({"start": m.start(), "end": m.end(), "category": cat, "score": 1.0, "weight": w})
        # "45.12.123456", "45_12_123456", "4719-596586" (после слова "паспорт")
        for m in PASSPORT_DOT_RE.finditer(text):
            cat, w = _pick_doc_category(text, m.start(), m.end())
            raw.append({"start": m.start(), "end": m.end(), "category": cat, "score": 0.9, "weight": w})
        for m in PASSPORT_HYPHEN_RE.finditer(text):
            frag = m.group(0)
            digits_pos = re.search(r"\d", frag)
            start = m.start() + (digits_pos.start() if digits_pos else 0)
            cat, w = _pick_doc_category(text, start, m.end())
            raw.append({"start": start, "end": m.end(), "category": cat, "score": 1.0, "weight": w})
        # "Номер документа 5550666"
        for m in DOC_NUMBER_ONLY_RE.finditer(text):
            frag = m.group(0)
            digits_pos = re.search(r"\d{6,7}", frag)
            start = m.start() + (digits_pos.start() if digits_pos else 0)
            raw.append({"start": start, "end": m.end(), "category": "doc_number", "score": 0.95})
        # Числа словами: "восемь три восемь пять ..." — телефон, номер документа
        # или СНИЛС/ИНН (11 цифр при слове снилс/инн рядом). Приоритет: явное
        # слово документа > маркер удостоверения > телефон
        for m in NUMBER_WORDS_SEQ_RE.finditer(text):
            window = text[max(0, m.start() - 60) : m.end() + 20]
            n_words = len(m.group(0).split())
            if re.search(r"сни[лn]с|страхов\w+ свидетельств\w+", window, re.IGNORECASE):
                cat = "snils"
            elif re.search(r"инн|идентификационн\w+ номер", window, re.IGNORECASE):
                cat = "inn"
            elif n_words == 10 or n_words == 11 or DOC_CONTEXT_RE.search(window):
                cat = "doc_number"
            else:
                cat = "phone"
            raw.append({"start": m.start(), "end": m.end(), "category": cat, "score": 0.85})
        for m in DRIVER_LICENSE_RE.finditer(text):
            window = text[max(0, m.start() - 60) : m.end()]
            has_ctx = bool(DOC_CONTEXT_RE.search(window))
            digits = m.group(0).replace(" ", "").replace("-", "")
            if len(digits) == 10 and not has_ctx:
                continue  # слитно без контекста — вероятно ИНН
            if has_ctx:
                raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.95, "weight": 1.1})

        # Военный билет: 2 буквы + 7 цифр (до doc_number — специфичнее)
        for m in MILITARY_ID_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.9})
        # Военный билет с маскировочными разделителями: "АВ_12_34567"
        for m in MILITARY_ID_MASKED_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.85})

        # ВУ / свидетельство о рождении: римские цифры+буквы+номер
        for m in DOC_RU_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "doc_number", "score": 0.8})

        # Адреса: улицы и дома (FN бенчмарка: "ул. ГОРЬКОГО", "Тверской бульвар",
        # "дом 107, кв. 274", "10с1", "9А"). Голые числа и "д. 1" — только при
        # адресном контексте (улица/дом-маркер рядом)
        for m in STREET_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "location", "score": 0.9})
        for m in STREET_TAIL_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "location", "score": 0.9})
        for m in HOUSE_RE.finditer(text):
            frag = m.group(0).strip()
            start, end = m.start(), m.end()
            window = text[max(0, start - 50) : end + 30]
            if not HOUSE_CONTEXT_RE.search(window):
                continue  # голые числа без адресного контекста не трогаем
            # "дом 10, офис 15" / "дом 34 кв. 118" — gold считает дом и офис
            # отдельными сущностями: разрезаем по запятым и кв/стр/оф-переходам
            if "," in frag or re.search(r"\s(?:кв|стр|оф)\.?\s", frag):
                pieces = re.split(r"\s*,\s*|\s+(?=(?:кв|стр|оф)\b)", frag)
                consumed = 0
                for part in pieces:
                    idx = frag.find(part, consumed)
                    piece_start = start + idx
                    consumed = idx + len(part)
                    if part.strip(", "):
                        raw.append({
                            "start": piece_start,
                            "end": piece_start + len(part),
                            "category": "location",
                            "score": 0.85,
                        })
                continue
            raw.append({"start": start, "end": end, "category": "location", "score": 0.85})
        for m in HOUSE_SHORT_RE.finditer(text):
            window = text[max(0, m.start() - 50) : m.end() + 30]
            if not HOUSE_CONTEXT_RE.search(window):
                continue
            raw.append({"start": m.start(), "end": m.end(), "category": "location", "score": 0.85})

        # ИНН (после счёта/паспорта — фильтруем те, что уже покрыты)
        for m in INN_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "inn", "score": 1.0})

        # Деньги
        for m in MONEY_RE.finditer(text):
            raw.append({"start": m.start(), "end": m.end(), "category": "money", "score": 1.0})

        return _dedupe_and_sort(raw)
