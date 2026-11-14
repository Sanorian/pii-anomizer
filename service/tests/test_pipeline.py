# -*- coding: utf-8 -*-
"""Тесты пайплайна анонимизации.

Запуск: ../venv/bin/python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detectors.regex_detector import RegexDetector, luhn_ok  # noqa: E402


# --- Regex-детектор -----------------------------------------------------------


class TestLuhn:
    def test_valid_card(self):
        assert luhn_ok("4539148803436467")  # классический валидный тестовый

    def test_invalid_card(self):
        assert not luhn_ok("2202203612345678")

    def test_wrong_length(self):
        assert not luhn_ok("1234")


class TestRegexDetector:
    def setup_method(self):
        self.det = RegexDetector()

    def test_phone_spaced(self):
        ents = self.det.detect("Звоните 8 950 784-55-53 сегодня")
        assert any(e["category"] == "phone" and e["text_span"] if False else e["category"] == "phone" for e in ents)

    def test_phone_plus7(self):
        ents = self.det.detect("Пишите +7 950 784 55 53")
        assert any(e["category"] == "phone" for e in ents)

    def test_phone_compact(self):
        ents = self.det.detect("номер 89507845553 один")
        assert any(e["category"] == "phone" for e in ents)

    def test_email(self):
        ents = self.det.detect("почта ivanov_2024@mail.ru ок")
        assert any(e["category"] == "email" for e in ents)

    def test_url(self):
        ents = self.det.detect("чат https://t.me/diplom_help здесь")
        assert any(e["category"] == "url" for e in ents)

    def test_url_www(self):
        ents = self.det.detect("сайт www.example.ru/page хороший")
        assert any(e["category"] == "url" for e in ents)

    def test_card_valid_luhn(self):
        ents = self.det.detect("карта 4539 1488 0343 6467 активна")
        assert any(e["category"] == "card" and (e["end"] - e["start"]) == 19 for e in ents)

    def test_card_invalid_luhn_still_masked(self):
        # 16-значная последовательность без Luhn — маскируем с меньшим score
        ents = self.det.detect("карта 2202 2036 1234 5678 активна")
        assert any(e["category"] == "card" for e in ents)

    def test_account_20_digits(self):
        ents = self.det.detect("счёт 40817810000000123456 в банке")
        assert any(e["category"] == "account" for e in ents)

    def test_snils(self):
        ents = self.det.detect("СНИЛС 112-233-445 95 выдан")
        assert any(e["category"] == "snils" for e in ents)

    def test_inn(self):
        ents = self.det.detect("ИНН 7712345678 выдан")
        assert any(e["category"] == "inn" for e in ents)

    def test_passport(self):
        ents = self.det.detect("паспорт 45 10 123456 выдан")
        assert any(e["category"] == "passport" for e in ents)

    def test_money_rubles(self):
        ents = self.det.detect("оплата 15 000 рублей")
        assert any(e["category"] == "money" for e in ents)

    def test_money_rub_short(self):
        ents = self.det.detect("цена 5000 руб.")
        assert any(e["category"] == "money" for e in ents)

    def test_oms_context(self):
        ents = self.det.detect("полис ОМС 1234 5678 9012 3456 выдан")
        assert any(e["category"] == "oms" for e in ents)

    def test_oms_equals_separator(self):
        ents = self.det.detect("ОМС: 1234=5678=9012 3456")
        assert any(e["category"] == "oms" for e in ents)

    def test_card_16_digits_without_context(self):
        # 16 цифр без контекста полиса = карта (даже без Luhn)
        ents = self.det.detect("число 1234567890123456 без контекста")
        assert any(e["category"] == "card" for e in ents)

    def test_doc_number_driver_license_ctx(self):
        ents = self.det.detect("права 7714 345678 выданы")
        assert any(e["category"] == "doc_number" for e in ents)

    def test_doc_number_driver_license_words(self):
        ents = self.det.detect("ВУ серии 77 14, номер 345678")
        assert any(e["category"] == "doc_number" for e in ents)

    def test_doc_number_military_id(self):
        ents = self.det.detect("военный билет АВ 9998887")
        assert any(e["category"] == "doc_number" for e in ents)

    def test_doc_number_birth_cert(self):
        ents = self.det.detect("свидетельство II-КЗ 123456 выдано")
        assert any(e["category"] == "doc_number" for e in ents)

    def test_passport_not_doc_number_without_ctx(self):
        ents = self.det.detect("паспорт 45 10 123456 выдан")
        assert any(e["category"] == "passport" for e in ents)
        assert not any(e["category"] == "doc_number" for e in ents)

    def test_inn_not_doc_number(self):
        ents = self.det.detect("ИНН 7712345678 выдан")
        assert any(e["category"] == "inn" for e in ents)
        assert not any(e["category"] == "doc_number" for e in ents)

    def test_negative_no_entities(self):
        ents = self.det.detect("Добрый день! Отправил курсовую работу на оценку.")
        assert ents == []

    def test_negative_numbers_not_phone(self):
        # короткая последовательность цифр — не телефон
        ents = self.det.detect("задание 123 выполнено")
        assert not any(e["category"] == "phone" for e in ents)


# --- Пайплайн (требует модели) ------------------------------------------------


@pytest.fixture(scope="module")
def pipeline():
    from app.detectors.gliner_detector import GlinerDetector
    from app.detectors.natasha_detector import NatashaDetector
    from app.pipeline import AnonymizationPipeline

    root = Path(__file__).resolve().parent.parent.parent
    gliner = GlinerDetector(root / "models" / "gliner_ru_finetuned", device="cpu").load()
    natasha = NatashaDetector().load()
    return AnonymizationPipeline(RegexDetector(), gliner, natasha)


class TestPipeline:
    def test_spam_message(self, pipeline):
        text = (
            "Здравствуйте. Я ваша одногруппница. Напишите в вацап на номер "
            "8 950 784 55 53. Вас добавят в беседу группы. УНИВЕРСИТЕТ ПСИХОЛОГИЯ ВУЗ"
        )
        res = pipeline.run(text, threshold=0.45)
        masked, ents = mask_text(text, res.entities)
        assert "[PHONE]" in masked
        assert "[ORG]" in masked
        cats = {e["category"] for e in ents}
        assert "phone" in cats
        assert "organization" in cats

    def test_person_and_contacts(self, pipeline):
        text = "Меня зовут Валерия, помогаю студентам университета. Пишите на ivanov_2024@mail.ru"
        res = pipeline.run(text)
        masked, ents = mask_text(text, res.entities)
        assert "[PERSON]" in masked
        assert "[EMAIL]" in masked
        assert "[ORG]" in masked

    def test_financial(self, pipeline):
        text = "Оплата 5000 рублей на карту 4539 1488 0343 6467, паспорт 45 10 123456"
        res = pipeline.run(text)
        masked, _ = mask_text(text, res.entities)
        assert "[MONEY]" in masked
        assert "[CARD]" in masked
        assert "[PASSPORT]" in masked

    def test_ham_not_masked(self, pipeline):
        text = "Добрый день! Отправил курсовую работу на оценку. Проверьте пожалуйста."
        res = pipeline.run(text)
        masked, ents = mask_text(text, res.entities)
        assert masked == text  # ничего не замаскировано
        assert ents == []

    def test_no_overlap_double_masking(self, pipeline):
        text = "Звоните +7 950 784 55 53 или пишите smirnov.a@gmail.com"
        res = pipeline.run(text)
        ents = res.entities
        # сущности не пересекаются
        for i, a in enumerate(ents):
            for b in ents[i + 1 :]:
                assert a["end"] <= b["start"] or b["end"] <= a["start"]

    def test_categories_filter(self, pipeline):
        text = "Звоните +7 950 784 55 53, Иван Петров ждёт"
        res = pipeline.run(text, categories={"phone"})
        cats = {e["category"] for e in res.entities}
        assert cats == {"phone"}

    def test_offsets_correct(self, pipeline):
        text = "Пишите на ivanov_2024@mail.ru скорее"
        res = pipeline.run(text)
        for e in res.entities:
            assert text[e["start"] : e["end"]]  # спан непустой
            original = text[e["start"] : e["end"]]
            assert "@" in original or original.isdigit()


# Импорт mask_text здесь, чтобы fixture-модуль видел
from app.pipeline import mask_text  # noqa: E402
