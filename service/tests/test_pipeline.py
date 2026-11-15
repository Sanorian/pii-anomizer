# -*- coding: utf-8 -*-
"""Тесты пайплайна анонимизации.

Запуск: ../venv/bin/python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import psycopg
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detectors.regex_detector import RegexDetector, luhn_ok  # noqa: E402
from app.vault import MemoryVault  # noqa: E402


class _FakePipeline:
    """Заглушка AnonymizationPipeline с заранее заданными сущностями."""

    def __init__(self, entities):
        self._entities = entities

    def run(self, text, threshold=0.45, categories=None):
        class R:
            pass

        r = R()
        r.entities = [e for e in self._entities if categories is None or e["category"] in categories]
        return r


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
        assert "[PHONE_1]" in masked
        assert "[ORG_1]" in masked
        cats = {e["category"] for e in ents}
        assert "phone" in cats
        assert "organization" in cats

    def test_person_and_contacts(self, pipeline):
        text = "Меня зовут Валерия, помогаю студентам университета. Пишите на ivanov_2024@mail.ru"
        res = pipeline.run(text)
        masked, ents = mask_text(text, res.entities)
        assert "[PERSON_1]" in masked
        assert "[EMAIL_1]" in masked
        assert "[ORG_1]" in masked

    def test_financial(self, pipeline):
        text = "Оплата 5000 рублей на карту 4539 1488 0343 6467, паспорт 45 10 123456"
        res = pipeline.run(text)
        masked, _ = mask_text(text, res.entities)
        assert "[MONEY_1]" in masked
        assert "[CARD_1]" in masked
        assert "[PASSPORT_1]" in masked

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


# --- Guardrails-клиент --------------------------------------------------------


class TestGuardrailsClient:
    """scan_text: fail-open при недоступности и отключении."""

    def test_disabled_returns_none(self, monkeypatch):
        from app import guardrails_client as gc

        monkeypatch.setattr(gc, "GUARDRAILS_ENABLED", False)
        assert gc.scan_text("любой текст") is None

    def test_unreachable_fails_open(self, monkeypatch):
        from app import guardrails_client as gc

        # несуществующий порт — connection refused, scan_text вернёт None
        monkeypatch.setattr(gc, "GUARDRAILS_URL", "http://127.0.0.1:59999")
        monkeypatch.setattr(gc, "GUARDRAILS_ENABLED", True)
        monkeypatch.setattr(gc, "TIMEOUT_S", 1.0)
        assert gc.scan_text("любой текст") is None

    def test_ok_response_parsed(self, monkeypatch):
        from app import guardrails_client as gc

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "safe": False,
                    "decision": "block",
                    "score": 0.95,
                    "matched_rules": [{"rule": "test", "weight": 0.9}],
                }

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def post(self, url, **kw):
                return FakeResponse()

        monkeypatch.setattr(gc, "GUARDRAILS_ENABLED", True)
        monkeypatch.setattr(gc.httpx, "post", lambda *a, **kw: FakeResponse())
        res = gc.scan_text("текст")
        assert res["decision"] == "block" and res["score"] == 0.95


# --- Vault (memory-бэкенд) ----------------------------------------------------


class TestVaultMemory:
    def test_stable_ids_same_value(self):
        from app.vault import MemoryVault, new_vault_key

        v = MemoryVault()
        k = new_vault_key()
        assert v.get_or_assign(k, "person", "Иванов") == v.get_or_assign(k, "person", "Иванов")

    def test_different_values_different_ids(self):
        from app.vault import MemoryVault, new_vault_key

        v = MemoryVault()
        k = new_vault_key()
        p1 = v.get_or_assign(k, "person", "Иванов")
        p2 = v.get_or_assign(k, "person", "Петров")
        assert p1 == "[PERSON_1]" and p2 == "[PERSON_2]"

    def test_fingerprint_normalization(self):
        from app.vault import MemoryVault, new_vault_key

        v = MemoryVault()
        k = new_vault_key()
        # регистр и пробелы не влияют на ID
        assert v.get_or_assign(k, "person", "Иванов") == v.get_or_assign(k, "person", "иванов ")

    def test_categories_numbered_independently(self):
        from app.vault import MemoryVault, new_vault_key

        v = MemoryVault()
        k = new_vault_key()
        assert v.get_or_assign(k, "person", "Иванов") == "[PERSON_1]"
        assert v.get_or_assign(k, "email", "i@mail.ru") == "[EMAIL_1]"
        assert v.get_or_assign(k, "person", "Петров") == "[PERSON_2]"

    def test_ttl_expiry(self):
        from app.vault import MemoryVault, new_vault_key

        v = MemoryVault(ttl_hours=0)
        k = new_vault_key()
        v.get_or_assign(k, "person", "Иванов")
        import time

        time.sleep(0.05)
        assert not v.exists(k)  # ttl=0 -> сразу просрочено
        # exists() уже удалил просрочку, чистка больше не находит
        assert v.cleanup() == 0
        # новая запись и ручная чистка
        v.get_or_assign(new_vault_key(), "person", "Петров")
        v._data[k2 := next(iter(v._data))]["expires"] = (
            __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
            - __import__("datetime").timedelta(hours=1)
        )
        assert v.cleanup() == 1

    def test_delete_cascade(self):
        from app.vault import MemoryVault, new_vault_key

        v = MemoryVault()
        k = new_vault_key()
        p = v.get_or_assign(k, "phone", "+7 950 000 00 00")
        assert v.resolve(k, p) == "+7 950 000 00 00"
        v.delete(k)
        assert v.resolve(k, p) is None


class TestDeanonymizeText:
    def test_strict_and_loose(self):
        from app.vault import deanonymize_text

        mapping = {"[PERSON_1]": "Иванов", "[EMAIL_1]": "i@m.ru"}
        text, n = deanonymize_text(
            "[PERSON_1] -> PERSON_1, а [EMAIL_1] и [UNKNOWN_9]",
            lambda ph: mapping.get(ph),
        )
        assert "Иванов" in text and "i@m.ru" in text and "[UNKNOWN_9]" in text
        assert n == 3

    def test_no_false_positive_inside_word(self):
        from app.vault import deanonymize_text

        text, n = deanonymize_text("XPERSON_1Y", lambda ph: "Иванов")
        assert text == "XPERSON_1Y" and n == 0


# --- API round-trip (TestClient + memory vault) -------------------------------


@pytest.fixture()
def api(monkeypatch):
    from fastapi.testclient import TestClient

    from app import main as m

    monkeypatch.setattr(m, "vault", MemoryVault())
    # guardrails недоступен в тестовом окружении -> fail-open
    monkeypatch.setattr(m, "scan_text", lambda t: None)
    client = TestClient(m.app, raise_server_exceptions=True)
    # "модели загружены": pipeline-заглушку не ставим — используем реальный
    # pipeline fixture не можем (TestClient отдельный процесс lifespan не
    # запускает), поэтому тесты API с pipeline=None ожидают 503
    return client


class TestApiVault:
    def test_anonymize_returns_vault_key(self, api, monkeypatch):
        from app import main as m

        monkeypatch.setattr(
            m,
            "pipeline",
            _FakePipeline([{"start": 9, "end": 25, "category": "phone", "source": "regex", "score": 1.0}]),
        )
        r = api.post("/anonymize", json={"text": "Звони по +7 950 784 55 53"})
        assert r.status_code == 200
        data = r.json()
        assert data["vault_key"]
        assert data["anonymized_text"] == "Звони по [PHONE_1]"

    def test_roundtrip_same_session(self, api, monkeypatch):
        from app import main as m

        monkeypatch.setattr(
            m,
            "pipeline",
            _FakePipeline([{"start": 0, "end": 6, "category": "person", "source": "natasha", "score": 0.9}]),
        )
        r1 = api.post("/anonymize", json={"text": "Иванов пришёл"})
        k = r1.json()["vault_key"]
        assert r1.json()["anonymized_text"] == "[PERSON_1] пришёл"
        # второй запрос с тем же ключом: та же сущность -> тот же ID
        r2 = api.post("/anonymize", json={"text": "Иванов ушёл", "vault_key": k})
        assert r2.json()["anonymized_text"] == "[PERSON_1] ушёл"
        # деанонимизация
        r3 = api.post("/deanonymize", json={"text": "[PERSON_1] ушёл", "vault_key": k})
        assert r3.json() == {"text": "Иванов ушёл", "replaced": 1}

    def test_unknown_vault_key_404(self, api, monkeypatch):
        from app import main as m

        monkeypatch.setattr(m, "pipeline", _FakePipeline([]))
        r = api.post(
            "/anonymize",
            json={"text": "привет", "vault_key": "00000000-0000-0000-0000-000000000000"},
        )
        assert r.status_code == 404
        r2 = api.post("/deanonymize", json={"text": "x", "vault_key": "not-a-uuid"})
        assert r2.status_code == 422

    def test_deanonymize_unknown_placeholder_kept(self, api, monkeypatch):
        from app import main as m

        monkeypatch.setattr(
            m,
            "pipeline",
            _FakePipeline([{"start": 0, "end": 6, "category": "person", "source": "natasha", "score": 0.9}]),
        )
        k = api.post("/anonymize", json={"text": "Иванов тут"}).json()["vault_key"]
        r = api.post("/deanonymize", json={"text": "[PERSON_5] и [PERSON_1]", "vault_key": k})
        assert r.json()["text"] == "[PERSON_5] и Иванов"
