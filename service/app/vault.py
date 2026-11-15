# -*- coding: utf-8 -*-
"""Vault: хранение анонимизаций для деанонимизации (round-trip).

LLM видит номерованные плейсхолдеры ([PERSON_1], [EMAIL_1]...), ответ LLM
содержит те же плейсхолдеры; деанонимизация по ключу сессии восстанавливает
реальные значения.

Два бэкенда за одним интерфейсом:
- MemoryVault — работа без БД и юнит-тесты;
- PostgresVault — PostgreSQL (psycopg3), таблицы создаются при старте.

Ключ сессии — uuid4, выдаётся сервисом при первом /anonymize и передаётся
в последующих запросах вместе с /deanonymize.

Стабильные ID: одинаковая сущность в рамках vault → один и тот же
плейсхолдер («Иванов» во всех сообщениях сессии → всегда [PERSON_1]).
Стабильность даёт fingerprint = категория + нормализованное значение
(регистр/пробелы не важны). Ограничение: два разных человека с одинаковым
именем получат один плейсхолдер — разрешение тождеств не делаем.

TTL: сессии старше VAULT_TTL_HOURS (дефолт 24) просрочены; чистка ленивая,
при записи (DELETE ... WHERE expires_at <= now()).
"""

import os
import re
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone

# TTL vault-сессии (часы)
VAULT_TTL_HOURS = int(os.environ.get("VAULT_TTL_HOURS", "24"))

# Бэкенд: memory | postgres
VAULT_BACKEND = os.environ.get("VAULT_BACKEND", "memory")

# [PERSON_1] -> person; базовый [PERSON] без номера -> person
_PLACEHOLDER_RE = re.compile(r"^\[([A-Z][A-Z_]*?)(?:_\d+)?\]$")
# Строгий плейсхолдер для деанонимизации: [PERSON_1]
_STRICT_RE = re.compile(r"\[([A-Z][A-Z_]*_\d+)\]")
# Терпимый вариант (LLM иногда теряет скобки): PERSON_1
_LOOSE_RE = re.compile(r"(?<![\w\[])([A-Z][A-Z_]*_\d+)(?![\w\]])")


def new_vault_key() -> str:
    """Новый ключ vault-сессии (uuid4)."""
    return str(uuid.uuid4())


def is_valid_vault_key(key: str) -> bool:
    """Ключ должен быть корректным UUID (мы выдаём только uuid4)."""
    try:
        uuid.UUID(key)
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def placeholder_category(placeholder: str) -> str:
    """Категория из плейсхолдера: [PERSON_1] -> person, [PERSON] -> person."""
    m = _PLACEHOLDER_RE.match(placeholder)
    return m.group(1).lower() if m else placeholder.strip("[]").lower()


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def fingerprint(category: str, value: str) -> str:
    """Fingerprint = категория + нормализованное значение."""
    return f"{category}:{_normalize(value)}"


def deanonymize_text(text: str, resolve) -> tuple[str, int]:
    """Заменяет плейсхолдеры в тексте значениями из vault.

    resolve(placeholder) -> value | None. Нераспознанные плейсхолдеры
    остаются как есть. Возвращает (текст, число замен).
    """
    replacements = 0

    def _sub(m: re.Match) -> str:
        nonlocal replacements
        # группа без скобок (STRICT) или со скобками (LOOSE) -> нормализуем
        ph = m.group(1)
        if not ph.startswith("["):
            ph = f"[{ph}]"
        value = resolve(ph)
        if value is None:
            return m.group(0)
        replacements += 1
        return value

    out = _STRICT_RE.sub(_sub, text)
    out = _LOOSE_RE.sub(_sub, out)
    return out, replacements


class Vault(ABC):
    """Интерфейс vault: выдача плейсхолдеров / деанонимизация / TTL."""

    @abstractmethod
    def ensure(self, vault_key: str) -> None:
        """Создать vault-сессию, если ещё нет; обновляет TTL."""

    @abstractmethod
    def get_or_assign(self, vault_key: str, category: str, value: str) -> str:
        """Плейсхолдер для сущности: существующий по fingerprint или новый
        ([PERSON_1], [PERSON_2]... — нумерация своя на категорию)."""

    @abstractmethod
    def resolve(self, vault_key: str, placeholder: str) -> str | None:
        """Значение по плейсхолдеру или None."""

    @abstractmethod
    def exists(self, vault_key: str) -> bool:
        """Существует ли vault-сессия (и не просрочена ли)."""

    @abstractmethod
    def delete(self, vault_key: str) -> bool:
        """Удалить vault-сессию."""

    @abstractmethod
    def cleanup(self) -> int:
        """Удалить просроченные сессии. Возвращает число удалённых."""


class MemoryVault(Vault):
    """In-memory vault: dict + expires. Для тестов и работы без Postgres."""

    def __init__(self, ttl_hours: int = VAULT_TTL_HOURS) -> None:
        self.ttl = timedelta(hours=ttl_hours)
        # key -> {"expires": dt, "entries": {placeholder: (category, value)}}
        self._data: dict[str, dict] = {}

    def _expires(self) -> datetime:
        return datetime.now(timezone.utc) + self.ttl

    def _get(self, vault_key: str) -> dict | None:
        data = self._data.get(vault_key)
        if data and datetime.now(timezone.utc) > data["expires"]:
            self._data.pop(vault_key, None)
            return None
        return data

    def ensure(self, vault_key: str) -> None:
        self.cleanup()
        if vault_key not in self._data:
            self._data[vault_key] = {"expires": self._expires(), "entries": {}}

    def get_or_assign(self, vault_key: str, category: str, value: str) -> str:
        data = self._data.get(vault_key)
        if data is None:
            self.ensure(vault_key)
            data = self._data[vault_key]
        data["expires"] = self._expires()
        fp = fingerprint(category, value)
        for placeholder, (cat, val) in data["entries"].items():
            if cat == category and _normalize(val) == _normalize(value):
                return placeholder
        # новый номер: максимальный существующий для категории + 1
        nums = []
        for p, (cat, _) in data["entries"].items():
            m = re.fullmatch(r"\[[A-Z][A-Z_]*_(\d+)\]", p)
            if cat == category and m:
                nums.append(int(m.group(1)))
        n = max(nums, default=0) + 1
        placeholder = f"[{category.upper()}_{n}]"
        data["entries"][placeholder] = (category, value)
        return placeholder

    def resolve(self, vault_key: str, placeholder: str) -> str | None:
        data = self._get(vault_key)
        if not data:
            return None
        entry = data["entries"].get(placeholder)
        return entry[1] if entry else None

    def exists(self, vault_key: str) -> bool:
        return self._get(vault_key) is not None

    def delete(self, vault_key: str) -> bool:
        return self._data.pop(vault_key, None) is not None

    def cleanup(self) -> int:
        now = datetime.now(timezone.utc)
        expired = [k for k, v in self._data.items() if now > v["expires"]]
        for k in expired:
            self._data.pop(k, None)
        return len(expired)


class PostgresVault(Vault):
    """Vault на PostgreSQL (psycopg3), autocommit, идемпотентный DDL.

    Схема:
        vaults(key UUID PK, created_at, updated_at, expires_at)
        vault_entries(id PK, vault_key FK CASCADE, placeholder, category,
                      value, fingerprint, UNIQUE(vault_key, placeholder))
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS vaults (
        key UUID PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        expires_at TIMESTAMPTZ NOT NULL
    );
    CREATE TABLE IF NOT EXISTS vault_entries (
        id BIGSERIAL PRIMARY KEY,
        vault_key UUID NOT NULL REFERENCES vaults(key) ON DELETE CASCADE,
        placeholder TEXT NOT NULL,
        category TEXT NOT NULL,
        value TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        UNIQUE (vault_key, placeholder)
    );
    CREATE INDEX IF NOT EXISTS idx_vault_entries_fp
        ON vault_entries (vault_key, fingerprint);
    CREATE INDEX IF NOT EXISTS idx_vaults_expires ON vaults (expires_at);
    """

    def __init__(self, dsn: str, ttl_hours: int = VAULT_TTL_HOURS) -> None:
        self.dsn = dsn
        self.ttl = timedelta(hours=ttl_hours)
        self._conn = None

    def _get_conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg

            self._conn = psycopg.connect(self.dsn, autocommit=True)
        return self._conn

    def init_schema(self) -> None:
        """Создать таблицы (вызывается при старте сервиса)."""
        with self._get_conn().cursor() as cur:
            cur.execute(self.DDL)

    def ensure(self, vault_key: str) -> None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO vaults (key, expires_at) VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET
                    updated_at = now(), expires_at = EXCLUDED.expires_at
                """,
                (vault_key, datetime.now(timezone.utc) + self.ttl),
            )

    def get_or_assign(self, vault_key: str, category: str, value: str) -> str:
        conn = self._get_conn()
        self.ensure(vault_key)
        fp = fingerprint(category, value)
        with conn.cursor() as cur:
            # существующий плейсхолдер по fingerprint
            cur.execute(
                """
                SELECT placeholder FROM vault_entries
                WHERE vault_key = %s AND fingerprint = %s
                ORDER BY id LIMIT 1
                """,
                (vault_key, fp),
            )
            row = cur.fetchone()
            if row:
                return row[0]
            # новый номер: по числу уже выданных плейсхолдеров категории
            prefix = f"[{category.upper()}_"
            cur.execute(
                """
                SELECT COUNT(DISTINCT placeholder) FROM vault_entries
                WHERE vault_key = %s AND placeholder LIKE %s
                """,
                (vault_key, prefix + "%]"),
            )
            n = cur.fetchone()[0] + 1
            placeholder = f"[{category.upper()}_{n}]"
            cur.execute(
                """
                INSERT INTO vault_entries
                    (vault_key, placeholder, category, value, fingerprint)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (vault_key, placeholder) DO NOTHING
                """,
                (vault_key, placeholder, category, value, fp),
            )
            return placeholder

    def resolve(self, vault_key: str, placeholder: str) -> str | None:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT value FROM vault_entries
                WHERE vault_key = %s AND placeholder = %s
                ORDER BY id LIMIT 1
                """,
                (vault_key, placeholder),
            )
            row = cur.fetchone()
            return row[0] if row else None

    def exists(self, vault_key: str) -> bool:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM vaults WHERE key = %s AND expires_at > now()",
                (vault_key,),
            )
            return cur.fetchone() is not None

    def delete(self, vault_key: str) -> bool:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM vaults WHERE key = %s", (vault_key,))
            return cur.rowcount > 0

    def cleanup(self) -> int:
        conn = self._get_conn()
        with conn.cursor() as cur:
            cur.execute("DELETE FROM vaults WHERE expires_at <= now()")
            return cur.rowcount

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()


def get_vault() -> Vault:
    """Фабрика vault по VAULT_BACKEND (env): memory | postgres."""
    if VAULT_BACKEND == "postgres":
        dsn = os.environ.get(
            "VAULT_DSN",
            "postgresql://vault:vault@localhost:5432/vault",
        )
        return PostgresVault(dsn)
    return MemoryVault()
