# -*- coding: utf-8 -*-
"""Клиент guardrails-сервиса для пайплайна анонимизации.

Вызывается перед детекцией ПДн: если текст — промпт-инъекция,
основной сервис помечает это в ответе (report) или блокирует
обработку (block). Fail-open: недоступность guardrails не ломает
анонимизацию — в ответе ставится guardrails.status="unavailable".
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

GUARDRAILS_URL = os.environ.get("GUARDRAILS_URL", "http://guardrails:8000")
GUARDRAILS_ENABLED = os.environ.get("GUARDRAILS_ENABLED", "1") == "1"
GUARDRAILS_MODE = os.environ.get("GUARDRAILS_MODE", "report")  # report|block
TIMEOUT_S = float(os.environ.get("GUARDRAILS_TIMEOUT", "5"))


def scan_text(text: str) -> dict | None:
    """Прогон через guardrails. None = сервис недоступен/отключён."""
    if not GUARDRAILS_ENABLED:
        return None
    try:
        r = httpx.post(
            f"{GUARDRAILS_URL}/scan",
            json={"text": text},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.warning("guardrails недоступен: %s", e)
        return None
