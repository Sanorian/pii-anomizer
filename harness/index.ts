/**
 * PII Guard для Pi coding agent.
 *
 * Что делает:
 * 1. input        — скан ввода (и /skill:... до раскрытия!) на промпт-инъекции;
 *                   анонимизация ПДн перед отправкой в LLM.
 * 2. tool_call    — аргументы инструментов с ПДн маскируются (в файле/команде
 *                   реальные данные не нужны — они уходят в LLM-контекст).
 * 3. tool_result  — вывод инструментов: скан на инъекции (данные с веба/
 *                   файлов могут содержать инструкции) + анонимизация.
 * 4. message_end  — ответ ассистента деанонимизируется по vault-ключу
 *                   сессии: пользователь видит реальные данные.
 * 5. session_*    — vault-ключ живёт в сессии (переживает restart через
 *                   appendEntry), /pii-reset выдаёт новый ключ.
 *
 * Архитектура: плейсхолдеры [PERSON_1], [EMAIL_1]... стабильны в рамках
 * vault-сессии (хранится в PostgreSQL на стороне сервиса). Один и тот же
 * «Иванов» в разных сообщениях → один и тот же [PERSON_1] → ответ LLM
 * корректно деанонимизируется.
 *
 * Конфиг: PII_SERVICE_URL, GUARDRAILS_URL, PII_TIMEOUT_MS, PII_FAIL_OPEN
 * (см. client.ts).
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import {
  anonymize,
  deanonymize,
  guardrailsUrl,
  piiServiceUrl,
  scanInjections,
} from "./client.ts";

const VAULT_ENTRY_TYPE = "pii-guard-vault-key";
export default function piiGuardExtension(pi: ExtensionAPI) {
  // --- Состояние (на инстанс extension'а = на сессию Pi) -------------------

  const state = {
    vaultKey: null as string | null,
  };

  /** Восстановить vault-ключ из персистентных записей сессии. */
  function restoreVaultKeyFrom(entries: Array<unknown>): void {
    try {
      for (let i = entries.length - 1; i >= 0; i--) {
        const e = entries[i] as {
          type?: string;
          customType?: string;
          data?: { vault_key?: string };
        };
        if (e.type === "custom" && e.customType === VAULT_ENTRY_TYPE && e.data?.vault_key) {
          state.vaultKey = e.data.vault_key;
          return;
        }
      }
    } catch {
      // сессия может быть ещё не готова — ключ создастся при первом /anonymize
    }
    state.vaultKey = null;
  }

  /** Скан текста на инъекции. Возвращает строку-причину при block, иначе null. */
  async function injectionBlockReason(text: string, what: string): Promise<string | null> {
    const scan = await scanInjections(text);
    if (scan === null) return null; // fail-open: guardrails недоступен
    if (scan.decision === "block") {
      return (
        `[pii-guard] ${what} заблокирован: обнаружена промпт-инъекция ` +
        `(score ${scan.score})` +
        (scan.matched_rules.length
          ? ` — ${scan.matched_rules.map((r) => r.rule).join(", ")}`
          : "")
      );
    }
    return null;
  }

  // --- Жизненный цикл vault-ключа -----------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    restoreVaultKeyFrom(ctx.sessionManager.getEntries());
    if (state.vaultKey) {
      ctx.ui.notify(
        `pii-guard: vault-сессия восстановлена (${short(state.vaultKey)})`,
        "info",
      );
    }
  });

  pi.on("session_shutdown", async (_event, _ctx) => {
    // Ключ остаётся в записи сессии — восстановится при resume.
    // Активной чистки нет: TTL на стороне сервиса (24 ч по умолчанию).
    state.vaultKey = null;
  });


  // --- Ввод пользователя: скан + анонимизация ------------------------------

  pi.on("input", async (event, ctx) => {
    if (event.source === "extension") return { action: "continue" };

    const text = event.text;

    // 1. Инъекции в сыром вводе (включая /skill:foo — до раскрытия скилла)
    const blocked = await injectionBlockReason(text, "запрос");
    if (blocked) {
      ctx.ui.notify(blocked, "warning");
      return { action: "handled" };
    }

    // 2. Анонимизация ПДн
    const res = await anonymize(text, state.vaultKey);
    if (res === null) return { action: "continue" }; // fail-open
    state.vaultKey = res.vault_key;
    persistVaultKey(pi, res.vault_key);

    if (res.blocked) {
      // guardrails в block-режиме: сервис не обработал текст
      const msg = "[pii-guard] запрос заблокирован guardrails-сервисом";
      ctx.ui.notify(msg, "warning");
      return { action: "handled" };
    }

    if (res.entities.length > 0) {
      ctx.ui.notify(
        `pii-guard: замаскировано ${res.entities.length} сущностей ` +
          `(${Object.keys(res.stats).join(", ")})`,
        "info",
      );
    }
    return { action: "transform", text: res.anonymized_text };
  });

  // --- Вызов инструментов: аргументы с ПДн маскируются ---------------------

  pi.on("tool_call", async (event, ctx) => {
    const input = event.input as Record<string, unknown> | undefined;
    if (!input) return;
    // Текстовые поля аргументов: command (bash), path/content (write) и т.п.
    const fields = ["command", "content", "query", "text", "pattern"];
    for (const f of fields) {
      const v = input[f];
      if (typeof v !== "string" || !v) continue;
      const res = await anonymize(v, state.vaultKey);
      if (res === null) continue; // fail-open
      state.vaultKey = res.vault_key;
      if (res.vault_key) persistVaultKey(pi, res.vault_key);
      if (res.entities.length > 0) {
        input[f] = res.anonymized_text;
        ctx.ui.notify(
          `pii-guard: в ${event.toolName}.${f} замаскировано ${res.entities.length}`,
          "info",
        );
      }
    }
  });

  // --- Результаты инструментов: инъекции + анонимизация --------------------

  pi.on("tool_result", async (event, ctx) => {
    // Склейка текстового содержимого результата
    const content = event.content;
    if (!Array.isArray(content)) return;
    const texts = content
      .map((c) => (c && typeof c === "object" && "text" in c ? String((c as { text: unknown }).text) : null))
      .filter((t): t is string => t !== null && t.length > 0);
    if (texts.length === 0) return;
    const joined = texts.join("\n");

    // 1. Инъекции в данных (веб-страница, файл, вывод команды)
    const blocked = await injectionBlockReason(joined, `результат ${event.toolName}`);
    if (blocked) {
      ctx.ui.notify(blocked, "warning");
      return {
        content: [{ type: "text", text: blocked }],
        isError: true,
      };
    }

    // 2. Анонимизация ПДн в данных перед попаданием в LLM-контекст
    const res = await anonymize(joined, state.vaultKey);
    if (res === null || res.entities.length === 0) return;
    state.vaultKey = res.vault_key;
    if (res.vault_key) persistVaultKey(pi, res.vault_key);
    return {
      content: [{ type: "text", text: res.anonymized_text }],
    };
  });

  // --- Ответ ассистента: деанонимизация для пользователя -------------------

  pi.on("message_end", async (event) => {
    const msg = event.message;
    if (msg.role !== "assistant" || !state.vaultKey) return;
    if (!Array.isArray(msg.content)) return;

    let changed = false;
    const newContent = [];
    for (const block of msg.content) {
      if (
        block &&
        typeof block === "object" &&
        "type" in block &&
        block.type === "text" &&
        "text" in block
      ) {
        const res = await deanonymize(
          String(block.text),
          state.vaultKey,
        );
        if (res && res.replaced > 0) {
          changed = true;
          newContent.push({ ...block, text: res.text });
          continue;
        }
      }
      newContent.push(block);
    }
    if (changed) {
      return { message: { ...msg, content: newContent } };
    }
  });

  // --- Команда /pii-reset: новая vault-сессия ------------------------------

  pi.registerCommand("pii-reset", {
    description: "pii-guard: новая vault-сессия (сброс связи плейсхолдеров)",
    handler: async (_args, ctx) => {
      state.vaultKey = null;
      // первый же /anonymize выдаст новый ключ
      ctx.ui.notify("pii-guard: vault-сессия сброшена", "info");
    },
  });

  // --- Команда /pii-status -------------------------------------------------

  pi.registerCommand("pii-status", {
    description: "pii-guard: статус (URL сервисов, vault-ключ)",
    handler: async (_args, ctx) => {
      ctx.ui.notify(
        `pii-guard: service=${piiServiceUrl()}, guardrails=${guardrailsUrl()}, ` +
          `vault=${state.vaultKey ? short(state.vaultKey) : "нет (будет создан)"}`,
        "info",
      );
    },
  });
}

/** Сохранить vault-ключ в сессию (переживает restart). */
function persistVaultKey(pi: ExtensionAPI, key: string | null): void {
  if (!key) return;
  try {
    pi.appendEntry(VAULT_ENTRY_TYPE, { vault_key: key });
  } catch {
    // не критично: ключ живёт в памяти процесса
  }
}

function short(key: string): string {
  return key.slice(0, 8);
}
