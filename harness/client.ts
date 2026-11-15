/**
 * HTTP-клиент к API сервиса анонимизации и guardrails (Node 18+, fetch).
 *
 * Конфиг через env:
 * - PII_SERVICE_URL  (дефолт http://localhost:8001) — /anonymize, /deanonymize
 * - GUARDRAILS_URL   (дефолт http://localhost:8002) — /scan
 * - PII_TIMEOUT_MS   (дефолт 15000)
 * - PII_FAIL_OPEN    (дефолт "1"): недоступность API не ломает работу Pi
 *   (текст проходит без обработки); "0" — ошибки блокируют запрос.
 */

const DEFAULT_TIMEOUT_MS = 15_000;

export interface ScanResult {
  safe: boolean;
  decision: "allow" | "review" | "block";
  score: number;
  rules_score: number;
  ml_score: number | null;
  matched_rules: Array<{ rule: string; weight: number; match: string }>;
  latency_ms: number;
}

export interface EntitySpan {
  start: number;
  end: number;
  text: string;
  category: string;
  placeholder: string;
  source: string;
  score: number | null;
}

export interface AnonymizeResult {
  anonymized_text: string;
  entities: EntitySpan[];
  stats: Record<string, number>;
  blocked: boolean;
  vault_key: string | null;
  guardrails: {
    status: string;
    decision: string | null;
    score: number | null;
  } | null;
}

export interface DeanonymizeResult {
  text: string;
  replaced: number;
}

export class PiiApiError extends Error {
  constructor(
    message: string,
    public readonly status: number | null,
  ) {
    super(message);
  }
}

function env(name: string, fallback: string): string {
  // process может отсутствовать в тестовом окружении
  const p = typeof process !== "undefined" ? process.env : {};
  return p[name] ?? fallback;
}

export function piiServiceUrl(): string {
  return env("PII_SERVICE_URL", "http://localhost:8001");
}

export function guardrailsUrl(): string {
  return env("GUARDRAILS_URL", "http://localhost:8002");
}

function timeoutMs(): number {
  return Number(env("PII_TIMEOUT_MS", String(DEFAULT_TIMEOUT_MS)));
}

function failOpen(): boolean {
  return env("PII_FAIL_OPEN", "1") === "1";
}

async function postJson<T>(url: string, body: unknown): Promise<T | null> {
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), timeoutMs());
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      throw new PiiApiError(`${url} -> HTTP ${res.status}`, res.status);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(t);
  }
}

/** Скан на промпт-инъекции. null = сервис недоступен (fail-open). */
export async function scanInjections(text: string): Promise<ScanResult | null> {
  try {
    return await postJson<ScanResult>(`${guardrailsUrl()}/scan`, { text });
  } catch (e) {
    if (!failOpen()) throw e;
    return null;
  }
}

/** Анонимизация. null = сервис недоступен (fail-open). */
export async function anonymize(
  text: string,
  vaultKey?: string | null,
): Promise<AnonymizeResult | null> {
  try {
    return await postJson<AnonymizeResult>(`${piiServiceUrl()}/anonymize`, {
      text,
      ...(vaultKey ? { vault_key: vaultKey } : {}),
    });
  } catch (e) {
    if (!failOpen()) throw e;
    return null;
  }
}

/** Деанонимизация по vault-ключу. null = сервис недоступен. */
export async function deanonymize(
  text: string,
  vaultKey: string,
): Promise<DeanonymizeResult | null> {
  try {
    return await postJson<DeanonymizeResult>(
      `${piiServiceUrl()}/deanonymize`,
      { text, vault_key: vaultKey },
    );
  } catch (e) {
    if (!failOpen()) throw e;
    return null;
  }
}
