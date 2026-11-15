// Тестовые моки для pi API и fetch — общие для всех тестов.

import { vi } from "vitest";

export type Handler = (event: any, ctx: any) => Promise<any>;

export interface RegisteredCommand {
  description: string;
  handler: (args: string, ctx: any) => Promise<void>;
}

/** Мок ExtensionAPI: собирает обработчики событий и команды. */
export class FakePi {
  handlers = new Map<string, Handler>();
  commands = new Map<string, RegisteredCommand>();
  entries: Array<{ type: string; customType?: string; data?: any }> = [];

  on(event: string, handler: Handler) {
    this.handlers.set(event, handler);
    return () => {};
  }

  registerCommand(name: string, cmd: RegisteredCommand) {
    this.commands.set(name, cmd);
  }

  appendEntry(customType: string, data?: any) {
    this.entries.push({ type: "custom", customType, data });
  }

  sessionManager = {
    getEntries: () => this.entries,
  };

  async emit(event: string, ev: any, ctx: any = fakeCtx()) {
    const h = this.handlers.get(event);
    return h ? h(ev, ctx) : undefined;
  }
}

export function fakeCtx(overrides: Partial<any> = {}) {
  return {
    ui: {
      notify: vi.fn(),
      confirm: vi.fn(async () => true),
    },
    mode: "tui",
    hasUI: true,
    cwd: "/tmp",
    signal: new AbortController().signal,
    sessionManager: {
      getEntries: (): Array<unknown> => [],
      getSessionId: () => "test-session",
    },
    ...overrides,
  };
}

export function extensionInput(text: string, source = "interactive") {
  return { text, images: undefined, source, streamingBehavior: undefined };
}

type FetchResponse = { ok: boolean; status?: number; json: () => Promise<any> };
type FetchMock = (url: string, init?: RequestInit) => FetchResponse;

/** Все вызовы fetch (глобально, переживает повторные mockFetch). */
export const fetchCalls: Array<{ url: string; body: any }> = [];

/** Подмена global fetch с записью вызовов. */
export function mockFetch(routes: Array<{ match: RegExp; reply: FetchMock }>) {
  const impl = async (url: string | URL | Request, init?: RequestInit) => {
    const u = String(url);
    const body = init?.body ? JSON.parse(String(init.body)) : null;
    fetchCalls.push({ url: u, body });
    for (const r of routes) {
      if (r.match.test(u)) return r.reply(u, init);
    }
    return { ok: false, status: 404, json: async () => ({}) };
  };
  vi.stubGlobal("fetch", vi.fn(impl));
  return fetchCalls;
}

export const okScan = {
  safe: true,
  decision: "allow",
  score: 0.01,
  rules_score: 0,
  ml_score: 0.01,
  matched_rules: [],
  latency_ms: 5,
};

export const blockScan = {
  safe: false,
  decision: "block",
  score: 0.95,
  rules_score: 0.9,
  ml_score: 0.95,
  matched_rules: [{ rule: "ru:ignore_instructions", weight: 0.9, match: "Игнорируй" }],
  latency_ms: 5,
};
