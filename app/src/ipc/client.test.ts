import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { BrainClient } from "./client";
import type { LumiBrainEvent, OutboundEvent } from "./events";

// ---------------------------------------------------------------------------
// Minimal WebSocket mock
// ---------------------------------------------------------------------------

type WsEventName = "open" | "close" | "message" | "error";

class MockWebSocket {
  static instances: MockWebSocket[] = [];

  url: string;
  readyState: number = 0; // CONNECTING

  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onerror: (() => void) | null = null;

  sentMessages: string[] = [];
  closed = false;

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  /** Simulate server accepting the connection. */
  simulateOpen(): void {
    this.readyState = 1; // OPEN
    this.onopen?.();
  }

  /** Simulate the server closing / network drop. */
  simulateClose(): void {
    this.readyState = 3; // CLOSED
    this.onclose?.();
  }

  /** Push a JSON-encoded brain event to the client. */
  simulateMessage(data: string): void {
    this.onmessage?.({ data });
  }

  simulateError(): void {
    this.onerror?.();
  }

  send(data: string): void {
    this.sentMessages.push(data);
  }

  close(): void {
    this.closed = true;
    this.readyState = 3;
    this.onclose?.();
  }

  // Satisfy the addEventListener / removeEventListener surface so TypeScript
  // doesn't complain when the lib checks for it.
  addEventListener(_type: WsEventName, _listener: () => void): void {}
  removeEventListener(_type: WsEventName, _listener: () => void): void {}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// TODO: BACKOFF_STEPS_MS is not exported from client.ts — values hardcoded
//       here must stay in sync with the source file.
const BACKOFF_STEPS_MS = [1000, 2000, 4000, 8000];

function wireEvent(event: LumiBrainEvent): string {
  return JSON.stringify({
    event: event.event,
    payload: event.payload,
    timestamp: Date.now(),
    version: "1.0",
  });
}

function latestWs(): MockWebSocket {
  return MockWebSocket.instances[MockWebSocket.instances.length - 1];
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.useFakeTimers();
  vi.stubGlobal("WebSocket", MockWebSocket);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("BrainClient — state transitions", () => {
  it("starts in disconnected state", () => {
    const client = new BrainClient();
    expect(client.state).toBe("disconnected");
  });

  it("transitions to connecting immediately after connect()", () => {
    const client = new BrainClient();
    client.connect();
    expect(client.state).toBe("connecting");
  });

  it("transitions to connected when the WebSocket opens", () => {
    const client = new BrainClient();
    client.connect();
    latestWs().simulateOpen();
    expect(client.state).toBe("connected");
  });

  it("transitions to disconnected when the WebSocket closes", () => {
    const client = new BrainClient();
    client.connect();
    const ws = latestWs();
    ws.simulateOpen();
    expect(client.state).toBe("connected");

    // Prevent the automatic reconnect from spawning a new WS during teardown.
    // We cancel it by calling disconnect() after inspecting state.
    ws.simulateClose();
    expect(client.state).toBe("disconnected");
    client.disconnect(); // clean up pending timer
  });

  it("notifies all onStateChange handlers on every transition", () => {
    const states1: string[] = [];
    const states2: string[] = [];
    const client = new BrainClient();
    client.onStateChange((s) => states1.push(s));
    client.onStateChange((s) => states2.push(s));

    client.connect();
    latestWs().simulateOpen();

    expect(states1).toEqual(["connecting", "connected"]);
    expect(states2).toEqual(["connecting", "connected"]);
  });
});

// ---------------------------------------------------------------------------

describe("BrainClient — event dispatch", () => {
  it("calls onEvent handler with a correctly typed state_change event", () => {
    const client = new BrainClient();
    const received: LumiBrainEvent[] = [];
    client.onEvent((e) => received.push(e));
    client.connect();

    const event: LumiBrainEvent = {
      event: "state_change",
      payload: { state: "LISTENING" },
    };
    latestWs().simulateMessage(wireEvent(event));

    expect(received).toHaveLength(1);
    expect(received[0]).toMatchObject({
      event: "state_change",
      payload: { state: "LISTENING" },
    });
  });

  it("calls onEvent handler with a transcript event", () => {
    const client = new BrainClient();
    const received: LumiBrainEvent[] = [];
    client.onEvent((e) => received.push(e));
    client.connect();

    const event: LumiBrainEvent = {
      event: "transcript",
      payload: { text: "Hello, world!" },
    };
    latestWs().simulateMessage(wireEvent(event));

    expect(received[0]).toMatchObject({
      event: "transcript",
      payload: { text: "Hello, world!" },
    });
  });

  it("ignores messages with unknown event types", () => {
    const client = new BrainClient();
    const received: LumiBrainEvent[] = [];
    client.onEvent((e) => received.push(e));
    client.connect();

    latestWs().simulateMessage(
      JSON.stringify({
        event: "unknown_event_type",
        payload: {},
        timestamp: Date.now(),
        version: "1.0",
      })
    );

    expect(received).toHaveLength(0);
  });

  it("ignores malformed (non-JSON) messages", () => {
    const client = new BrainClient();
    const received: LumiBrainEvent[] = [];
    client.onEvent((e) => received.push(e));
    client.connect();

    expect(() =>
      latestWs().simulateMessage("not-valid-json{{")
    ).not.toThrow();
    expect(received).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------

describe("BrainClient — multiple handlers", () => {
  it("calls all registered onEvent handlers for each incoming event", () => {
    const client = new BrainClient();
    const calls1: LumiBrainEvent[] = [];
    const calls2: LumiBrainEvent[] = [];

    client.onEvent((e) => calls1.push(e));
    client.onEvent((e) => calls2.push(e));
    client.connect();

    const event: LumiBrainEvent = {
      event: "llm_token",
      payload: { token: "Hi", utterance_id: "u1" },
    };
    latestWs().simulateMessage(wireEvent(event));

    expect(calls1).toHaveLength(1);
    expect(calls2).toHaveLength(1);
    expect(calls1[0].event).toBe("llm_token");
    expect(calls2[0].event).toBe("llm_token");
  });

  it("accumulates multiple llm_token messages across all handlers", () => {
    const client = new BrainClient();
    const tokens: string[] = [];
    client.onEvent((e) => {
      if (e.event === "llm_token") tokens.push(e.payload.token);
    });
    client.connect();

    const ws = latestWs();
    (["Hello", " ", "world"] as const).forEach((token) => {
      const event: LumiBrainEvent = {
        event: "llm_token",
        payload: { token, utterance_id: "u1" },
      };
      ws.simulateMessage(wireEvent(event));
    });

    expect(tokens).toEqual(["Hello", " ", "world"]);
  });
});

// ---------------------------------------------------------------------------

describe("BrainClient — send()", () => {
  it("serialises and sends a valid outbound event when connected", () => {
    const client = new BrainClient();
    client.connect();
    const ws = latestWs();
    ws.simulateOpen();

    const outbound: OutboundEvent = {
      event: "user_text",
      payload: { text: "test message" },
    };
    client.send(outbound);

    expect(ws.sentMessages).toHaveLength(1);
    const parsed = JSON.parse(ws.sentMessages[0]) as {
      event: string;
      payload: { text: string };
      version: string;
    };
    expect(parsed.event).toBe("user_text");
    expect(parsed.payload.text).toBe("test message");
    expect(parsed.version).toBe("1.0");
  });

  it("silently drops send() while in connecting state", () => {
    const client = new BrainClient();
    client.connect(); // state = connecting; ws not open yet
    const ws = latestWs();

    const outbound: OutboundEvent = {
      event: "interrupt",
      payload: {},
    };
    client.send(outbound);

    expect(ws.sentMessages).toHaveLength(0);
  });

  it("silently drops send() after disconnect()", () => {
    const client = new BrainClient();
    client.connect();
    latestWs().simulateOpen();
    client.disconnect();

    const outbound: OutboundEvent = {
      event: "interrupt",
      payload: {},
    };
    client.send(outbound);

    // The WS created during connect() should not have received the message.
    expect(MockWebSocket.instances[0].sentMessages).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------

describe("BrainClient — reconnect backoff", () => {
  it("reconnects after first disconnect using BACKOFF_STEPS_MS[0] delay", () => {
    const client = new BrainClient();
    client.connect();
    const firstWs = latestWs();
    firstWs.simulateOpen();

    // Simulate network drop — triggers _scheduleReconnect
    firstWs.simulateClose();
    expect(MockWebSocket.instances).toHaveLength(1); // no new WS yet

    // Advance time by the first backoff step
    vi.advanceTimersByTime(BACKOFF_STEPS_MS[0]);
    expect(MockWebSocket.instances).toHaveLength(2);
    expect(client.state).toBe("connecting");

    client.disconnect();
  });

  it("escalates delay on the second reconnect attempt", () => {
    const client = new BrainClient();
    client.connect();

    // First connection established and lost
    latestWs().simulateOpen();
    latestWs().simulateClose();

    // First reconnect fires
    vi.advanceTimersByTime(BACKOFF_STEPS_MS[0]);
    expect(MockWebSocket.instances).toHaveLength(2);

    // Second connection also lost without ever opening → escalated delay
    latestWs().simulateClose();
    expect(MockWebSocket.instances).toHaveLength(2); // still waiting

    vi.advanceTimersByTime(BACKOFF_STEPS_MS[1]);
    expect(MockWebSocket.instances).toHaveLength(3);

    client.disconnect();
  });

  it("caps delay at the last BACKOFF_STEPS_MS value", () => {
    const client = new BrainClient();
    client.connect();

    // Exhaust all backoff steps
    const steps = BACKOFF_STEPS_MS.length + 2; // go beyond the array length
    for (let i = 0; i < steps; i++) {
      latestWs().simulateClose();
      const lastDelay = BACKOFF_STEPS_MS[BACKOFF_STEPS_MS.length - 1];
      vi.advanceTimersByTime(lastDelay);
    }

    // Should have created one WS per reconnect attempt (steps + 1 initial)
    expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(steps);

    client.disconnect();
  });

  it("resets reconnect attempt counter after a successful connection", () => {
    const client = new BrainClient();
    client.connect();

    // Fail once
    latestWs().simulateClose();
    vi.advanceTimersByTime(BACKOFF_STEPS_MS[0]);

    // Second attempt succeeds → resets counter
    latestWs().simulateOpen();
    expect(client.state).toBe("connected");

    // Fail again — should use the first backoff step again, not the second
    latestWs().simulateClose();
    expect(MockWebSocket.instances.length).toBe(2);

    vi.advanceTimersByTime(BACKOFF_STEPS_MS[0]);
    expect(MockWebSocket.instances.length).toBe(3);

    client.disconnect();
  });

  it("disconnect() cancels a pending reconnect timer", () => {
    const client = new BrainClient();
    client.connect();
    latestWs().simulateOpen();
    latestWs().simulateClose(); // schedules reconnect

    client.disconnect(); // should cancel the timer

    vi.advanceTimersByTime(BACKOFF_STEPS_MS[0] * 2);
    // No new WebSocket should be created after disconnect
    expect(MockWebSocket.instances).toHaveLength(1);
    expect(client.state).toBe("disconnected");
  });
});

// ---------------------------------------------------------------------------

describe("BrainClient — error handling", () => {
  it("closes the WebSocket when onerror fires", () => {
    const client = new BrainClient();
    client.connect();
    const ws = latestWs();
    ws.simulateOpen();

    ws.simulateError();

    expect(ws.closed).toBe(true);
    client.disconnect();
  });

  it("transitions to disconnected after an error triggers close", () => {
    const client = new BrainClient();
    client.connect();
    const ws = latestWs();
    ws.simulateOpen();

    // onerror calls ws.close() which fires onclose, which calls _setState("disconnected")
    ws.simulateError();

    expect(client.state).toBe("disconnected");
    client.disconnect();
  });
});
