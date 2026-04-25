import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import { useLumiState } from "./useLumiState";
import type { BrainClient } from "../ipc/client";
import type { LumiBrainEvent } from "../ipc/events";

// ---------------------------------------------------------------------------
// Stub BrainClient
// ---------------------------------------------------------------------------
// We need a minimal fake that lets us manually push events into the hook
// without any real WebSocket. We capture the handler registered via onEvent.

function createMockClient(): BrainClient & {
  _pushEvent: (e: LumiBrainEvent) => void;
} {
  let registeredHandler: ((e: LumiBrainEvent) => void) | null = null;

  const mock = {
    onEvent(handler: (e: LumiBrainEvent) => void): void {
      registeredHandler = handler;
    },
    _pushEvent(e: LumiBrainEvent): void {
      if (registeredHandler === null) {
        throw new Error("No handler registered — did useLumiState mount?");
      }
      registeredHandler(e);
    },
  } as unknown as BrainClient & { _pushEvent: (e: LumiBrainEvent) => void };

  return mock;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("useLumiState — initial state", () => {
  it("returns correct defaults on mount", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    expect(result.current.brainState).toBe("IDLE");
    expect(result.current.transcript).toBe("");
    expect(result.current.streamingTokens).toBe("");
    expect(result.current.currentUtterance).toBe("");
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — state_change event", () => {
  it("updates brainState when state_change is received", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "state_change",
        payload: { state: "LISTENING" },
      });
    });

    expect(result.current.brainState).toBe("LISTENING");
  });

  it("updates brainState through all valid states", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    const states = ["LISTENING", "PROCESSING", "SPEAKING", "IDLE"] as const;
    for (const state of states) {
      act(() => {
        client._pushEvent({ event: "state_change", payload: { state } });
      });
      expect(result.current.brainState).toBe(state);
    }
  });

  it("does not mutate other state fields when brainState changes", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "transcript",
        payload: { text: "existing text" },
      });
    });
    act(() => {
      client._pushEvent({
        event: "state_change",
        payload: { state: "PROCESSING" },
      });
    });

    expect(result.current.transcript).toBe("existing text");
    expect(result.current.streamingTokens).toBe("");
    expect(result.current.currentUtterance).toBe("");
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — transcript event", () => {
  it("sets transcript when transcript event arrives", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "transcript",
        payload: { text: "Hello from user" },
      });
    });

    expect(result.current.transcript).toBe("Hello from user");
  });

  it("replaces transcript on subsequent transcript events", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "first" } });
    });
    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "second" } });
    });

    // useLumiState sets transcript directly — not appended
    expect(result.current.transcript).toBe("second");
  });

  it("handles empty string transcript", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "hello" } });
    });
    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "" } });
    });

    expect(result.current.transcript).toBe("");
  });

  it("handles special characters in transcript", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    const specialText = "Hello! 😊 <script>alert('xss')</script> & \"quotes\"";
    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: specialText } });
    });

    expect(result.current.transcript).toBe(specialText);
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — llm_token events", () => {
  it("accumulates streaming tokens", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "llm_token",
        payload: { token: "Hello", utterance_id: "u1" },
      });
    });
    act(() => {
      client._pushEvent({
        event: "llm_token",
        payload: { token: " world", utterance_id: "u1" },
      });
    });

    expect(result.current.streamingTokens).toBe("Hello world");
  });

  it("appends tokens from different utterance_ids", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "llm_token",
        payload: { token: "A", utterance_id: "u1" },
      });
    });
    act(() => {
      client._pushEvent({
        event: "llm_token",
        payload: { token: "B", utterance_id: "u2" },
      });
    });

    expect(result.current.streamingTokens).toBe("AB");
  });

  it("does not change other fields while accumulating tokens", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "llm_token",
        payload: { token: "token", utterance_id: "u1" },
      });
    });

    expect(result.current.brainState).toBe("IDLE");
    expect(result.current.transcript).toBe("");
    expect(result.current.currentUtterance).toBe("");
  });

  it("accumulates a large number of tokens correctly", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    const tokenCount = 200;
    act(() => {
      for (let i = 0; i < tokenCount; i++) {
        client._pushEvent({
          event: "llm_token",
          payload: { token: `t${i} `, utterance_id: "u1" },
        });
      }
    });

    const parts = result.current.streamingTokens.trim().split(" ");
    expect(parts).toHaveLength(tokenCount);
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — tts_start event", () => {
  it("sets currentUtterance from tts_start text", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "tts_start",
        payload: { text: "Speaking now", duration_ms: 1500 },
      });
    });

    expect(result.current.currentUtterance).toBe("Speaking now");
  });

  it("clears streamingTokens when tts_start fires", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    // Accumulate some tokens first
    act(() => {
      client._pushEvent({
        event: "llm_token",
        payload: { token: "some tokens", utterance_id: "u1" },
      });
    });
    expect(result.current.streamingTokens).toBe("some tokens");

    // tts_start should flush them
    act(() => {
      client._pushEvent({
        event: "tts_start",
        payload: { text: "Finalised text", duration_ms: 2000 },
      });
    });

    expect(result.current.streamingTokens).toBe("");
    expect(result.current.currentUtterance).toBe("Finalised text");
  });

  it("does not mutate transcript or brainState on tts_start", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "user said hi" } });
    });
    act(() => {
      client._pushEvent({ event: "state_change", payload: { state: "SPEAKING" } });
    });
    act(() => {
      client._pushEvent({
        event: "tts_start",
        payload: { text: "response text", duration_ms: 1000 },
      });
    });

    expect(result.current.transcript).toBe("user said hi");
    expect(result.current.brainState).toBe("SPEAKING");
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — tts_stop event", () => {
  it("clears currentUtterance when tts_stop fires", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({
        event: "tts_start",
        payload: { text: "Speaking", duration_ms: 1000 },
      });
    });
    expect(result.current.currentUtterance).toBe("Speaking");

    act(() => {
      client._pushEvent({
        event: "tts_stop",
        payload: {},
      });
    });

    expect(result.current.currentUtterance).toBe("");
  });

  it("does not affect other state fields on tts_stop", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "preserved" } });
    });
    act(() => {
      client._pushEvent({ event: "state_change", payload: { state: "IDLE" } });
    });
    act(() => {
      client._pushEvent({
        event: "tts_start",
        payload: { text: "hello", duration_ms: 500 },
      });
    });
    act(() => {
      client._pushEvent({ event: "tts_stop", payload: {} });
    });

    expect(result.current.transcript).toBe("preserved");
    expect(result.current.brainState).toBe("IDLE");
    expect(result.current.streamingTokens).toBe("");
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — full conversation flow", () => {
  it("models a complete turn: listening → tokens → tts → done", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    act(() => {
      client._pushEvent({ event: "state_change", payload: { state: "LISTENING" } });
    });
    expect(result.current.brainState).toBe("LISTENING");

    act(() => {
      client._pushEvent({ event: "transcript", payload: { text: "what time is it?" } });
    });
    expect(result.current.transcript).toBe("what time is it?");

    act(() => {
      client._pushEvent({ event: "state_change", payload: { state: "PROCESSING" } });
    });

    act(() => {
      ["It", " is", " noon."].forEach((token) => {
        client._pushEvent({
          event: "llm_token",
          payload: { token, utterance_id: "u1" },
        });
      });
    });
    expect(result.current.streamingTokens).toBe("It is noon.");

    act(() => {
      client._pushEvent({
        event: "tts_start",
        payload: { text: "It is noon.", duration_ms: 1200 },
      });
    });
    expect(result.current.currentUtterance).toBe("It is noon.");
    expect(result.current.streamingTokens).toBe("");

    act(() => {
      client._pushEvent({ event: "state_change", payload: { state: "SPEAKING" } });
    });

    act(() => {
      client._pushEvent({ event: "tts_stop", payload: {} });
    });
    expect(result.current.currentUtterance).toBe("");

    act(() => {
      client._pushEvent({ event: "state_change", payload: { state: "IDLE" } });
    });
    expect(result.current.brainState).toBe("IDLE");
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — ignored event types", () => {
  it("does not change state when tts_viseme arrives", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    const before = { ...result.current };

    act(() => {
      client._pushEvent({
        event: "tts_viseme",
        payload: { viseme: "AA", duration_ms: 80 },
      });
    });

    expect(result.current).toEqual(before);
  });

  it("does not change state when error event arrives", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    const before = { ...result.current };

    act(() => {
      client._pushEvent({
        event: "error",
        payload: { code: "E001", message: "something went wrong" },
      });
    });

    expect(result.current).toEqual(before);
  });
});

// ---------------------------------------------------------------------------

describe("useLumiState — hook stability", () => {
  it("re-registers the onEvent handler when the client reference changes", () => {
    const client1 = createMockClient();
    let client = client1;

    const { result, rerender } = renderHook(() => useLumiState(client));

    // Confirm handler works with client1
    act(() => {
      client1._pushEvent({ event: "transcript", payload: { text: "from c1" } });
    });
    expect(result.current.transcript).toBe("from c1");

    // Swap to a new client
    const client2 = createMockClient();
    client = client2;
    rerender();

    act(() => {
      client2._pushEvent({ event: "transcript", payload: { text: "from c2" } });
    });
    expect(result.current.transcript).toBe("from c2");
  });

  it("returns a stable state reference shape (all four fields always present)", () => {
    const client = createMockClient();
    const { result } = renderHook(() => useLumiState(client));

    const state = result.current;
    expect(state).toHaveProperty("brainState");
    expect(state).toHaveProperty("transcript");
    expect(state).toHaveProperty("streamingTokens");
    expect(state).toHaveProperty("currentUtterance");
  });
});
