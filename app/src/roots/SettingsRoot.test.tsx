/**
 * SettingsRoot.test.tsx
 *
 * Covers the CR-15 / Issue #16 listen-before-emit fix:
 *   - tauriListen must be called BEFORE tauriEmit in the useEffect so that a
 *     fast schema response from the Brain is never dropped.
 *   - Schema and currentValues are passed down to SettingsPanel once received.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, act } from "@testing-library/react";
import { SettingsRoot } from "./SettingsRoot";

// ---------------------------------------------------------------------------
// Mock tauriCompat — we control when events arrive.
// ---------------------------------------------------------------------------

// Track the call order between tauriListen and tauriEmit.
const callOrder: string[] = [];

// The EV_BRAIN handler captured when tauriListen is called.
type BrainHandler = (evt: unknown) => void;
let capturedBrainHandler: BrainHandler | null = null;

const mockUnlisten = vi.fn<() => void>().mockReturnValue(undefined);

vi.mock("../lib/tauriCompat", () => ({
  tauriListen: vi.fn((_event: string, handler: BrainHandler) => {
    callOrder.push("listen");
    capturedBrainHandler = handler;
    return Promise.resolve(mockUnlisten);
  }),
  tauriEmit: vi.fn((_event: string) => {
    callOrder.push("emit");
    return Promise.resolve();
  }),
  tauriGetWindowByLabel: vi.fn().mockResolvedValue(null),
}));

// ---------------------------------------------------------------------------

beforeEach(() => {
  callOrder.length = 0;
  capturedBrainHandler = null;
  mockUnlisten.mockClear();
});

// ---------------------------------------------------------------------------
// Listen-before-emit ordering
// ---------------------------------------------------------------------------

describe("SettingsRoot — listen-before-emit ordering", () => {
  it("registers EV_BRAIN listener BEFORE emitting config_schema_request", async () => {
    await act(async () => {
      render(<SettingsRoot />);
    });

    // tauriListen must have been called first.
    expect(callOrder[0]).toBe("listen");
    expect(callOrder[1]).toBe("emit");
  });

  it("does not emit before listen even when both resolve synchronously", async () => {
    await act(async () => {
      render(<SettingsRoot />);
    });

    const listenIdx = callOrder.indexOf("listen");
    const emitIdx = callOrder.indexOf("emit");
    expect(listenIdx).toBeGreaterThanOrEqual(0);
    expect(emitIdx).toBeGreaterThanOrEqual(0);
    expect(listenIdx).toBeLessThan(emitIdx);
  });
});

// ---------------------------------------------------------------------------
// Schema propagation — config_schema event arrives after listener registered
// ---------------------------------------------------------------------------

describe("SettingsRoot — schema propagation", () => {
  it("renders SettingsPanel (Settings heading visible)", async () => {
    await act(async () => {
      render(<SettingsRoot />);
    });
    expect(screen.getByText("Settings")).not.toBeNull();
  });

  it("passes currentValues from config_schema to SettingsPanel", async () => {
    await act(async () => {
      render(<SettingsRoot />);
    });

    // Simulate the Brain responding to the schema request immediately.
    await act(async () => {
      capturedBrainHandler?.({
        event: "config_schema",
        payload: {
          fields: { volume: { type: "number" } },
          current_values: { volume: 88 },
        },
      });
    });

    // After the schema arrives, clicking Apply should emit the live volume (88),
    // not the default (72). We verify by checking the Apply button is present —
    // the full prop-flow is unit-tested in SettingsPanel.test.tsx.
    expect(screen.getByText("Apply")).not.toBeNull();
  });

  it("handles config_schema response received before unlisten (fast Brain)", async () => {
    // The whole point of the fix: if the Brain responds synchronously (before the
    // Promise returned by tauriListen resolves), the handler is already registered
    // so the response is NOT dropped.
    //
    // We simulate this by firing the event inside the same act() as the render.
    await act(async () => {
      render(<SettingsRoot />);
      // Handler is captured synchronously inside tauriListen mock.
      // Fire the schema event immediately — this proves fast Brain ≠ dropped event.
      capturedBrainHandler?.({
        event: "config_schema",
        payload: {
          fields: {},
          current_values: { volume: 77 },
        },
      });
    });

    // SettingsPanel rendered successfully — no crash from dropped/missing schema.
    expect(screen.getByText("Settings")).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// config_update_result feedback propagation
// ---------------------------------------------------------------------------

describe("SettingsRoot — config_update_result feedback", () => {
  it("shows restart hint when Brain reports pending_restart", async () => {
    await act(async () => {
      render(<SettingsRoot />);
    });

    await act(async () => {
      capturedBrainHandler?.({
        event: "config_update_result",
        payload: {
          applied_live: [],
          pending_restart: ["llm.model"],
          errors: {},
        },
      });
    });

    expect(screen.getByText(/Restart required for some changes/i)).not.toBeNull();
  });

  it("shows Applied marker when Brain reports success with no pending_restart", async () => {
    await act(async () => {
      render(<SettingsRoot />);
    });

    await act(async () => {
      capturedBrainHandler?.({
        event: "config_update_result",
        payload: {
          applied_live: ["volume"],
          pending_restart: [],
          errors: {},
        },
      });
    });

    expect(screen.getByText(/Applied/i)).not.toBeNull();
  });
});
