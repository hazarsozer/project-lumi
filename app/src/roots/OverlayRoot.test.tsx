/**
 * OverlayRoot.test.tsx
 *
 * Covers the CR-17 / Issue #18 unlisten-on-unmount fix:
 *   - The unlisten handle must always be invoked on unmount, even when unmount
 *     happens before the listen() Promise resolves (the "cancelled-flag race").
 *   - Repeated mount/unmount must not accumulate listeners.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, act } from "@testing-library/react";
import { OverlayRoot } from "./OverlayRoot";

// ---------------------------------------------------------------------------
// Mock useBrainSocket — minimal stub so OverlayRoot mounts cleanly.
// ---------------------------------------------------------------------------

vi.mock("../state/useBrainSocket", () => ({
  useBrainSocket: () => ({
    client: {
      onEvent: vi.fn(() => () => {}),
      send: vi.fn(),
    },
    connectionState: "connected",
  }),
}));

vi.mock("../state/useLumiState", () => ({
  useLumiState: () => ({
    brainState: "idle",
    systemStatus: null,
  }),
}));

// ---------------------------------------------------------------------------
// Mock tauriCompat — we control when (and whether) the Promise resolves.
// ---------------------------------------------------------------------------

// Each call to tauriListen gets its own resolvable Promise so we can
// test both "resolve before unmount" and "resolve after unmount" cases.
type ListenResolve = (fn: () => void) => void;
const pendingResolves: ListenResolve[] = [];
const mockUnlisten = vi.fn<() => void>().mockReturnValue(undefined);

vi.mock("../lib/tauriCompat", () => ({
  tauriListen: vi.fn((_event: string, _handler: unknown) => {
    return new Promise<() => void>((resolve) => {
      pendingResolves.push(resolve);
    });
  }),
  tauriEmit: vi.fn().mockResolvedValue(undefined),
  tauriGetWindowByLabel: vi.fn().mockResolvedValue(null),
}));

// ---------------------------------------------------------------------------

beforeEach(() => {
  pendingResolves.length = 0;
  mockUnlisten.mockClear();
});

// ---------------------------------------------------------------------------
// Case 1: Normal flow — resolve BEFORE unmount.
// Unlisten should be called when the component unmounts.
// ---------------------------------------------------------------------------

describe("OverlayRoot — unlisten called on normal unmount (resolve before unmount)", () => {
  it("calls unlisten for each listener when the component unmounts", async () => {
    const { unmount } = render(<OverlayRoot />);

    // OverlayRoot registers 3 tauriListen calls (EV_SEND, EV_CONFIG_SCHEMA_REQUEST,
    // EV_CONFIG_UPDATE). Resolve all of them with mockUnlisten.
    await act(async () => {
      for (const resolve of pendingResolves) {
        resolve(mockUnlisten);
      }
    });

    const resolvedCount = pendingResolves.length;
    expect(resolvedCount).toBeGreaterThanOrEqual(3);

    // No unlisten calls yet — component is still mounted.
    expect(mockUnlisten).not.toHaveBeenCalled();

    // Unmount → each effect cleanup must invoke unlisten.
    act(() => { unmount(); });
    expect(mockUnlisten).toHaveBeenCalledTimes(resolvedCount);
  });
});

// ---------------------------------------------------------------------------
// Case 2: Unmount BEFORE the listen() Promise resolves.
// The unlisten handle must still be called when the Promise eventually resolves.
// ---------------------------------------------------------------------------

describe("OverlayRoot — unlisten called even when unmount happens before listen() resolves", () => {
  it("calls unlisten after the Promise resolves, even if unmount already happened", async () => {
    const { unmount } = render(<OverlayRoot />);

    // Unmount immediately — Promise has NOT resolved yet.
    act(() => { unmount(); });

    // Unlisten has not been called yet (Promise still pending).
    expect(mockUnlisten).not.toHaveBeenCalled();

    // Now resolve the Promises — the effects must detect unmount and call
    // unlisten immediately upon resolution.
    await act(async () => {
      for (const resolve of pendingResolves) {
        resolve(mockUnlisten);
      }
    });

    // All resolved Promises must have triggered unlisten.
    expect(mockUnlisten).toHaveBeenCalledTimes(pendingResolves.length);
    expect(pendingResolves.length).toBeGreaterThanOrEqual(3);
  });
});

// ---------------------------------------------------------------------------
// Case 3: Repeated mount/unmount does not accumulate listeners.
// Each cycle must produce exactly N unlisten calls (no leaks).
// ---------------------------------------------------------------------------

describe("OverlayRoot — repeated mount/unmount does not accumulate listeners", () => {
  it("unlisten call count equals listener count across two mount/unmount cycles", async () => {
    // First cycle
    const { unmount: unmount1 } = render(<OverlayRoot />);
    await act(async () => {
      for (const resolve of pendingResolves) {
        resolve(mockUnlisten);
      }
    });
    act(() => { unmount1(); });
    const afterFirstUnmount = mockUnlisten.mock.calls.length;
    expect(afterFirstUnmount).toBeGreaterThanOrEqual(3);

    // Reset for second cycle
    mockUnlisten.mockClear();
    pendingResolves.length = 0;

    // Second cycle
    const { unmount: unmount2 } = render(<OverlayRoot />);
    await act(async () => {
      for (const resolve of pendingResolves) {
        resolve(mockUnlisten);
      }
    });
    act(() => { unmount2(); });
    const afterSecondUnmount = mockUnlisten.mock.calls.length;

    // Same count as first cycle — no accumulation.
    expect(afterSecondUnmount).toBe(afterFirstUnmount);
  });
});
