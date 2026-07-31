/**
 * SettingsPanel.test.tsx
 *
 * Covers the three fixes from CR-15 / Issue #16:
 *   1. Panel renders from currentValues props — not hardcoded DEFAULT_SETTINGS.
 *   2. Footer "Requires restart" only appears with a pending restart result.
 *   3. resolveInitialVals fallback logic.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { SettingsPanel } from "./SettingsPanel";

// ---------------------------------------------------------------------------
// Suppress React act() warnings caused by ContextTab's internal useState
// (Sources list renders on mount but has no async side-effects relevant here).
// ---------------------------------------------------------------------------

// SettingsTabs import heavy UI but no async side effects — no extra mocking needed.

const noop = () => {};

// ---------------------------------------------------------------------------
// resolveInitialVals — tested via rendered output
// ---------------------------------------------------------------------------

describe("SettingsPanel — renders from currentValues prop (not DEFAULT_SETTINGS)", () => {
  it("uses DEFAULT_SETTINGS when currentValues is undefined", () => {
    // DEFAULT_SETTINGS.volume = 72 — we can indirectly verify via handleApply.
    const captured: Array<Record<string, unknown>> = [];
    render(
      <SettingsPanel
        onUpdate={(changes) => { captured.push(changes); }}
        onClose={noop}
      />,
    );

    screen.getByText("Apply").click();

    expect(captured).toHaveLength(1);
    expect(captured[0]["volume"]).toBe(72);
    expect(captured[0]["temperature"]).toBe(0.68);
  });

  it("uses currentValues from Brain when they are provided", () => {
    const liveValues: Record<string, unknown> = {
      volume: 55,
      temperature: 0.42,
      model: "lumi-custom",
    };

    const captured: Array<Record<string, unknown>> = [];
    render(
      <SettingsPanel
        currentValues={liveValues}
        onUpdate={(changes) => { captured.push(changes); }}
        onClose={noop}
      />,
    );

    screen.getByText("Apply").click();

    expect(captured).toHaveLength(1);
    // Live values should override defaults.
    expect(captured[0]["volume"]).toBe(55);
    expect(captured[0]["temperature"]).toBe(0.42);
    expect(captured[0]["model"]).toBe("lumi-custom");
  });

  it("falls back to DEFAULT_SETTINGS for keys absent in currentValues", () => {
    // Only override one field — the rest should still be the defaults.
    const liveValues: Record<string, unknown> = { volume: 99 };

    const captured: Array<Record<string, unknown>> = [];
    render(
      <SettingsPanel
        currentValues={liveValues}
        onUpdate={(changes) => { captured.push(changes); }}
        onClose={noop}
      />,
    );

    screen.getByText("Apply").click();

    expect(captured[0]["volume"]).toBe(99);
    // logLevel is NOT in liveValues — should remain the default "warn".
    expect(captured[0]["logLevel"]).toBe("warn");
    // gpuLayers is NOT in liveValues — default is 32.
    expect(captured[0]["gpuLayers"]).toBe(32);
  });

  it("does NOT use DEFAULT_SETTINGS when currentValues is fully provided", () => {
    // Provide a volume that differs from the default (72) — verify it sticks.
    const liveValues: Record<string, unknown> = { volume: 10 };

    const captured: Array<Record<string, unknown>> = [];
    render(
      <SettingsPanel
        currentValues={liveValues}
        onUpdate={(changes) => { captured.push(changes); }}
        onClose={noop}
      />,
    );

    screen.getByText("Apply").click();

    // volume should be 10 (from Brain), not 72 (from DEFAULT_SETTINGS).
    expect(captured[0]["volume"]).toBe(10);
  });
});

// ---------------------------------------------------------------------------
// Footer visibility — "Requires restart" must not appear in idle state
// ---------------------------------------------------------------------------

describe("SettingsPanel — footer restart hint is conditional", () => {
  it("shows NO restart hint when updateResult is undefined (idle state)", () => {
    render(<SettingsPanel onUpdate={noop} onClose={noop} />);
    expect(screen.queryByText(/Requires restart/i)).toBeNull();
    expect(screen.queryByText(/Restart required/i)).toBeNull();
  });

  it("shows NO restart hint when updateResult is null (cleared after 4 s)", () => {
    render(<SettingsPanel onUpdate={noop} onClose={noop} updateResult={null} />);
    expect(screen.queryByText(/Requires restart/i)).toBeNull();
    expect(screen.queryByText(/Restart required/i)).toBeNull();
  });

  it("shows restart hint only when updateResult has pending_restart entries", () => {
    render(
      <SettingsPanel
        onUpdate={noop}
        onClose={noop}
        updateResult={{ applied_live: [], pending_restart: ["llm.model"], errors: {} }}
      />,
    );
    expect(screen.getByText(/Restart required for some changes/i)).not.toBeNull();
  });

  it("shows success marker (✓ Applied) when updateResult has no pending_restart and no errors", () => {
    render(
      <SettingsPanel
        onUpdate={noop}
        onClose={noop}
        updateResult={{ applied_live: ["volume"], pending_restart: [], errors: {} }}
      />,
    );
    expect(screen.getByText(/Applied/i)).not.toBeNull();
    expect(screen.queryByText(/Restart required/i)).toBeNull();
  });

  it("shows error message when updateResult.errors is non-empty", () => {
    render(
      <SettingsPanel
        onUpdate={noop}
        onClose={noop}
        updateResult={{
          applied_live: [],
          pending_restart: [],
          errors: { model: "invalid path" },
        }}
      />,
    );
    expect(screen.getByText(/Error:.*invalid path/i)).not.toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Basic render smoke test
// ---------------------------------------------------------------------------

describe("SettingsPanel — basic render", () => {
  it("renders the Settings header", () => {
    render(<SettingsPanel onUpdate={noop} onClose={noop} />);
    expect(screen.getByText("Settings")).not.toBeNull();
  });

  it("renders all seven sidebar tabs", () => {
    render(<SettingsPanel onUpdate={noop} onClose={noop} />);
    for (const tab of ["General", "Voice", "Model", "Context", "Privacy", "Appearance", "Advanced"]) {
      expect(screen.getByText(tab)).not.toBeNull();
    }
  });

  it("calls onClose when the × button is clicked", () => {
    const onClose = vi.fn();
    render(<SettingsPanel onUpdate={noop} onClose={onClose} />);
    screen.getByText("×").click();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("calls onUpdate with persist=false when Apply is clicked", () => {
    const onUpdate = vi.fn();
    render(<SettingsPanel onUpdate={onUpdate} onClose={noop} />);
    screen.getByText("Apply").click();
    expect(onUpdate).toHaveBeenCalledOnce();
    expect(onUpdate.mock.calls[0][1]).toBe(false);
  });

  it("calls onUpdate with persist=true and onClose when Save is clicked", () => {
    const onUpdate = vi.fn();
    const onClose = vi.fn();
    render(<SettingsPanel onUpdate={onUpdate} onClose={onClose} />);
    screen.getByText("Save").click();
    expect(onUpdate).toHaveBeenCalledOnce();
    expect(onUpdate.mock.calls[0][1]).toBe(true);
    expect(onClose).toHaveBeenCalledOnce();
  });
});
