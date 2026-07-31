/**
 * CompactOverlay.test.tsx
 *
 * Covers the fix from CR-19 / Issue #20:
 *   The click-to-interrupt handler (onMicClick) must fire reliably.
 *   The clickable mic element must NOT carry data-tauri-drag-region so the
 *   Tauri drag region does not swallow click events.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { CompactOverlay } from "./CompactOverlay";

// ---------------------------------------------------------------------------
// Mock LumiAvatar — it imports PNG assets that jsdom cannot handle.
// ---------------------------------------------------------------------------
vi.mock("./LumiAvatar", () => ({
  LumiAvatar: () => <div data-testid="lumi-avatar" />,
}));

// ---------------------------------------------------------------------------

const noop = () => {};

const BASE_PROPS = {
  brainState: "idle" as const,
  micAvailable: true,
  onSettingsClick: noop,
  onChatClick: noop,
  onMicClick: noop,
};

// ---------------------------------------------------------------------------
// CR-19 / Issue #20 — drag region separated from click-to-interrupt
// ---------------------------------------------------------------------------

describe("CompactOverlay — drag region separated from click-to-interrupt (CR-19 / Issue #20)", () => {
  it("calls onMicClick when the mic control is clicked", () => {
    const onMicClick = vi.fn();
    render(<CompactOverlay {...BASE_PROPS} onMicClick={onMicClick} />);

    const micButton = screen.getByRole("button", { name: /interrupt lumi/i });
    micButton.click();

    expect(onMicClick).toHaveBeenCalledOnce();
  });

  it("the clickable mic element does NOT have data-tauri-drag-region", () => {
    render(<CompactOverlay {...BASE_PROPS} />);

    const micButton = screen.getByRole("button", { name: /interrupt lumi/i });
    expect(micButton.hasAttribute("data-tauri-drag-region")).toBe(false);
  });

  it("the drag region element exists separately from the clickable mic element", () => {
    const { container } = render(<CompactOverlay {...BASE_PROPS} />);

    const dragRegion = container.querySelector("[data-tauri-drag-region]");
    const micButton = screen.getByRole("button", { name: /interrupt lumi/i });

    expect(dragRegion).not.toBeNull();
    // The drag region element itself is NOT the mic button
    expect(dragRegion).not.toBe(micButton);
    // The mic button is a descendant of the drag region (nested, not the same element)
    expect(dragRegion!.contains(micButton)).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Basic render smoke tests
// ---------------------------------------------------------------------------

describe("CompactOverlay — basic render", () => {
  it("renders without crashing", () => {
    render(<CompactOverlay {...BASE_PROPS} />);
    expect(screen.getByTestId("lumi-avatar")).not.toBeNull();
  });

  it("calls onSettingsClick when settings button is clicked", () => {
    const onSettingsClick = vi.fn();
    render(<CompactOverlay {...BASE_PROPS} onSettingsClick={onSettingsClick} />);

    screen.getByTitle("Settings").click();
    expect(onSettingsClick).toHaveBeenCalledOnce();
  });

  it("calls onChatClick when chat button is clicked", () => {
    const onChatClick = vi.fn();
    render(<CompactOverlay {...BASE_PROPS} onChatClick={onChatClick} />);

    screen.getByTitle("Chat").click();
    expect(onChatClick).toHaveBeenCalledOnce();
  });

  it("shows amber status dot when mic is unavailable", () => {
    const { container } = render(
      <CompactOverlay {...BASE_PROPS} micAvailable={false} />
    );
    // The status dot has a title when mic is unavailable
    const dot = container.querySelector('[title="Microphone unavailable"]');
    expect(dot).not.toBeNull();
  });
});
