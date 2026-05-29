/**
 * ChatPanel.test.tsx
 *
 * Covers the fix from CR-16 / Issue #17:
 *   The status indicator in ChatPanel must reflect the real Brain connection
 *   state (connected / connecting / disconnected) instead of a hardcoded
 *   "● Online" string.
 */
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ChatPanel } from "./ChatPanel";

const noop = () => {};

const BASE_PROPS = {
  messages: [],
  brainState: "idle" as const,
  onSend: noop,
  onClose: noop,
};

// ---------------------------------------------------------------------------
// Status indicator reflects connection state
// ---------------------------------------------------------------------------

describe("ChatPanel — status indicator reflects connection state (CR-16 / Issue #17)", () => {
  it("shows '● Online' when connectionState is connected", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="connected" />);
    expect(screen.getByText("● Online")).not.toBeNull();
  });

  it("shows '○ Offline' when connectionState is disconnected", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="disconnected" />);
    expect(screen.getByText("○ Offline")).not.toBeNull();
    expect(screen.queryByText("● Online")).toBeNull();
  });

  it("shows '◌ Connecting' when connectionState is connecting", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="connecting" />);
    expect(screen.getByText("◌ Connecting")).not.toBeNull();
    expect(screen.queryByText("● Online")).toBeNull();
  });

  it("does NOT show Online when disconnected (regression guard)", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="disconnected" />);
    expect(screen.queryByText("● Online")).toBeNull();
  });

  it("does NOT show Online when connecting (regression guard)", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="connecting" />);
    expect(screen.queryByText("● Online")).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Basic render smoke tests
// ---------------------------------------------------------------------------

describe("ChatPanel — basic render", () => {
  it("renders the Lumi heading", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="connected" />);
    expect(screen.getByText("Lumi")).not.toBeNull();
  });

  it("calls onClose when the × button is clicked", () => {
    const onClose = vi.fn();
    render(<ChatPanel {...BASE_PROPS} connectionState="connected" onClose={onClose} />);
    screen.getByText("×").click();
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("renders with an empty message list without crashing", () => {
    render(<ChatPanel {...BASE_PROPS} connectionState="connected" messages={[]} />);
    expect(screen.getByText("Lumi")).not.toBeNull();
  });

  it("renders provided messages in the chat list", () => {
    const messages = [
      { id: "1", role: "user" as const, text: "Hello Lumi" },
      { id: "2", role: "lumi" as const, text: "Hi there!" },
    ];
    render(<ChatPanel {...BASE_PROPS} connectionState="connected" messages={messages} />);
    expect(screen.getByText("Hello Lumi")).not.toBeNull();
    expect(screen.getByText("Hi there!")).not.toBeNull();
  });
});
