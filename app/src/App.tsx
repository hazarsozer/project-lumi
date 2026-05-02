import { useMemo } from "react";
import { OverlayRoot } from "./roots/OverlayRoot";
import { ChatRoot } from "./roots/ChatRoot";
import { SettingsRoot } from "./roots/SettingsRoot";

// ── Window routing ────────────────────────────────────────────────────────────
type WindowKind = "overlay" | "chat" | "settings";

function getWindowKind(): WindowKind {
  const w = new URLSearchParams(window.location.search).get("window");
  if (w === "chat" || w === "settings") return w;
  return "overlay";
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const kind = useMemo(getWindowKind, []);
  if (kind === "chat") return <ChatRoot />;
  if (kind === "settings") return <SettingsRoot />;
  return <OverlayRoot />;
}
