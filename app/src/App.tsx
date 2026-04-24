import { useCallback, useEffect, useMemo, useState } from "react";
import { getAllWebviewWindows } from "@tauri-apps/api/webviewWindow";
import { emit, listen } from "@tauri-apps/api/event";
import { useBrainSocket } from "./state/useBrainSocket";
import { useLumiState } from "./state/useLumiState";
import { CompactOverlay } from "./components/CompactOverlay";
import { ChatPanel, type Message } from "./components/ChatPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import type { LumiBrainEvent } from "./ipc/events";
import type { AvatarStateKey } from "./styles/tokens";

// ── Window routing ────────────────────────────────────────────────────────────
type WindowKind = "overlay" | "chat" | "settings";

function getWindowKind(): WindowKind {
  const w = new URLSearchParams(window.location.search).get("window");
  if (w === "chat" || w === "settings") return w;
  return "overlay";
}

// ── Tauri event names ────────────────────────────────────────────────────────
const EV_BRAIN = "lumi://brain-event";
const EV_SEND = "lumi://send-text";
const EV_CONFIG_UPDATE = "lumi://config-update";

// ── Overlay root — owns the WS client ─────────────────────────────────────────
function OverlayRoot() {
  const { client } = useBrainSocket();
  const state = useLumiState(client);

  // Re-broadcast every brain event to Chat/Settings windows via Tauri event bus
  useEffect(() => {
    client.onEvent((evt: LumiBrainEvent) => {
      void emit(EV_BRAIN, evt);
    });
  }, [client]);

  // Forward send-text requests from Chat window to Brain
  useEffect(() => {
    const p = listen<{ text: string }>(EV_SEND, (e) => {
      client.send({ event: "user_text", payload: { text: e.payload.text } });
    });
    return () => { void p.then((fn) => fn()); };
  }, [client]);

  // Forward config-update requests from Settings window to Brain
  useEffect(() => {
    const p = listen<{ changes: Record<string, unknown>; persist: boolean }>(
      EV_CONFIG_UPDATE,
      (e) => {
        client.send({ event: "config_update", payload: e.payload });
      },
    );
    return () => { void p.then((fn) => fn()); };
  }, [client]);

  const toggleWindow = useCallback(async (label: "chat" | "settings") => {
    const all = await getAllWebviewWindows();
    const target = all.find((w) => w.label === label);
    if (!target) return;
    if (await target.isVisible()) {
      await target.hide();
    } else {
      await target.show();
      await target.setFocus();
    }
  }, []);

  const avatarState = (state.brainState.toLowerCase() as AvatarStateKey);

  return (
    <CompactOverlay
      brainState={avatarState}
      onSettingsClick={() => { void toggleWindow("settings"); }}
      onChatClick={() => { void toggleWindow("chat"); }}
      onMicClick={() => {
        client.send({ event: "interrupt", payload: {} });
      }}
    />
  );
}

// ── Chat root — listens via Tauri event bus ───────────────────────────────────
function ChatRoot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState("");
  const [brainState, setBrainState] = useState<AvatarStateKey>("idle");

  useEffect(() => {
    const p = listen<LumiBrainEvent>(EV_BRAIN, (e) => {
      const evt = e.payload;
      switch (evt.event) {
        case "state_change":
          setBrainState(evt.payload.state.toLowerCase() as AvatarStateKey);
          break;
        case "transcript":
          setMessages((prev) => [
            ...prev,
            { id: crypto.randomUUID(), role: "user", text: evt.payload.text },
          ]);
          break;
        case "llm_token":
          setStreaming((prev) => prev + evt.payload.token);
          break;
        case "tts_start":
          setMessages((prev) => [
            ...prev,
            { id: crypto.randomUUID(), role: "lumi", text: evt.payload.text },
          ]);
          setStreaming("");
          break;
      }
    });
    return () => { void p.then((fn) => fn()); };
  }, []);

  const handleSend = useCallback((text: string) => {
    void emit(EV_SEND, { text });
  }, []);

  const handleClose = useCallback(async () => {
    const all = await getAllWebviewWindows();
    await all.find((w) => w.label === "chat")?.hide();
  }, []);

  return (
    <ChatPanel
      messages={messages}
      streamingTokens={streaming}
      brainState={brainState}
      onSend={handleSend}
      onClose={handleClose}
    />
  );
}

// ── Settings root ─────────────────────────────────────────────────────────────
function SettingsRoot() {
  const handleUpdate = useCallback(
    (changes: Record<string, unknown>, persist: boolean) => {
      void emit(EV_CONFIG_UPDATE, { changes, persist });
    },
    [],
  );

  const handleClose = useCallback(async () => {
    const all = await getAllWebviewWindows();
    await all.find((w) => w.label === "settings")?.hide();
  }, []);

  return <SettingsPanel onUpdate={handleUpdate} onClose={handleClose} />;
}

// ── Root ──────────────────────────────────────────────────────────────────────
export default function App() {
  const kind = useMemo(getWindowKind, []);
  if (kind === "chat") return <ChatRoot />;
  if (kind === "settings") return <SettingsRoot />;
  return <OverlayRoot />;
}
