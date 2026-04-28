import { useCallback, useEffect, useMemo, useState } from "react";
import { tauriEmit, tauriListen, tauriGetWindowByLabel } from "./lib/tauriCompat";
import { useBrainSocket } from "./state/useBrainSocket";
import { useLumiState } from "./state/useLumiState";
import { CompactOverlay } from "./components/CompactOverlay";
import { ChatPanel, type Message } from "./components/ChatPanel";
import { SettingsPanel } from "./components/SettingsPanel";
import type { LumiBrainEvent } from "./ipc/events";
import type { AvatarStateKey } from "./styles/tokens";

// Safe mapping from any string → AvatarStateKey with 'idle' fallback
const VALID_AVATAR_STATES = ["idle", "listening", "processing", "speaking"] as const;
const toAvatarState = (s: string): AvatarStateKey =>
  (VALID_AVATAR_STATES as readonly string[]).includes(s.toLowerCase())
    ? (s.toLowerCase() as AvatarStateKey)
    : "idle";

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

interface SystemStatus {
  tts_available: boolean;
  rag_available: boolean;
  mic_available: boolean;
  llm_available: boolean;
}

// ── Overlay root — owns the WS client ─────────────────────────────────────────
function OverlayRoot() {
  const { client } = useBrainSocket();
  const state = useLumiState(client);
  const [sysStatus, setSysStatus] = useState<SystemStatus | null>(null);

  // Re-broadcast every brain event to Chat/Settings windows via Tauri event bus
  useEffect(() => {
    const unsub = client.onEvent((evt: LumiBrainEvent) => {
      if (evt.event === "system_status") {
        setSysStatus(evt.payload);
      }
      void tauriEmit(EV_BRAIN, evt);
    });
    return unsub;
  }, [client]);

  // Forward send-text requests from Chat window to Brain
  useEffect(() => {
    let cancelled = false;
    const p = tauriListen<{ text: string }>(EV_SEND, (payload) => {
      client.send({ event: "user_text", payload: { text: payload.text } });
    });
    return () => {
      cancelled = true;
      // If the promise resolves after unmount, skip the unlisten call —
      // the listener was never fully established before the component was torn down.
      void p.then((fn) => { if (cancelled) return; fn(); });
    };
  }, [client]);

  // Forward config-update requests from Settings window to Brain
  useEffect(() => {
    let cancelled = false;
    const p = tauriListen<{ changes: Record<string, unknown>; persist: boolean }>(
      EV_CONFIG_UPDATE,
      (payload) => {
        client.send({ event: "config_update", payload });
      },
    );
    return () => {
      cancelled = true;
      void p.then((fn) => { if (cancelled) return; fn(); });
    };
  }, [client]);

  const toggleWindow = useCallback(async (label: "chat" | "settings") => {
    const target = await tauriGetWindowByLabel(label);
    if (!target) return;
    if (await target.isVisible()) {
      await target.hide();
    } else {
      await target.show();
      await target.setFocus();
    }
  }, []);

  const avatarState = toAvatarState(state.brainState);

  return (
    <CompactOverlay
      brainState={avatarState}
      micAvailable={sysStatus === null || sysStatus.mic_available}
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
    let cancelled = false;
    const p = tauriListen<LumiBrainEvent>(EV_BRAIN, (evt) => {
      switch (evt.event) {
        case "state_change":
          setBrainState(toAvatarState(evt.payload.state));
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
    return () => {
      cancelled = true;
      void p.then((fn) => { if (cancelled) return; fn(); });
    };
  }, []);

  const handleSend = useCallback((text: string) => {
    void tauriEmit(EV_SEND, { text });
  }, []);

  const handleClose = useCallback(async () => {
    const win = await tauriGetWindowByLabel("chat");
    await win?.hide();
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
      void tauriEmit(EV_CONFIG_UPDATE, { changes, persist });
    },
    [],
  );

  const handleClose = useCallback(async () => {
    const win = await tauriGetWindowByLabel("settings");
    await win?.hide();
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
