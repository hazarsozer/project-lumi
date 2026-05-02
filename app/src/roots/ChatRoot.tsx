import { useCallback, useEffect, useRef, useState } from "react";
import { tauriEmit, tauriListen, tauriGetWindowByLabel } from "../lib/tauriCompat";
import { ChatPanel, type Message } from "../components/ChatPanel";
import type { LumiBrainEvent } from "../ipc/events";
import type { AvatarStateKey } from "../styles/tokens";
import { EV_BRAIN, EV_SEND } from "../ipc/eventNames";

// Safe mapping from BrainState → AvatarStateKey with 'idle' fallback
const VALID_AVATAR_STATES = ["idle", "listening", "processing", "speaking"] as const;
const toAvatarState = (s: string): AvatarStateKey =>
  (VALID_AVATAR_STATES as readonly string[]).includes(s)
    ? (s as AvatarStateKey)
    : "idle";

// ── Chat root — listens via Tauri event bus ───────────────────────────────────
export function ChatRoot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streaming, setStreaming] = useState("");
  const [brainState, setBrainState] = useState<AvatarStateKey>("idle");
  // Pending citations from the most recent rag_retrieval event — attached to
  // the next tts_start message and then cleared.
  const pendingCitations = useRef<string[]>([]);

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
        case "rag_retrieval":
          pendingCitations.current = evt.payload.top_doc_paths;
          break;
        case "tts_start": {
          const citations = pendingCitations.current.length > 0
            ? [...pendingCitations.current]
            : undefined;
          pendingCitations.current = [];
          setMessages((prev) => [
            ...prev,
            { id: crypto.randomUUID(), role: "lumi", text: evt.payload.text, citations },
          ]);
          setStreaming("");
          break;
        }
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
