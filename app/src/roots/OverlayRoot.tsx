import { useCallback, useEffect, useState } from "react";
import { tauriEmit, tauriListen, tauriGetWindowByLabel } from "../lib/tauriCompat";
import { useBrainSocket } from "../state/useBrainSocket";
import { useLumiState } from "../state/useLumiState";
import { CompactOverlay } from "../components/CompactOverlay";
import { SetupPanel } from "../components/SetupPanel";
import type { LumiBrainEvent } from "../ipc/events";
import type { AvatarStateKey } from "../styles/tokens";
import { EV_BRAIN, EV_SEND, EV_CONFIG_UPDATE, EV_CONFIG_SCHEMA_REQUEST } from "../ipc/eventNames";

// Safe mapping from BrainState → AvatarStateKey with 'idle' fallback
const VALID_AVATAR_STATES = ["idle", "listening", "processing", "speaking"] as const;
const toAvatarState = (s: string): AvatarStateKey =>
  (VALID_AVATAR_STATES as readonly string[]).includes(s)
    ? (s as AvatarStateKey)
    : "idle";

// ── Overlay root — owns the WS client ─────────────────────────────────────────
export function OverlayRoot() {
  const { client } = useBrainSocket();
  const state = useLumiState(client);
  const { systemStatus } = state;
  const [setupDismissed, setSetupDismissed] = useState(false);

  // Re-broadcast every brain event to Chat/Settings windows via Tauri event bus
  useEffect(() => {
    const unsub = client.onEvent((evt: LumiBrainEvent) => {
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
      void p.then((fn) => { if (cancelled) return; fn(); });
    };
  }, [client]);

  // Forward config-schema-request from Settings window to Brain
  useEffect(() => {
    let cancelled = false;
    const p = tauriListen<Record<string, never>>(EV_CONFIG_SCHEMA_REQUEST, () => {
      client.send({ event: "config_schema_request", payload: {} });
    });
    return () => {
      cancelled = true;
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
  const showSetup =
    !setupDismissed &&
    systemStatus?.setup_required === true &&
    (systemStatus?.missing_items?.length ?? 0) > 0;

  return (
    <>
      {showSetup && (
        <SetupPanel
          missingItems={systemStatus!.missing_items}
          onDismiss={() => setSetupDismissed(true)}
        />
      )}
      <CompactOverlay
        brainState={avatarState}
        micAvailable={systemStatus === null || (systemStatus?.mic_available ?? true)}
        onSettingsClick={() => { void toggleWindow("settings"); }}
        onChatClick={() => { void toggleWindow("chat"); }}
        onMicClick={() => {
          client.send({ event: "interrupt", payload: {} });
        }}
      />
    </>
  );
}
