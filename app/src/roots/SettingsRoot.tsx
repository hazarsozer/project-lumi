import { useCallback, useEffect, useState } from "react";
import { tauriEmit, tauriListen, tauriGetWindowByLabel } from "../lib/tauriCompat";
import { SettingsPanel } from "../components/SettingsPanel";
import { EV_BRAIN, EV_CONFIG_UPDATE, EV_CONFIG_SCHEMA_REQUEST } from "../ipc/eventNames";
import type { LumiBrainEvent } from "../ipc/events";

interface ConfigUpdateResult {
  applied_live: string[];
  pending_restart: string[];
  errors: Record<string, string>;
}

// ── Settings root ─────────────────────────────────────────────────────────────
export function SettingsRoot() {
  const [configSchema, setConfigSchema] = useState<Record<string, unknown> | undefined>();
  const [currentValues, setCurrentValues] = useState<Record<string, unknown> | undefined>();
  const [updateResult, setUpdateResult] = useState<ConfigUpdateResult | null>(null);

  // Request config schema from Brain on mount, and listen for schema + update results.
  useEffect(() => {
    let cancelled = false;

    // Ask Brain to send the current config schema.
    void tauriEmit(EV_CONFIG_SCHEMA_REQUEST, {});

    const p = tauriListen<LumiBrainEvent>(EV_BRAIN, (evt) => {
      if (evt.event === "config_schema") {
        setConfigSchema(evt.payload.fields);
        setCurrentValues(evt.payload.current_values);
      }
      if (evt.event === "config_update_result") {
        setUpdateResult(evt.payload as ConfigUpdateResult);
        // Clear feedback after 4 s
        setTimeout(() => { if (!cancelled) setUpdateResult(null); }, 4000);
      }
    });

    return () => {
      cancelled = true;
      void p.then((fn) => { if (cancelled) return; fn(); });
    };
  }, []);

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

  return (
    <SettingsPanel
      configSchema={configSchema}
      currentValues={currentValues}
      updateResult={updateResult}
      onUpdate={handleUpdate}
      onClose={handleClose}
    />
  );
}
