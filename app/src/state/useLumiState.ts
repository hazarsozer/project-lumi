import { useEffect, useState } from "react";
import type { IBrainClient } from "../ipc/client";
import type { BrainState, LumiBrainEvent } from "../ipc/events";

export interface SystemStatus {
  tts_available: boolean;
  rag_available: boolean;
  mic_available: boolean;
  llm_available: boolean;
  setup_required: boolean;
  missing_items: string[];
}

interface LumiState {
  brainState: BrainState;
  transcript: string;
  streamingTokens: string;
  currentUtterance: string;
  systemStatus: SystemStatus | null;
}

const INITIAL_STATE: LumiState = {
  brainState: "idle",
  transcript: "",
  streamingTokens: "",
  currentUtterance: "",
  systemStatus: null,
};

export function useLumiState(client: IBrainClient): LumiState {
  const [lumiState, setLumiState] = useState<LumiState>(INITIAL_STATE);

  useEffect(() => {
    function handleEvent(e: LumiBrainEvent): void {
      switch (e.event) {
        case "state_change":
          setLumiState((prev) => ({ ...prev, brainState: e.payload.state }));
          break;

        case "transcript":
          setLumiState((prev) => ({ ...prev, transcript: e.payload.text }));
          break;

        case "llm_token":
          setLumiState((prev) => ({
            ...prev,
            streamingTokens: prev.streamingTokens + e.payload.token,
          }));
          break;

        case "tts_start":
          setLumiState((prev) => ({
            ...prev,
            currentUtterance: e.payload.text,
            streamingTokens: "",
          }));
          break;

        case "tts_stop":
          setLumiState((prev) => ({ ...prev, currentUtterance: "" }));
          break;

        case "system_status":
          setLumiState((prev) => ({
            ...prev,
            systemStatus: {
              tts_available: e.payload.tts_available,
              rag_available: e.payload.rag_available,
              mic_available: e.payload.mic_available,
              llm_available: e.payload.llm_available,
              setup_required: e.payload.setup_required,
              missing_items: e.payload.missing_items,
            },
          }));
          break;
      }
    }

    const unsub = client.onEvent(handleEvent);
    return unsub;
  }, [client]);

  return lumiState;
}
