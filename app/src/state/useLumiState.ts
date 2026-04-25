import { useEffect, useState } from "react";
import type { BrainClient } from "../ipc/client";
import type { BrainState, LumiBrainEvent } from "../ipc/events";

interface LumiState {
  brainState: BrainState;
  transcript: string;
  streamingTokens: string;
  currentUtterance: string;
}

const INITIAL_STATE: LumiState = {
  brainState: "IDLE",
  transcript: "",
  streamingTokens: "",
  currentUtterance: "",
};

export function useLumiState(client: BrainClient): LumiState {
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
      }
    }

    const unsub = client.onEvent(handleEvent);
    return unsub;
  }, [client]);

  return lumiState;
}
