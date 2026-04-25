import { useEffect, useRef, useState } from "react";
import { BrainClient } from "../ipc/client";
import { MockBrainClient } from "../ipc/mockClient";

const WS_URL = "ws://127.0.0.1:5556";
const IS_MOCK = import.meta.env.VITE_MOCK_WS === "true";

export function useBrainSocket(): {
  client: BrainClient;
  connectionState: string;
} {
  // In mock mode we use MockBrainClient which has an identical public API.
  // The cast to BrainClient is safe because all callers (useLumiState, App.tsx)
  // only access the shared public methods: connect, disconnect, send, onEvent,
  // onStateChange, and the state getter.
  const clientRef = useRef<BrainClient | null>(null);
  const [connectionState, setConnectionState] = useState<string>("disconnected");

  if (clientRef.current === null) {
    clientRef.current = IS_MOCK
      ? (new MockBrainClient() as unknown as BrainClient)
      : new BrainClient(WS_URL);
  }

  useEffect(() => {
    const client = clientRef.current!;
    const unsub = client.onStateChange(setConnectionState);
    client.connect();
    return () => {
      unsub();
      client.disconnect();
    };
  }, []);

  return { client: clientRef.current, connectionState };
}
