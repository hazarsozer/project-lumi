import { useEffect, useRef, useState } from "react";
import { BrainClient, type ConnectionState, type IBrainClient } from "../ipc/client";
import { MockBrainClient } from "../ipc/mockClient";

const WS_URL = "ws://127.0.0.1:5556";
const IS_MOCK = import.meta.env.VITE_MOCK_WS === "true";

export function useBrainSocket(): {
  client: IBrainClient;
  connectionState: ConnectionState;
} {
  const clientRef = useRef<IBrainClient | null>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");

  if (clientRef.current === null) {
    clientRef.current = IS_MOCK
      ? new MockBrainClient()
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
