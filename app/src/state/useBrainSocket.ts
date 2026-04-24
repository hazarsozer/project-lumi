import { useEffect, useRef, useState } from "react";
import { BrainClient } from "../ipc/client";

const WS_URL = "ws://127.0.0.1:5556";

export function useBrainSocket(): {
  client: BrainClient;
  connectionState: string;
} {
  const clientRef = useRef<BrainClient | null>(null);
  const [connectionState, setConnectionState] = useState<string>("disconnected");

  if (clientRef.current === null) {
    clientRef.current = new BrainClient(WS_URL);
  }

  useEffect(() => {
    const client = clientRef.current!;
    client.onStateChange(setConnectionState);
    client.connect();
    return () => {
      client.disconnect();
    };
  }, []);

  return { client: clientRef.current, connectionState };
}
