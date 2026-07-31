import { invoke } from "@tauri-apps/api/core";
import type { LumiBrainEvent, OutboundEvent, WireMessage } from "./events";

export type ConnectionState = "connecting" | "connected" | "disconnected";

// Pre-read the IPC bearer token at module load so the first connect() call
// finds it already resolved. Without this, _sendHelloAck() would issue a cold
// Tauri invoke ON the WebSocket critical path, which can exceed the Brain's
// 3 s HANDSHAKE_TIMEOUT_S and trigger a 1008 disconnect before the ack lands.
//
// The promise is kept module-level so all BrainClient instances share it.
// If the token file is missing the promise resolves to null (dev mode / race).
let _cachedTokenPromise: Promise<string | null> = Promise.resolve(
  invoke<string>("read_ipc_token")
).catch(() => null);

/** Public API shared by BrainClient and MockBrainClient. */
export interface IBrainClient {
  readonly state: ConnectionState;
  connect(): void;
  disconnect(): void;
  send(event: OutboundEvent): void;
  onEvent(handler: (e: LumiBrainEvent) => void): () => void;
  onStateChange(handler: (s: ConnectionState) => void): () => void;
}

export const BACKOFF_STEPS_MS = [1000, 2000, 4000, 8000];
const KNOWN_BRAIN_EVENTS = new Set([
  "state_change", "tts_start", "tts_viseme", "tts_stop",
  "transcript", "llm_token", "rag_retrieval", "rag_status",
  "system_status", "error", "config_schema", "config_update_result",
]);

function isLumiBrainEvent(raw: WireMessage): raw is WireMessage & LumiBrainEvent {
  return KNOWN_BRAIN_EVENTS.has(raw.event);
}

const MAX_OUTBOUND_QUEUE = 32;

export class BrainClient implements IBrainClient {
  private ws: WebSocket | null = null;
  private handlers: Array<(e: LumiBrainEvent) => void> = [];
  private stateHandlers: Array<(s: ConnectionState) => void> = [];
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private reconnectAttempt = 0;
  private _state: ConnectionState = "disconnected";
  private _outboundQueue: string[] = [];

  constructor(private readonly url: string = "ws://127.0.0.1:5556") {}

  get state(): ConnectionState {
    return this._state;
  }

  connect(): void {
    this._setState("connecting");
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.reconnectAttempt = 0;
      this._setState("connected");
      // Flush queued messages in order
      const queued = this._outboundQueue.splice(0);
      for (const msg of queued) {
        ws.send(msg);
      }
    };
    ws.onmessage = (ev: MessageEvent<string>) => this._dispatch(ev.data);
    ws.onclose = (ev: CloseEvent) => {
      this._setState("disconnected");
      // Code 1008 = policy violation (auth failure). Do not retry — the token
      // file may be missing or stale; a retry loop would be pointless noise.
      // Re-prime the cached token promise so that if the Brain restarts with a
      // new token, the next manual connect() will send the fresh token.
      if (ev.code === 1008) {
        _cachedTokenPromise = Promise.resolve(
          invoke<string>("read_ipc_token")
        ).catch(() => null);
      } else {
        this._scheduleReconnect();
      }
    };
    ws.onerror = () => {
      this._notifyError("WebSocket error");
      ws.close();
    };
  }

  private _dispatch(data: string): void {
    let raw: unknown;
    try { raw = JSON.parse(data); } catch { return; }

    if (typeof raw !== "object" || raw === null) {
      console.warn("[BrainClient] received non-object frame, discarding");
      return;
    }

    const obj = raw as Record<string, unknown>;

    // Brain's hello frame uses "type" not "event" — intercept it and respond
    // with hello_ack carrying the IPC bearer token for authentication.
    if (obj["type"] === "hello") {
      void this._sendHelloAck();
      return;
    }

    // Runtime validation: require event (string) and payload (object).
    if (typeof obj["event"] !== "string" || typeof obj["payload"] !== "object" || obj["payload"] === null) {
      console.warn("[BrainClient] received frame missing required fields, discarding:", obj["event"]);
      return;
    }

    const wire: WireMessage = {
      event: obj["event"],
      payload: obj["payload"] as Record<string, unknown>,
      timestamp: typeof obj["timestamp"] === "number" ? obj["timestamp"] : 0,
      version: "1.0",
    };

    if (!isLumiBrainEvent(wire)) return;
    const narrowed = wire as unknown as LumiBrainEvent;
    for (const h of this.handlers) h(narrowed);
  }

  private async _sendHelloAck(): Promise<void> {
    // Await the module-level pre-fetched token promise. On first connect the
    // token is usually already resolved; subsequent reconnects always find it
    // resolved. This keeps the Tauri invoke OFF the WS handshake critical path.
    const token = await _cachedTokenPromise;
    const ack: Record<string, unknown> = {
      type: "hello_ack",
      version: "1.0",
      status: "ok",
      ...(token !== null ? { token } : {}),
    };
    if (this.ws !== null && this._state === "connected") {
      this.ws.send(JSON.stringify(ack));
    }
  }

  private _notifyError(message: string): void {
    const errEvent: LumiBrainEvent = { event: "error", payload: { code: "ws_error", message } };
    for (const h of this.handlers) h(errEvent);
  }

  disconnect(): void {
    this._cancelReconnect();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this._setState("disconnected");
  }

  send(event: OutboundEvent): void {
    const wire: WireMessage = {
      event: event.event,
      payload: event.payload as Record<string, unknown>,
      timestamp: Date.now(),
      version: "1.0",
    };
    const serialized = JSON.stringify(wire);
    if (this._state === "connected" && this.ws) {
      this.ws.send(serialized);
    } else {
      if (this._outboundQueue.length < MAX_OUTBOUND_QUEUE) {
        this._outboundQueue.push(serialized);
      }
    }
  }

  onEvent(handler: (e: LumiBrainEvent) => void): () => void {
    this.handlers.push(handler);
    return () => {
      this.handlers = this.handlers.filter((h) => h !== handler);
    };
  }

  onStateChange(handler: (s: ConnectionState) => void): () => void {
    this.stateHandlers.push(handler);
    return () => {
      this.stateHandlers = this.stateHandlers.filter((h) => h !== handler);
    };
  }

  private _setState(next: ConnectionState): void {
    this._state = next;
    for (const h of this.stateHandlers) h(next);
  }

  private _scheduleReconnect(): void {
    const delayMs =
      BACKOFF_STEPS_MS[Math.min(this.reconnectAttempt, BACKOFF_STEPS_MS.length - 1)];
    this.reconnectAttempt += 1;
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, delayMs);
  }

  private _cancelReconnect(): void {
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }
}
