import type { LumiBrainEvent, OutboundEvent, WireMessage } from "./events";

type ConnectionState = "connecting" | "connected" | "disconnected";

export const BACKOFF_STEPS_MS = [1000, 2000, 4000, 8000];
const KNOWN_BRAIN_EVENTS = new Set([
  "state_change", "tts_start", "tts_viseme", "tts_stop",
  "transcript", "llm_token", "rag_retrieval", "rag_status",
  "error", "config_schema", "config_update_result",
]);

function isLumiBrainEvent(raw: WireMessage): raw is WireMessage & LumiBrainEvent {
  return KNOWN_BRAIN_EVENTS.has(raw.event);
}

const MAX_OUTBOUND_QUEUE = 32;

export class BrainClient {
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
    ws.onclose = () => { this._setState("disconnected"); this._scheduleReconnect(); };
    ws.onerror = () => ws.close();
  }

  private _dispatch(data: string): void {
    let wire: WireMessage;
    try { wire = JSON.parse(data) as WireMessage; } catch { return; }
    if (!isLumiBrainEvent(wire)) return;
    const typed = wire as unknown as LumiBrainEvent;
    for (const h of this.handlers) h(typed);
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
