import type { LumiBrainEvent, OutboundEvent } from "./events";
import type { ConnectionState, IBrainClient } from "./client";

// Scripted conversation token sequence for streaming simulation
const STREAM_TOKENS = [
  "The", " weather", " today", " is", " sunny", " and", " 22", "°C", "."
];

/**
 * MockBrainClient — drop-in replacement for BrainClient when VITE_MOCK_WS=true.
 *
 * Matches the public API of BrainClient exactly so it can be used wherever
 * BrainClient is expected. Runs a scripted event loop with setTimeout/setInterval
 * — no real WebSocket involved.
 */
export class MockBrainClient implements IBrainClient {
  private handlers: Array<(e: LumiBrainEvent) => void> = [];
  private stateHandlers: Array<(s: ConnectionState) => void> = [];
  private _state: ConnectionState = "disconnected";
  private timers: ReturnType<typeof setTimeout>[] = [];
  private intervals: ReturnType<typeof setInterval>[] = [];

  get state(): ConnectionState {
    return this._state;
  }

  connect(): void {
    this._setState("connecting");
    // Simulate async connection handshake
    this._schedule(() => {
      this._setState("connected");
      this._startCycle();
    }, 300);
  }

  disconnect(): void {
    for (const t of this.timers) clearTimeout(t);
    this.timers = [];
    for (const id of this.intervals) clearInterval(id);
    this.intervals = [];
    this._setState("disconnected");
  }

  send(_event: OutboundEvent): void {
    // No-op in mock mode — outbound events are silently dropped
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

  // ── private helpers ──────────────────────────────────────────────────────────

  private _emit(event: LumiBrainEvent): void {
    for (const h of this.handlers) h(event);
  }

  private _setState(next: ConnectionState): void {
    this._state = next;
    for (const h of this.stateHandlers) h(next);
  }

  private _schedule(fn: () => void, delayMs: number): void {
    this.timers.push(setTimeout(fn, delayMs));
  }

  /**
   * Scripted conversation cycle:
   *
   *  0 ms  — state_change IDLE
   *  2 s   — state_change LISTENING
   *  4 s   — state_change PROCESSING + transcript
   *  5 s   — llm_token stream (one token every 120 ms)
   *  8 s   — state_change SPEAKING + tts_start
   * 10 s   — tts_stop + state_change IDLE
   * 11 s   — repeat
   */
  private _startCycle(): void {
    const CYCLE_MS = 11_000;

    const runCycle = () => {
      // 0 ms — IDLE
      this._emit({ event: "state_change", payload: { state: "IDLE" } });

      // 2 s — LISTENING
      this._schedule(() => {
        this._emit({ event: "state_change", payload: { state: "LISTENING" } });
      }, 2_000);

      // 4 s — PROCESSING + user transcript
      this._schedule(() => {
        this._emit({ event: "state_change", payload: { state: "PROCESSING" } });
        this._emit({
          event: "transcript",
          payload: { text: "What is the weather today?" },
        });
      }, 4_000);

      // 5 s — stream LLM tokens, one per 120 ms
      STREAM_TOKENS.forEach((token, i) => {
        this._schedule(() => {
          this._emit({
            event: "llm_token",
            payload: { token, utterance_id: "mock-utt-1" },
          });
        }, 5_000 + i * 120);
      });

      // 8 s — SPEAKING + tts_start
      this._schedule(() => {
        this._emit({ event: "state_change", payload: { state: "SPEAKING" } });
        this._emit({
          event: "tts_start",
          payload: {
            text: "The weather today is sunny and 22°C.",
            duration_ms: 2_000,
          },
        });
      }, 8_000);

      // 10 s — tts_stop + IDLE
      this._schedule(() => {
        this._emit({ event: "tts_stop", payload: {} });
        this._emit({ event: "state_change", payload: { state: "IDLE" } });
      }, 10_000);
    };

    // Run immediately, then repeat every CYCLE_MS
    runCycle();
    this.intervals.push(setInterval(runCycle, CYCLE_MS));
  }
}
