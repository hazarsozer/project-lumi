export interface WireMessage {
  event: string;
  payload: Record<string, unknown>;
  timestamp: number;
  version: "1.0";
}

// Brain → React

export type BrainState = "IDLE" | "LISTENING" | "PROCESSING" | "SPEAKING";

export type LumiBrainEvent =
  | { event: "state_change"; payload: { state: BrainState } }
  | { event: "tts_start"; payload: { text: string; duration_ms: number } }
  | { event: "tts_viseme"; payload: { viseme: string; duration_ms: number } }
  | { event: "tts_stop"; payload: Record<string, never> }
  | { event: "transcript"; payload: { text: string } }
  | { event: "llm_token"; payload: { token: string; utterance_id: string } }
  | {
      event: "rag_retrieval";
      payload: {
        query: string;
        hit_count: number;
        latency_ms: number;
        top_doc_paths: string[];
      };
    }
  | {
      event: "rag_status";
      payload: {
        enabled: boolean;
        doc_count: number;
        chunk_count: number;
        last_indexed: string;
      };
    }
  | { event: "error"; payload: { code: string; message: string } }
  | {
      event: "config_schema";
      payload: {
        fields: Record<string, unknown>;
        current_values: Record<string, unknown>;
      };
    }
  | {
      event: "config_update_result";
      payload: {
        applied_live: string[];
        pending_restart: string[];
        errors: Record<string, string>;
      };
    };

// React → Brain

export type OutboundEvent =
  | { event: "interrupt"; payload: Record<string, never> }
  | { event: "user_text"; payload: { text: string } }
  | { event: "rag_set_enabled"; payload: { enabled: boolean } }
  | { event: "config_schema_request"; payload: Record<string, never> }
  | {
      event: "config_update";
      payload: { changes: Record<string, unknown>; persist: boolean };
    };
