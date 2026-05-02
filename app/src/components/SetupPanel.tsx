/**
 * First-run setup guidance panel — shown when the Brain reports missing
 * model files or packages.
 *
 * Provides a summary of what is missing and copy-pasteable shell commands
 * for each item.  The user can dismiss to run in degraded mode, or follow
 * the instructions and restart.
 *
 * Note: this is a static guidance panel — it does not perform downloads or
 * test the microphone directly.  Those features are planned for a future
 * release once the Brain sidecar is bundled into the installer.
 */

import { useState } from "react";

// ── Download reference ────────────────────────────────────────────────────────

const DOWNLOAD_STEPS = `# 1. LLM model (~2.4 GB — Phi-3.5-mini Q4_K_M)
huggingface-cli download bartowski/Phi-3.5-mini-instruct-GGUF \\
  Phi-3.5-mini-instruct-Q4_K_M.gguf --local-dir models/llm/

# 2. TTS model (~80 MB — Kokoro v1.0)
#    Download kokoro-v1_0.onnx and voices.bin from:
#    https://github.com/thewh1teagle/kokoro-onnx/releases
#    and place them in models/tts/

# 3. Restart Lumi after downloading.`;

// ── Component ─────────────────────────────────────────────────────────────────

interface Props {
  missingItems: string[];
  onDismiss: () => void;
}

export function SetupPanel({ missingItems, onDismiss }: Props) {
  const [copied, setCopied] = useState(false);

  function copyCommands() {
    void navigator.clipboard.writeText(DOWNLOAD_STEPS).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div style={styles.overlay}>
      <div style={styles.panel}>
        <h2 style={styles.title}>Lumi Setup Required</h2>

        <p style={styles.desc}>
          The following model files were not found. Lumi will run in degraded
          mode until they are downloaded.
        </p>

        <ul style={styles.list}>
          {missingItems.map((item, i) => (
            <li key={i} style={styles.listItem}>
              <code style={styles.code}>{item}</code>
            </li>
          ))}
        </ul>

        <details style={styles.details}>
          <summary style={styles.summary}>Download commands</summary>
          <pre style={styles.pre}>{DOWNLOAD_STEPS}</pre>
          <button onClick={copyCommands} style={styles.copyBtn}>
            {copied ? "Copied!" : "Copy to clipboard"}
          </button>
        </details>

        <div style={styles.actions}>
          <button onClick={onDismiss} style={styles.dismissBtn}>
            Continue in degraded mode
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  overlay: {
    position: "fixed" as const,
    inset: 0,
    background: "rgba(0, 0, 0, 0.85)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
  },
  panel: {
    background: "#1a1a2e",
    border: "1px solid #4a4a6a",
    borderRadius: 12,
    padding: "2rem",
    maxWidth: 560,
    width: "90vw",
    color: "#e0e0e0",
    fontFamily: "system-ui, sans-serif",
  },
  title: {
    margin: "0 0 0.75rem",
    fontSize: "1.25rem",
    color: "#ff9f43",
  },
  desc: {
    margin: "0 0 1rem",
    lineHeight: 1.5,
    fontSize: "0.9rem",
    color: "#b0b0c0",
  },
  list: {
    margin: "0 0 1rem",
    paddingLeft: "1.25rem",
    display: "flex",
    flexDirection: "column" as const,
    gap: "0.5rem",
  },
  listItem: {
    fontSize: "0.8rem",
  },
  code: {
    background: "#0d0d1a",
    padding: "0.25rem 0.5rem",
    borderRadius: 4,
    fontFamily: "monospace",
    whiteSpace: "pre-wrap" as const,
    wordBreak: "break-all" as const,
    color: "#a8e6cf",
  },
  details: {
    margin: "0 0 1.5rem",
  },
  summary: {
    cursor: "pointer",
    color: "#a0a0c0",
    fontSize: "0.85rem",
    marginBottom: "0.5rem",
  },
  pre: {
    background: "#0d0d1a",
    padding: "0.75rem",
    borderRadius: 6,
    fontSize: "0.75rem",
    overflowX: "auto" as const,
    color: "#c0ffc0",
    marginBottom: "0.5rem",
  },
  copyBtn: {
    background: "transparent",
    border: "1px solid #4a4a6a",
    color: "#a0a0c0",
    borderRadius: 6,
    padding: "0.25rem 0.75rem",
    cursor: "pointer",
    fontSize: "0.8rem",
  },
  actions: {
    display: "flex",
    justifyContent: "flex-end",
  },
  dismissBtn: {
    background: "#2a2a4a",
    border: "1px solid #4a4a6a",
    color: "#c0c0d8",
    borderRadius: 8,
    padding: "0.5rem 1.25rem",
    cursor: "pointer",
    fontSize: "0.9rem",
  },
} as const;
