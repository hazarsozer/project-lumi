// ChatPanel — 380×540 floating chat window
import { useState, useRef, useEffect } from 'react';
import { tokens as T, AvatarStateKey } from '../styles/tokens';

export interface Message {
  id: string;
  role: 'user' | 'lumi';
  text: string;
  timestamp?: string;
  citations?: string[];
}

export interface ChatPanelProps {
  messages: Message[];
  streamingTokens?: string;
  brainState: AvatarStateKey;
  onSend: (text: string) => void;
  onClose: () => void;
}

export function ChatPanel({ messages, streamingTokens, brainState, onSend, onClose }: ChatPanelProps) {
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<HTMLDivElement>(null);
  const isProcessing = brainState === 'processing' || brainState === 'speaking';

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, streamingTokens]);

  function handleSend() {
    const text = draft.trim();
    if (!text) return;
    onSend(text);
    setDraft('');
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  return (
    <div style={{
      width: 380, height: 540, display: 'flex', flexDirection: 'column',
      background: T.colors.surface,
      borderRadius: T.radius.xl, border: `1px solid ${T.colors.border}`,
      boxShadow: T.shadow.lg, overflow: 'hidden',
      animation: 'fadeIn 280ms ease',
    }}>
      <ChatHeader onClose={onClose} />

      {/* Message list */}
      <div ref={scrollRef} style={{ flex: 1, overflowY: 'auto', padding: T.space.lg, display: 'flex', flexDirection: 'column', gap: T.space.lg }}>
        {messages.map((m) => <ChatBubble key={m.id} message={m} />)}

        {/* Streaming token display */}
        {streamingTokens && brainState === 'processing' && (
          <StreamingBubble text={streamingTokens} />
        )}

        {/* Typing indicator when processing with no tokens yet */}
        {isProcessing && !streamingTokens && <TypingIndicator />}
      </div>

      <ChatInputBar
        draft={draft}
        onChange={setDraft}
        onKeyDown={handleKeyDown}
        onSend={handleSend}
      />
    </div>
  );
}

// ── Sub-components ──────────────────────────────────────────

function ChatHeader({ onClose }: { onClose: () => void }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: `${T.space.md}px ${T.space.lg}px`, borderBottom: `1px solid ${T.colors.borderSub}`, flexShrink: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div style={{ width: 28, height: 28, borderRadius: '50%', background: T.colors.surfaceTop, border: `1.5px solid ${T.colors.accentBlue}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, boxShadow: T.shadow.glowBlue }}>✦</div>
        <div>
          <div style={{ fontSize: T.font.lg, fontWeight: 600, color: T.colors.textPri, lineHeight: 1 }}>Lumi</div>
          <div style={{ fontSize: T.font.xs, color: T.colors.accentGreen, marginTop: 2 }}>● Online</div>
        </div>
      </div>
      <div onClick={onClose} style={{ width: 26, height: 26, borderRadius: T.radius.md, background: T.colors.surfaceTop, border: `1px solid ${T.colors.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 12, color: T.colors.textSec }}>×</div>
    </div>
  );
}

function ChatBubble({ message }: { message: Message }) {
  const isLumi = message.role === 'lumi';
  return (
    <div style={{ display: 'flex', flexDirection: isLumi ? 'row' : 'row-reverse', gap: 10, alignItems: 'flex-end' }}>
      {isLumi && (
        <div style={{ width: 28, height: 28, borderRadius: '50%', background: T.colors.surfaceTop, border: `1px solid ${T.colors.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 12 }}>✦</div>
      )}
      <div style={{ maxWidth: '80%' }}>
        <div style={{
          padding: `${T.space.sm}px ${T.space.md}px`,
          borderRadius: isLumi ? `${T.radius.lg}px ${T.radius.lg}px ${T.radius.lg}px 4px` : `${T.radius.lg}px ${T.radius.lg}px 4px ${T.radius.lg}px`,
          background: isLumi ? T.colors.surfaceUp : 'oklch(62% 0.18 222 / 0.18)',
          border: `1px solid ${isLumi ? T.colors.borderSub : 'oklch(62% 0.18 222 / 0.3)'}`,
          fontSize: T.font.md, color: T.colors.textPri, lineHeight: 1.55,
        }}>
          {message.text}
          {message.citations && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5, marginTop: 8 }}>
              {message.citations.map((c) => (
                <span key={c} style={{ fontSize: T.font.xs, color: T.colors.accentBlue, background: 'oklch(62% 0.18 222 / 0.12)', border: '1px solid oklch(62% 0.18 222 / 0.25)', padding: '2px 7px', borderRadius: T.radius.pill }}>{c}</span>
              ))}
            </div>
          )}
        </div>
        {message.timestamp && (
          <div style={{ fontSize: T.font.xs, color: T.colors.textMuted, marginTop: 3, textAlign: isLumi ? 'left' : 'right', paddingLeft: isLumi ? 2 : 0, paddingRight: isLumi ? 0 : 2 }}>{message.timestamp}</div>
        )}
      </div>
    </div>
  );
}

function StreamingBubble({ text }: { text: string }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'row', gap: 10, alignItems: 'flex-end' }}>
      <div style={{ width: 28, height: 28, borderRadius: '50%', background: T.colors.surfaceTop, border: `1px solid ${T.colors.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, fontSize: 12 }}>✦</div>
      <div style={{ maxWidth: '80%', padding: `${T.space.sm}px ${T.space.md}px`, borderRadius: `${T.radius.lg}px ${T.radius.lg}px ${T.radius.lg}px 4px`, background: T.colors.surfaceUp, border: `1px solid ${T.colors.borderSub}`, fontSize: T.font.md, color: T.colors.textPri, lineHeight: 1.55 }}>
        {text}
        <span style={{ display: 'inline-block', width: 2, height: '1em', background: T.colors.accentBlue, marginLeft: 2, verticalAlign: 'text-bottom', animation: 'lumiDot 0.8s ease-in-out infinite' }} />
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 10 }}>
      <div style={{ width: 28, height: 28, borderRadius: '50%', background: T.colors.surfaceTop, border: `1px solid ${T.colors.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12 }}>✦</div>
      <div style={{ padding: `${T.space.sm}px ${T.space.md}px`, borderRadius: `${T.radius.lg}px ${T.radius.lg}px ${T.radius.lg}px 4px`, background: T.colors.surfaceUp, border: `1px solid ${T.colors.borderSub}`, display: 'flex', gap: 5, alignItems: 'center' }}>
        {[0, 1, 2].map((i) => (
          <div key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: T.colors.textMuted, animation: `lumiDot 1.2s ${i * 0.2}s ease-in-out infinite` }} />
        ))}
      </div>
    </div>
  );
}

interface ChatInputBarProps {
  draft: string;
  onChange: (v: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLInputElement>) => void;
  onSend: () => void;
}

function ChatInputBar({ draft, onChange, onKeyDown, onSend }: ChatInputBarProps) {
  return (
    <div style={{ padding: T.space.md, borderTop: `1px solid ${T.colors.borderSub}`, flexShrink: 0 }}>
      <div style={{ display: 'flex', gap: T.space.sm, alignItems: 'center', background: T.colors.surfaceTop, borderRadius: T.radius.lg, border: `1px solid ${T.colors.border}`, padding: `${T.space.sm}px ${T.space.md}px` }}>
        <input
          value={draft}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask Lumi something…"
          style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', fontSize: T.font.md, color: T.colors.textSec, fontFamily: 'inherit' }}
        />
        <div style={{ display: 'flex', gap: T.space.sm, flexShrink: 0 }}>
          <div style={{ width: 28, height: 28, borderRadius: T.radius.md, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 14, color: T.colors.textMuted }}>🎙</div>
          <div onClick={onSend} style={{ width: 28, height: 28, borderRadius: T.radius.md, background: T.colors.accentBlue, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 12 }}>
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
              <path d="M2 6h8M7 3l3 3-3 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        </div>
      </div>
    </div>
  );
}
