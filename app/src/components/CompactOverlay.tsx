// CompactOverlay — always-on-top 140px wide overlay with character + button tray
// data-tauri-drag-region is placed on the character area (transparent bg, safe to drag)
import { useState } from 'react';
import { tokens as T, AVATAR_STATES, AvatarStateKey } from '../styles/tokens';
import { LumiAvatar } from './LumiAvatar';

export interface CompactOverlayProps {
  brainState: AvatarStateKey;
  onSettingsClick: () => void;
  onChatClick: () => void;
  onMicClick: () => void;
}

export function CompactOverlay({ brainState, onSettingsClick, onChatClick, onMicClick }: CompactOverlayProps) {
  const st = AVATAR_STATES[brainState];
  const [hoverSettings, setHoverSettings] = useState(false);
  const [hoverChat, setHoverChat] = useState(false);

  const buttons: Array<{
    icon: string;
    hover: boolean;
    setHover: (v: boolean) => void;
    action: () => void;
    tip: string;
  }> = [
    { icon: '⚙', hover: hoverSettings, setHover: setHoverSettings, action: onSettingsClick, tip: 'Settings' },
    { icon: '💬', hover: hoverChat,     setHover: setHoverChat,     action: onChatClick,     tip: 'Chat'     },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', userSelect: 'none', position: 'relative' }}>

      {/* Character portrait — drag region (transparent, no pointer-event conflicts with child buttons) */}
      <div
        data-tauri-drag-region
        style={{ zIndex: 2, position: 'relative' }}
        onClick={onMicClick}
      >
        <LumiAvatar state={brainState} />
      </div>

      {/* Button tray — pill shaped, sits below character with 28px overlap handled by avatar's negative margin */}
      <div style={{
        position: 'relative', zIndex: 1,
        display: 'flex', alignItems: 'center', gap: T.space.sm,
        padding: `${T.space.md}px ${T.space.lg}px`,
        background: 'oklch(14% 0.022 245 / 0.95)',
        borderRadius: T.radius.pill,
        border: '1px solid oklch(30% 0.035 245 / 0.7)',
        boxShadow: `${T.shadow.md}, inset 0 1px 0 oklch(100% 0 0 / 0.05)`,
        backdropFilter: 'blur(18px)',
      }}>
        {/* Status dot */}
        <div style={{
          width: 6, height: 6, borderRadius: '50%',
          background: st.color,
          boxShadow: st.glow,
          opacity: 0.9,
          flexShrink: 0,
        }} />

        <span style={{
          fontSize: T.font.sm, color: T.colors.textSec,
          letterSpacing: '0.02em', marginRight: T.space.sm,
          whiteSpace: 'nowrap',
        }}>
          {st.label}
        </span>

        {/* Divider */}
        <div style={{ width: 1, height: 18, background: T.colors.borderSub, flexShrink: 0 }} />

        {/* Action buttons */}
        {buttons.map(({ icon, hover, setHover, action, tip }) => (
          <div
            key={icon}
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
            onClick={(e) => { e.stopPropagation(); action(); }}
            title={tip}
            style={{
              width: 34, height: 34, borderRadius: T.radius.md,
              background: hover ? T.colors.surfaceTop : T.colors.surfaceUp,
              border: `1px solid ${hover ? T.colors.border : T.colors.borderSub}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer',
              transition: 'background 0.15s, border-color 0.15s',
              fontSize: 15,
            }}
          >
            {icon}
          </div>
        ))}
      </div>
    </div>
  );
}
