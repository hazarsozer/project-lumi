// Lumi character avatar — SVG placeholder with state-driven glow + animation
// Replace the SVG placeholder by dropping a transparent PNG/WebP asset.
import { useState, useEffect } from 'react';
import { AVATAR_STATES, AvatarStateKey } from '../styles/tokens';

interface LumiAvatarProps {
  state?: AvatarStateKey;
}

export function LumiAvatar({ state = 'idle' }: LumiAvatarProps) {
  const st = AVATAR_STATES[state];
  const isActive = state !== 'idle';
  const [pulse, setPulse] = useState(false);

  useEffect(() => {
    if (state !== 'idle') return;
    const id = setInterval(() => setPulse((p) => !p), 2800);
    return () => clearInterval(id);
  }, [state]);

  return (
    <div style={{
      width: 140, height: 210,
      opacity: isActive ? 1 : st.opacity,
      transition: 'opacity 0.5s, filter 0.5s',
      filter: isActive
        ? `drop-shadow(0 0 14px ${st.color})`
        : `drop-shadow(0 0 6px ${st.color}80)`,
      marginBottom: -28,
      zIndex: 2,
      position: 'relative',
      animation: state === 'listening' ? 'lumiFloat 1.4s ease-in-out infinite' : 'none',
    }}>
      <CharacterPlaceholder state={state} st={st} isActive={isActive} pulse={pulse} />
    </div>
  );
}

// ── SVG character silhouette ────────────────────────────────
// Replace with: <img src={characterAsset} style={{ width:'100%', height:'100%' }} />

interface PlaceholderProps {
  state: AvatarStateKey;
  st: typeof AVATAR_STATES[AvatarStateKey];
  isActive: boolean;
  pulse: boolean;
}

function CharacterPlaceholder({ state, st, isActive, pulse }: PlaceholderProps) {
  const hue = state === 'idle' ? 222 : state === 'listening' ? 152 : state === 'processing' ? 65 : 240;

  return (
    <svg viewBox="0 0 120 200" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ width: '100%', height: '100%' }}>
      <defs>
        <pattern id="charStripe" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke={st.color} strokeWidth="0.5" strokeOpacity="0.12" />
        </pattern>
        <radialGradient id="charFade" cx="50%" cy="80%" r="60%">
          <stop offset="0%" stopColor={st.color} stopOpacity="0.08" />
          <stop offset="100%" stopColor={st.color} stopOpacity="0" />
        </radialGradient>
        <radialGradient id="groundGlow" cx="50%" cy="100%" r="50%">
          <stop offset="0%" stopColor={st.color} stopOpacity="0.35" />
          <stop offset="100%" stopColor={st.color} stopOpacity="0" />
        </radialGradient>
      </defs>

      <ellipse cx="60" cy="196" rx="44" ry="8" fill="url(#groundGlow)" />
      <rect x="0" y="0" width="120" height="200" fill="url(#charStripe)" rx="8" />
      <rect x="0" y="0" width="120" height="200" fill="url(#charFade)" rx="8" />

      {/* Legs */}
      <rect x="38" y="138" width="18" height="52" rx="8" fill={st.color} fillOpacity="0.12" stroke={st.color} strokeOpacity="0.2" strokeWidth="0.8" />
      <rect x="64" y="138" width="18" height="52" rx="8" fill={st.color} fillOpacity="0.12" stroke={st.color} strokeOpacity="0.2" strokeWidth="0.8" />
      {/* Skirt */}
      <path d="M30 108 Q60 122 90 108 L86 148 Q60 160 34 148 Z" fill={st.color} fillOpacity="0.14" stroke={st.color} strokeOpacity="0.22" strokeWidth="0.8" />
      {/* Torso */}
      <path d="M38 68 Q60 60 82 68 L86 110 Q60 118 34 110 Z" fill={st.color} fillOpacity="0.18" stroke={st.color} strokeOpacity="0.28" strokeWidth="0.8" />
      {/* Arms */}
      <path d="M38 72 Q18 88 20 108" stroke={st.color} strokeOpacity="0.3" strokeWidth="10" strokeLinecap="round" fill="none" />
      <path d="M82 72 Q102 88 100 108" stroke={st.color} strokeOpacity="0.3" strokeWidth="10" strokeLinecap="round" fill="none" />
      {/* Neck */}
      <rect x="53" y="52" width="14" height="20" rx="6" fill={st.color} fillOpacity="0.2" stroke={st.color} strokeOpacity="0.25" strokeWidth="0.8" />
      {/* Head */}
      <ellipse cx="60" cy="36" rx="24" ry="26" fill={st.color} fillOpacity="0.18" stroke={st.color} strokeOpacity="0.35" strokeWidth="1" />
      {/* Hair */}
      <path d="M36 28 Q40 6 60 8 Q80 6 84 28" fill={st.color} fillOpacity="0.28" stroke={st.color} strokeOpacity="0.4" strokeWidth="0.8" />
      <path d="M36 28 Q30 16 34 8" stroke={st.color} strokeOpacity="0.3" strokeWidth="5" strokeLinecap="round" fill="none" />
      <path d="M84 28 Q90 16 86 8" stroke={st.color} strokeOpacity="0.3" strokeWidth="5" strokeLinecap="round" fill="none" />
      {/* Eyes */}
      <ellipse cx="50" cy="36" rx="5" ry={isActive ? 5 : 3.5} fill={st.color} fillOpacity="0.8" />
      <ellipse cx="70" cy="36" rx="5" ry={isActive ? 5 : 3.5} fill={st.color} fillOpacity="0.8" />
      <circle cx="52" cy="33" r="1.5" fill="white" fillOpacity="0.8" />
      <circle cx="72" cy="33" r="1.5" fill="white" fillOpacity="0.8" />
      {/* Mouth */}
      {state === 'speaking'   && <path d="M53 46 Q60 52 67 46" stroke={st.color} strokeOpacity="0.8" strokeWidth="1.5" strokeLinecap="round" fill="none" />}
      {state === 'idle'       && <path d="M54 46 Q60 49 66 46" stroke={st.color} strokeOpacity="0.5" strokeWidth="1.2" strokeLinecap="round" fill="none" />}
      {state === 'listening'  && <circle cx="60" cy="47" r="2.5" fill={st.color} fillOpacity="0.6" />}
      {state === 'processing' && <path d="M55 47 h10" stroke={st.color} strokeOpacity="0.6" strokeWidth="1.5" strokeLinecap="round" />}

      <text x="60" y="116" textAnchor="middle" fontSize="7" fill={st.color} fillOpacity="0.35" fontFamily="monospace" letterSpacing="0.5">character art</text>

      {/* Idle ambient pulse on head glow — controlled by pulse state */}
      {state === 'idle' && (
        <ellipse cx="60" cy="36" rx="28" ry="30"
          fill="none" stroke={st.color}
          strokeOpacity={pulse ? 0.18 : 0.06}
          strokeWidth="2"
          style={{ transition: 'stroke-opacity 1.8s' }}
        />
      )}

      {/* Suppress unused var warning from hue — used in future radial gradient */}
      <circle cx="0" cy="0" r="0" fill={`oklch(22% 0.04 ${hue})`} />
    </svg>
  );
}
