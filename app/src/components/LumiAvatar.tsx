import { useEffect, useState } from 'react';
import { AVATAR_STATES, AvatarStateKey } from '../styles/tokens';
import lumiIdle from '../assets/lumi-idle.png';
import lumiSpeaking from '../assets/lumi-speaking.png';

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

  const src = state === 'speaking' ? lumiSpeaking : lumiIdle;

  return (
    <div
      style={{
        width: 140,
        height: 140,
        opacity: isActive ? 1 : st.opacity,
        transition: 'opacity 0.5s, filter 0.5s',
        filter: isActive
          ? `drop-shadow(0 0 14px ${st.color})`
          : `drop-shadow(0 0 6px ${st.color}80) drop-shadow(0 0 ${pulse ? 18 : 8}px ${st.color}${pulse ? '50' : '20'})`,
        marginBottom: -8,
        zIndex: 2,
        position: 'relative',
        animation: state === 'listening' ? 'lumiFloat 1.4s ease-in-out infinite' : 'none',
      }}
    >
      <img
        src={src}
        alt="Lumi"
        style={{
          width: '100%',
          height: '100%',
          imageRendering: 'pixelated',
          display: 'block',
          objectFit: 'contain',
        }}
      />
    </div>
  );
}
