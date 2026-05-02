// Lumi Design Tokens — mirrors the T object from lumi-components.jsx
// All values are final and pixel-faithful to the design handoff.

export const tokens = {
  colors: {
    bg:          'oklch(11% 0.018 245)',
    surface:     'oklch(15.5% 0.022 245)',
    surfaceUp:   'oklch(19% 0.026 245)',
    surfaceTop:  'oklch(23% 0.03 245)',
    border:      'oklch(28% 0.03 245)',
    borderSub:   'oklch(22% 0.025 245)',
    textPri:     'oklch(91% 0.01 240)',
    textSec:     'oklch(58% 0.025 240)',
    textMuted:   'oklch(36% 0.022 240)',
    accentBlue:  'oklch(62% 0.18 222)',
    accentGreen: 'oklch(64% 0.18 152)',
    accentAmber: 'oklch(72% 0.17 65)',
    accentWhite: 'oklch(94% 0.012 240)',
    danger:      'oklch(60% 0.18 25)',
  },
  space: { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 },
  radius: { sm: 4, md: 8, lg: 12, xl: 16, pill: 999 },
  font: { xs: 10, sm: 11, md: 13, lg: 15, xl: 18, xxl: 22 },
  opacity: { idle: 0.42, active: 1, disabled: 0.32 },
  shadow: {
    sm:        '0 2px 8px oklch(0% 0 0 / 0.45)',
    md:        '0 4px 20px oklch(0% 0 0 / 0.55)',
    lg:        '0 8px 36px oklch(0% 0 0 / 0.65)',
    glowBlue:  '0 0 18px oklch(62% 0.18 222 / 0.45)',
    glowGreen: '0 0 18px oklch(64% 0.18 152 / 0.50)',
    glowAmber: '0 0 18px oklch(72% 0.17 65 / 0.45)',
    glowWhite: '0 0 22px oklch(94% 0.012 240 / 0.55)',
  },
} as const;

// Per-state derived values used by avatar and overlay.
// BrainState (events.ts) and AvatarStateKey share the same lowercase values.
export type AvatarStateKey = 'idle' | 'listening' | 'processing' | 'speaking';

export interface StateStyle {
  color: string;
  glow: string;
  label: string;
  opacity: number;
}

export const AVATAR_STATES: Record<AvatarStateKey, StateStyle> = {
  idle: {
    color:   tokens.colors.accentBlue,
    glow:    tokens.shadow.glowBlue,
    label:   'Idle',
    opacity: tokens.opacity.idle,
  },
  listening: {
    color:   tokens.colors.accentGreen,
    glow:    tokens.shadow.glowGreen,
    label:   'Listening',
    opacity: tokens.opacity.active,
  },
  processing: {
    color:   tokens.colors.accentAmber,
    glow:    tokens.shadow.glowAmber,
    label:   'Processing',
    opacity: tokens.opacity.active,
  },
  speaking: {
    color:   tokens.colors.accentWhite,
    glow:    tokens.shadow.glowWhite,
    label:   'Speaking',
    opacity: tokens.opacity.active,
  },
};

// Keyframe CSS injected once at app root — no build step required
export const GLOBAL_KEYFRAMES = `
@keyframes lumiPulse {
  0%, 100% { opacity: 0.45; transform: scale(1); }
  50%       { opacity: 0.9;  transform: scale(1.06); }
}
@keyframes lumiFloat {
  0%, 100% { transform: translateY(0); }
  50%       { transform: translateY(-5px); }
}
@keyframes lumiDot {
  0%, 100% { transform: translateY(0);   opacity: 0.4; }
  50%       { transform: translateY(-4px); opacity: 1; }
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes lumiStatePop {
  0%   { transform: scale(0.95); opacity: 0.6; }
  60%  { transform: scale(1.03); opacity: 1; }
  100% { transform: scale(1);    opacity: 1; }
}
`;
