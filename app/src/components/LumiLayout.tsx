// LumiLayout — Checkbox and SettingRow layout components
// Separated from LumiControls.tsx to keep both files under 200 lines.
import { tokens as T } from '../styles/tokens';

// ── Checkbox ────────────────────────────────────────────────

interface LumiCheckboxProps {
  checked: boolean;
  label: string;
  disabled?: boolean;
  onChange?: (value: boolean) => void;
}

export function LumiCheckbox({ checked, label, disabled = false }: LumiCheckboxProps) {
  return (
    <label style={{
      display: 'flex', alignItems: 'center', gap: 8,
      cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? T.opacity.disabled : 1,
    }}>
      <div style={{
        width: 15, height: 15, borderRadius: T.radius.sm,
        background: checked ? T.colors.accentBlue : 'transparent',
        border: `1.5px solid ${checked ? T.colors.accentBlue : T.colors.border}`,
        display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        transition: 'background 0.15s',
      }}>
        {checked && (
          <svg width="9" height="7" viewBox="0 0 9 7" fill="none">
            <path d="M1 3.5l2.5 2.5L8 1" stroke={T.colors.bg} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </div>
      <span style={{ fontSize: T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri }}>
        {label}
      </span>
    </label>
  );
}

// ── Setting Row ─────────────────────────────────────────────

interface SettingRowProps {
  label: string;
  desc?: string;
  control: React.ReactNode;
  restart?: boolean;
}

export function SettingRow({ label, desc, control, restart = false }: SettingRowProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: `${T.space.sm}px ${T.space.md}px`, borderRadius: T.radius.md,
      gap: T.space.lg,
    }}>
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ fontSize: T.font.md, color: T.colors.textPri }}>{label}</span>
          {restart && (
            <span
              title="Requires restart"
              style={{
                fontSize: T.font.xs, color: T.colors.accentAmber,
                background: 'oklch(72% 0.17 65 / 0.15)',
                padding: '1px 5px', borderRadius: T.radius.pill,
              }}
            >
              ↻
            </span>
          )}
        </div>
        {desc && (
          <div style={{ fontSize: T.font.sm, color: T.colors.textMuted, marginTop: 2 }}>
            {desc}
          </div>
        )}
      </div>
      <div style={{ flexShrink: 0 }}>{control}</div>
    </div>
  );
}
