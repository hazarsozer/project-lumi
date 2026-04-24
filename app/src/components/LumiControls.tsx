// Lumi shared UI controls — Toggle, Slider, Dropdown, Input variants, Checkbox, SettingRow
import { tokens as T } from '../styles/tokens';

// ── Toggle ──────────────────────────────────────────────────

interface LumiToggleProps {
  checked: boolean;
  onChange: (value: boolean) => void;
  disabled?: boolean;
}

export function LumiToggle({ checked, onChange, disabled = false }: LumiToggleProps) {
  return (
    <div
      onClick={() => !disabled && onChange(!checked)}
      style={{
        width: 36, height: 20, borderRadius: T.radius.pill,
        background: checked ? T.colors.accentBlue : T.colors.surfaceTop,
        position: 'relative', cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.22s', opacity: disabled ? T.opacity.disabled : 1,
        flexShrink: 0,
      }}
    >
      <div style={{
        position: 'absolute', top: 3, left: checked ? 19 : 3,
        width: 14, height: 14, borderRadius: T.radius.pill,
        background: checked ? T.colors.textPri : T.colors.textMuted,
        transition: 'left 0.22s, background 0.22s',
        boxShadow: checked ? T.shadow.glowBlue : 'none',
      }} />
    </div>
  );
}

// ── Slider ──────────────────────────────────────────────────

interface LumiSliderProps {
  value: number;
  min: number;
  max: number;
  label?: string;
  minLabel?: string;
  maxLabel?: string;
  disabled?: boolean;
  onChange?: (value: number) => void;
}

export function LumiSlider({ value, min, max, label, minLabel, maxLabel, disabled = false }: LumiSliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, opacity: disabled ? T.opacity.disabled : 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        {label && <span style={{ fontSize: T.font.sm, color: T.colors.textSec }}>{label}</span>}
        <span style={{ fontSize: T.font.sm, color: T.colors.accentBlue, fontVariantNumeric: 'tabular-nums' }}>{value}</span>
      </div>
      <div style={{ position: 'relative', height: 4, borderRadius: T.radius.pill, background: T.colors.surfaceTop }}>
        <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, borderRadius: T.radius.pill, background: T.colors.accentBlue }} />
        <div style={{ position: 'absolute', top: '50%', left: `${pct}%`, transform: 'translate(-50%,-50%)', width: 12, height: 12, borderRadius: T.radius.pill, background: T.colors.textPri, border: `2px solid ${T.colors.accentBlue}`, cursor: disabled ? 'not-allowed' : 'ew-resize' }} />
      </div>
      {(minLabel || maxLabel) && (
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontSize: T.font.xs, color: T.colors.textMuted }}>{minLabel}</span>
          <span style={{ fontSize: T.font.xs, color: T.colors.textMuted }}>{maxLabel}</span>
        </div>
      )}
    </div>
  );
}

// ── Dropdown ────────────────────────────────────────────────

interface LumiDropdownProps {
  value: string;
  options: string[];
  disabled?: boolean;
  onChange?: (value: string) => void;
}

export function LumiDropdown({ value, disabled = false }: LumiDropdownProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '7px 10px', borderRadius: T.radius.md, background: T.colors.surfaceTop,
      border: `1px solid ${T.colors.border}`, cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? T.opacity.disabled : 1, minWidth: 160,
    }}>
      <span style={{ fontSize: T.font.md, color: T.colors.textPri }}>{value}</span>
      <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
        <path d="M1 1l4 4 4-4" stroke={T.colors.textMuted} strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    </div>
  );
}

// ── Text Input ──────────────────────────────────────────────

interface LumiInputProps {
  value: string;
  placeholder?: string;
  disabled?: boolean;
  type?: string;
  suffix?: string;
}

export function LumiInput({ value, placeholder, disabled = false, type = 'text', suffix }: LumiInputProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '7px 10px', borderRadius: T.radius.md, background: T.colors.surfaceTop,
      border: `1px solid ${T.colors.border}`,
      opacity: disabled ? T.opacity.disabled : 1, flex: 1,
    }}>
      <input readOnly value={value} placeholder={placeholder} type={type} style={{
        background: 'transparent', border: 'none', outline: 'none', flex: 1,
        fontSize: T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri,
        fontFamily: 'inherit', cursor: disabled ? 'not-allowed' : 'text',
      }} />
      {suffix && <span style={{ fontSize: T.font.sm, color: T.colors.textMuted }}>{suffix}</span>}
    </div>
  );
}

// ── Number Input ────────────────────────────────────────────

interface LumiNumberInputProps {
  value: number;
  min?: number;
  max?: number;
  disabled?: boolean;
}

export function LumiNumberInput({ value, disabled = false }: LumiNumberInputProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center',
      borderRadius: T.radius.md, background: T.colors.surfaceTop,
      border: `1px solid ${T.colors.border}`, overflow: 'hidden',
      opacity: disabled ? T.opacity.disabled : 1, width: 90,
    }}>
      <input readOnly value={value} type="number" style={{
        background: 'transparent', border: 'none', outline: 'none', flex: 1,
        fontSize: T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri,
        fontFamily: 'inherit', textAlign: 'center', padding: '7px 4px',
        cursor: disabled ? 'not-allowed' : 'text',
      }} />
      <div style={{ display: 'flex', flexDirection: 'column', borderLeft: `1px solid ${T.colors.border}` }}>
        {(['+', '−'] as const).map((s) => (
          <div key={s} style={{ padding: '3px 7px', cursor: 'pointer', color: T.colors.textSec, fontSize: 10, lineHeight: '11px', borderBottom: s === '+' ? `1px solid ${T.colors.border}` : 'none' }}>{s}</div>
        ))}
      </div>
    </div>
  );
}

// ── File Path Input ─────────────────────────────────────────

interface LumiFilePathProps {
  value: string;
  disabled?: boolean;
}

export function LumiFilePath({ value, disabled = false }: LumiFilePathProps) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 0, opacity: disabled ? T.opacity.disabled : 1, flex: 1 }}>
      <div style={{
        flex: 1, padding: '7px 10px', borderRadius: `${T.radius.md}px 0 0 ${T.radius.md}px`,
        background: T.colors.surfaceTop, border: `1px solid ${T.colors.border}`, borderRight: 'none',
        fontSize: T.font.sm, color: T.colors.textSec, fontFamily: 'monospace',
        overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
      }}>{value}</div>
      <div style={{
        padding: '7px 10px', background: T.colors.surfaceUp,
        border: `1px solid ${T.colors.border}`, borderRadius: `0 ${T.radius.md}px ${T.radius.md}px 0`,
        fontSize: T.font.sm, color: T.colors.textSec, cursor: 'pointer', whiteSpace: 'nowrap',
      }}>Browse</div>
    </div>
  );
}

// LumiCheckbox and SettingRow live in LumiLayout.tsx to keep this file under 200 lines.
export { LumiCheckbox, SettingRow } from './LumiLayout';
