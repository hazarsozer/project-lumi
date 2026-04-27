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
  step?: number;
  label?: string;
  minLabel?: string;
  maxLabel?: string;
  disabled?: boolean;
  onChange?: (value: number) => void;
}

export function LumiSlider({ value, min, max, step = 1, label, minLabel, maxLabel, disabled = false, onChange }: LumiSliderProps) {
  const pct = ((value - min) / (max - min)) * 100;
  const displayValue = step < 1 ? value.toFixed(step < 0.1 ? 2 : 1) : String(value);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, opacity: disabled ? T.opacity.disabled : 1 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        {label && <span style={{ fontSize: T.font.sm, color: T.colors.textSec }}>{label}</span>}
        <span style={{ fontSize: T.font.sm, color: T.colors.accentBlue, fontVariantNumeric: 'tabular-nums' }}>{displayValue}</span>
      </div>
      <div style={{ position: 'relative', height: 4, borderRadius: T.radius.pill, background: T.colors.surfaceTop }}>
        <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, borderRadius: T.radius.pill, background: T.colors.accentBlue }} />
        <div style={{ position: 'absolute', top: '50%', left: `${pct}%`, transform: 'translate(-50%,-50%)', width: 12, height: 12, borderRadius: T.radius.pill, background: T.colors.textPri, border: `2px solid ${T.colors.accentBlue}`, pointerEvents: 'none' }} />
        {/* Transparent native range input layered on top for drag interaction */}
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange?.(Number(e.target.value))}
          style={{ position: 'absolute', top: -6, left: 0, width: '100%', height: 16, opacity: 0, cursor: disabled ? 'not-allowed' : 'ew-resize', margin: 0, padding: 0 }}
        />
      </div>
      {(minLabel ?? maxLabel) && (
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

export function LumiDropdown({ value, options, disabled = false, onChange }: LumiDropdownProps) {
  return (
    <div style={{ position: 'relative', minWidth: 160 }}>
      {/* Visual layer */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '7px 10px', borderRadius: T.radius.md, background: T.colors.surfaceTop,
        border: `1px solid ${T.colors.border}`,
        opacity: disabled ? T.opacity.disabled : 1,
        pointerEvents: 'none',
      }}>
        <span style={{ fontSize: T.font.md, color: T.colors.textPri }}>{value}</span>
        <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
          <path d="M1 1l4 4 4-4" stroke={T.colors.textMuted} strokeWidth="1.5" strokeLinecap="round" />
        </svg>
      </div>
      {/* Native select — invisible, captures all interaction */}
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.value)}
        style={{
          position: 'absolute', inset: 0, opacity: 0,
          cursor: disabled ? 'not-allowed' : 'pointer',
          width: '100%', height: '100%',
        }}
      >
        {options.map((opt) => <option key={opt} value={opt}>{opt}</option>)}
      </select>
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
  onChange?: (value: string) => void;
}

export function LumiInput({ value, placeholder, disabled = false, type = 'text', suffix, onChange }: LumiInputProps) {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '7px 10px', borderRadius: T.radius.md, background: T.colors.surfaceTop,
      border: `1px solid ${T.colors.border}`,
      opacity: disabled ? T.opacity.disabled : 1, flex: 1,
    }}>
      <input
        value={value}
        placeholder={placeholder}
        type={type}
        disabled={disabled}
        onChange={(e) => onChange?.(e.target.value)}
        style={{
          background: 'transparent', border: 'none', outline: 'none', flex: 1,
          fontSize: T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri,
          fontFamily: 'inherit', cursor: disabled ? 'not-allowed' : 'text',
        }}
      />
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
  onChange?: (value: number) => void;
}

export function LumiNumberInput({ value, min, max, disabled = false, onChange }: LumiNumberInputProps) {
  function adjust(delta: number) {
    if (disabled) return;
    const next = value + delta;
    if (min !== undefined && next < min) return;
    if (max !== undefined && next > max) return;
    onChange?.(next);
  }
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
        cursor: 'default',
      }} />
      <div style={{ display: 'flex', flexDirection: 'column', borderLeft: `1px solid ${T.colors.border}` }}>
        {(['+', '−'] as const).map((s) => (
          <div
            key={s}
            onClick={() => adjust(s === '+' ? 1 : -1)}
            style={{
              padding: '3px 7px', cursor: disabled ? 'not-allowed' : 'pointer',
              color: T.colors.textSec, fontSize: 10, lineHeight: '11px',
              borderBottom: s === '+' ? `1px solid ${T.colors.border}` : 'none',
              userSelect: 'none',
            }}
          >
            {s}
          </div>
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
