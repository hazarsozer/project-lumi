// Lumi Design System — Shared Components
// Tokens, controls, avatar, chat bubbles, settings rows

const T = {
  colors: {
    bg:         'oklch(11% 0.018 245)',
    surface:    'oklch(15.5% 0.022 245)',
    surfaceUp:  'oklch(19% 0.026 245)',
    surfaceTop: 'oklch(23% 0.03 245)',
    border:     'oklch(28% 0.03 245)',
    borderSub:  'oklch(22% 0.025 245)',
    textPri:    'oklch(91% 0.01 240)',
    textSec:    'oklch(58% 0.025 240)',
    textMuted:  'oklch(36% 0.022 240)',
    accentBlue:  'oklch(62% 0.18 222)',
    accentGreen: 'oklch(64% 0.18 152)',
    accentAmber: 'oklch(72% 0.17 65)',
    accentWhite: 'oklch(94% 0.012 240)',
    danger:      'oklch(60% 0.18 25)',
  },
  space: { xs:4, sm:8, md:12, lg:16, xl:24, xxl:32 },
  radius: { sm:4, md:8, lg:12, xl:16, pill:999 },
  font: { xs:10, sm:11, md:13, lg:15, xl:18, xxl:22 },
  opacity: { idle:0.42, active:1, disabled:0.32 },
  shadow: {
    sm:  '0 2px 8px oklch(0% 0 0 / 0.45)',
    md:  '0 4px 20px oklch(0% 0 0 / 0.55)',
    lg:  '0 8px 36px oklch(0% 0 0 / 0.65)',
    glowBlue:  '0 0 18px oklch(62% 0.18 222 / 0.45)',
    glowGreen: '0 0 18px oklch(64% 0.18 152 / 0.50)',
    glowAmber: '0 0 18px oklch(72% 0.17 65 / 0.45)',
    glowWhite: '0 0 22px oklch(94% 0.012 240 / 0.55)',
  },
};

// ── Toggle ────────────────────────────────────────────────
function LumiToggle({ checked, onChange, disabled }) {
  const on = checked;
  return (
    <div
      onClick={() => !disabled && onChange(!on)}
      style={{
        width: 36, height: 20, borderRadius: T.radius.pill,
        background: on ? T.colors.accentBlue : T.colors.surfaceTop,
        position: 'relative', cursor: disabled ? 'not-allowed' : 'pointer',
        transition: 'background 0.22s', opacity: disabled ? T.opacity.disabled : 1,
        flexShrink: 0,
      }}
    >
      <div style={{
        position:'absolute', top:3, left: on ? 19 : 3,
        width:14, height:14, borderRadius: T.radius.pill,
        background: on ? T.colors.textPri : T.colors.textMuted,
        transition: 'left 0.22s, background 0.22s',
        boxShadow: on ? T.shadow.glowBlue : 'none',
      }}/>
    </div>
  );
}

// ── Slider ────────────────────────────────────────────────
function LumiSlider({ value, min, max, label, minLabel, maxLabel, disabled, onChange }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:4, opacity: disabled ? T.opacity.disabled : 1 }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        {label && <span style={{ fontSize:T.font.sm, color:T.colors.textSec }}>{label}</span>}
        <span style={{ fontSize:T.font.sm, color:T.colors.accentBlue, fontVariantNumeric:'tabular-nums' }}>{value}</span>
      </div>
      <div style={{ position:'relative', height:4, borderRadius:T.radius.pill, background:T.colors.surfaceTop }}>
        <div style={{ position:'absolute', left:0, top:0, height:'100%', width:`${pct}%`, borderRadius:T.radius.pill, background:T.colors.accentBlue }}/>
        <div style={{ position:'absolute', top:'50%', left:`${pct}%`, transform:'translate(-50%,-50%)', width:12, height:12, borderRadius:T.radius.pill, background:T.colors.textPri, border:`2px solid ${T.colors.accentBlue}`, cursor: disabled ? 'not-allowed' : 'ew-resize' }}/>
      </div>
      {(minLabel || maxLabel) && (
        <div style={{ display:'flex', justifyContent:'space-between' }}>
          <span style={{ fontSize:T.font.xs, color:T.colors.textMuted }}>{minLabel}</span>
          <span style={{ fontSize:T.font.xs, color:T.colors.textMuted }}>{maxLabel}</span>
        </div>
      )}
    </div>
  );
}

// ── Dropdown ──────────────────────────────────────────────
function LumiDropdown({ value, options, disabled }) {
  return (
    <div style={{
      display:'flex', alignItems:'center', justifyContent:'space-between',
      padding:'7px 10px', borderRadius:T.radius.md, background:T.colors.surfaceTop,
      border:`1px solid ${T.colors.border}`, cursor: disabled ? 'not-allowed' : 'pointer',
      opacity: disabled ? T.opacity.disabled : 1, minWidth:160,
    }}>
      <span style={{ fontSize:T.font.md, color:T.colors.textPri }}>{value}</span>
      <svg width="10" height="6" viewBox="0 0 10 6" fill="none">
        <path d="M1 1l4 4 4-4" stroke={T.colors.textMuted} strokeWidth="1.5" strokeLinecap="round"/>
      </svg>
    </div>
  );
}

// ── Text Input ────────────────────────────────────────────
function LumiInput({ value, placeholder, disabled, type='text', suffix }) {
  return (
    <div style={{
      display:'flex', alignItems:'center', gap:6,
      padding:'7px 10px', borderRadius:T.radius.md, background:T.colors.surfaceTop,
      border:`1px solid ${T.colors.border}`,
      opacity: disabled ? T.opacity.disabled : 1, flex:1,
    }}>
      <input readOnly value={value} placeholder={placeholder} type={type} style={{
        background:'transparent', border:'none', outline:'none', flex:1,
        fontSize:T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri,
        fontFamily:'inherit', cursor: disabled ? 'not-allowed' : 'text',
      }}/>
      {suffix && <span style={{ fontSize:T.font.sm, color:T.colors.textMuted }}>{suffix}</span>}
    </div>
  );
}

// ── Number Input ──────────────────────────────────────────
function LumiNumberInput({ value, min, max, disabled }) {
  return (
    <div style={{
      display:'flex', alignItems:'center',
      borderRadius:T.radius.md, background:T.colors.surfaceTop,
      border:`1px solid ${T.colors.border}`, overflow:'hidden',
      opacity: disabled ? T.opacity.disabled : 1, width:90,
    }}>
      <input readOnly value={value} type="number" style={{
        background:'transparent', border:'none', outline:'none', flex:1,
        fontSize:T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri,
        fontFamily:'inherit', textAlign:'center', padding:'7px 4px',
        cursor: disabled ? 'not-allowed' : 'text', MozAppearance:'textfield',
      }}/>
      <div style={{ display:'flex', flexDirection:'column', borderLeft:`1px solid ${T.colors.border}` }}>
        {['+','−'].map(s=>(
          <div key={s} style={{ padding:'3px 7px', cursor:'pointer', color:T.colors.textSec, fontSize:10, lineHeight:'11px', borderBottom: s==='+' ? `1px solid ${T.colors.border}` : 'none' }}>{s}</div>
        ))}
      </div>
    </div>
  );
}

// ── File Path Input ───────────────────────────────────────
function LumiFilePath({ value, disabled }) {
  return (
    <div style={{ display:'flex', alignItems:'center', gap:0, opacity: disabled ? T.opacity.disabled : 1, flex:1 }}>
      <div style={{
        flex:1, padding:'7px 10px', borderRadius:`${T.radius.md}px 0 0 ${T.radius.md}px`,
        background:T.colors.surfaceTop, border:`1px solid ${T.colors.border}`, borderRight:'none',
        fontSize:T.font.sm, color:T.colors.textSec, fontFamily:'monospace', overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap',
      }}>{value}</div>
      <div style={{
        padding:'7px 10px', background:T.colors.surfaceUp,
        border:`1px solid ${T.colors.border}`, borderRadius:`0 ${T.radius.md}px ${T.radius.md}px 0`,
        fontSize:T.font.sm, color:T.colors.textSec, cursor:'pointer', whiteSpace:'nowrap',
      }}>Browse</div>
    </div>
  );
}

// ── Multi-select Checkbox ─────────────────────────────────
function LumiCheckbox({ checked, label, disabled }) {
  return (
    <label style={{ display:'flex', alignItems:'center', gap:8, cursor: disabled ? 'not-allowed' : 'pointer', opacity: disabled ? T.opacity.disabled : 1 }}>
      <div style={{
        width:15, height:15, borderRadius:T.radius.sm,
        background: checked ? T.colors.accentBlue : 'transparent',
        border:`1.5px solid ${checked ? T.colors.accentBlue : T.colors.border}`,
        display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0,
        transition:'background 0.15s',
      }}>
        {checked && <svg width="9" height="7" viewBox="0 0 9 7" fill="none"><path d="M1 3.5l2.5 2.5L8 1" stroke={T.colors.bg} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>}
      </div>
      <span style={{ fontSize:T.font.md, color: disabled ? T.colors.textMuted : T.colors.textPri }}>{label}</span>
    </label>
  );
}

// ── Icon Button ───────────────────────────────────────────
function LumiIconBtn({ icon, state='normal', tooltip }) {
  const bg = state==='active' ? T.colors.surfaceTop : state==='hover' ? T.colors.surfaceUp : 'transparent';
  const col = state==='disabled' ? T.colors.textMuted : state==='active' ? T.colors.accentBlue : T.colors.textSec;
  return (
    <div title={tooltip} style={{
      width:32, height:32, borderRadius:T.radius.md,
      background:bg, display:'flex', alignItems:'center', justifyContent:'center',
      cursor: state==='disabled' ? 'not-allowed' : 'pointer',
      opacity: state==='disabled' ? T.opacity.disabled : 1,
      border: state==='active' ? `1px solid ${T.colors.border}` : '1px solid transparent',
      transition:'background 0.15s',
    }}>
      <span style={{ fontSize:18, lineHeight:1, color:col, userSelect:'none' }}>{icon}</span>
    </div>
  );
}

// ── Setting Row ───────────────────────────────────────────
function SettingRow({ label, desc, control, restart }) {
  return (
    <div style={{
      display:'flex', alignItems:'center', justifyContent:'space-between',
      padding:`${T.space.sm}px ${T.space.md}px`, borderRadius:T.radius.md,
      gap:T.space.lg,
    }}>
      <div style={{ flex:1, minWidth:0 }}>
        <div style={{ display:'flex', alignItems:'center', gap:6 }}>
          <span style={{ fontSize:T.font.md, color:T.colors.textPri }}>{label}</span>
          {restart && <span title="Requires restart" style={{ fontSize:T.font.xs, color:T.colors.accentAmber, background:'oklch(72% 0.17 65 / 0.15)', padding:'1px 5px', borderRadius:T.radius.pill }}>↻</span>}
        </div>
        {desc && <div style={{ fontSize:T.font.sm, color:T.colors.textMuted, marginTop:2 }}>{desc}</div>}
      </div>
      <div style={{ flexShrink:0 }}>{control}</div>
    </div>
  );
}

// Export to window
Object.assign(window, {
  T, LumiToggle, LumiSlider, LumiDropdown, LumiInput, LumiNumberInput,
  LumiFilePath, LumiCheckbox, LumiIconBtn, SettingRow
});
