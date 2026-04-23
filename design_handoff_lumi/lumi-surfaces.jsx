// Lumi Surfaces — Compact Overlay, Chat Panel, Settings Panel

// ── Avatar Glow States ────────────────────────────────────
const STATES = {
  idle:       { color: T.colors.accentBlue,  glow: T.shadow.glowBlue,  label:'Idle',       opacity: T.opacity.idle },
  listening:  { color: T.colors.accentGreen, glow: T.shadow.glowGreen, label:'Listening',  opacity: 1 },
  processing: { color: T.colors.accentAmber, glow: T.shadow.glowAmber, label:'Processing', opacity: 1 },
  speaking:   { color: T.colors.accentWhite, glow: T.shadow.glowWhite, label:'Speaking',   opacity: 1 },
};

// ── Lumi Avatar ───────────────────────────────────────────
function LumiAvatar({ state='idle', size=100 }) {
  const st = STATES[state];
  const [pulse, setPulse] = React.useState(false);
  React.useEffect(() => {
    if (state !== 'idle') return;
    const id = setInterval(() => setPulse(p => !p), 2800);
    return () => clearInterval(id);
  }, [state]);

  return (
    <div style={{
      width: size, height: size, borderRadius: '50%',
      position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center',
      transition: 'opacity 0.4s',
      opacity: st.opacity,
    }}>
      {/* Outer glow ring */}
      <div style={{
        position: 'absolute', inset: -6, borderRadius: '50%',
        boxShadow: st.glow,
        border: `1.5px solid ${st.color}`,
        opacity: state==='idle' ? (pulse ? 0.5 : 0.25) : 0.75,
        transition: 'opacity 1.8s, box-shadow 0.4s, border-color 0.4s',
        animation: state==='listening' ? 'lumiPulse 1.1s ease-in-out infinite' : 'none',
      }}/>
      {/* Avatar face — soft gradient portrait area */}
      <div style={{
        width: size, height: size, borderRadius: '50%',
        background: `radial-gradient(ellipse at 38% 35%, oklch(28% 0.04 245) 0%, oklch(14% 0.022 245) 70%)`,
        border: `1.5px solid ${st.color}`,
        overflow: 'hidden', display:'flex', alignItems:'center', justifyContent:'center',
        boxShadow: `inset 0 1px 0 oklch(100% 0 0 / 0.06), ${st.glow}`,
        transition: 'border-color 0.4s, box-shadow 0.4s',
        position:'relative',
      }}>
        {/* Abstract face: eyes + subtle glow core */}
        <svg width={size*0.55} height={size*0.55} viewBox="0 0 60 60" fill="none">
          {/* Core orb */}
          <circle cx="30" cy="30" r="18" fill={`oklch(22% 0.04 ${state==='idle'?222:state==='listening'?152:state==='processing'?65:240})`}/>
          <circle cx="30" cy="30" r="10" fill={st.color} opacity="0.18"/>
          <circle cx="30" cy="30" r="4" fill={st.color} opacity="0.55"/>
          {/* Eyes */}
          <ellipse cx="23" cy="27" rx="2.5" ry={state==='processing'?1.5:2.5} fill={st.color} opacity="0.9"/>
          <ellipse cx="37" cy="27" rx="2.5" ry={state==='processing'?1.5:2.5} fill={st.color} opacity="0.9"/>
          {/* Mouth — varies by state */}
          {state==='speaking' && <path d="M24 36 Q30 41 36 36" stroke={st.color} strokeWidth="1.5" strokeLinecap="round" fill="none" opacity="0.8"/>}
          {state==='idle'     && <path d="M25 36 Q30 39 35 36" stroke={st.color} strokeWidth="1.2" strokeLinecap="round" fill="none" opacity="0.5"/>}
          {state==='listening'&& <circle cx="30" cy="36" r="2" fill={st.color} opacity="0.6"/>}
          {state==='processing'&& <path d="M26 36 h8" stroke={st.color} strokeWidth="1.5" strokeLinecap="round" opacity="0.6"/>}
        </svg>
      </div>
    </div>
  );
}

// ── Compact Overlay ───────────────────────────────────────
// Layout: character avatar floats above the button tray,
// overlapping it. No shared background rectangle.
function CompactOverlay({ avatarState='idle', onChat, onSettings }) {
  const [hoverChat, setHoverChat] = React.useState(false);
  const [hoverSettings, setHoverSettings] = React.useState(false);
  const st = STATES[avatarState];
  const isActive = avatarState !== 'idle';

  // Silhouette placeholder — suggests a seated/standing character
  // Real asset would be a PNG with transparency dropped in here
  const CharacterPlaceholder = () => (
    <svg viewBox="0 0 120 200" fill="none" xmlns="http://www.w3.org/2000/svg"
      style={{ width:'100%', height:'100%' }}>
      {/* Subtle stripes backdrop — "drop character art here" */}
      <defs>
        <pattern id="charStripe" patternUnits="userSpaceOnUse" width="6" height="6" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="6" stroke={st.color} strokeWidth="0.5" strokeOpacity="0.12"/>
        </pattern>
        <radialGradient id="charFade" cx="50%" cy="80%" r="60%">
          <stop offset="0%" stopColor={st.color} stopOpacity="0.08"/>
          <stop offset="100%" stopColor={st.color} stopOpacity="0"/>
        </radialGradient>
        {/* Glow beneath feet */}
        <radialGradient id="groundGlow" cx="50%" cy="100%" r="50%">
          <stop offset="0%" stopColor={st.color} stopOpacity="0.35"/>
          <stop offset="100%" stopColor={st.color} stopOpacity="0"/>
        </radialGradient>
      </defs>

      {/* Ground glow */}
      <ellipse cx="60" cy="196" rx="44" ry="8" fill="url(#groundGlow)"/>

      {/* Character silhouette — body */}
      <rect x="0" y="0" width="120" height="200" fill="url(#charStripe)" rx="8"/>
      <rect x="0" y="0" width="120" height="200" fill="url(#charFade)" rx="8"/>

      {/* Legs */}
      <rect x="38" y="138" width="18" height="52" rx="8" fill={st.color} fillOpacity="0.12" stroke={st.color} strokeOpacity="0.2" strokeWidth="0.8"/>
      <rect x="64" y="138" width="18" height="52" rx="8" fill={st.color} fillOpacity="0.12" stroke={st.color} strokeOpacity="0.2" strokeWidth="0.8"/>

      {/* Skirt / lower body */}
      <path d="M30 108 Q60 122 90 108 L86 148 Q60 160 34 148 Z" fill={st.color} fillOpacity="0.14" stroke={st.color} strokeOpacity="0.22" strokeWidth="0.8"/>

      {/* Torso */}
      <path d="M38 68 Q60 60 82 68 L86 110 Q60 118 34 110 Z" fill={st.color} fillOpacity="0.18" stroke={st.color} strokeOpacity="0.28" strokeWidth="0.8"/>

      {/* Left arm */}
      <path d="M38 72 Q18 88 20 108" stroke={st.color} strokeOpacity="0.3" strokeWidth="10" strokeLinecap="round" fill="none"/>
      {/* Right arm */}
      <path d="M82 72 Q102 88 100 108" stroke={st.color} strokeOpacity="0.3" strokeWidth="10" strokeLinecap="round" fill="none"/>

      {/* Neck */}
      <rect x="53" y="52" width="14" height="20" rx="6" fill={st.color} fillOpacity="0.2" stroke={st.color} strokeOpacity="0.25" strokeWidth="0.8"/>

      {/* Head */}
      <ellipse cx="60" cy="36" rx="24" ry="26" fill={st.color} fillOpacity="0.18" stroke={st.color} strokeOpacity="0.35" strokeWidth="1"/>

      {/* Hair — simple manga top */}
      <path d="M36 28 Q40 6 60 8 Q80 6 84 28" fill={st.color} fillOpacity="0.28" stroke={st.color} strokeOpacity="0.4" strokeWidth="0.8"/>
      <path d="M36 28 Q30 16 34 8" stroke={st.color} strokeOpacity="0.3" strokeWidth="5" strokeLinecap="round" fill="none"/>
      <path d="M84 28 Q90 16 86 8" stroke={st.color} strokeOpacity="0.3" strokeWidth="5" strokeLinecap="round" fill="none"/>

      {/* Eyes */}
      <ellipse cx="50" cy="36" rx="5" ry={isActive ? 5 : 3.5} fill={st.color} fillOpacity="0.8"/>
      <ellipse cx="70" cy="36" rx="5" ry={isActive ? 5 : 3.5} fill={st.color} fillOpacity="0.8"/>
      {/* Eye shine */}
      <circle cx="52" cy="33" r="1.5" fill="white" fillOpacity="0.8"/>
      <circle cx="72" cy="33" r="1.5" fill="white" fillOpacity="0.8"/>

      {/* Mouth */}
      {avatarState === 'speaking' && <path d="M53 46 Q60 52 67 46" stroke={st.color} strokeOpacity="0.8" strokeWidth="1.5" strokeLinecap="round" fill="none"/>}
      {avatarState === 'idle' && <path d="M54 46 Q60 49 66 46" stroke={st.color} strokeOpacity="0.5" strokeWidth="1.2" strokeLinecap="round" fill="none"/>}
      {avatarState === 'listening' && <circle cx="60" cy="47" r="2.5" fill={st.color} fillOpacity="0.6"/>}
      {avatarState === 'processing' && <path d="M55 47 h10" stroke={st.color} strokeOpacity="0.6" strokeWidth="1.5" strokeLinecap="round"/>}

      {/* Label */}
      <text x="60" y="116" textAnchor="middle" fontSize="7" fill={st.color} fillOpacity="0.35" fontFamily="monospace" letterSpacing="0.5">character art</text>
    </svg>
  );

  return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', userSelect:'none', position:'relative' }}>

      {/* ── Character portrait — no background ── */}
      <div style={{
        width: 140, height: 210,
        opacity: isActive ? 1 : T.opacity.idle,
        transition: 'opacity 0.5s, filter 0.5s',
        filter: isActive
          ? `drop-shadow(0 0 14px ${st.color})`
          : `drop-shadow(0 0 6px ${st.color}80)`,
        // Sits above tray — overlaps by ~28px via negative margin
        marginBottom: -28,
        zIndex: 2,
        position: 'relative',
        // Listening state: subtle bounce
        animation: avatarState === 'listening' ? 'lumiFloat 1.4s ease-in-out infinite' : 'none',
      }}>
        <CharacterPlaceholder/>
      </div>

      {/* ── Button tray ── */}
      <div style={{
        position: 'relative', zIndex: 1,
        display: 'flex', alignItems: 'center', gap: T.space.sm,
        padding: `${T.space.md}px ${T.space.lg}px`,
        background: 'oklch(14% 0.022 245 / 0.95)',
        borderRadius: T.radius.pill,
        border: `1px solid oklch(30% 0.035 245 / 0.7)`,
        boxShadow: `${T.shadow.md}, inset 0 1px 0 oklch(100% 0 0 / 0.05)`,
        backdropFilter: 'blur(18px)',
      }}>
        {/* Status dot */}
        <div style={{
          width:6, height:6, borderRadius:'50%',
          background: st.color,
          boxShadow: st.glow,
          opacity: 0.9,
        }}/>

        <span style={{ fontSize: T.font.sm, color: T.colors.textSec, letterSpacing:'0.02em', marginRight: T.space.sm }}>
          {st.label}
        </span>

        {/* Divider */}
        <div style={{ width:1, height:18, background: T.colors.borderSub }}/>

        {/* Buttons */}
        {[
          { icon:'⚙', hover:hoverSettings, setHover:setHoverSettings, action:onSettings, tip:'Settings' },
          { icon:'💬', hover:hoverChat,     setHover:setHoverChat,     action:onChat,     tip:'Chat'     },
        ].map(({ icon, hover, setHover, action, tip }) => (
          <div key={icon}
            onMouseEnter={() => setHover(true)}
            onMouseLeave={() => setHover(false)}
            onClick={action}
            title={tip}
            style={{
              width: 34, height: 34, borderRadius: T.radius.md,
              background: hover ? T.colors.surfaceTop : T.colors.surfaceUp,
              border: `1px solid ${hover ? T.colors.border : T.colors.borderSub}`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              cursor: 'pointer',
              transition: 'background 0.15s, border-color 0.15s',
              fontSize: 15,
            }}>
            {icon}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Chat Bubble ───────────────────────────────────────────
function ChatBubble({ from, text, time, citations }) {
  const isLumi = from === 'lumi';
  return (
    <div style={{ display:'flex', flexDirection: isLumi?'row':'row-reverse', gap:10, alignItems:'flex-end' }}>
      {isLumi && (
        <div style={{ width:28, height:28, borderRadius:'50%', background:T.colors.surfaceTop, border:`1px solid ${T.colors.border}`, display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0, fontSize:12 }}>✦</div>
      )}
      <div style={{ maxWidth:'80%' }}>
        <div style={{
          padding:`${T.space.sm}px ${T.space.md}px`,
          borderRadius: isLumi ? `${T.radius.lg}px ${T.radius.lg}px ${T.radius.lg}px 4px` : `${T.radius.lg}px ${T.radius.lg}px 4px ${T.radius.lg}px`,
          background: isLumi ? T.colors.surfaceUp : `oklch(62% 0.18 222 / 0.18)`,
          border: `1px solid ${isLumi ? T.colors.borderSub : 'oklch(62% 0.18 222 / 0.3)'}`,
          fontSize: T.font.md, color: T.colors.textPri, lineHeight:1.55,
        }}>
          {text}
          {citations && (
            <div style={{ display:'flex', flexWrap:'wrap', gap:5, marginTop:8 }}>
              {citations.map(c => (
                <span key={c} style={{ fontSize:T.font.xs, color:T.colors.accentBlue, background:'oklch(62% 0.18 222 / 0.12)', border:'1px solid oklch(62% 0.18 222 / 0.25)', padding:'2px 7px', borderRadius:T.radius.pill }}>{c}</span>
              ))}
            </div>
          )}
        </div>
        <div style={{ fontSize:T.font.xs, color:T.colors.textMuted, marginTop:3, textAlign: isLumi?'left':'right', paddingLeft: isLumi?2:0, paddingRight: isLumi?0:2 }}>{time}</div>
      </div>
    </div>
  );
}

// ── Chat Panel ────────────────────────────────────────────
const CHAT_MSGS = [
  { id:1, from:'lumi', text:'Hello! I\'m Lumi. How can I help you today?', time:'2:31 PM' },
  { id:2, from:'user', text:'Can you summarize my meeting notes from this morning?', time:'2:31 PM' },
  { id:3, from:'lumi', text:'Sure! Your 9 AM standup covered three key items: the auth bug in staging, the Q3 roadmap review, and the new onboarding flow design. Ryan is blocked on the API issue — he needs your input.', time:'2:32 PM', citations:['notes.md', 'calendar.json'] },
  { id:4, from:'user', text:'What\'s the auth bug status?', time:'2:33 PM' },
  { id:5, from:'lumi', text:'The bug is a JWT expiry edge case. Brandon opened PR #418 this morning — it\'s awaiting review. Staging should be unblocked once it merges.', time:'2:33 PM', citations:['notes.md'] },
];

function ChatPanel({ onClose }) {
  return (
    <div style={{
      width:380, height:540, display:'flex', flexDirection:'column',
      background: T.colors.surface,
      borderRadius:T.radius.xl, border:`1px solid ${T.colors.border}`,
      boxShadow: T.shadow.lg, overflow:'hidden',
    }}>
      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:`${T.space.md}px ${T.space.lg}px`, borderBottom:`1px solid ${T.colors.borderSub}`, flexShrink:0 }}>
        <div style={{ display:'flex', alignItems:'center', gap:10 }}>
          <div style={{ width:28, height:28, borderRadius:'50%', background:T.colors.surfaceTop, border:`1.5px solid ${T.colors.accentBlue}`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:12, boxShadow:T.shadow.glowBlue }}>✦</div>
          <div>
            <div style={{ fontSize:T.font.lg, fontWeight:600, color:T.colors.textPri, lineHeight:1 }}>Lumi</div>
            <div style={{ fontSize:T.font.xs, color:T.colors.accentGreen, marginTop:2 }}>● Online</div>
          </div>
        </div>
        <div onClick={onClose} style={{ width:26, height:26, borderRadius:T.radius.md, background:T.colors.surfaceTop, border:`1px solid ${T.colors.border}`, display:'flex', alignItems:'center', justifyContent:'center', cursor:'pointer', fontSize:12, color:T.colors.textSec }}>×</div>
      </div>

      {/* Messages */}
      <div style={{ flex:1, overflowY:'auto', padding:T.space.lg, display:'flex', flexDirection:'column', gap:T.space.lg }}>
        {CHAT_MSGS.map(m => <ChatBubble key={m.id} {...m}/>)}
        {/* Typing indicator */}
        <div style={{ display:'flex', alignItems:'flex-end', gap:10 }}>
          <div style={{ width:28, height:28, borderRadius:'50%', background:T.colors.surfaceTop, border:`1px solid ${T.colors.border}`, display:'flex', alignItems:'center', justifyContent:'center', fontSize:12 }}>✦</div>
          <div style={{ padding:`${T.space.sm}px ${T.space.md}px`, borderRadius:`${T.radius.lg}px ${T.radius.lg}px ${T.radius.lg}px 4px`, background:T.colors.surfaceUp, border:`1px solid ${T.colors.borderSub}`, display:'flex', gap:5, alignItems:'center' }}>
            {[0,1,2].map(i=>(
              <div key={i} style={{ width:5, height:5, borderRadius:'50%', background:T.colors.textMuted, animation:`lumiDot 1.2s ${i*0.2}s ease-in-out infinite` }}/>
            ))}
          </div>
        </div>
      </div>

      {/* Input */}
      <div style={{ padding:T.space.md, borderTop:`1px solid ${T.colors.borderSub}`, flexShrink:0 }}>
        <div style={{ display:'flex', gap:T.space.sm, alignItems:'center', background:T.colors.surfaceTop, borderRadius:T.radius.lg, border:`1px solid ${T.colors.border}`, padding:`${T.space.sm}px ${T.space.md}px` }}>
          <input readOnly placeholder="Ask Lumi something…" style={{ flex:1, background:'transparent', border:'none', outline:'none', fontSize:T.font.md, color:T.colors.textSec, fontFamily:'inherit' }}/>
          <div style={{ display:'flex', gap:T.space.sm, flexShrink:0 }}>
            <div style={{ width:28, height:28, borderRadius:T.radius.md, display:'flex', alignItems:'center', justifyContent:'center', cursor:'pointer', fontSize:14, color:T.colors.textMuted }}>🎙</div>
            <div style={{ width:28, height:28, borderRadius:T.radius.md, background:T.colors.accentBlue, display:'flex', alignItems:'center', justifyContent:'center', cursor:'pointer', fontSize:12 }}>
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M2 6h8M7 3l3 3-3 3" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/></svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Settings Panel ────────────────────────────────────────
const TABS = ['General','Voice','Model','Context','Privacy','Appearance','Advanced'];

function SettingsPanel({ onClose }) {
  const [tab, setTab] = React.useState(0);
  const [vals, setVals] = React.useState({
    autostart:true, notifications:false, trayIcon:true,
    voiceActivation:true, volume:72, micSensitivity:45,
    model:'lumi-2-local', temperature:0.68, maxTokens:2048,
    contextFiles:true, calendarAccess:false, clipboardWatch:true,
    telemetry:false, crashReports:true,
    theme:'Dark', fontSize:13, compactMode:false,
    devTools:false, logLevel:'warn', gpuAccel:true,
  });
  const set = (k,v) => setVals(p=>({...p,[k]:v}));

  const tabContent = [
    // General
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Launch at login" desc="Start Lumi when your computer starts" restart control={<LumiToggle checked={vals.autostart} onChange={v=>set('autostart',v)}/>}/>
      <SettingRow label="Desktop notifications" desc="Show system notifications for responses" control={<LumiToggle checked={vals.notifications} onChange={v=>set('notifications',v)}/>}/>
      <SettingRow label="System tray icon" desc="Show Lumi in the menu bar / taskbar tray" control={<LumiToggle checked={vals.trayIcon} onChange={v=>set('trayIcon',v)}/>}/>
      <SettingRow label="Language" control={<LumiDropdown value="English (US)" options={['English (US)','English (UK)','French','German','Japanese']}/>}/>
      <SettingRow label="Update channel" control={<LumiDropdown value="Stable" options={['Stable','Beta','Nightly']}/>}/>
      <SettingRow label="Data folder" desc="Where Lumi stores local data" control={<LumiFilePath value="/Users/alex/Library/Application Support/Lumi"/>}/>
    </div>,
    // Voice
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Wake word activation" desc='Respond to "Hey Lumi"' control={<LumiToggle checked={vals.voiceActivation} onChange={v=>set('voiceActivation',v)}/>}/>
      <SettingRow label="Output volume" control={<div style={{width:180}}><LumiSlider value={vals.volume} min={0} max={100} minLabel="0%" maxLabel="100%" onChange={v=>set('volume',v)}/></div>}/>
      <SettingRow label="Mic sensitivity" control={<div style={{width:180}}><LumiSlider value={vals.micSensitivity} min={0} max={100} minLabel="Low" maxLabel="High" onChange={v=>set('micSensitivity',v)}/></div>}/>
      <SettingRow label="Voice" control={<LumiDropdown value="Lumi Default" options={['Lumi Default','Aria','Echo','Nova']}/>}/>
      <SettingRow label="Microphone" control={<LumiDropdown value="Built-in Microphone" options={['Built-in Microphone','External Mic','Airpods Pro']}/>}/>
      <SettingRow label="Speaking rate" control={<div style={{width:180}}><LumiSlider value={1.0} min={0.5} max={2.0} minLabel="0.5×" maxLabel="2×"/></div>}/>
    </div>,
    // Model
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Active model" desc="The language model Lumi uses for responses" restart control={<LumiDropdown value="lumi-2-local" options={['lumi-2-local','lumi-2-cloud','lumi-1-fast','Custom']}/>}/>
      <SettingRow label="Temperature" desc="Higher = more creative, lower = more focused" control={<div style={{width:180}}><LumiSlider value={0.68} min={0} max={1} minLabel="Focused" maxLabel="Creative"/></div>}/>
      <SettingRow label="Max tokens" control={<LumiNumberInput value={2048} min={256} max={8192}/>}/>
      <SettingRow label="System prompt" desc="Custom instructions for Lumi's personality" control={<LumiInput value="" placeholder="You are a helpful assistant…"/>}/>
      <SettingRow label="Model path" desc="Path to local GGUF model file" restart control={<LumiFilePath value="/models/lumi-2-q4_K_M.gguf"/>}/>
      <SettingRow label="GPU acceleration" restart control={<LumiToggle checked={vals.gpuAccel} onChange={v=>set('gpuAccel',v)}/>}/>
    </div>,
    // Context
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Context sources" desc="What Lumi can read to understand your work" control={null}/>
      <div style={{ paddingLeft:T.space.md, paddingRight:T.space.md, display:'flex', flexDirection:'column', gap:T.space.sm }}>
        {[['Files & Folders',true],['Calendar',false],['Clipboard',true],['Browser history',false],['Notes.app',true]].map(([l,c])=>(
          <LumiCheckbox key={l} label={l} checked={c}/>
        ))}
      </div>
      <SettingRow label="Watched folder" control={<LumiFilePath value="/Users/alex/Documents/Work"/>}/>
      <SettingRow label="Context window size" control={<div style={{width:180}}><LumiSlider value={4096} min={512} max={16384} minLabel="512" maxLabel="16k"/></div>}/>
    </div>,
    // Privacy
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Send usage analytics" desc="Anonymous telemetry to improve Lumi" control={<LumiToggle checked={vals.telemetry} onChange={v=>set('telemetry',v)}/>}/>
      <SettingRow label="Send crash reports" control={<LumiToggle checked={vals.crashReports} onChange={v=>set('crashReports',v)}/>}/>
      <SettingRow label="Store conversation history" desc="Persists chats locally between sessions" control={<LumiToggle checked={true} onChange={()=>{}}/>}/>
      <SettingRow label="History retention" control={<LumiDropdown value="30 days" options={['7 days','30 days','90 days','Forever','Never']}/>}/>
      <SettingRow label="Encryption at rest" desc="Encrypt local data with your login password" restart control={<LumiToggle checked={false} onChange={()=>{}}/>}/>
      <div style={{ margin:`${T.space.sm}px ${T.space.md}px`, padding:T.space.md, borderRadius:T.radius.md, background:'oklch(60% 0.18 25 / 0.08)', border:`1px solid oklch(60% 0.18 25 / 0.25)` }}>
        <div style={{ fontSize:T.font.sm, color:T.colors.danger }}>Clear all conversation history — this cannot be undone.</div>
        <div style={{ marginTop:T.space.sm, padding:'6px 14px', borderRadius:T.radius.md, background:'oklch(60% 0.18 25 / 0.18)', border:`1px solid ${T.colors.danger}`, display:'inline-flex', cursor:'pointer', fontSize:T.font.sm, color:T.colors.danger }}>Clear History</div>
      </div>
    </div>,
    // Appearance
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Theme" control={<LumiDropdown value="Dark" options={['Dark','Darker','Midnight']}/>}/>
      <SettingRow label="Font size" control={<LumiNumberInput value={13} min={10} max={20}/>}/>
      <SettingRow label="Compact mode" desc="Reduce spacing and padding" control={<LumiToggle checked={vals.compactMode} onChange={v=>set('compactMode',v)}/>}/>
      <SettingRow label="Window opacity" desc="Overlay transparency when idle" control={<div style={{width:180}}><LumiSlider value={42} min={10} max={100} minLabel="10%" maxLabel="100%"/></div>}/>
      <SettingRow label="Accent color" control={
        <div style={{ display:'flex', gap:6 }}>
          {[T.colors.accentBlue,T.colors.accentGreen,T.colors.accentAmber,'oklch(65% 0.18 310)'].map((c,i)=>(
            <div key={i} style={{ width:18, height:18, borderRadius:'50%', background:c, border: i===0?`2px solid ${T.colors.textPri}`:'2px solid transparent', cursor:'pointer' }}/>
          ))}
        </div>
      }/>
      <SettingRow label="Avatar position" control={<LumiDropdown value="Bottom Left" options={['Bottom Left','Bottom Right','Top Left','Top Right']}/>}/>
    </div>,
    // Advanced
    <div style={{ display:'flex', flexDirection:'column', gap:2 }}>
      <SettingRow label="Developer tools" desc="Enable DevTools and verbose logging" control={<LumiToggle checked={vals.devTools} onChange={v=>set('devTools',v)}/>}/>
      <SettingRow label="Log level" control={<LumiDropdown value="warn" options={['error','warn','info','debug','verbose']}/>}/>
      <SettingRow label="GPU layers" desc="Number of model layers offloaded to GPU" restart control={<LumiNumberInput value={32} min={0} max={80}/>}/>
      <SettingRow label="Thread count" desc="CPU threads for inference" restart control={<LumiNumberInput value={8} min={1} max={32}/>}/>
      <SettingRow label="Extension folder" control={<LumiFilePath value="/Users/alex/.lumi/extensions"/>}/>
      <SettingRow label="Reset all settings" desc="Restore defaults — requires restart" restart control={<div style={{ padding:'6px 12px', borderRadius:T.radius.md, border:`1px solid ${T.colors.border}`, fontSize:T.font.sm, color:T.colors.textSec, cursor:'pointer' }}>Reset</div>}/>
    </div>,
  ];

  return (
    <div style={{
      width:680, height:540, display:'flex', flexDirection:'column',
      background:T.colors.surface, borderRadius:T.radius.xl,
      border:`1px solid ${T.colors.border}`, boxShadow:T.shadow.lg, overflow:'hidden',
    }}>
      {/* Header */}
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:`${T.space.md}px ${T.space.lg}px`, borderBottom:`1px solid ${T.colors.borderSub}`, flexShrink:0 }}>
        <span style={{ fontSize:T.font.lg, fontWeight:600, color:T.colors.textPri }}>Settings</span>
        <div onClick={onClose} style={{ width:26, height:26, borderRadius:T.radius.md, background:T.colors.surfaceTop, border:`1px solid ${T.colors.border}`, display:'flex', alignItems:'center', justifyContent:'center', cursor:'pointer', fontSize:12, color:T.colors.textSec }}>×</div>
      </div>

      <div style={{ display:'flex', flex:1, overflow:'hidden' }}>
        {/* Sidebar tabs */}
        <div style={{ width:140, borderRight:`1px solid ${T.colors.borderSub}`, padding:`${T.space.sm}px`, display:'flex', flexDirection:'column', gap:2, flexShrink:0 }}>
          {TABS.map((t,i)=>(
            <div key={t} onClick={()=>setTab(i)} style={{
              padding:`${T.space.sm}px ${T.space.md}px`, borderRadius:T.radius.md,
              background: tab===i ? T.colors.surfaceTop : 'transparent',
              border: tab===i ? `1px solid ${T.colors.border}` : '1px solid transparent',
              fontSize:T.font.md, color: tab===i ? T.colors.textPri : T.colors.textSec,
              cursor:'pointer', transition:'background 0.15s',
            }}>{t}</div>
          ))}
        </div>

        {/* Tab content */}
        <div style={{ flex:1, overflowY:'auto', padding:`${T.space.sm}px 0` }}>
          {tabContent[tab]}
        </div>
      </div>

      {/* Footer */}
      <div style={{ display:'flex', alignItems:'center', justifyContent:'flex-end', gap:T.space.sm, padding:`${T.space.md}px ${T.space.lg}px`, borderTop:`1px solid ${T.colors.borderSub}`, flexShrink:0 }}>
        <div style={{ fontSize:T.font.sm, color:T.colors.textMuted, flex:1 }}>↻ Requires restart</div>
        {['Cancel','Apply','Save'].map((l,i)=>(
          <div key={l} style={{
            padding:`7px ${T.space.lg}px`, borderRadius:T.radius.md, cursor:'pointer',
            fontSize:T.font.md, fontWeight: i===2 ? 600 : 400,
            background: i===2 ? T.colors.accentBlue : i===1 ? T.colors.surfaceTop : 'transparent',
            border: i===2 ? 'none' : `1px solid ${T.colors.border}`,
            color: i===2 ? 'white' : T.colors.textSec,
            boxShadow: i===2 ? T.shadow.glowBlue : 'none',
          }}>{l}</div>
        ))}
      </div>
    </div>
  );
}

Object.assign(window, { LumiAvatar, CompactOverlay, ChatPanel, SettingsPanel, STATES, TABS });
