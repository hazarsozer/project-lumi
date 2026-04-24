// Settings tab content — one render function per tab
// Kept separate from SettingsPanel to keep both files under 200 lines.
import { tokens as T } from '../styles/tokens';
import {
  LumiToggle, LumiSlider, LumiDropdown, LumiInput,
  LumiNumberInput, LumiFilePath, LumiCheckbox, SettingRow,
} from './LumiControls';

export interface SettingsValues {
  autostart: boolean;
  notifications: boolean;
  trayIcon: boolean;
  voiceActivation: boolean;
  volume: number;
  micSensitivity: number;
  model: string;
  temperature: number;
  maxTokens: number;
  contextFiles: boolean;
  calendarAccess: boolean;
  clipboardWatch: boolean;
  telemetry: boolean;
  crashReports: boolean;
  theme: string;
  fontSize: number;
  compactMode: boolean;
  devTools: boolean;
  logLevel: string;
  gpuAccel: boolean;
}

export const DEFAULT_SETTINGS: SettingsValues = {
  autostart: true, notifications: false, trayIcon: true,
  voiceActivation: true, volume: 72, micSensitivity: 45,
  model: 'lumi-2-local', temperature: 0.68, maxTokens: 2048,
  contextFiles: true, calendarAccess: false, clipboardWatch: true,
  telemetry: false, crashReports: true,
  theme: 'Dark', fontSize: 13, compactMode: false,
  devTools: false, logLevel: 'warn', gpuAccel: true,
};

type Setter = <K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) => void;

function Col({ children }: { children: React.ReactNode }) {
  return <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>{children}</div>;
}

export function GeneralTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Launch at login" desc="Start Lumi when your computer starts" restart control={<LumiToggle checked={vals.autostart} onChange={(v) => set('autostart', v)} />} />
      <SettingRow label="Desktop notifications" desc="Show system notifications for responses" control={<LumiToggle checked={vals.notifications} onChange={(v) => set('notifications', v)} />} />
      <SettingRow label="System tray icon" desc="Show Lumi in the menu bar / taskbar tray" control={<LumiToggle checked={vals.trayIcon} onChange={(v) => set('trayIcon', v)} />} />
      <SettingRow label="Language" control={<LumiDropdown value="English (US)" options={['English (US)', 'English (UK)', 'French', 'German', 'Japanese']} />} />
      <SettingRow label="Update channel" control={<LumiDropdown value="Stable" options={['Stable', 'Beta', 'Nightly']} />} />
      <SettingRow label="Data folder" desc="Where Lumi stores local data" control={<LumiFilePath value="/Users/alex/Library/Application Support/Lumi" />} />
    </Col>
  );
}

export function VoiceTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Wake word activation" desc='Respond to "Hey Lumi"' control={<LumiToggle checked={vals.voiceActivation} onChange={(v) => set('voiceActivation', v)} />} />
      <SettingRow label="Output volume" control={<div style={{ width: 180 }}><LumiSlider value={vals.volume} min={0} max={100} minLabel="0%" maxLabel="100%" onChange={(v) => set('volume', v)} /></div>} />
      <SettingRow label="Mic sensitivity" control={<div style={{ width: 180 }}><LumiSlider value={vals.micSensitivity} min={0} max={100} minLabel="Low" maxLabel="High" onChange={(v) => set('micSensitivity', v)} /></div>} />
      <SettingRow label="Voice" control={<LumiDropdown value="Lumi Default" options={['Lumi Default', 'Aria', 'Echo', 'Nova']} />} />
      <SettingRow label="Microphone" control={<LumiDropdown value="Built-in Microphone" options={['Built-in Microphone', 'External Mic', 'Airpods Pro']} />} />
      <SettingRow label="Speaking rate" control={<div style={{ width: 180 }}><LumiSlider value={1.0} min={0.5} max={2.0} minLabel="0.5×" maxLabel="2×" /></div>} />
    </Col>
  );
}

export function ModelTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Active model" desc="The language model Lumi uses for responses" restart control={<LumiDropdown value={vals.model} options={['lumi-2-local', 'lumi-2-cloud', 'lumi-1-fast', 'Custom']} />} />
      <SettingRow label="Temperature" desc="Higher = more creative, lower = more focused" control={<div style={{ width: 180 }}><LumiSlider value={vals.temperature} min={0} max={1} minLabel="Focused" maxLabel="Creative" onChange={(v) => set('temperature', v)} /></div>} />
      <SettingRow label="Max tokens" control={<LumiNumberInput value={vals.maxTokens} min={256} max={8192} />} />
      <SettingRow label="System prompt" desc="Custom instructions for Lumi's personality" control={<LumiInput value="" placeholder="You are a helpful assistant…" />} />
      <SettingRow label="Model path" desc="Path to local GGUF model file" restart control={<LumiFilePath value="/models/lumi-2-q4_K_M.gguf" />} />
      <SettingRow label="GPU acceleration" restart control={<LumiToggle checked={vals.gpuAccel} onChange={(v) => set('gpuAccel', v)} />} />
    </Col>
  );
}

const CONTEXT_SOURCES: Array<[string, boolean]> = [
  ['Files & Folders', true], ['Calendar', false], ['Clipboard', true],
  ['Browser history', false], ['Notes.app', true],
];

export function ContextTab(_: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Context sources" desc="What Lumi can read to understand your work" control={null} />
      <div style={{ paddingLeft: T.space.md, paddingRight: T.space.md, display: 'flex', flexDirection: 'column', gap: T.space.sm }}>
        {CONTEXT_SOURCES.map(([label, checked]) => (
          <LumiCheckbox key={label} label={label} checked={checked} />
        ))}
      </div>
      <SettingRow label="Watched folder" control={<LumiFilePath value="/Users/alex/Documents/Work" />} />
      <SettingRow label="Context window size" control={<div style={{ width: 180 }}><LumiSlider value={4096} min={512} max={16384} minLabel="512" maxLabel="16k" /></div>} />
    </Col>
  );
}

export function PrivacyTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Send usage analytics" desc="Anonymous telemetry to improve Lumi" control={<LumiToggle checked={vals.telemetry} onChange={(v) => set('telemetry', v)} />} />
      <SettingRow label="Send crash reports" control={<LumiToggle checked={vals.crashReports} onChange={(v) => set('crashReports', v)} />} />
      <SettingRow label="Store conversation history" desc="Persists chats locally between sessions" control={<LumiToggle checked={true} onChange={() => {}} />} />
      <SettingRow label="History retention" control={<LumiDropdown value="30 days" options={['7 days', '30 days', '90 days', 'Forever', 'Never']} />} />
      <SettingRow label="Encryption at rest" desc="Encrypt local data with your login password" restart control={<LumiToggle checked={false} onChange={() => {}} />} />
      <div style={{ margin: `${T.space.sm}px ${T.space.md}px`, padding: T.space.md, borderRadius: T.radius.md, background: 'oklch(60% 0.18 25 / 0.08)', border: '1px solid oklch(60% 0.18 25 / 0.25)' }}>
        <div style={{ fontSize: T.font.sm, color: T.colors.danger }}>Clear all conversation history — this cannot be undone.</div>
        <div style={{ marginTop: T.space.sm, padding: '6px 14px', borderRadius: T.radius.md, background: 'oklch(60% 0.18 25 / 0.18)', border: `1px solid ${T.colors.danger}`, display: 'inline-flex', cursor: 'pointer', fontSize: T.font.sm, color: T.colors.danger }}>Clear History</div>
      </div>
    </Col>
  );
}

export function AppearanceTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  const accentOptions = [T.colors.accentBlue, T.colors.accentGreen, T.colors.accentAmber, 'oklch(65% 0.18 310)'];
  return (
    <Col>
      <SettingRow label="Theme" control={<LumiDropdown value={vals.theme} options={['Dark', 'Darker', 'Midnight']} />} />
      <SettingRow label="Font size" control={<LumiNumberInput value={vals.fontSize} min={10} max={20} />} />
      <SettingRow label="Compact mode" desc="Reduce spacing and padding" control={<LumiToggle checked={vals.compactMode} onChange={(v) => set('compactMode', v)} />} />
      <SettingRow label="Window opacity" desc="Overlay transparency when idle" control={<div style={{ width: 180 }}><LumiSlider value={42} min={10} max={100} minLabel="10%" maxLabel="100%" /></div>} />
      <SettingRow label="Accent color" control={
        <div style={{ display: 'flex', gap: 6 }}>
          {accentOptions.map((c, i) => (
            <div key={i} style={{ width: 18, height: 18, borderRadius: '50%', background: c, border: i === 0 ? `2px solid ${T.colors.textPri}` : '2px solid transparent', cursor: 'pointer' }} />
          ))}
        </div>
      } />
      <SettingRow label="Avatar position" control={<LumiDropdown value="Bottom Left" options={['Bottom Left', 'Bottom Right', 'Top Left', 'Top Right']} />} />
    </Col>
  );
}

export function AdvancedTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Developer tools" desc="Enable DevTools and verbose logging" control={<LumiToggle checked={vals.devTools} onChange={(v) => set('devTools', v)} />} />
      <SettingRow label="Log level" control={<LumiDropdown value={vals.logLevel} options={['error', 'warn', 'info', 'debug', 'verbose']} />} />
      <SettingRow label="GPU layers" desc="Number of model layers offloaded to GPU" restart control={<LumiNumberInput value={32} min={0} max={80} />} />
      <SettingRow label="Thread count" desc="CPU threads for inference" restart control={<LumiNumberInput value={8} min={1} max={32} />} />
      <SettingRow label="Extension folder" control={<LumiFilePath value="/Users/alex/.lumi/extensions" />} />
      <SettingRow label="Reset all settings" desc="Restore defaults — requires restart" restart control={<div style={{ padding: '6px 12px', borderRadius: T.radius.md, border: `1px solid ${T.colors.border}`, fontSize: T.font.sm, color: T.colors.textSec, cursor: 'pointer' }}>Reset</div>} />
    </Col>
  );
}
