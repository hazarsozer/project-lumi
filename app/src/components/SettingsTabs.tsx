// Settings tab content — one render function per tab
// Kept separate from SettingsPanel to keep both files under 200 lines.
import { useState } from 'react';
import { tokens as T } from '../styles/tokens';
import {
  LumiToggle, LumiSlider, LumiDropdown, LumiInput,
  LumiNumberInput, LumiFilePath, LumiCheckbox, SettingRow,
} from './LumiControls';

export interface SettingsValues {
  autostart: boolean;
  notifications: boolean;
  trayIcon: boolean;
  language: string;
  updateChannel: string;
  voiceActivation: boolean;
  volume: number;
  micSensitivity: number;
  voice: string;
  mic: string;
  speakingRate: number;
  model: string;
  temperature: number;
  maxTokens: number;
  systemPrompt: string;
  contextFiles: boolean;
  calendarAccess: boolean;
  clipboardWatch: boolean;
  contextWindowSize: number;
  telemetry: boolean;
  crashReports: boolean;
  theme: string;
  fontSize: number;
  compactMode: boolean;
  windowOpacity: number;
  devTools: boolean;
  logLevel: string;
  gpuAccel: boolean;
  gpuLayers: number;
  threadCount: number;
}

export const DEFAULT_SETTINGS: SettingsValues = {
  autostart: true, notifications: false, trayIcon: true,
  language: 'English (US)', updateChannel: 'Stable',
  voiceActivation: true, volume: 72, micSensitivity: 45,
  voice: 'Lumi Default', mic: 'Built-in Microphone', speakingRate: 1.0,
  model: 'lumi-2-local', temperature: 0.68, maxTokens: 2048,
  systemPrompt: '',
  contextFiles: true, calendarAccess: false, clipboardWatch: true,
  contextWindowSize: 4096,
  telemetry: false, crashReports: true,
  theme: 'Dark', fontSize: 13, compactMode: false, windowOpacity: 42,
  devTools: false, logLevel: 'warn', gpuAccel: true,
  gpuLayers: 32, threadCount: 8,
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
      <SettingRow label="Language" control={<LumiDropdown value={vals.language} options={['English (US)', 'English (UK)', 'French', 'German', 'Japanese']} onChange={(v) => set('language', v)} />} />
      <SettingRow label="Update channel" control={<LumiDropdown value={vals.updateChannel} options={['Stable', 'Beta', 'Nightly']} onChange={(v) => set('updateChannel', v)} />} />
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
      <SettingRow label="Voice" control={<LumiDropdown value={vals.voice} options={['Lumi Default', 'Aria', 'Echo', 'Nova']} onChange={(v) => set('voice', v)} />} />
      <SettingRow label="Microphone" control={<LumiDropdown value={vals.mic} options={['Built-in Microphone', 'External Mic', 'Airpods Pro']} onChange={(v) => set('mic', v)} />} />
      <SettingRow label="Speaking rate" control={<div style={{ width: 180 }}><LumiSlider value={vals.speakingRate} min={0.5} max={2.0} step={0.1} minLabel="0.5×" maxLabel="2×" onChange={(v) => set('speakingRate', v)} /></div>} />
    </Col>
  );
}

export function ModelTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Active model" desc="The language model Lumi uses for responses" restart control={<LumiDropdown value={vals.model} options={['lumi-2-local', 'lumi-2-cloud', 'lumi-1-fast', 'Custom']} onChange={(v) => set('model', v)} />} />
      <SettingRow label="Temperature" desc="Higher = more creative, lower = more focused" control={<div style={{ width: 180 }}><LumiSlider value={vals.temperature} min={0} max={1} step={0.01} minLabel="Focused" maxLabel="Creative" onChange={(v) => set('temperature', v)} /></div>} />
      <SettingRow label="Max tokens" control={<LumiNumberInput value={vals.maxTokens} min={256} max={8192} onChange={(v) => set('maxTokens', v)} />} />
      <SettingRow label="System prompt" desc="Custom instructions for Lumi's personality" control={<LumiInput value={vals.systemPrompt} placeholder="You are a helpful assistant…" onChange={(v) => set('systemPrompt', v)} />} />
      <SettingRow label="Model path" desc="Path to local GGUF model file" restart control={<LumiFilePath value="/models/lumi-2-q4_K_M.gguf" />} />
      <SettingRow label="GPU acceleration" restart control={<LumiToggle checked={vals.gpuAccel} onChange={(v) => set('gpuAccel', v)} />} />
    </Col>
  );
}

const INITIAL_CONTEXT_SOURCES: Array<[string, boolean]> = [
  ['Files & Folders', true], ['Calendar', false], ['Clipboard', true],
  ['Browser history', false], ['Notes.app', true],
];

export function ContextTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  const [sources, setSources] = useState<Array<[string, boolean]>>(INITIAL_CONTEXT_SOURCES);

  function toggleSource(label: string) {
    setSources((prev) => prev.map(([l, c]) => l === label ? [l, !c] : [l, c]));
  }

  return (
    <Col>
      <SettingRow label="Context sources" desc="What Lumi can read to understand your work" control={null} />
      <div style={{ paddingLeft: T.space.md, paddingRight: T.space.md, display: 'flex', flexDirection: 'column', gap: T.space.sm }}>
        {sources.map(([label, checked]) => (
          <LumiCheckbox key={label} label={label} checked={checked} onChange={() => toggleSource(label)} />
        ))}
      </div>
      <SettingRow label="Watched folder" control={<LumiFilePath value="/Users/alex/Documents/Work" />} />
      <SettingRow label="Context window size" control={<div style={{ width: 180 }}><LumiSlider value={vals.contextWindowSize} min={512} max={16384} minLabel="512" maxLabel="16k" onChange={(v) => set('contextWindowSize', v)} /></div>} />
    </Col>
  );
}

export function PrivacyTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  const [retention, setRetention] = useState('30 days');
  return (
    <Col>
      <SettingRow label="Send usage analytics" desc="Anonymous telemetry to improve Lumi" control={<LumiToggle checked={vals.telemetry} onChange={(v) => set('telemetry', v)} />} />
      <SettingRow label="Send crash reports" control={<LumiToggle checked={vals.crashReports} onChange={(v) => set('crashReports', v)} />} />
      {/* not yet wired to config schema */}
      <SettingRow label="Store conversation history" desc="Persists chats locally between sessions" control={<LumiToggle checked={true} onChange={() => {}} disabled />} />
      <SettingRow label="History retention" control={<LumiDropdown value={retention} options={['7 days', '30 days', '90 days', 'Forever', 'Never']} onChange={setRetention} />} />
      {/* not yet wired to config schema */}
      <SettingRow label="Encryption at rest" desc="Encrypt local data with your login password" restart control={<LumiToggle checked={false} onChange={() => {}} disabled />} />
      <div style={{ margin: `${T.space.sm}px ${T.space.md}px`, padding: T.space.md, borderRadius: T.radius.md, background: 'oklch(60% 0.18 25 / 0.08)', border: '1px solid oklch(60% 0.18 25 / 0.25)' }}>
        <div style={{ fontSize: T.font.sm, color: T.colors.danger }}>Clear all conversation history — this cannot be undone.</div>
        <div style={{ marginTop: T.space.sm, padding: '6px 14px', borderRadius: T.radius.md, background: 'oklch(60% 0.18 25 / 0.18)', border: `1px solid ${T.colors.danger}`, display: 'inline-flex', cursor: 'pointer', fontSize: T.font.sm, color: T.colors.danger }}>Clear History</div>
      </div>
    </Col>
  );
}

export function AppearanceTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  const [accentIdx, setAccentIdx] = useState(0);
  const [avatarPos, setAvatarPos] = useState('Bottom Left');
  const accentOptions = [T.colors.accentBlue, T.colors.accentGreen, T.colors.accentAmber, 'oklch(65% 0.18 310)'];
  return (
    <Col>
      <SettingRow label="Theme" control={<LumiDropdown value={vals.theme} options={['Dark', 'Darker', 'Midnight']} onChange={(v) => set('theme', v)} />} />
      <SettingRow label="Font size" control={<LumiNumberInput value={vals.fontSize} min={10} max={20} onChange={(v) => set('fontSize', v)} />} />
      <SettingRow label="Compact mode" desc="Reduce spacing and padding" control={<LumiToggle checked={vals.compactMode} onChange={(v) => set('compactMode', v)} />} />
      <SettingRow label="Window opacity" desc="Overlay transparency when idle" control={<div style={{ width: 180 }}><LumiSlider value={vals.windowOpacity} min={10} max={100} minLabel="10%" maxLabel="100%" onChange={(v) => set('windowOpacity', v)} /></div>} />
      <SettingRow label="Accent color" control={
        <div style={{ display: 'flex', gap: 6 }}>
          {accentOptions.map((c, i) => (
            <div key={i} onClick={() => setAccentIdx(i)} style={{ width: 18, height: 18, borderRadius: '50%', background: c, border: i === accentIdx ? `2px solid ${T.colors.textPri}` : '2px solid transparent', cursor: 'pointer' }} />
          ))}
        </div>
      } />
      <SettingRow label="Avatar position" control={<LumiDropdown value={avatarPos} options={['Bottom Left', 'Bottom Right', 'Top Left', 'Top Right']} onChange={setAvatarPos} />} />
    </Col>
  );
}

export function AdvancedTab({ vals, set }: { vals: SettingsValues; set: Setter }) {
  return (
    <Col>
      <SettingRow label="Developer tools" desc="Enable DevTools and verbose logging" control={<LumiToggle checked={vals.devTools} onChange={(v) => set('devTools', v)} />} />
      <SettingRow label="Log level" control={<LumiDropdown value={vals.logLevel} options={['error', 'warn', 'info', 'debug', 'verbose']} onChange={(v) => set('logLevel', v)} />} />
      <SettingRow label="GPU layers" desc="Number of model layers offloaded to GPU" restart control={<LumiNumberInput value={vals.gpuLayers} min={0} max={80} onChange={(v) => set('gpuLayers', v)} />} />
      <SettingRow label="Thread count" desc="CPU threads for inference" restart control={<LumiNumberInput value={vals.threadCount} min={1} max={32} onChange={(v) => set('threadCount', v)} />} />
      <SettingRow label="Extension folder" control={<LumiFilePath value="/Users/alex/.lumi/extensions" />} />
      <SettingRow label="Reset all settings" desc="Restore defaults — requires restart" restart control={<div style={{ padding: '6px 12px', borderRadius: T.radius.md, border: `1px solid ${T.colors.border}`, fontSize: T.font.sm, color: T.colors.textSec, cursor: 'pointer' }}>Reset</div>} />
    </Col>
  );
}
