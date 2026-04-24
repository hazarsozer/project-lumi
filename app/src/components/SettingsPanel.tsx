// SettingsPanel — 680×540 floating settings window with 7 sidebar tabs
import { useState } from 'react';
import { tokens as T } from '../styles/tokens';
import {
  SettingsValues, DEFAULT_SETTINGS,
  GeneralTab, VoiceTab, ModelTab, ContextTab,
  PrivacyTab, AppearanceTab, AdvancedTab,
} from './SettingsTabs';

const TABS = ['General', 'Voice', 'Model', 'Context', 'Privacy', 'Appearance', 'Advanced'] as const;
type TabName = typeof TABS[number];

export interface SettingsPanelProps {
  /** Config schema + current values received from the backend via config_schema event. */
  configSchema?: Record<string, unknown>;
  currentValues?: Record<string, unknown>;
  /** Called when a setting changes — emits a config_update IPC event. */
  onUpdate: (changes: Record<string, unknown>, persist: boolean) => void;
  onClose: () => void;
}

export function SettingsPanel({ onUpdate, onClose }: SettingsPanelProps) {
  const [activeTab, setActiveTab] = useState<TabName>('General');
  const [vals, setVals] = useState<SettingsValues>(DEFAULT_SETTINGS);

  function set<K extends keyof SettingsValues>(key: K, value: SettingsValues[K]) {
    setVals((prev) => ({ ...prev, [key]: value }));
  }

  function handleApply() {
    onUpdate(vals as unknown as Record<string, unknown>, false);
  }

  function handleSave() {
    onUpdate(vals as unknown as Record<string, unknown>, true);
    onClose();
  }

  const tabProps = { vals, set };

  const tabContent: Record<TabName, React.ReactNode> = {
    General:    <GeneralTab    {...tabProps} />,
    Voice:      <VoiceTab      {...tabProps} />,
    Model:      <ModelTab      {...tabProps} />,
    Context:    <ContextTab    {...tabProps} />,
    Privacy:    <PrivacyTab    {...tabProps} />,
    Appearance: <AppearanceTab {...tabProps} />,
    Advanced:   <AdvancedTab   {...tabProps} />,
  };

  return (
    <div style={{
      width: 680, height: 540, display: 'flex', flexDirection: 'column',
      background: T.colors.surface, borderRadius: T.radius.xl,
      border: `1px solid ${T.colors.border}`, boxShadow: T.shadow.lg, overflow: 'hidden',
      animation: 'fadeIn 280ms ease',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: `${T.space.md}px ${T.space.lg}px`, borderBottom: `1px solid ${T.colors.borderSub}`, flexShrink: 0 }}>
        <span style={{ fontSize: T.font.lg, fontWeight: 600, color: T.colors.textPri }}>Settings</span>
        <div onClick={onClose} style={{ width: 26, height: 26, borderRadius: T.radius.md, background: T.colors.surfaceTop, border: `1px solid ${T.colors.border}`, display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', fontSize: 12, color: T.colors.textSec }}>×</div>
      </div>

      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Sidebar */}
        <div style={{ width: 140, borderRight: `1px solid ${T.colors.borderSub}`, padding: T.space.sm, display: 'flex', flexDirection: 'column', gap: 2, flexShrink: 0 }}>
          {TABS.map((tab) => (
            <div
              key={tab}
              onClick={() => setActiveTab(tab)}
              style={{
                padding: `${T.space.sm}px ${T.space.md}px`, borderRadius: T.radius.md,
                background: activeTab === tab ? T.colors.surfaceTop : 'transparent',
                border: activeTab === tab ? `1px solid ${T.colors.border}` : '1px solid transparent',
                fontSize: T.font.md, color: activeTab === tab ? T.colors.textPri : T.colors.textSec,
                cursor: 'pointer', transition: 'background 0.15s',
              }}
            >
              {tab}
            </div>
          ))}
        </div>

        {/* Tab content */}
        <div style={{ flex: 1, overflowY: 'auto', padding: `${T.space.sm}px 0` }}>
          {tabContent[activeTab]}
        </div>
      </div>

      {/* Footer */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: T.space.sm, padding: `${T.space.md}px ${T.space.lg}px`, borderTop: `1px solid ${T.colors.borderSub}`, flexShrink: 0 }}>
        <div style={{ fontSize: T.font.sm, color: T.colors.textMuted, flex: 1 }}>↻ Requires restart</div>
        <FooterButton label="Cancel" onClick={onClose} variant="ghost" />
        <FooterButton label="Apply"  onClick={handleApply} variant="secondary" />
        <FooterButton label="Save"   onClick={handleSave}  variant="primary" />
      </div>
    </div>
  );
}

// ── Footer button ───────────────────────────────────────────

interface FooterButtonProps {
  label: string;
  onClick: () => void;
  variant: 'ghost' | 'secondary' | 'primary';
}

function FooterButton({ label, onClick, variant }: FooterButtonProps) {
  const isPrimary   = variant === 'primary';
  const isSecondary = variant === 'secondary';
  return (
    <div
      onClick={onClick}
      style={{
        padding: `7px ${T.space.lg}px`, borderRadius: T.radius.md, cursor: 'pointer',
        fontSize: T.font.md, fontWeight: isPrimary ? 600 : 400,
        background: isPrimary ? T.colors.accentBlue : isSecondary ? T.colors.surfaceTop : 'transparent',
        border: isPrimary ? 'none' : `1px solid ${T.colors.border}`,
        color: isPrimary ? 'white' : T.colors.textSec,
        boxShadow: isPrimary ? T.shadow.glowBlue : 'none',
      }}
    >
      {label}
    </div>
  );
}
