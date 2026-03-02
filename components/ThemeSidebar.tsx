'use client';

import { useTheme } from '@/contexts/ThemeContext';
import { Theme, presets } from '@/lib/theme';

type ColorFieldKey = keyof Omit<Theme, 'radius'>;

function SectionTitle({ children }: { children: string }) {
  return (
    <div
      style={{
        fontSize: '9px',
        color: '#475569',
        textTransform: 'uppercase',
        letterSpacing: '0.12em',
        fontWeight: 700,
        marginBottom: '2px',
      }}
    >
      {children}
    </div>
  );
}

function Divider() {
  return <div style={{ height: '1px', backgroundColor: '#1A2035' }} />;
}

function ColorRow({ label, field }: { label: string; field: ColorFieldKey }) {
  const { theme, updateTheme } = useTheme();
  const value = theme[field] as string;

  return (
    <div
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '8px',
      }}
    >
      <span
        style={{
          fontSize: '11px',
          color: '#94A3B8',
          letterSpacing: '0.01em',
          flex: 1,
        }}
      >
        {label}
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
        <span
          style={{
            fontSize: '10px',
            color: '#475569',
            fontFamily: 'monospace',
          }}
        >
          {value}
        </span>
        <input
          type="color"
          value={value}
          onChange={(e) => updateTheme({ [field]: e.target.value })}
          style={{
            width: '28px',
            height: '24px',
            padding: '2px',
            border: '1px solid #1E2739',
            borderRadius: '4px',
            backgroundColor: '#0D1117',
            cursor: 'pointer',
            appearance: 'none',
            WebkitAppearance: 'none',
          }}
        />
      </div>
    </div>
  );
}

export function ThemeSidebar() {
  const { theme, updateTheme, setTheme } = useTheme();

  return (
    <div
      style={{
        width: '216px',
        height: '100vh',
        backgroundColor: '#0D1117',
        borderRight: '1px solid #1A2035',
        display: 'flex',
        flexDirection: 'column',
        padding: '18px 14px',
        gap: '16px',
        overflowY: 'auto',
        flexShrink: 0,
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      {/* Header */}
      <div>
        <div
          style={{
            fontSize: '14px',
            fontWeight: 700,
            color: '#E2E8F0',
            letterSpacing: '-0.01em',
            marginBottom: '2px',
          }}
        >
          IG Finance
        </div>
        <div style={{ fontSize: '11px', color: '#334155' }}>
          Post Builder — Part 1
        </div>
      </div>

      <Divider />

      {/* Colors */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <SectionTitle>Canvas Colors</SectionTitle>
        <ColorRow label="Background" field="background" />
        <ColorRow label="Panel" field="panel" />
        <ColorRow label="Divider" field="divider" />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <SectionTitle>Accent Colors</SectionTitle>
        <ColorRow label="Bull (Positive)" field="bull" />
        <ColorRow label="Bear (Negative)" field="bear" />
        <ColorRow label="Text Primary" field="textPrimary" />
        <ColorRow label="Text Muted" field="textMuted" />
      </div>

      <Divider />

      {/* Border radius */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <SectionTitle>Border Radius</SectionTitle>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <span style={{ fontSize: '11px', color: '#94A3B8' }}>Radius</span>
          <span
            style={{
              fontSize: '11px',
              color: '#64748B',
              fontFamily: 'monospace',
            }}
          >
            {theme.radius}px
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={20}
          step={1}
          value={theme.radius}
          onChange={(e) => updateTheme({ radius: parseInt(e.target.value) })}
          style={{
            width: '100%',
            accentColor: '#22D3A5',
            cursor: 'pointer',
          }}
        />
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span style={{ fontSize: '9px', color: '#334155' }}>Sharp</span>
          <span style={{ fontSize: '9px', color: '#334155' }}>Round</span>
        </div>
      </div>

      <Divider />

      {/* Presets */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        <SectionTitle>Presets</SectionTitle>
        {presets.map((preset) => (
          <button
            key={preset.name}
            onClick={() => setTheme(preset.theme)}
            style={{
              padding: '8px 10px',
              backgroundColor: '#111827',
              border: '1px solid #1E2739',
              borderRadius: '6px',
              color: '#94A3B8',
              fontSize: '11px',
              cursor: 'pointer',
              textAlign: 'left',
              fontWeight: 500,
              fontFamily: 'inherit',
              transition: 'all 0.15s',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = '#22D3A5';
              e.currentTarget.style.color = '#E2E8F0';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#1E2739';
              e.currentTarget.style.color = '#94A3B8';
            }}
          >
            {preset.name}
          </button>
        ))}
      </div>

      {/* Footer note */}
      <div style={{ marginTop: 'auto', paddingTop: '8px' }}>
        <div
          style={{
            fontSize: '9px',
            color: '#1E293B',
            lineHeight: 1.5,
          }}
        >
          Changes apply live to the preview canvas.
        </div>
      </div>
    </div>
  );
}
