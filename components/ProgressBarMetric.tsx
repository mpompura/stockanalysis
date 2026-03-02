'use client';

import { useTheme } from '@/contexts/ThemeContext';

type Props = {
  label: string;
  /** Normalised 0–100 fill level */
  fillPct: number;
  displayValue: string;
  color?: string;
};

export function ProgressBarMetric({ label, fillPct, displayValue, color }: Props) {
  const { theme } = useTheme();
  const clamped = Math.min(Math.max(fillPct, 0), 100);
  const barColor = color ?? theme.bull;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span
          style={{
            fontSize: '10px',
            color: theme.textMuted,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            fontWeight: 500,
          }}
        >
          {label}
        </span>
        <span
          style={{
            fontSize: '11px',
            color: barColor,
            fontWeight: 600,
            fontFamily: 'monospace',
          }}
        >
          {displayValue}
        </span>
      </div>
      <div
        style={{
          height: '3px',
          backgroundColor: theme.divider,
          borderRadius: '2px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            width: `${clamped}%`,
            height: '100%',
            backgroundColor: barColor,
            borderRadius: '2px',
          }}
        />
      </div>
    </div>
  );
}
