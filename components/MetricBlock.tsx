'use client';

import { useTheme } from '@/contexts/ThemeContext';
import { CSSProperties } from 'react';

type Size = 'xs' | 'sm' | 'md' | 'lg' | 'xl';

type Props = {
  label: string;
  value: string;
  valueColor?: string;
  size?: Size;
  align?: 'left' | 'right' | 'center';
  style?: CSSProperties;
};

const VALUE_SIZES: Record<Size, string> = {
  xs: '12px',
  sm: '15px',
  md: '20px',
  lg: '28px',
  xl: '38px',
};

const LABEL_SIZES: Record<Size, string> = {
  xs: '9px',
  sm: '10px',
  md: '11px',
  lg: '11px',
  xl: '12px',
};

export function MetricBlock({
  label,
  value,
  valueColor,
  size = 'md',
  align = 'left',
  style,
}: Props) {
  const { theme } = useTheme();

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '4px',
        textAlign: align,
        ...style,
      }}
    >
      <span
        style={{
          fontSize: LABEL_SIZES[size],
          color: theme.textMuted,
          textTransform: 'uppercase',
          letterSpacing: '0.07em',
          fontWeight: 500,
          lineHeight: 1,
        }}
      >
        {label}
      </span>
      <span
        style={{
          fontSize: VALUE_SIZES[size],
          color: valueColor ?? theme.textPrimary,
          fontWeight: 700,
          lineHeight: 1.15,
          letterSpacing: size === 'xl' || size === 'lg' ? '-0.02em' : '0',
        }}
      >
        {value}
      </span>
    </div>
  );
}
