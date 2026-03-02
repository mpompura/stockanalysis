'use client';

import { useTheme } from '@/contexts/ThemeContext';
import { ReactNode, CSSProperties } from 'react';

type Props = {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
};

export function CardShell({ children, className = '', style }: Props) {
  const { theme } = useTheme();

  return (
    <div
      className={className}
      style={{
        backgroundColor: theme.panel,
        borderRadius: theme.radius,
        border: `1px solid ${theme.divider}`,
        overflow: 'hidden',
        ...style,
      }}
    >
      {children}
    </div>
  );
}
