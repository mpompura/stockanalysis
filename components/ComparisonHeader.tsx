'use client';

import { useTheme } from '@/contexts/ThemeContext';
import { TickerData } from '@/lib/data';
import { MetricBlock } from './MetricBlock';
import {
  formatMoneyAbbrev,
  formatPct,
  formatSignedPct,
} from '@/lib/utils';

type Props = {
  left: TickerData;
  right: TickerData;
};

export function ComparisonHeader({ left, right }: Props) {
  const { theme } = useTheme();

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1px 1fr',
        height: '100%',
        gap: 0,
      }}
    >
      <TickerPanel data={left} side="left" />

      {/* Vertical divider */}
      <div style={{ backgroundColor: theme.divider, margin: '8px 0' }} />

      <TickerPanel data={right} side="right" />
    </div>
  );
}

function TickerPanel({
  data,
  side,
}: {
  data: TickerData;
  side: 'left' | 'right';
}) {
  const { theme } = useTheme();
  const isLeft = side === 'left';
  const textAlign = isLeft ? 'right' : 'left';
  const padding = isLeft ? '0 28px 0 0' : '0 0 0 28px';
  const accentColor = isLeft ? theme.bull : theme.bear;

  return (
    <div
      style={{
        padding,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        height: '100%',
      }}
    >
      {/* Ticker identity */}
      <div style={{ textAlign }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            justifyContent: isLeft ? 'flex-end' : 'flex-start',
            marginBottom: '2px',
          }}
        >
          <div
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              backgroundColor: accentColor,
              order: isLeft ? 1 : 0,
            }}
          />
          <div
            style={{
              fontSize: '34px',
              fontWeight: 800,
              color: theme.textPrimary,
              letterSpacing: '-0.025em',
              lineHeight: 1,
            }}
          >
            {data.ticker}
          </div>
        </div>
        <div
          style={{
            fontSize: '11px',
            color: theme.textMuted,
            letterSpacing: '0.01em',
          }}
        >
          {data.name}
        </div>
      </div>

      {/* CAGR highlight */}
      <div style={{ textAlign }}>
        <div
          style={{
            fontSize: '10px',
            color: theme.textMuted,
            textTransform: 'uppercase',
            letterSpacing: '0.1em',
            fontWeight: 500,
            marginBottom: '2px',
          }}
        >
          5Y Revenue CAGR
        </div>
        <div
          style={{
            fontSize: '46px',
            fontWeight: 800,
            color: accentColor,
            lineHeight: 1,
            letterSpacing: '-0.025em',
          }}
        >
          {formatSignedPct(data.cagr5Y, 1)}
        </div>
      </div>

      {/* Key metrics */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '10px 16px',
        }}
      >
        {[
          { label: 'Market Cap', value: formatMoneyAbbrev(data.marketCap) },
          { label: 'TTM Revenue', value: formatMoneyAbbrev(data.ttmRevenue) },
          { label: 'TTM FCF', value: formatMoneyAbbrev(data.ttmFCF) },
          {
            label: 'FCF Margin',
            value: formatPct(data.fcfMargin),
            color: accentColor,
          },
        ].map((m, i) => (
          <MetricBlock
            key={i}
            label={m.label}
            value={m.value}
            valueColor={m.color}
            size="sm"
            align={
              isLeft ? (i % 2 === 0 ? 'right' : 'left') : i % 2 === 0 ? 'right' : 'left'
            }
          />
        ))}
      </div>
    </div>
  );
}
