'use client';

import { useTheme } from '@/contexts/ThemeContext';
import { mockLeft, mockRight } from '@/lib/data';
import { ComparisonHeader } from './ComparisonHeader';
import { LineChartCard } from './LineChartCard';
import { BarChartCard } from './BarChartCard';
import { CardShell } from './CardShell';
import { ProgressBarMetric } from './ProgressBarMetric';
import { MetricBlock } from './MetricBlock';
import { formatPct, formatSignedPct } from '@/lib/utils';

// ─── Layout constants ───────────────────────────────────────────────
const W = 1080;
const H = 1350;
const PAD = 26;
const GAP = 12;

// ─── Section label strip ─────────────────────────────────────────────
function SectionLabel({ children }: { children: string }) {
  const { theme } = useTheme();
  return (
    <div
      style={{
        fontSize: '9px',
        textTransform: 'uppercase',
        letterSpacing: '0.14em',
        color: theme.textMuted,
        fontWeight: 600,
        opacity: 0.6,
        marginBottom: '4px',
      }}
    >
      {children}
    </div>
  );
}

// ─── Section wrapper ─────────────────────────────────────────────────
function Section({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return <div style={{ display: 'flex', flexDirection: 'column', ...style }}>{children}</div>;
}

// ─── Card title ───────────────────────────────────────────────────────
function CardTitle({ children }: { children: string }) {
  const { theme } = useTheme();
  return (
    <div
      style={{
        fontSize: '10px',
        color: theme.textMuted,
        textTransform: 'uppercase',
        letterSpacing: '0.09em',
        fontWeight: 600,
        marginBottom: '10px',
        flexShrink: 0,
      }}
    >
      {children}
    </div>
  );
}

// ─── Inline divider ───────────────────────────────────────────────────
function InlineDivider() {
  const { theme } = useTheme();
  return <div style={{ height: '1px', backgroundColor: theme.divider, margin: '10px 0' }} />;
}

// ─── Ticker badge ─────────────────────────────────────────────────────
function TickerBadge({ ticker, color }: { ticker: string; color: string }) {
  return (
    <div
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '5px',
        marginBottom: '8px',
      }}
    >
      <div
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: color,
        }}
      />
      <span
        style={{
          fontSize: '10px',
          color,
          fontWeight: 700,
          letterSpacing: '0.08em',
        }}
      >
        {ticker}
      </span>
    </div>
  );
}

// ─── Reinvestment card ────────────────────────────────────────────────
function ReinvestmentCard() {
  const { theme } = useTheme();
  return (
    <CardShell
      style={{
        padding: '14px',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      <CardTitle>Reinvestment vs Cash</CardTitle>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        {/* NVDA */}
        <div>
          <TickerBadge ticker={mockLeft.ticker} color={theme.bull} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            <ProgressBarMetric
              label="CapEx / Revenue"
              fillPct={(mockLeft.capexPct / 20) * 100}
              displayValue={formatPct(mockLeft.capexPct)}
              color={theme.textMuted}
            />
            <ProgressBarMetric
              label="FCF Margin"
              fillPct={mockLeft.fcfMargin}
              displayValue={formatPct(mockLeft.fcfMargin)}
              color={theme.bull}
            />
          </div>
        </div>

        <InlineDivider />

        {/* MSFT */}
        <div>
          <TickerBadge ticker={mockRight.ticker} color={theme.bear} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            <ProgressBarMetric
              label="CapEx / Revenue"
              fillPct={(mockRight.capexPct / 20) * 100}
              displayValue={formatPct(mockRight.capexPct)}
              color={theme.textMuted}
            />
            <ProgressBarMetric
              label="FCF Margin"
              fillPct={mockRight.fcfMargin}
              displayValue={formatPct(mockRight.fcfMargin)}
              color={theme.bear}
            />
          </div>
        </div>
      </div>
    </CardShell>
  );
}

// ─── Profitability card ───────────────────────────────────────────────
function ProfitabilityCard() {
  const { theme } = useTheme();
  return (
    <CardShell
      style={{
        padding: '14px',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      <CardTitle>Profitability</CardTitle>

      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
        {/* NVDA */}
        <div>
          <TickerBadge ticker={mockLeft.ticker} color={theme.bull} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            <ProgressBarMetric
              label="Gross Margin"
              fillPct={mockLeft.grossMargin}
              displayValue={formatPct(mockLeft.grossMargin)}
              color={theme.bull}
            />
            <ProgressBarMetric
              label="Op. Margin"
              fillPct={mockLeft.opMargin}
              displayValue={formatPct(mockLeft.opMargin)}
              color={theme.bull}
            />
            <ProgressBarMetric
              label="Net Margin"
              fillPct={mockLeft.netMargin}
              displayValue={formatPct(mockLeft.netMargin)}
              color={theme.bull}
            />
          </div>
        </div>

        <InlineDivider />

        {/* MSFT */}
        <div>
          <TickerBadge ticker={mockRight.ticker} color={theme.bear} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            <ProgressBarMetric
              label="Gross Margin"
              fillPct={mockRight.grossMargin}
              displayValue={formatPct(mockRight.grossMargin)}
              color={theme.bear}
            />
            <ProgressBarMetric
              label="Op. Margin"
              fillPct={mockRight.opMargin}
              displayValue={formatPct(mockRight.opMargin)}
              color={theme.bear}
            />
            <ProgressBarMetric
              label="Net Margin"
              fillPct={mockRight.netMargin}
              displayValue={formatPct(mockRight.netMargin)}
              color={theme.bear}
            />
          </div>
        </div>
      </div>
    </CardShell>
  );
}

// ─── Risk card ────────────────────────────────────────────────────────
function RiskCard() {
  const { theme } = useTheme();

  const riskItems = (ticker: string, color: string, beta: number, dte: number, ic: number) => (
    <div>
      <TickerBadge ticker={ticker} color={color} />
      <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
        <ProgressBarMetric
          label="Beta"
          fillPct={Math.min((beta / 3) * 100, 100)}
          displayValue={`${beta.toFixed(1)}x`}
          color={beta > 1.5 ? theme.bear : theme.bull}
        />
        <ProgressBarMetric
          label="Debt / Equity"
          fillPct={Math.min((dte / 2) * 100, 100)}
          displayValue={`${dte.toFixed(1)}x`}
          color={dte > 1 ? theme.bear : theme.bull}
        />
        <ProgressBarMetric
          label="Interest Coverage"
          fillPct={Math.min((ic / 100) * 100, 100)}
          displayValue={`${ic}x`}
          color={theme.bull}
        />
      </div>
    </div>
  );

  return (
    <CardShell
      style={{
        padding: '14px',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
      }}
    >
      <CardTitle>Risk Profile</CardTitle>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', flex: 1 }}>
        {riskItems(mockLeft.ticker, theme.bull, mockLeft.beta, mockLeft.debtToEquity, mockLeft.interestCoverage)}
        {riskItems(mockRight.ticker, theme.bear, mockRight.beta, mockRight.debtToEquity, mockRight.interestCoverage)}
      </div>
    </CardShell>
  );
}

// ─── Dashboard ────────────────────────────────────────────────────────
export function Dashboard() {
  const { theme } = useTheme();

  return (
    <div
      style={{
        width: `${W}px`,
        height: `${H}px`,
        backgroundColor: theme.background,
        display: 'grid',
        gridTemplateRows: '30px 248px 248px 252px 224px 26px',
        padding: `${PAD}px`,
        gap: `${GAP}px`,
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        overflow: 'hidden',
        boxSizing: 'border-box',
      }}
    >
      {/* ── Brand header ─────────────────────────────────── */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div
            style={{
              width: '18px',
              height: '18px',
              borderRadius: '4px',
              background: `linear-gradient(135deg, ${theme.bull}, ${theme.bear})`,
            }}
          />
          <span
            style={{
              fontSize: '12px',
              fontWeight: 700,
              color: theme.textPrimary,
              letterSpacing: '-0.01em',
            }}
          >
            IG Finance Post Builder
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span
            style={{
              fontSize: '9px',
              color: theme.textMuted,
              letterSpacing: '0.12em',
              textTransform: 'uppercase',
              opacity: 0.7,
            }}
          >
            Stock Comparison
          </span>
          <div
            style={{
              width: '5px',
              height: '5px',
              borderRadius: '50%',
              backgroundColor: theme.bull,
            }}
          />
          <div
            style={{
              width: '5px',
              height: '5px',
              borderRadius: '50%',
              backgroundColor: theme.bear,
            }}
          />
        </div>
      </div>

      {/* ── Section 1: Ticker comparison header ──────────── */}
      <CardShell style={{ padding: '20px 22px', overflow: 'hidden' }}>
        <ComparisonHeader left={mockLeft} right={mockRight} />
      </CardShell>

      {/* ── Section 2: Revenue & Op Income charts ─────────── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: `${GAP}px`,
        }}
      >
        <LineChartCard
          title="Revenue"
          leftData={mockLeft.revenueHistory}
          rightData={mockRight.revenueHistory}
          leftTicker={mockLeft.ticker}
          rightTicker={mockRight.ticker}
          formatY={(v) => `$${v}B`}
          chartHeight={188}
        />
        <LineChartCard
          title="Operating Income"
          leftData={mockLeft.opIncomeHistory}
          rightData={mockRight.opIncomeHistory}
          leftTicker={mockLeft.ticker}
          rightTicker={mockRight.ticker}
          formatY={(v) => `$${v}B`}
          chartHeight={188}
        />
      </div>

      {/* ── Section 3: Valuation / Reinvestment / Profitability */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr',
          gap: `${GAP}px`,
        }}
      >
        <BarChartCard
          title="Valuation"
          data={[
            { category: 'P/E', leftValue: mockLeft.pe, rightValue: mockRight.pe },
            { category: 'EV/EBITDA', leftValue: mockLeft.evEbitda, rightValue: mockRight.evEbitda },
            { category: 'Peer P/E', leftValue: mockLeft.peerPe, rightValue: mockRight.peerPe },
          ]}
          leftLabel={mockLeft.ticker}
          rightLabel={mockRight.ticker}
          formatY={(v) => `${v}x`}
          chartHeight={180}
        />
        <ReinvestmentCard />
        <ProfitabilityCard />
      </div>

      {/* ── Section 4: Total Return & Risk ────────────────── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: `${GAP}px`,
        }}
      >
        <BarChartCard
          title="Total Return — Annualised"
          data={[
            {
              category: '1Y',
              leftValue: mockLeft.return1Y,
              rightValue: mockRight.return1Y,
            },
            {
              category: '3Y',
              leftValue: mockLeft.return3Y,
              rightValue: mockRight.return3Y,
            },
            {
              category: '5Y',
              leftValue: mockLeft.return5Y,
              rightValue: mockRight.return5Y,
            },
          ]}
          leftLabel={mockLeft.ticker}
          rightLabel={mockRight.ticker}
          formatY={(v) => `${v}%`}
          chartHeight={155}
        />
        <RiskCard />
      </div>

      {/* ── Footer ────────────────────────────────────────── */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span
          style={{
            fontSize: '9px',
            color: theme.textMuted,
            opacity: 0.4,
            letterSpacing: '0.03em',
          }}
        >
          Mock data only — not financial advice
        </span>
        <span
          style={{
            fontSize: '9px',
            color: theme.textMuted,
            opacity: 0.4,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
          }}
        >
          @IGFinanceBuilder
        </span>
      </div>
    </div>
  );
}
