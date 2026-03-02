'use client';

import { useDashboard } from '@/contexts/DashboardContext';
import { useTheme } from '@/contexts/ThemeContext';
import { CompanyBlock } from '@/lib/schema';
import { formatMoneyAbbrev, formatPct, formatSignedPct } from '@/lib/utils';
import { ComparisonHeader } from '@/components/ComparisonHeader';
import { LineChartCard } from '@/components/LineChartCard';
import { BarChartCard } from '@/components/BarChartCard';
import { CardShell } from '@/components/CardShell';
import { ProgressBarMetric } from '@/components/ProgressBarMetric';

const PAD = 26;
const GAP = 12;

// ─── Local card sub-components ───────────────────────────────────────────────

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

function InlineDivider() {
  const { theme } = useTheme();
  return (
    <div style={{ height: '1px', backgroundColor: theme.divider, margin: '8px 0' }} />
  );
}

function TickerBadge({ ticker, color }: { ticker: string; color: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '5px', marginBottom: '7px' }}>
      <div
        style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: color }}
      />
      <span style={{ fontSize: '10px', color, fontWeight: 700, letterSpacing: '0.08em' }}>
        {ticker}
      </span>
    </div>
  );
}

function ReinvestmentCard({ left, right }: { left: CompanyBlock; right?: CompanyBlock }) {
  const { theme } = useTheme();
  return (
    <CardShell
      style={{ padding: '14px', display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <CardTitle>Reinvestment vs Cash</CardTitle>
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <TickerBadge ticker={left.ticker} color={theme.bull} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            <ProgressBarMetric
              label="CapEx / Revenue"
              fillPct={Math.min(((left.reinvestment?.capexPct ?? 0) / 20) * 100, 100)}
              displayValue={formatPct(left.reinvestment?.capexPct)}
              color={theme.textMuted}
            />
            <ProgressBarMetric
              label="R&D / Revenue"
              fillPct={Math.min(((left.reinvestment?.rndPct ?? 0) / 25) * 100, 100)}
              displayValue={formatPct(left.reinvestment?.rndPct)}
              color={theme.textMuted}
            />
            <ProgressBarMetric
              label="FCF Margin"
              fillPct={left.reinvestment?.fcfMargin ?? 0}
              displayValue={formatPct(left.reinvestment?.fcfMargin)}
              color={theme.bull}
            />
          </div>
        </div>

        {right && (
          <>
            <InlineDivider />
            <div>
              <TickerBadge ticker={right.ticker} color={theme.bear} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
                <ProgressBarMetric
                  label="CapEx / Revenue"
                  fillPct={Math.min(((right.reinvestment?.capexPct ?? 0) / 20) * 100, 100)}
                  displayValue={formatPct(right.reinvestment?.capexPct)}
                  color={theme.textMuted}
                />
                <ProgressBarMetric
                  label="R&D / Revenue"
                  fillPct={Math.min(((right.reinvestment?.rndPct ?? 0) / 25) * 100, 100)}
                  displayValue={formatPct(right.reinvestment?.rndPct)}
                  color={theme.textMuted}
                />
                <ProgressBarMetric
                  label="FCF Margin"
                  fillPct={right.reinvestment?.fcfMargin ?? 0}
                  displayValue={formatPct(right.reinvestment?.fcfMargin)}
                  color={theme.bear}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </CardShell>
  );
}

function ProfitabilityCard({ left, right }: { left: CompanyBlock; right?: CompanyBlock }) {
  const { theme } = useTheme();
  return (
    <CardShell
      style={{ padding: '14px', display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <CardTitle>Profitability</CardTitle>
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        }}
      >
        <div>
          <TickerBadge ticker={left.ticker} color={theme.bull} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
            <ProgressBarMetric
              label="Gross Margin"
              fillPct={left.profitability?.grossMargin ?? 0}
              displayValue={formatPct(left.profitability?.grossMargin)}
              color={theme.bull}
            />
            <ProgressBarMetric
              label="Op. Margin"
              fillPct={left.profitability?.opMargin ?? 0}
              displayValue={formatPct(left.profitability?.opMargin)}
              color={theme.bull}
            />
            <ProgressBarMetric
              label="Net Margin"
              fillPct={left.profitability?.netMargin ?? 0}
              displayValue={formatPct(left.profitability?.netMargin)}
              color={theme.bull}
            />
          </div>
        </div>

        {right && (
          <>
            <InlineDivider />
            <div>
              <TickerBadge ticker={right.ticker} color={theme.bear} />
              <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
                <ProgressBarMetric
                  label="Gross Margin"
                  fillPct={right.profitability?.grossMargin ?? 0}
                  displayValue={formatPct(right.profitability?.grossMargin)}
                  color={theme.bear}
                />
                <ProgressBarMetric
                  label="Op. Margin"
                  fillPct={right.profitability?.opMargin ?? 0}
                  displayValue={formatPct(right.profitability?.opMargin)}
                  color={theme.bear}
                />
                <ProgressBarMetric
                  label="Net Margin"
                  fillPct={right.profitability?.netMargin ?? 0}
                  displayValue={formatPct(right.profitability?.netMargin)}
                  color={theme.bear}
                />
              </div>
            </div>
          </>
        )}
      </div>
    </CardShell>
  );
}

function RiskCard({ left, right }: { left: CompanyBlock; right?: CompanyBlock }) {
  const { theme } = useTheme();

  const riskRows = (company: CompanyBlock, color: string) => (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '7px' }}>
      <ProgressBarMetric
        label="Beta"
        fillPct={Math.min(((company.risk?.beta ?? 0) / 3) * 100, 100)}
        displayValue={company.risk?.beta != null ? `${company.risk.beta.toFixed(1)}x` : '—'}
        color={(company.risk?.beta ?? 0) > 1.5 ? theme.bear : theme.bull}
      />
      <ProgressBarMetric
        label="Debt / Equity"
        fillPct={Math.min(((company.risk?.debtToEquity ?? 0) / 2) * 100, 100)}
        displayValue={
          company.risk?.debtToEquity != null ? `${company.risk.debtToEquity.toFixed(1)}x` : '—'
        }
        color={(company.risk?.debtToEquity ?? 0) > 1 ? theme.bear : theme.bull}
      />
      <ProgressBarMetric
        label="Int. Coverage"
        fillPct={Math.min(((company.risk?.interestCoverage ?? 0) / 100) * 100, 100)}
        displayValue={
          company.risk?.interestCoverage != null ? `${company.risk.interestCoverage}x` : '—'
        }
        color={theme.bull}
      />
    </div>
  );

  return (
    <CardShell
      style={{ padding: '14px', display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <CardTitle>Risk Profile</CardTitle>
      <div
        style={{ display: 'grid', gridTemplateColumns: right ? '1fr 1fr' : '1fr', gap: '16px', flex: 1 }}
      >
        <div>
          <TickerBadge ticker={left.ticker} color={theme.bull} />
          {riskRows(left, theme.bull)}
        </div>
        {right && (
          <div>
            <TickerBadge ticker={right.ticker} color={theme.bear} />
            {riskRows(right, theme.bear)}
          </div>
        )}
      </div>
    </CardShell>
  );
}

// ─── Main template ───────────────────────────────────────────────────────────

export function CompareTemplate({ width, height }: { width: number; height: number }) {
  const { data } = useDashboard();
  const { theme } = useTheme();
  const { left, right, meta } = data;

  const leftRevenue = left.series?.revenue ?? [];
  const rightRevenue = right?.series?.revenue ?? [];
  const leftOpIncome = left.series?.opIncome ?? [];
  const rightOpIncome = right?.series?.opIncome ?? [];

  const valuationData = [
    {
      category: 'P/E',
      leftValue: left.valuation?.pe ?? 0,
      rightValue: right?.valuation?.pe ?? 0,
    },
    {
      category: 'EV/EBITDA',
      leftValue: left.valuation?.evEbitda ?? 0,
      rightValue: right?.valuation?.evEbitda ?? 0,
    },
    {
      category: 'Peer P/E',
      leftValue: left.valuation?.peerPe ?? 0,
      rightValue: right?.valuation?.peerPe ?? 0,
    },
  ].filter((d) => d.leftValue > 0 || d.rightValue > 0);

  const returnsData = [
    {
      category: '1Y',
      leftValue: left.returns?.return1Y ?? 0,
      rightValue: right?.returns?.return1Y ?? 0,
    },
    {
      category: '3Y',
      leftValue: left.returns?.return3Y ?? 0,
      rightValue: right?.returns?.return3Y ?? 0,
    },
    {
      category: '5Y',
      leftValue: left.returns?.return5Y ?? 0,
      rightValue: right?.returns?.return5Y ?? 0,
    },
  ].filter((d) => d.leftValue !== 0 || d.rightValue !== 0);

  return (
    <div
      style={{
        width: `${width}px`,
        height: `${height}px`,
        backgroundColor: theme.background,
        display: 'grid',
        gridTemplateRows: `30px 1fr 1fr 1fr 1fr 24px`,
        padding: `${PAD}px`,
        gap: `${GAP}px`,
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
        overflow: 'hidden',
        boxSizing: 'border-box',
      }}
    >
      {/* Brand header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
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
            {meta.title ?? 'Stock Comparison'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {meta.date && (
            <span
              style={{
                fontSize: '9px',
                color: theme.textMuted,
                letterSpacing: '0.08em',
                opacity: 0.7,
              }}
            >
              {meta.date}
            </span>
          )}
          <div style={{ width: '5px', height: '5px', borderRadius: '50%', backgroundColor: theme.bull }} />
          <div style={{ width: '5px', height: '5px', borderRadius: '50%', backgroundColor: theme.bear }} />
        </div>
      </div>

      {/* Section 1: Comparison header */}
      <CardShell style={{ padding: '20px 22px', overflow: 'hidden', height: '100%' }}>
        <ComparisonHeader left={left} right={right} />
      </CardShell>

      {/* Section 2: Revenue + Op Income charts */}
      <div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: `${GAP}px`, height: '100%' }}
      >
        <LineChartCard
          title="Revenue"
          leftData={leftRevenue}
          rightData={rightRevenue}
          leftTicker={left.ticker}
          rightTicker={right?.ticker ?? ''}
          formatY={(v) => `$${v}B`}
        />
        <LineChartCard
          title="Operating Income"
          leftData={leftOpIncome}
          rightData={rightOpIncome}
          leftTicker={left.ticker}
          rightTicker={right?.ticker ?? ''}
          formatY={(v) => `$${v}B`}
        />
      </div>

      {/* Section 3: Valuation / Reinvestment / Profitability */}
      <div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: `${GAP}px`, height: '100%' }}
      >
        <BarChartCard
          title="Valuation"
          data={valuationData}
          leftLabel={left.ticker}
          rightLabel={right?.ticker ?? ''}
          formatY={(v) => `${v}x`}
        />
        <ReinvestmentCard left={left} right={right} />
        <ProfitabilityCard left={left} right={right} />
      </div>

      {/* Section 4: Returns + Risk */}
      <div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: `${GAP}px`, height: '100%' }}
      >
        <BarChartCard
          title="Total Return — Annualised"
          data={returnsData}
          leftLabel={left.ticker}
          rightLabel={right?.ticker ?? ''}
          formatY={(v) => `${v}%`}
        />
        <RiskCard left={left} right={right} />
      </div>

      {/* Footer */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: '9px', color: theme.textMuted, opacity: 0.35 }}>
          Mock data — not financial advice
        </span>
        <span
          style={{
            fontSize: '9px',
            color: theme.textMuted,
            opacity: 0.35,
            letterSpacing: '0.1em',
            textTransform: 'uppercase',
          }}
        >
          {meta.watermark ?? '@IGFinanceBuilder'}
        </span>
      </div>
    </div>
  );
}
