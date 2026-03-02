'use client';

import { useDashboard } from '@/contexts/DashboardContext';
import { useTheme } from '@/contexts/ThemeContext';
import { CompanyBlock } from '@/lib/schema';
import {
  formatMoneyAbbrev,
  formatPct,
  formatSignedPct,
  formatMultiple,
} from '@/lib/utils';
import { LineChartCard } from '@/components/LineChartCard';
import { BarChartCard } from '@/components/BarChartCard';
import { CardShell } from '@/components/CardShell';
import { ProgressBarMetric } from '@/components/ProgressBarMetric';
import { MetricBlock } from '@/components/MetricBlock';

const PAD = 26;
const GAP = 12;

// ─── Helper sub-components ───────────────────────────────────────────────────

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

// ─── Hero section ────────────────────────────────────────────────────────────

function HeroSection({ company }: { company: CompanyBlock }) {
  const { theme } = useTheme();
  const m = company.metrics;

  const kpis = [
    { label: 'Market Cap', value: formatMoneyAbbrev(m?.marketCap) },
    { label: 'TTM Revenue', value: formatMoneyAbbrev(m?.ttmRevenue) },
    { label: 'TTM FCF', value: formatMoneyAbbrev(m?.ttmFCF) },
    { label: 'FCF Margin', value: formatPct(m?.fcfMargin), color: theme.bull },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '0',
        height: '100%',
      }}
    >
      {/* Left: identity + CAGR */}
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          paddingRight: '28px',
          borderRight: `1px solid ${theme.divider}`,
        }}
      >
        {/* Ticker + name */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
            <div
              style={{
                width: '10px',
                height: '10px',
                borderRadius: '50%',
                backgroundColor: theme.bull,
              }}
            />
            <span
              style={{
                fontSize: '44px',
                fontWeight: 800,
                color: theme.textPrimary,
                letterSpacing: '-0.03em',
                lineHeight: 1,
              }}
            >
              {company.ticker}
            </span>
          </div>
          <div style={{ fontSize: '13px', color: theme.textMuted, letterSpacing: '0.01em' }}>
            {company.name}
          </div>
        </div>

        {/* CAGR hero */}
        <div>
          <div
            style={{
              fontSize: '10px',
              color: theme.textMuted,
              textTransform: 'uppercase',
              letterSpacing: '0.12em',
              fontWeight: 500,
              marginBottom: '2px',
            }}
          >
            5-Year Revenue CAGR
          </div>
          <div
            style={{
              fontSize: '58px',
              fontWeight: 800,
              color: theme.bull,
              letterSpacing: '-0.03em',
              lineHeight: 1,
            }}
          >
            {formatSignedPct(m?.cagr5Y, 1)}
          </div>
        </div>
      </div>

      {/* Right: KPI grid */}
      <div
        style={{
          paddingLeft: '28px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
        }}
      >
        <div
          style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px 20px', flex: 1, alignContent: 'center' }}
        >
          {kpis.map((kpi, i) => (
            <MetricBlock
              key={i}
              label={kpi.label}
              value={kpi.value}
              valueColor={kpi.color}
              size="lg"
            />
          ))}
        </div>

        {/* Margin badges */}
        <div
          style={{
            display: 'flex',
            gap: '8px',
            flexWrap: 'wrap',
          }}
        >
          {[
            { label: 'Gross', value: company.profitability?.grossMargin },
            { label: 'Op', value: company.profitability?.opMargin },
            { label: 'Net', value: company.profitability?.netMargin },
          ].map((m, i) => (
            <div
              key={i}
              style={{
                padding: '4px 10px',
                backgroundColor: `${theme.bull}15`,
                border: `1px solid ${theme.bull}30`,
                borderRadius: theme.radius / 2,
                display: 'flex',
                gap: '5px',
                alignItems: 'center',
              }}
            >
              <span style={{ fontSize: '9px', color: theme.textMuted, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {m.label}
              </span>
              <span style={{ fontSize: '11px', color: theme.bull, fontWeight: 700 }}>
                {formatPct(m.value)}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Profitability card (single) ─────────────────────────────────────────────

function SingleProfitabilityCard({ company }: { company: CompanyBlock }) {
  const { theme } = useTheme();
  const p = company.profitability;

  const rows = [
    { label: 'Gross Margin', value: p?.grossMargin },
    { label: 'Op. Margin', value: p?.opMargin },
    { label: 'Net Margin', value: p?.netMargin },
    { label: 'ROIC', value: p?.roic, max: 80 },
    { label: 'ROE', value: p?.roe, max: 200 },
  ].filter((r) => r.value != null);

  return (
    <CardShell
      style={{ padding: '14px', display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <CardTitle>Profitability</CardTitle>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, justifyContent: 'center' }}>
        {rows.length === 0 ? (
          <span style={{ fontSize: '12px', color: theme.textMuted }}>—</span>
        ) : (
          rows.map((r, i) => (
            <ProgressBarMetric
              key={i}
              label={r.label}
              fillPct={Math.min((r.value! / (r.max ?? 100)) * 100, 100)}
              displayValue={formatPct(r.value)}
              color={theme.bull}
            />
          ))
        )}
      </div>
    </CardShell>
  );
}

// ─── Risk card (single) ───────────────────────────────────────────────────────

function SingleRiskCard({ company }: { company: CompanyBlock }) {
  const { theme } = useTheme();
  const r = company.risk;

  return (
    <CardShell
      style={{ padding: '14px', display: 'flex', flexDirection: 'column', height: '100%' }}
    >
      <CardTitle>Risk Profile</CardTitle>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, justifyContent: 'center' }}>
        <ProgressBarMetric
          label="Beta"
          fillPct={Math.min(((r?.beta ?? 0) / 3) * 100, 100)}
          displayValue={r?.beta != null ? `${r.beta.toFixed(1)}x` : '—'}
          color={(r?.beta ?? 0) > 1.5 ? theme.bear : theme.bull}
        />
        <ProgressBarMetric
          label="Debt / Equity"
          fillPct={Math.min(((r?.debtToEquity ?? 0) / 2) * 100, 100)}
          displayValue={r?.debtToEquity != null ? `${r.debtToEquity.toFixed(1)}x` : '—'}
          color={(r?.debtToEquity ?? 0) > 1 ? theme.bear : theme.bull}
        />
        <ProgressBarMetric
          label="Interest Coverage"
          fillPct={Math.min(((r?.interestCoverage ?? 0) / 100) * 100, 100)}
          displayValue={r?.interestCoverage != null ? `${r.interestCoverage}x` : '—'}
          color={theme.bull}
        />
      </div>

      {/* Summary badges */}
      <div style={{ display: 'flex', gap: '6px', marginTop: '10px', flexWrap: 'wrap' }}>
        {r?.beta != null && (
          <span
            style={{
              padding: '3px 8px',
              backgroundColor: (r.beta > 1.5 ? `${theme.bear}15` : `${theme.bull}15`),
              border: `1px solid ${r.beta > 1.5 ? `${theme.bear}30` : `${theme.bull}30`}`,
              borderRadius: '4px',
              fontSize: '9px',
              color: r.beta > 1.5 ? theme.bear : theme.bull,
              fontWeight: 600,
            }}
          >
            {r.beta > 1.5 ? 'High Volatility' : r.beta < 0.8 ? 'Defensive' : 'Moderate Beta'}
          </span>
        )}
        {r?.debtToEquity != null && (
          <span
            style={{
              padding: '3px 8px',
              backgroundColor: (r.debtToEquity > 1 ? `${theme.bear}15` : `${theme.bull}15`),
              border: `1px solid ${r.debtToEquity > 1 ? `${theme.bear}30` : `${theme.bull}30`}`,
              borderRadius: '4px',
              fontSize: '9px',
              color: r.debtToEquity > 1 ? theme.bear : theme.bull,
              fontWeight: 600,
            }}
          >
            {r.debtToEquity > 1 ? 'Leveraged' : 'Low Leverage'}
          </span>
        )}
      </div>
    </CardShell>
  );
}

// ─── Main template ────────────────────────────────────────────────────────────

export function SingleTemplate({ width, height }: { width: number; height: number }) {
  const { data } = useDashboard();
  const { theme } = useTheme();
  const { left: company, meta } = data;

  const revenue = company.series?.revenue ?? [];
  const opIncome = company.series?.opIncome ?? [];
  const fcf = company.series?.fcf ?? [];

  // Decide chart count (show FCF only if data exists)
  const hasFcf = fcf.length > 0;
  const chartCols = hasFcf ? '1fr 1fr 1fr' : '1fr 1fr';

  const valuationData = [
    { category: 'P/E', leftValue: company.valuation?.pe ?? 0, rightValue: company.valuation?.peerPe ?? 0 },
    { category: 'EV/EBITDA', leftValue: company.valuation?.evEbitda ?? 0, rightValue: company.valuation?.peerEvEbitda ?? 0 },
    { category: 'P/S', leftValue: company.valuation?.priceSales ?? 0, rightValue: 0 },
    { category: 'P/B', leftValue: company.valuation?.priceToBook ?? 0, rightValue: 0 },
  ].filter((d) => d.leftValue > 0 || d.rightValue > 0);

  const returnsData = [
    { category: '1Y', leftValue: company.returns?.return1Y ?? 0, rightValue: 0 },
    { category: '3Y', leftValue: company.returns?.return3Y ?? 0, rightValue: 0 },
    { category: '5Y', leftValue: company.returns?.return5Y ?? 0, rightValue: 0 },
  ].filter((d) => d.leftValue !== 0);

  return (
    <div
      style={{
        width: `${width}px`,
        height: `${height}px`,
        backgroundColor: theme.background,
        display: 'grid',
        gridTemplateRows: `30px 1fr 1.8fr 1.4fr 24px`,
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
              background: `linear-gradient(135deg, ${theme.bull}CC, ${theme.bull}40)`,
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
            {meta.title ?? 'Deep Dive'}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          {meta.subtitle && (
            <span
              style={{ fontSize: '10px', color: theme.textMuted, opacity: 0.6 }}
            >
              {meta.subtitle}
            </span>
          )}
          {meta.date && (
            <span
              style={{
                fontSize: '9px',
                color: theme.textMuted,
                opacity: 0.5,
                letterSpacing: '0.06em',
              }}
            >
              {meta.date}
            </span>
          )}
        </div>
      </div>

      {/* Section 1: Hero KPI row */}
      <CardShell style={{ padding: '22px 24px', overflow: 'hidden', height: '100%' }}>
        <HeroSection company={company} />
      </CardShell>

      {/* Section 2: 3 charts */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: chartCols,
          gap: `${GAP}px`,
          height: '100%',
        }}
      >
        <LineChartCard
          title="Revenue"
          leftData={revenue}
          rightData={[]}
          leftTicker={company.ticker}
          rightTicker=""
          formatY={(v) => `$${v}B`}
        />
        <LineChartCard
          title="Operating Income"
          leftData={opIncome}
          rightData={[]}
          leftTicker={company.ticker}
          rightTicker=""
          formatY={(v) => `$${v}B`}
        />
        {hasFcf && (
          <LineChartCard
            title="Free Cash Flow"
            leftData={fcf}
            rightData={[]}
            leftTicker={company.ticker}
            rightTicker=""
            formatY={(v) => `$${v}B`}
          />
        )}
      </div>

      {/* Section 3: 4 metric cards */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1fr 1fr',
          gap: `${GAP}px`,
          height: '100%',
        }}
      >
        <BarChartCard
          title="Valuation"
          data={valuationData.length > 0 ? valuationData : [{ category: '—', leftValue: 0, rightValue: 0 }]}
          leftLabel={company.ticker}
          rightLabel="Peers"
          formatY={(v) => (v > 0 ? `${v}x` : '')}
        />
        <SingleProfitabilityCard company={company} />
        <BarChartCard
          title="Total Return"
          data={returnsData.length > 0 ? returnsData : [{ category: '—', leftValue: 0, rightValue: 0 }]}
          leftLabel={company.ticker}
          rightLabel=""
          formatY={(v) => (v !== 0 ? `${v}%` : '')}
        />
        <SingleRiskCard company={company} />
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
