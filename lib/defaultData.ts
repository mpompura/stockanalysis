import { DashboardData } from './schema';

// ─── Compare: NVDA vs MSFT ───────────────────────────────────────────────────

export const compareDefault: DashboardData = {
  meta: {
    title: 'Stock Comparison',
    subtitle: 'NVDA vs MSFT — FY2020–TTM',
    watermark: '@IGFinanceBuilder',
    date: 'Q1 2026',
  },
  left: {
    ticker: 'NVDA',
    name: 'NVIDIA Corporation',
    metrics: {
      marketCap: 2650,
      cagr5Y: 52.3,
      ttmRevenue: 130.5,
      ttmFCF: 60.8,
      fcfMargin: 46.6,
    },
    series: {
      revenue: [
        { year: 'FY20', value: 10.9 },
        { year: 'FY21', value: 16.7 },
        { year: 'FY22', value: 26.9 },
        { year: 'FY23', value: 44.9 },
        { year: 'FY24', value: 60.9 },
        { year: 'TTM', value: 130.5 },
      ],
      opIncome: [
        { year: 'FY20', value: 2.8 },
        { year: 'FY21', value: 4.5 },
        { year: 'FY22', value: 10.0 },
        { year: 'FY23', value: 22.6 },
        { year: 'FY24', value: 32.3 },
        { year: 'TTM', value: 83.8 },
      ],
      fcf: [
        { year: 'FY20', value: 4.7 },
        { year: 'FY21', value: 8.1 },
        { year: 'FY22', value: 17.5 },
        { year: 'FY23', value: 26.9 },
        { year: 'FY24', value: 45.3 },
        { year: 'TTM', value: 60.8 },
      ],
    },
    valuation: { pe: 35.2, evEbitda: 28.1, peerPe: 28.5, peerEvEbitda: 22.4 },
    reinvestment: { capexPct: 3.2, rndPct: 11.4, fcfMargin: 46.6 },
    profitability: { grossMargin: 75.0, opMargin: 64.2, netMargin: 55.7 },
    returns: { return1Y: 180.5, return3Y: 95.2, return5Y: 68.3 },
    risk: { beta: 2.1, debtToEquity: 0.5, interestCoverage: 85 },
  },
  right: {
    ticker: 'MSFT',
    name: 'Microsoft Corporation',
    metrics: {
      marketCap: 3200,
      cagr5Y: 15.2,
      ttmRevenue: 245.1,
      ttmFCF: 80.3,
      fcfMargin: 32.8,
    },
    series: {
      revenue: [
        { year: 'FY20', value: 143.0 },
        { year: 'FY21', value: 168.1 },
        { year: 'FY22', value: 198.3 },
        { year: 'FY23', value: 211.9 },
        { year: 'FY24', value: 245.1 },
        { year: 'TTM', value: 253.8 },
      ],
      opIncome: [
        { year: 'FY20', value: 52.9 },
        { year: 'FY21', value: 69.9 },
        { year: 'FY22', value: 83.4 },
        { year: 'FY23', value: 88.5 },
        { year: 'FY24', value: 109.4 },
        { year: 'TTM', value: 119.8 },
      ],
      fcf: [
        { year: 'FY20', value: 45.2 },
        { year: 'FY21', value: 56.1 },
        { year: 'FY22', value: 65.1 },
        { year: 'FY23', value: 59.5 },
        { year: 'FY24', value: 74.1 },
        { year: 'TTM', value: 80.3 },
      ],
    },
    valuation: { pe: 32.1, evEbitda: 24.3, peerPe: 28.5, peerEvEbitda: 22.4 },
    reinvestment: { capexPct: 12.3, rndPct: 14.1, fcfMargin: 32.8 },
    profitability: { grossMargin: 70.1, opMargin: 44.6, netMargin: 37.2 },
    returns: { return1Y: 15.2, return3Y: 18.5, return5Y: 22.1 },
    risk: { beta: 0.9, debtToEquity: 0.3, interestCoverage: 45 },
  },
};

// ─── Single: AAPL Deep Dive ──────────────────────────────────────────────────

export const singleDefault: DashboardData = {
  meta: {
    title: 'Deep Dive',
    subtitle: 'AAPL — Apple Inc.',
    watermark: '@IGFinanceBuilder',
    date: 'Q1 2026',
  },
  left: {
    ticker: 'AAPL',
    name: 'Apple Inc.',
    metrics: {
      marketCap: 3100,
      cagr5Y: 8.4,
      ttmRevenue: 391.0,
      ttmFCF: 110.5,
      fcfMargin: 28.3,
    },
    series: {
      revenue: [
        { year: 'FY20', value: 274.5 },
        { year: 'FY21', value: 365.8 },
        { year: 'FY22', value: 394.3 },
        { year: 'FY23', value: 383.3 },
        { year: 'FY24', value: 391.0 },
        { year: 'TTM', value: 398.5 },
      ],
      opIncome: [
        { year: 'FY20', value: 66.3 },
        { year: 'FY21', value: 108.9 },
        { year: 'FY22', value: 119.4 },
        { year: 'FY23', value: 114.3 },
        { year: 'FY24', value: 123.2 },
        { year: 'TTM', value: 127.8 },
      ],
      fcf: [
        { year: 'FY20', value: 73.4 },
        { year: 'FY21', value: 93.0 },
        { year: 'FY22', value: 111.4 },
        { year: 'FY23', value: 99.6 },
        { year: 'FY24', value: 108.8 },
        { year: 'TTM', value: 110.5 },
      ],
    },
    valuation: { pe: 28.5, evEbitda: 22.1, peerPe: 25.0, peerEvEbitda: 20.0 },
    reinvestment: { capexPct: 2.8, rndPct: 8.1, fcfMargin: 28.3 },
    profitability: { grossMargin: 46.2, opMargin: 31.5, netMargin: 26.4, roic: 54.2, roe: 158.4 },
    returns: { return1Y: 12.5, return3Y: 11.2, return5Y: 18.9 },
    risk: { beta: 1.2, debtToEquity: 1.4, interestCoverage: 32 },
  },
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

export type TemplateId = 'compare' | 'single';

export const templateDefaults: Record<TemplateId, DashboardData> = {
  compare: compareDefault,
  single: singleDefault,
};

export const templateLabels: Record<TemplateId, string> = {
  compare: 'Compare 2 Stocks',
  single: 'Single Deep Dive',
};

export type CanvasPresetId = '1080x1350' | '1080x1080';

export const canvasPresets: Record<CanvasPresetId, { w: number; h: number; label: string }> = {
  '1080x1350': { w: 1080, h: 1350, label: '1080 × 1350 (Portrait IG)' },
  '1080x1080': { w: 1080, h: 1080, label: '1080 × 1080 (Square IG)' },
};
