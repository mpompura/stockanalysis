export type YearlyPoint = {
  year: string;
  value: number;
};

export type TickerData = {
  ticker: string;
  name: string;
  marketCap: number;       // billions
  cagr5Y: number;          // percent
  ttmRevenue: number;      // billions
  ttmFCF: number;          // billions
  revenueHistory: YearlyPoint[];
  opIncomeHistory: YearlyPoint[];
  pe: number;
  evEbitda: number;
  peerPe: number;
  peerEvEbitda: number;
  capexPct: number;        // % of revenue
  fcfMargin: number;       // %
  grossMargin: number;     // %
  opMargin: number;        // %
  netMargin: number;       // %
  return1Y: number;        // %
  return3Y: number;        // % annualized
  return5Y: number;        // % annualized
  beta: number;
  debtToEquity: number;
  interestCoverage: number;
};

export const mockLeft: TickerData = {
  ticker: 'NVDA',
  name: 'NVIDIA Corporation',
  marketCap: 2650,
  cagr5Y: 52.3,
  ttmRevenue: 130.5,
  ttmFCF: 60.8,
  revenueHistory: [
    { year: 'FY20', value: 10.9 },
    { year: 'FY21', value: 16.7 },
    { year: 'FY22', value: 26.9 },
    { year: 'FY23', value: 44.9 },
    { year: 'FY24', value: 60.9 },
    { year: 'TTM', value: 130.5 },
  ],
  opIncomeHistory: [
    { year: 'FY20', value: 2.8 },
    { year: 'FY21', value: 4.5 },
    { year: 'FY22', value: 10.0 },
    { year: 'FY23', value: 22.6 },
    { year: 'FY24', value: 32.3 },
    { year: 'TTM', value: 83.8 },
  ],
  pe: 35.2,
  evEbitda: 28.1,
  peerPe: 28.5,
  peerEvEbitda: 22.4,
  capexPct: 3.2,
  fcfMargin: 46.6,
  grossMargin: 75.0,
  opMargin: 64.2,
  netMargin: 55.7,
  return1Y: 180.5,
  return3Y: 95.2,
  return5Y: 68.3,
  beta: 2.1,
  debtToEquity: 0.5,
  interestCoverage: 85,
};

export const mockRight: TickerData = {
  ticker: 'MSFT',
  name: 'Microsoft Corporation',
  marketCap: 3200,
  cagr5Y: 15.2,
  ttmRevenue: 245.1,
  ttmFCF: 80.3,
  revenueHistory: [
    { year: 'FY20', value: 143.0 },
    { year: 'FY21', value: 168.1 },
    { year: 'FY22', value: 198.3 },
    { year: 'FY23', value: 211.9 },
    { year: 'FY24', value: 245.1 },
    { year: 'TTM', value: 253.8 },
  ],
  opIncomeHistory: [
    { year: 'FY20', value: 52.9 },
    { year: 'FY21', value: 69.9 },
    { year: 'FY22', value: 83.4 },
    { year: 'FY23', value: 88.5 },
    { year: 'FY24', value: 109.4 },
    { year: 'TTM', value: 119.8 },
  ],
  pe: 32.1,
  evEbitda: 24.3,
  peerPe: 28.5,
  peerEvEbitda: 22.4,
  capexPct: 12.3,
  fcfMargin: 32.8,
  grossMargin: 70.1,
  opMargin: 44.6,
  netMargin: 37.2,
  return1Y: 15.2,
  return3Y: 18.5,
  return5Y: 22.1,
  beta: 0.9,
  debtToEquity: 0.3,
  interestCoverage: 45,
};
