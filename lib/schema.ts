// ─── Primitive types ────────────────────────────────────────────────────────

export type YearlyPoint = {
  year: string;
  value: number;
};

// ─── Sub-block types (all fields optional) ──────────────────────────────────

export type MetricsBlock = {
  marketCap?: number;   // billions
  cagr5Y?: number;      // percent
  ttmRevenue?: number;  // billions
  ttmFCF?: number;      // billions
  fcfMargin?: number;   // percent
};

export type SeriesBlock = {
  revenue?: YearlyPoint[];
  opIncome?: YearlyPoint[];
  netIncome?: YearlyPoint[];
  fcf?: YearlyPoint[];
};

export type ValuationBlock = {
  pe?: number;
  evEbitda?: number;
  peerPe?: number;
  peerEvEbitda?: number;
  priceToBook?: number;
  priceSales?: number;
};

export type ReinvestmentBlock = {
  capexPct?: number;  // % of revenue
  rndPct?: number;    // % of revenue
  fcfMargin?: number; // %
};

export type ProfitabilityBlock = {
  grossMargin?: number; // %
  opMargin?: number;    // %
  netMargin?: number;   // %
  roic?: number;        // %
  roe?: number;         // %
};

export type ReturnsBlock = {
  return1Y?: number; // % annualised
  return3Y?: number;
  return5Y?: number;
};

export type RiskBlock = {
  beta?: number;
  debtToEquity?: number;
  interestCoverage?: number;
};

// ─── Top-level types ─────────────────────────────────────────────────────────

export type CompanyBlock = {
  /** Required: short symbol, e.g. "NVDA" */
  ticker: string;
  /** Required: full company name */
  name: string;
  metrics?: MetricsBlock;
  series?: SeriesBlock;
  valuation?: ValuationBlock;
  reinvestment?: ReinvestmentBlock;
  profitability?: ProfitabilityBlock;
  returns?: ReturnsBlock;
  risk?: RiskBlock;
};

export type DashboardMeta = {
  title?: string;
  subtitle?: string;
  watermark?: string;
  date?: string;
};

export type DashboardData = {
  meta: DashboardMeta;
  /** Always required — primary company / left panel */
  left: CompanyBlock;
  /** Optional — second company for comparison template */
  right?: CompanyBlock;
};

// ─── Validation ──────────────────────────────────────────────────────────────

export type ValidationResult = {
  valid: boolean;
  errors: string[];
};

function validateCompanyBlock(
  obj: Record<string, unknown>,
  path: string,
  errors: string[]
): void {
  if (!obj.ticker || typeof obj.ticker !== 'string' || obj.ticker.trim() === '') {
    errors.push(`"${path}.ticker" must be a non-empty string`);
  }
  if (!obj.name || typeof obj.name !== 'string' || obj.name.trim() === '') {
    errors.push(`"${path}.name" must be a non-empty string`);
  }

  // Optional numeric fields — warn if present but wrong type
  const numericPaths: string[] = [
    'metrics.marketCap', 'metrics.cagr5Y', 'metrics.ttmRevenue',
    'metrics.ttmFCF', 'metrics.fcfMargin',
    'valuation.pe', 'valuation.evEbitda',
    'profitability.grossMargin', 'profitability.opMargin', 'profitability.netMargin',
    'returns.return1Y', 'returns.return3Y', 'returns.return5Y',
    'risk.beta', 'risk.debtToEquity', 'risk.interestCoverage',
  ];

  for (const fieldPath of numericPaths) {
    const [section, field] = fieldPath.split('.');
    const sectionObj = obj[section];
    if (sectionObj && typeof sectionObj === 'object') {
      const val = (sectionObj as Record<string, unknown>)[field];
      if (val !== undefined && typeof val !== 'number') {
        errors.push(`"${path}.${fieldPath}" must be a number (got ${typeof val})`);
      }
    }
  }

  // Validate series arrays
  if (obj.series && typeof obj.series === 'object') {
    const series = obj.series as Record<string, unknown>;
    for (const key of ['revenue', 'opIncome', 'netIncome', 'fcf']) {
      const arr = series[key];
      if (arr !== undefined) {
        if (!Array.isArray(arr)) {
          errors.push(`"${path}.series.${key}" must be an array`);
        } else {
          for (let i = 0; i < arr.length; i++) {
            const pt = arr[i];
            if (!pt || typeof pt !== 'object') {
              errors.push(`"${path}.series.${key}[${i}]" must be an object`);
            } else {
              if (typeof (pt as Record<string, unknown>).year !== 'string') {
                errors.push(`"${path}.series.${key}[${i}].year" must be a string`);
              }
              if (typeof (pt as Record<string, unknown>).value !== 'number') {
                errors.push(`"${path}.series.${key}[${i}].value" must be a number`);
              }
            }
          }
        }
      }
    }
  }
}

export function validateDashboardData(obj: unknown): ValidationResult {
  const errors: string[] = [];

  if (!obj || typeof obj !== 'object' || Array.isArray(obj)) {
    return { valid: false, errors: ['Root value must be a JSON object'] };
  }

  const d = obj as Record<string, unknown>;

  // meta
  if (!d.meta || typeof d.meta !== 'object' || Array.isArray(d.meta)) {
    errors.push('"meta" field is missing or not an object');
  }

  // left (required)
  if (!d.left || typeof d.left !== 'object' || Array.isArray(d.left)) {
    errors.push('"left" field is missing or not an object');
  } else {
    validateCompanyBlock(d.left as Record<string, unknown>, 'left', errors);
  }

  // right (optional)
  if (d.right !== undefined) {
    if (typeof d.right !== 'object' || Array.isArray(d.right)) {
      errors.push('"right" field must be an object if present');
    } else {
      validateCompanyBlock(d.right as Record<string, unknown>, 'right', errors);
    }
  }

  return { valid: errors.length === 0, errors };
}
