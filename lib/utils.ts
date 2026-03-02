export function formatMoneyAbbrev(value: number | null | undefined): string {
  if (value == null || isNaN(value as number)) return '—';
  const v = value as number;
  const abs = Math.abs(v);
  if (abs >= 1_000) return `$${(v / 1_000).toFixed(1)}T`;
  if (abs >= 1) return `$${v.toFixed(1)}B`;
  return `$${(v * 1_000).toFixed(0)}M`;
}

export function formatPct(
  value: number | null | undefined,
  decimals = 1
): string {
  if (value == null || isNaN(value as number)) return '—';
  return `${(value as number).toFixed(decimals)}%`;
}

export function formatSignedPct(
  value: number | null | undefined,
  decimals = 1
): string {
  if (value == null || isNaN(value as number)) return '—';
  const v = value as number;
  const sign = v >= 0 ? '+' : '';
  return `${sign}${v.toFixed(decimals)}%`;
}

export function formatMultiple(
  value: number | null | undefined,
  decimals = 1
): string {
  if (value == null || isNaN(value as number)) return '—';
  return `${(value as number).toFixed(decimals)}x`;
}
