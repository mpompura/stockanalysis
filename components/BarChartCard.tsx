'use client';

import ReactEcharts from 'echarts-for-react';
import { useTheme } from '@/contexts/ThemeContext';
import { CardShell } from './CardShell';

export type BarEntry = {
  category: string;
  leftValue: number;
  rightValue: number;
};

type Props = {
  title: string;
  data: BarEntry[];
  leftLabel: string;
  rightLabel: string;
  formatY?: (v: number) => string;
  /** Fixed pixel height. Omit to fill parent. */
  chartHeight?: number;
};

export function BarChartCard({
  title,
  data,
  leftLabel,
  rightLabel,
  formatY,
  chartHeight,
}: Props) {
  const { theme } = useTheme();
  const yFmt = formatY ?? ((v: number) => `${v}`);
  const hasRight = data.some((d) => d.rightValue !== 0) && rightLabel;

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: {
      top: hasRight ? 28 : 20,
      right: 10,
      bottom: 28,
      left: 10,
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: data.map((d) => d.category),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: theme.textMuted,
        fontSize: 10,
        fontFamily: 'Inter, system-ui, sans-serif',
      },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: theme.textMuted,
        fontSize: 9,
        fontFamily: 'Inter, system-ui, sans-serif',
        formatter: yFmt,
      },
      splitLine: {
        lineStyle: { color: theme.divider, type: 'solid' as const, opacity: 0.6 },
      },
    },
    tooltip: {
      trigger: 'axis',
      backgroundColor: theme.panel,
      borderColor: theme.divider,
      borderWidth: 1,
      textStyle: { color: theme.textPrimary, fontSize: 11 },
      formatter: (
        params: Array<{ marker: string; seriesName: string; value: number }>
      ) =>
        params
          .filter((p) => p.value !== 0 || !hasRight)
          .map((p) => `${p.marker} ${p.seriesName}: ${yFmt(p.value)}`)
          .join('<br/>'),
    },
    legend: hasRight
      ? {
          top: 2,
          right: 4,
          textStyle: { color: theme.textMuted, fontSize: 9 },
          itemWidth: 10,
          itemHeight: 6,
        }
      : { show: false },
    series: [
      {
        name: leftLabel || 'Value',
        type: 'bar',
        data: data.map((d) => (d.leftValue !== 0 ? d.leftValue : null)),
        itemStyle: { color: theme.bull, borderRadius: [4, 4, 0, 0] },
        barMaxWidth: hasRight ? 22 : 32,
        barGap: '20%',
        label: {
          show: true,
          position: 'top',
          color: theme.bull,
          fontSize: 9,
          fontWeight: 600,
          formatter: (p: { value: number | null }) =>
            p.value != null ? yFmt(p.value) : '',
        },
      },
      ...(hasRight
        ? [
            {
              name: rightLabel,
              type: 'bar',
              data: data.map((d) => (d.rightValue !== 0 ? d.rightValue : null)),
              itemStyle: { color: theme.bear, borderRadius: [4, 4, 0, 0] },
              barMaxWidth: 22,
              label: {
                show: true,
                position: 'top',
                color: theme.bear,
                fontSize: 9,
                fontWeight: 600,
                formatter: (p: { value: number | null }) =>
                  p.value != null ? yFmt(p.value) : '',
              },
            },
          ]
        : []),
    ],
  };

  return (
    <CardShell
      style={{
        padding: '14px 14px 10px',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        boxSizing: 'border-box',
      }}
    >
      <div
        style={{
          fontSize: '10px',
          color: theme.textMuted,
          textTransform: 'uppercase',
          letterSpacing: '0.09em',
          fontWeight: 600,
          marginBottom: '4px',
          flexShrink: 0,
        }}
      >
        {title}
      </div>

      {data.length === 0 || data.every((d) => d.leftValue === 0 && d.rightValue === 0) ? (
        <div
          style={{
            flex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: theme.textMuted,
            fontSize: '13px',
            opacity: 0.4,
          }}
        >
          No data
        </div>
      ) : (
        <div style={{ flex: 1, minHeight: 0 }}>
          <ReactEcharts
            option={option}
            style={{
              height: chartHeight != null ? `${chartHeight}px` : '100%',
              width: '100%',
            }}
            notMerge
          />
        </div>
      )}
    </CardShell>
  );
}
