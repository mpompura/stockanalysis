'use client';

import ReactEcharts from 'echarts-for-react';
import { useTheme } from '@/contexts/ThemeContext';
import { YearlyPoint } from '@/lib/schema';
import { CardShell } from './CardShell';

type Props = {
  title: string;
  leftData: YearlyPoint[];
  rightData: YearlyPoint[];
  leftTicker: string;
  rightTicker: string;
  formatY?: (v: number) => string;
  /** Fixed pixel height. Omit to fill parent (requires parent to have a defined height). */
  chartHeight?: number;
};

export function LineChartCard({
  title,
  leftData,
  rightData,
  leftTicker,
  rightTicker,
  formatY,
  chartHeight,
}: Props) {
  const { theme } = useTheme();
  const yFmt = formatY ?? ((v: number) => `$${v}B`);

  // Merge x-axis categories from both series
  const categories = Array.from(
    new Set([...leftData.map((d) => d.year), ...rightData.map((d) => d.year)])
  );

  const hasRight = rightData.length > 0 && rightTicker;

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: {
      top: hasRight ? 28 : 20,
      right: 12,
      bottom: 28,
      left: 50,
      containLabel: false,
    },
    xAxis: {
      type: 'category',
      data: categories.length > 0 ? categories : leftData.map((d) => d.year),
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: {
        color: theme.textMuted,
        fontSize: 9,
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
        lineStyle: {
          color: theme.divider,
          type: 'solid' as const,
          opacity: 0.6,
        },
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
          .map((p) => `${p.marker} ${p.seriesName}: ${yFmt(p.value)}`)
          .join('<br/>'),
    },
    legend: hasRight
      ? {
          top: 2,
          right: 4,
          textStyle: { color: theme.textMuted, fontSize: 9 },
          itemWidth: 14,
          itemHeight: 2,
        }
      : { show: false },
    series: [
      ...(leftData.length > 0
        ? [
            {
              name: leftTicker,
              type: 'line',
              data: leftData.map((d) => d.value),
              smooth: 0.4,
              symbol: 'circle',
              symbolSize: 4,
              lineStyle: { color: theme.bull, width: 2.5 },
              itemStyle: { color: theme.bull },
              areaStyle: {
                color: {
                  type: 'linear',
                  x: 0, y: 0, x2: 0, y2: 1,
                  colorStops: [
                    { offset: 0, color: `${theme.bull}28` },
                    { offset: 1, color: `${theme.bull}04` },
                  ],
                },
              },
            },
          ]
        : []),
      ...(rightData.length > 0 && rightTicker
        ? [
            {
              name: rightTicker,
              type: 'line',
              data: rightData.map((d) => d.value),
              smooth: 0.4,
              symbol: 'circle',
              symbolSize: 4,
              lineStyle: { color: theme.bear, width: 2.5 },
              itemStyle: { color: theme.bear },
              areaStyle: {
                color: {
                  type: 'linear',
                  x: 0, y: 0, x2: 0, y2: 1,
                  colorStops: [
                    { offset: 0, color: `${theme.bear}28` },
                    { offset: 1, color: `${theme.bear}04` },
                  ],
                },
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

      {leftData.length === 0 ? (
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
