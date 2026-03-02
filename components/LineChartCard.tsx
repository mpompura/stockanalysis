'use client';

import ReactEcharts from 'echarts-for-react';
import { useTheme } from '@/contexts/ThemeContext';
import { YearlyPoint } from '@/lib/data';
import { CardShell } from './CardShell';

type Props = {
  title: string;
  leftData: YearlyPoint[];
  rightData: YearlyPoint[];
  leftTicker: string;
  rightTicker: string;
  formatY?: (v: number) => string;
  chartHeight?: number;
};

export function LineChartCard({
  title,
  leftData,
  rightData,
  leftTicker,
  rightTicker,
  formatY,
  chartHeight = 190,
}: Props) {
  const { theme } = useTheme();

  const yFmt = formatY ?? ((v: number) => `$${v}B`);

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: {
      top: 28,
      right: 12,
      bottom: 28,
      left: 48,
      containLabel: false,
    },
    xAxis: {
      type: 'category',
      data: leftData.map((d) => d.year),
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
          width: 1,
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
      formatter: (params: Array<{ marker: string; seriesName: string; value: number }>) =>
        params
          .map((p) => `${p.marker} ${p.seriesName}: ${yFmt(p.value)}`)
          .join('<br/>'),
    },
    legend: {
      top: 2,
      right: 4,
      textStyle: { color: theme.textMuted, fontSize: 9 },
      itemWidth: 14,
      itemHeight: 2,
    },
    series: [
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
    ],
  };

  return (
    <CardShell
      style={{
        padding: '14px 14px 10px',
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
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
      <ReactEcharts
        option={option}
        style={{ height: `${chartHeight}px`, width: '100%', flex: 1 }}
        notMerge
      />
    </CardShell>
  );
}
