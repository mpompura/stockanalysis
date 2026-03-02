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
  chartHeight?: number;
};

export function BarChartCard({
  title,
  data,
  leftLabel,
  rightLabel,
  formatY,
  chartHeight = 180,
}: Props) {
  const { theme } = useTheme();
  const yFmt = formatY ?? ((v: number) => `${v}`);

  const option = {
    backgroundColor: 'transparent',
    animation: false,
    grid: {
      top: 28,
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
      formatter: (params: Array<{ marker: string; seriesName: string; value: number }>) =>
        params
          .map((p) => `${p.marker} ${p.seriesName}: ${yFmt(p.value)}`)
          .join('<br/>'),
    },
    legend: {
      top: 2,
      right: 4,
      textStyle: { color: theme.textMuted, fontSize: 9 },
      itemWidth: 10,
      itemHeight: 6,
    },
    series: [
      {
        name: leftLabel,
        type: 'bar',
        data: data.map((d) => d.leftValue),
        itemStyle: {
          color: theme.bull,
          borderRadius: [4, 4, 0, 0],
        },
        barMaxWidth: 22,
        barGap: '20%',
        label: {
          show: true,
          position: 'top',
          color: theme.bull,
          fontSize: 9,
          fontWeight: 600,
          formatter: (p: { value: number }) => yFmt(p.value),
        },
      },
      {
        name: rightLabel,
        type: 'bar',
        data: data.map((d) => d.rightValue),
        itemStyle: {
          color: theme.bear,
          borderRadius: [4, 4, 0, 0],
        },
        barMaxWidth: 22,
        label: {
          show: true,
          position: 'top',
          color: theme.bear,
          fontSize: 9,
          fontWeight: 600,
          formatter: (p: { value: number }) => yFmt(p.value),
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
