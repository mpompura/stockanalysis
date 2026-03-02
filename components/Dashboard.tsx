'use client';

import { useDashboard } from '@/contexts/DashboardContext';
import { CompareTemplate } from './templates/CompareTemplate';
import { SingleTemplate } from './templates/SingleTemplate';

export function Dashboard() {
  const { template, canvasSize } = useDashboard();

  if (template === 'single') {
    return <SingleTemplate width={canvasSize.w} height={canvasSize.h} />;
  }

  return <CompareTemplate width={canvasSize.w} height={canvasSize.h} />;
}
