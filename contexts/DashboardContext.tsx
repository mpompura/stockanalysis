'use client';

import React, { createContext, useContext, useState } from 'react';
import { DashboardData } from '@/lib/schema';
import {
  TemplateId,
  CanvasPresetId,
  canvasPresets,
  templateDefaults,
  compareDefault,
} from '@/lib/defaultData';

export type { TemplateId, CanvasPresetId };

type CanvasSize = { w: number; h: number };

type DashboardContextType = {
  data: DashboardData;
  setData: (data: DashboardData) => void;

  template: TemplateId;
  setTemplate: (t: TemplateId) => void;

  canvasSizeId: CanvasPresetId;
  setCanvasSizeId: (id: CanvasPresetId) => void;
  canvasSize: CanvasSize;
};

const DashboardContext = createContext<DashboardContextType>({
  data: compareDefault,
  setData: () => {},
  template: 'compare',
  setTemplate: () => {},
  canvasSizeId: '1080x1350',
  setCanvasSizeId: () => {},
  canvasSize: canvasPresets['1080x1350'],
});

export function DashboardProvider({ children }: { children: React.ReactNode }) {
  const [data, setData] = useState<DashboardData>(compareDefault);
  const [template, setTemplateState] = useState<TemplateId>('compare');
  const [canvasSizeId, setCanvasSizeIdState] = useState<CanvasPresetId>('1080x1350');

  /** Switching templates also loads the appropriate default data */
  const setTemplate = (t: TemplateId) => {
    setTemplateState(t);
    setData(templateDefaults[t]);
  };

  const setCanvasSizeId = (id: CanvasPresetId) => {
    setCanvasSizeIdState(id);
  };

  return (
    <DashboardContext.Provider
      value={{
        data,
        setData,
        template,
        setTemplate,
        canvasSizeId,
        setCanvasSizeId,
        canvasSize: canvasPresets[canvasSizeId],
      }}
    >
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard() {
  return useContext(DashboardContext);
}
