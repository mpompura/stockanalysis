'use client';

import { useState } from 'react';
import { Dashboard } from '@/components/Dashboard';
import { ThemeSidebar } from '@/components/ThemeSidebar';
import { JsonPanel } from '@/components/JsonPanel';
import { useDashboard } from '@/contexts/DashboardContext';
import { canvasPresets, templateLabels, CanvasPresetId, TemplateId } from '@/lib/defaultData';

// ─── Style constants ──────────────────────────────────────────────────────────

const selectStyle: React.CSSProperties = {
  padding: '4px 8px',
  backgroundColor: '#111827',
  border: '1px solid #1E2739',
  borderRadius: '6px',
  color: '#94A3B8',
  fontSize: '11px',
  cursor: 'pointer',
  fontFamily: 'Inter, system-ui, sans-serif',
  outline: 'none',
  appearance: 'none',
  WebkitAppearance: 'none',
  backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5' viewBox='0 0 8 5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%2364748B'/%3E%3C/svg%3E")`,
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 8px center',
  paddingRight: '24px',
};

const zoomBtnStyle: React.CSSProperties = {
  width: '24px',
  height: '24px',
  backgroundColor: '#111827',
  border: '1px solid #1E2739',
  borderRadius: '4px',
  color: '#64748B',
  fontSize: '14px',
  fontWeight: 600,
  cursor: 'pointer',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  lineHeight: 1,
  fontFamily: 'monospace',
};

function toolbarBtnStyle(active: boolean): React.CSSProperties {
  return {
    padding: '4px 10px',
    backgroundColor: active ? '#0D2E28' : '#111827',
    border: `1px solid ${active ? '#22D3A5' : '#1E2739'}`,
    borderRadius: '6px',
    color: active ? '#22D3A5' : '#64748B',
    fontSize: '11px',
    cursor: 'pointer',
    fontFamily: 'inherit',
    fontWeight: active ? 600 : 400,
    whiteSpace: 'nowrap',
  };
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  const { template, setTemplate, canvasSizeId, setCanvasSizeId, canvasSize } =
    useDashboard();

  const [zoom, setZoom] = useState(0.55);
  const [jsonOpen, setJsonOpen] = useState(false);

  const handleSizeChange = (id: CanvasPresetId) => {
    setCanvasSizeId(id);
    // Auto-fit zoom for the new canvas size
    setZoom(id === '1080x1080' ? 0.62 : 0.55);
  };

  const scaledW = Math.round(canvasSize.w * zoom);
  const scaledH = Math.round(canvasSize.h * zoom);

  return (
    <div
      style={{
        display: 'flex',
        height: '100vh',
        overflow: 'hidden',
        backgroundColor: '#070A0F',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      {/* ── Theme sidebar ────────────────────────────────── */}
      <ThemeSidebar />

      {/* ── Main area ──────────────────────────────────── */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
        }}
      >
        {/* ── Toolbar ──────────────────────────────────── */}
        <div
          style={{
            height: '46px',
            backgroundColor: '#0D1117',
            borderBottom: '1px solid #1A2035',
            display: 'flex',
            alignItems: 'center',
            padding: '0 14px',
            gap: '10px',
            flexShrink: 0,
          }}
        >
          {/* Template selector */}
          <ToolbarGroup label="Template">
            <select
              value={template}
              onChange={(e) => setTemplate(e.target.value as TemplateId)}
              style={selectStyle}
            >
              {(Object.entries(templateLabels) as [TemplateId, string][]).map(
                ([id, label]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                )
              )}
            </select>
          </ToolbarGroup>

          <ToolbarDivider />

          {/* Size selector */}
          <ToolbarGroup label="Size">
            <select
              value={canvasSizeId}
              onChange={(e) => handleSizeChange(e.target.value as CanvasPresetId)}
              style={selectStyle}
            >
              {(Object.entries(canvasPresets) as [CanvasPresetId, { label: string }][]).map(
                ([id, { label }]) => (
                  <option key={id} value={id}>
                    {label}
                  </option>
                )
              )}
            </select>
          </ToolbarGroup>

          {/* Canvas badge */}
          <div
            style={{
              padding: '3px 8px',
              backgroundColor: '#0A1020',
              border: '1px solid #141C2E',
              borderRadius: '4px',
              fontSize: '9px',
              color: '#334155',
              fontFamily: 'monospace',
            }}
          >
            {canvasSize.w} × {canvasSize.h}
          </div>

          <ToolbarDivider />

          {/* JSON panel toggle */}
          <button
            onClick={() => setJsonOpen((v) => !v)}
            style={toolbarBtnStyle(jsonOpen)}
          >
            {jsonOpen ? '✕  Close JSON' : '{  }  Edit JSON'}
          </button>

          {/* Zoom — pushed right */}
          <div
            style={{
              marginLeft: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
            }}
          >
            <button
              onClick={() => setZoom((z) => Math.max(0.2, +(z - 0.05).toFixed(2)))}
              style={zoomBtnStyle}
            >
              −
            </button>
            <input
              type="range"
              min={0.2}
              max={1.0}
              step={0.05}
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              style={{ width: '100px', cursor: 'pointer' }}
            />
            <button
              onClick={() => setZoom((z) => Math.min(1.0, +(z + 0.05).toFixed(2)))}
              style={zoomBtnStyle}
            >
              +
            </button>
            <span
              style={{
                fontSize: '11px',
                color: '#64748B',
                fontFamily: 'monospace',
                minWidth: '36px',
                textAlign: 'right',
              }}
            >
              {Math.round(zoom * 100)}%
            </span>
          </div>
        </div>

        {/* ── Middle row: canvas + optional JSON panel ─── */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>
          {/* Canvas scroll area */}
          <div
            style={{
              flex: 1,
              overflow: 'auto',
              backgroundColor: '#070A0F',
              backgroundImage:
                'radial-gradient(circle, #1a2035 1px, transparent 1px)',
              backgroundSize: '28px 28px',
            }}
          >
            <div
              style={{
                display: 'flex',
                justifyContent: 'center',
                padding: '40px',
                minHeight: `${scaledH + 80}px`,
                minWidth: `${scaledW + 80}px`,
              }}
            >
              {/* Size proxy — holds the correct scroll dimensions */}
              <div
                style={{
                  position: 'relative',
                  width: `${scaledW}px`,
                  height: `${scaledH}px`,
                  flexShrink: 0,
                  boxShadow:
                    '0 0 0 1px rgba(255,255,255,0.06), 0 24px 80px rgba(0,0,0,0.7)',
                  borderRadius: '3px',
                }}
              >
                {/* Canvas scaled from top-left */}
                <div
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    transformOrigin: 'top left',
                    transform: `scale(${zoom})`,
                  }}
                >
                  <Dashboard />
                </div>
              </div>
            </div>
          </div>

          {/* JSON panel — pushes the canvas left when open */}
          {jsonOpen && <JsonPanel onClose={() => setJsonOpen(false)} />}
        </div>

        {/* ── Status bar ──────────────────────────────────── */}
        <StatusBar />
      </div>
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ToolbarGroup({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
      <span
        style={{
          fontSize: '9px',
          color: '#334155',
          textTransform: 'uppercase',
          letterSpacing: '0.1em',
        }}
      >
        {label}
      </span>
      {children}
    </div>
  );
}

function ToolbarDivider() {
  return (
    <div
      style={{
        width: '1px',
        height: '18px',
        backgroundColor: '#1A2035',
        flexShrink: 0,
      }}
    />
  );
}

function StatusBar() {
  const { template, canvasSizeId, data } = useDashboard();

  return (
    <div
      style={{
        height: '28px',
        backgroundColor: '#0D1117',
        borderTop: '1px solid #1A2035',
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        gap: '12px',
        flexShrink: 0,
      }}
    >
      <StatusDot color="#22D3A5" label={data.left.ticker} />
      {data.right && <StatusDot color="#F45B69" label={data.right.ticker} />}

      <div style={{ width: '1px', height: '10px', backgroundColor: '#1A2035' }} />

      <span style={{ fontSize: '10px', color: '#1E293B' }}>
        {templateLabels[template]}
      </span>
      <span style={{ fontSize: '10px', color: '#1E293B' }}>{canvasSizeId}</span>

      <span style={{ fontSize: '10px', color: '#1E293B', marginLeft: 'auto' }}>
        Part 2 — JSON · Templates · Sizes
      </span>
    </div>
  );
}

function StatusDot({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
      <div
        style={{ width: '6px', height: '6px', borderRadius: '50%', backgroundColor: color }}
      />
      <span style={{ fontSize: '10px', color: '#334155', fontWeight: 600 }}>
        {label}
      </span>
    </div>
  );
}
