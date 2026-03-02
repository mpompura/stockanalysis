'use client';

import { useState } from 'react';
import { Dashboard } from '@/components/Dashboard';
import { ThemeSidebar } from '@/components/ThemeSidebar';

const CANVAS_W = 1080;
const CANVAS_H = 1350;

export default function Home() {
  const [zoom, setZoom] = useState(0.55);

  const scaledW = Math.round(CANVAS_W * zoom);
  const scaledH = Math.round(CANVAS_H * zoom);

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
      {/* ── Left sidebar ───────────────────────────────────── */}
      <ThemeSidebar />

      {/* ── Main area ──────────────────────────────────────── */}
      <div
        style={{
          flex: 1,
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
          minWidth: 0,
        }}
      >
        {/* Toolbar */}
        <div
          style={{
            height: '46px',
            backgroundColor: '#0D1117',
            borderBottom: '1px solid #1A2035',
            display: 'flex',
            alignItems: 'center',
            padding: '0 18px',
            gap: '14px',
            flexShrink: 0,
          }}
        >
          {/* Canvas badge */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              backgroundColor: '#111827',
              border: '1px solid #1A2035',
              borderRadius: '6px',
            }}
          >
            <span style={{ fontSize: '10px', color: '#475569' }}>Canvas</span>
            <span
              style={{
                fontSize: '10px',
                color: '#64748B',
                fontFamily: 'monospace',
              }}
            >
              {CANVAS_W} × {CANVAS_H}
            </span>
          </div>

          {/* Zoom controls — pushed to right */}
          <div
            style={{
              marginLeft: 'auto',
              display: 'flex',
              alignItems: 'center',
              gap: '10px',
            }}
          >
            <button
              onClick={() => setZoom((z) => Math.max(0.2, +(z - 0.05).toFixed(2)))}
              style={zoomBtnStyle}
              title="Zoom out"
            >
              −
            </button>
            <input
              type="range"
              min={0.25}
              max={1.0}
              step={0.05}
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
              style={{ width: '120px', cursor: 'pointer' }}
            />
            <button
              onClick={() => setZoom((z) => Math.min(1.0, +(z + 0.05).toFixed(2)))}
              style={zoomBtnStyle}
              title="Zoom in"
            >
              +
            </button>
            <span
              style={{
                fontSize: '11px',
                color: '#64748B',
                fontFamily: 'monospace',
                minWidth: '38px',
                textAlign: 'right',
              }}
            >
              {Math.round(zoom * 100)}%
            </span>
            <button onClick={() => setZoom(0.55)} style={resetBtnStyle}>
              Reset
            </button>
          </div>
        </div>

        {/* ── Canvas scroll area ─────────────────────────────── */}
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
          {/* Centred wrapper — sets correct scroll dimensions */}
          <div
            style={{
              display: 'flex',
              justifyContent: 'center',
              padding: '40px',
              minHeight: `${scaledH + 80}px`,
              minWidth: `${scaledW + 80}px`,
            }}
          >
            {/* Size-proxy: takes up scaled dimensions in flow */}
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
              {/* Actual canvas — scaled from top-left */}
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

        {/* ── Status bar ─────────────────────────────────────── */}
        <div
          style={{
            height: '28px',
            backgroundColor: '#0D1117',
            borderTop: '1px solid #1A2035',
            display: 'flex',
            alignItems: 'center',
            padding: '0 16px',
            gap: '14px',
            flexShrink: 0,
          }}
        >
          <StatusDot color="#22D3A5" label="NVDA" />
          <StatusDot color="#F45B69" label="MSFT" />
          <span
            style={{
              fontSize: '10px',
              color: '#1E293B',
              marginLeft: 'auto',
              letterSpacing: '0.02em',
            }}
          >
            Part 1 — Static Mock Data
          </span>
        </div>
      </div>
    </div>
  );
}

function StatusDot({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
      <div
        style={{
          width: '6px',
          height: '6px',
          borderRadius: '50%',
          backgroundColor: color,
        }}
      />
      <span style={{ fontSize: '10px', color: '#334155', fontWeight: 600 }}>
        {label}
      </span>
    </div>
  );
}

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

const resetBtnStyle: React.CSSProperties = {
  padding: '3px 10px',
  backgroundColor: '#111827',
  border: '1px solid #1E2739',
  borderRadius: '4px',
  color: '#64748B',
  fontSize: '10px',
  cursor: 'pointer',
  fontFamily: 'inherit',
};
