'use client';

import { useState, useEffect, useRef } from 'react';
import { useDashboard } from '@/contexts/DashboardContext';
import { validateDashboardData, DashboardData } from '@/lib/schema';
import { templateDefaults } from '@/lib/defaultData';

type Props = {
  onClose: () => void;
};

export function JsonPanel({ onClose }: Props) {
  const { data, setData, template } = useDashboard();
  const [jsonText, setJsonText] = useState('');
  const [errors, setErrors] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync textarea when data changes from outside (e.g. template switch)
  useEffect(() => {
    setJsonText(JSON.stringify(data, null, 2));
    setErrors([]);
    setStatus('idle');
  }, [data]);

  const handleApply = () => {
    setErrors([]);
    setStatus('idle');

    let parsed: unknown;
    try {
      parsed = JSON.parse(jsonText);
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Invalid JSON';
      setErrors([`JSON parse error: ${msg}`]);
      setStatus('error');
      return;
    }

    const result = validateDashboardData(parsed);
    if (!result.valid) {
      setErrors(result.errors);
      setStatus('error');
      return;
    }

    setData(parsed as DashboardData);
    setStatus('success');
    setTimeout(() => setStatus('idle'), 2500);
  };

  const handleReset = () => {
    const def = templateDefaults[template];
    setJsonText(JSON.stringify(def, null, 2));
    setData(def);
    setErrors([]);
    setStatus('idle');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Tab inserts 2 spaces
    if (e.key === 'Tab') {
      e.preventDefault();
      const ta = textareaRef.current!;
      const start = ta.selectionStart;
      const end = ta.selectionEnd;
      const newVal = jsonText.slice(0, start) + '  ' + jsonText.slice(end);
      setJsonText(newVal);
      requestAnimationFrame(() => {
        ta.selectionStart = ta.selectionEnd = start + 2;
      });
    }
    // Ctrl+Enter or Cmd+Enter applies
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      handleApply();
    }
  };

  const lineCount = jsonText.split('\n').length;

  return (
    <div
      style={{
        width: '340px',
        height: '100%',
        backgroundColor: '#080C14',
        borderLeft: '1px solid #1A2035',
        display: 'flex',
        flexDirection: 'column',
        fontFamily: 'Inter, system-ui, sans-serif',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          height: '46px',
          borderBottom: '1px solid #1A2035',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 14px',
          flexShrink: 0,
        }}
      >
        <div>
          <div style={{ fontSize: '12px', fontWeight: 700, color: '#E2E8F0' }}>
            JSON Data Editor
          </div>
          <div style={{ fontSize: '10px', color: '#334155' }}>
            Ctrl+Enter to apply
          </div>
        </div>
        <button
          onClick={onClose}
          style={{
            width: '24px',
            height: '24px',
            backgroundColor: 'transparent',
            border: '1px solid #1E2739',
            borderRadius: '4px',
            color: '#475569',
            fontSize: '14px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            lineHeight: 1,
          }}
          title="Close panel"
        >
          ×
        </button>
      </div>

      {/* Schema hint */}
      <div
        style={{
          padding: '8px 14px',
          borderBottom: '1px solid #1A2035',
          flexShrink: 0,
        }}
      >
        <div style={{ fontSize: '9px', color: '#334155', lineHeight: 1.6, fontFamily: 'monospace' }}>
          <span style={{ color: '#475569' }}>Required: </span>
          meta · left.ticker · left.name
          <br />
          <span style={{ color: '#475569' }}>Optional blocks: </span>
          metrics · series · valuation ·<br />
          &nbsp;&nbsp;&nbsp;reinvestment · profitability · returns · risk
          <br />
          <span style={{ color: '#475569' }}>right</span> block optional (compare template)
        </div>
      </div>

      {/* Textarea */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', position: 'relative' }}>
        {/* Line numbers */}
        <div
          style={{
            width: '36px',
            backgroundColor: '#06090F',
            borderRight: '1px solid #141C2E',
            overflowY: 'hidden',
            paddingTop: '12px',
            flexShrink: 0,
          }}
        >
          {Array.from({ length: lineCount }, (_, i) => (
            <div
              key={i}
              style={{
                fontSize: '10px',
                color: '#263354',
                textAlign: 'right',
                paddingRight: '8px',
                lineHeight: '1.5',
                fontFamily: 'monospace',
                height: '15px',
              }}
            >
              {i + 1}
            </div>
          ))}
        </div>

        <textarea
          ref={textareaRef}
          value={jsonText}
          onChange={(e) => {
            setJsonText(e.target.value);
            if (status !== 'idle') setStatus('idle');
          }}
          onKeyDown={handleKeyDown}
          spellCheck={false}
          style={{
            flex: 1,
            resize: 'none',
            backgroundColor: '#06090F',
            color: '#94A3B8',
            fontFamily: 'ui-monospace, "Cascadia Code", "Fira Code", monospace',
            fontSize: '11px',
            lineHeight: '1.5',
            border: 'none',
            outline: 'none',
            padding: '12px 12px',
            overflowY: 'auto',
            overflowX: 'auto',
            whiteSpace: 'pre',
            height: '100%',
          }}
        />
      </div>

      {/* Status / errors */}
      {(status === 'error' || status === 'success') && (
        <div
          style={{
            padding: '10px 14px',
            borderTop: '1px solid #1A2035',
            flexShrink: 0,
            maxHeight: '120px',
            overflowY: 'auto',
          }}
        >
          {status === 'success' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ color: '#22D3A5', fontSize: '12px' }}>✓</span>
              <span style={{ fontSize: '11px', color: '#22D3A5', fontWeight: 600 }}>
                Data applied successfully
              </span>
            </div>
          )}
          {status === 'error' && (
            <div>
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  marginBottom: '6px',
                }}
              >
                <span style={{ color: '#F45B69', fontSize: '12px' }}>✕</span>
                <span style={{ fontSize: '11px', color: '#F45B69', fontWeight: 600 }}>
                  Validation failed
                </span>
              </div>
              {errors.map((err, i) => (
                <div
                  key={i}
                  style={{
                    fontSize: '10px',
                    color: '#94A3B8',
                    paddingLeft: '16px',
                    lineHeight: 1.6,
                    fontFamily: 'monospace',
                  }}
                >
                  • {err}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div
        style={{
          padding: '10px 14px',
          borderTop: '1px solid #1A2035',
          display: 'flex',
          gap: '8px',
          flexShrink: 0,
        }}
      >
        <button
          onClick={handleReset}
          style={{
            flex: 1,
            padding: '8px',
            backgroundColor: '#111827',
            border: '1px solid #1E2739',
            borderRadius: '6px',
            color: '#64748B',
            fontSize: '11px',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontWeight: 500,
          }}
        >
          Reset Default
        </button>
        <button
          onClick={handleApply}
          style={{
            flex: 2,
            padding: '8px',
            backgroundColor: '#0D2E28',
            border: '1px solid #22D3A5',
            borderRadius: '6px',
            color: '#22D3A5',
            fontSize: '11px',
            cursor: 'pointer',
            fontFamily: 'inherit',
            fontWeight: 600,
            letterSpacing: '0.02em',
          }}
        >
          Apply Data ↵
        </button>
      </div>
    </div>
  );
}
