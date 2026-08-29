'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

type Prediction = { operational_rul: number; risk_severity: string; window_end_cycle: number };
type Alert = { id: number; asset_id: string; severity: string; status: string; active: boolean };
type WorkOrder = { id: number; asset_id: string; status: string; assigned_to: string | null };
type Asset = { asset_id: string; name: string; telemetry_count: number; prediction_count: number; latest_cycle: number | null; latest_prediction: Prediction | null; active_alert: Alert | null; active_work_order: WorkOrder | null };

const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
const demoAssets: Asset[] = [
  { asset_id: 'fd001-unit-81', name: 'Turbofan 81', telemetry_count: 240, prediction_count: 211, latest_cycle: 240, latest_prediction: { operational_rul: 7.4, risk_severity: 'CRITICAL', window_end_cycle: 240 }, active_alert: { id: 14, asset_id: 'fd001-unit-81', severity: 'CRITICAL', status: 'OPEN', active: true }, active_work_order: { id: 7, asset_id: 'fd001-unit-81', status: 'OPEN', assigned_to: null } },
  { asset_id: 'fd001-unit-36', name: 'Turbofan 36', telemetry_count: 180, prediction_count: 151, latest_cycle: 180, latest_prediction: { operational_rul: 24.8, risk_severity: 'HIGH', window_end_cycle: 180 }, active_alert: { id: 11, asset_id: 'fd001-unit-36', severity: 'HIGH', status: 'ACKNOWLEDGED', active: true }, active_work_order: null },
  { asset_id: 'fd001-unit-12', name: 'Turbofan 12', telemetry_count: 126, prediction_count: 97, latest_cycle: 126, latest_prediction: { operational_rul: 58.2, risk_severity: 'MEDIUM', window_end_cycle: 126 }, active_alert: { id: 9, asset_id: 'fd001-unit-12', severity: 'MEDIUM', status: 'OPEN', active: true }, active_work_order: null },
  { asset_id: 'fd001-unit-4', name: 'Turbofan 04', telemetry_count: 92, prediction_count: 63, latest_cycle: 92, latest_prediction: { operational_rul: 104.6, risk_severity: 'NORMAL', window_end_cycle: 92 }, active_alert: null, active_work_order: null },
];

function severityClass(severity = 'NORMAL') { return `severity severity-${severity.toLowerCase()}`; }

export default function Home() {
  const [assets, setAssets] = useState<Asset[]>(demoAssets);
  const [mode, setMode] = useState<'connecting' | 'live' | 'demo'>('connecting');
  const [selectedId, setSelectedId] = useState(demoAssets[0].asset_id);
  const [updatedAt, setUpdatedAt] = useState(new Date());

  const refresh = useCallback(async () => {
    try {
      const listResponse = await fetch(`${API_BASE}/v1/assets`, { cache: 'no-store' });
      if (!listResponse.ok) throw new Error('API unavailable');
      const list: { asset_id: string }[] = await listResponse.json();
      const details = await Promise.all(list.map(async ({ asset_id }) => {
        const response = await fetch(`${API_BASE}/v1/assets/${encodeURIComponent(asset_id)}`, { cache: 'no-store' });
        if (!response.ok) throw new Error('Asset detail unavailable');
        return response.json() as Promise<Asset>;
      }));
      setAssets(details);
      if (details.length) {
        setSelectedId((current) => details.some((asset) => asset.asset_id === current) ? current : details[0].asset_id);
      }
      setMode('live');
    } catch { setAssets(demoAssets); setMode('demo'); }
    setUpdatedAt(new Date());
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);
  const selected = assets.find((asset) => asset.asset_id === selectedId) || assets[0];
  const counts = useMemo(() => assets.reduce((acc, asset) => { const severity = asset.latest_prediction?.risk_severity || 'NORMAL'; acc[severity] = (acc[severity] || 0) + 1; return acc; }, {} as Record<string, number>), [assets]);
  const activeAlerts = assets.filter((asset) => asset.active_alert).length;
  const openOrders = assets.filter((asset) => asset.active_work_order && asset.active_work_order.status !== 'RESOLVED').length;

  return <main>
    <header className="topbar"><div className="brand"><span className="brand-mark">AR</span><div><strong>Asset Reliability</strong><span>Operations console</span></div></div><div className="top-actions"><span className={`connection ${mode}`}><i />{mode === 'live' ? 'Live API' : mode === 'demo' ? 'Demo snapshot' : 'Connecting'}</span><button className="refresh" onClick={() => void refresh()} aria-label="Refresh operations data">↻ <span>Refresh</span></button></div></header>
    <section className="page-heading"><div><p className="eyebrow">Fleet overview</p><h1>Reliability control room</h1><p>Prioritize maintenance by predicted remaining life and active operational risk.</p></div><div className="timestamp"><span>Last updated</span><strong>{updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</strong></div></section>
    <section className="metric-grid" aria-label="Fleet health summary">
      <article className="metric primary"><span className="metric-label">Monitored assets</span><strong>{assets.length}</strong><small>{assets.reduce((sum, asset) => sum + asset.telemetry_count, 0).toLocaleString()} telemetry cycles</small></article>
      <article className="metric"><span className="metric-label">Critical</span><strong className="critical-text">{counts.CRITICAL || 0}</strong><small>Immediate intervention</small></article>
      <article className="metric"><span className="metric-label">Active alerts</span><strong>{activeAlerts}</strong><small>{counts.HIGH || 0} high-risk assets</small></article>
      <article className="metric"><span className="metric-label">Open work orders</span><strong>{openOrders}</strong><small>Maintenance response</small></article>
    </section>
    <section className="workspace">
      <article className="panel fleet-panel"><div className="panel-title"><div><p className="eyebrow">Priority queue</p><h2>Asset health</h2></div><span>{assets.length} assets</span></div><div className="asset-list">{assets.map((asset) => { const prediction = asset.latest_prediction; const severity = prediction?.risk_severity || 'NORMAL'; return <button key={asset.asset_id} className={`asset-row ${asset.asset_id === selected?.asset_id ? 'selected' : ''}`} onClick={() => setSelectedId(asset.asset_id)}><span className={severityClass(severity)}>{severity}</span><span className="asset-name"><strong>{asset.name}</strong><small>{asset.asset_id} · Cycle {asset.latest_cycle ?? '—'}</small></span><span className="rul"><strong>{prediction ? prediction.operational_rul.toFixed(1) : '—'}</strong><small>cycles RUL</small></span></button>; })}</div></article>
      <aside className="panel detail-panel">{selected && <><div className="panel-title"><div><p className="eyebrow">Selected asset</p><h2>{selected.name}</h2></div><span className={severityClass(selected.latest_prediction?.risk_severity)}>{selected.latest_prediction?.risk_severity || 'NO DATA'}</span></div><div className="rul-hero"><span>Remaining useful life</span><div><strong>{selected.latest_prediction?.operational_rul.toFixed(1) || '—'}</strong><em>cycles</em></div><p>GRU model v1.0.0 · through cycle {selected.latest_cycle || '—'}</p></div><div className="detail-stats"><div><span>Predictions</span><strong>{selected.prediction_count}</strong></div><div><span>Telemetry</span><strong>{selected.telemetry_count}</strong></div></div><div className="workflow-card"><div className="workflow-heading"><span>Alert workflow</span><strong>{selected.active_alert?.status || 'CLEAR'}</strong></div>{selected.active_alert ? <><p>{selected.active_alert.severity} risk alert #{selected.active_alert.id}</p><div className="progress"><i /><i className={selected.active_alert.status !== 'OPEN' ? 'done' : ''} /><i className={selected.active_alert.status === 'RESOLVED' ? 'done' : ''} /></div></> : <p>No active maintenance alert.</p>}</div><div className="workflow-card"><div className="workflow-heading"><span>Work order</span><strong>{selected.active_work_order?.status || 'NOT REQUIRED'}</strong></div><p>{selected.active_work_order ? `Order #${selected.active_work_order.id} · ${selected.active_work_order.assigned_to || 'Unassigned'}` : 'Created automatically at critical risk.'}</p></div></>}</aside>
    </section>
    {mode === 'demo' && <p className="demo-note">Showing an identified FD001 demonstration snapshot. Start the backend to view persisted operational data.</p>}
  </main>;
}
