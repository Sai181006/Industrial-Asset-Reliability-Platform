'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';

type Prediction = { operational_rul: number; risk_severity: string; window_end_cycle: number };
type Alert = { id: number; asset_id: string; severity: string; status: string; active: boolean; last_updated_by?: string | null };
type WorkOrder = { id: number; asset_id: string; status: string; assigned_to: string | null; notes?: string | null; last_updated_by?: string | null };
type Asset = { asset_id: string; name: string; telemetry_count: number; prediction_count: number; latest_cycle: number | null; latest_prediction: Prediction | null; active_alert: Alert | null; active_work_order: WorkOrder | null };
type Principal = { subject: string; role: string };

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
  const [token, setToken] = useState('');
  const [principal, setPrincipal] = useState<Principal | null>(null);
  const [assignee, setAssignee] = useState('');
  const [notes, setNotes] = useState('');
  const [workflowMessage, setWorkflowMessage] = useState('');
  const [workflowBusy, setWorkflowBusy] = useState(false);

  useEffect(() => {
    const timer = window.setTimeout(() => setToken(window.sessionStorage.getItem('reliability-api-token') || ''), 0);
    return () => window.clearTimeout(timer);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const headers = token ? { Authorization: `Bearer ${token}` } : undefined;
      const identityResponse = await fetch(`${API_BASE}/v1/auth/me`, { cache: 'no-store', headers });
      if (!identityResponse.ok) throw new Error('Authentication required');
      const identity = await identityResponse.json() as Principal;
      const listResponse = await fetch(`${API_BASE}/v1/assets`, { cache: 'no-store', headers });
      if (!listResponse.ok) throw new Error('API unavailable');
      const list: { asset_id: string }[] = await listResponse.json();
      const details = await Promise.all(list.map(async ({ asset_id }) => {
        const response = await fetch(`${API_BASE}/v1/assets/${encodeURIComponent(asset_id)}`, { cache: 'no-store', headers });
        if (!response.ok) throw new Error('Asset detail unavailable');
        return response.json() as Promise<Asset>;
      }));
      setAssets(details);
      if (details.length) {
        setSelectedId((current) => details.some((asset) => asset.asset_id === current) ? current : details[0].asset_id);
      }
      setMode('live');
      setPrincipal(identity);
    } catch { setAssets(demoAssets); setMode('demo'); setPrincipal(null); }
    setUpdatedAt(new Date());
  }, [token]);

  const connect = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const value = new FormData(event.currentTarget).get('token')?.toString().trim() || '';
    if (value) window.sessionStorage.setItem('reliability-api-token', value);
    else window.sessionStorage.removeItem('reliability-api-token');
    setToken(value);
  };

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
  }, [refresh]);
  const selected = assets.find((asset) => asset.asset_id === selectedId) || assets[0];
  const canOperate = mode === 'live' && (principal?.role === 'operator' || principal?.role === 'admin');

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setAssignee(selected?.active_work_order?.assigned_to || '');
      setNotes(selected?.active_work_order?.notes || '');
      setWorkflowMessage('');
    }, 0);
    return () => window.clearTimeout(timer);
  }, [selectedId, selected?.active_work_order?.assigned_to, selected?.active_work_order?.notes]);

  const updateWorkflow = async (path: string, body: Record<string, string>) => {
    setWorkflowBusy(true);
    setWorkflowMessage('');
    try {
      const response = await fetch(`${API_BASE}${path}`, {
        method: 'PATCH',
        headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Workflow update failed' })) as { detail?: string };
        throw new Error(error.detail || 'Workflow update failed');
      }
      setWorkflowMessage('Workflow updated successfully.');
      await refresh();
    } catch (error) {
      setWorkflowMessage(error instanceof Error ? error.message : 'Workflow update failed');
    } finally { setWorkflowBusy(false); }
  };

  const advanceAlert = () => {
    const alert = selected?.active_alert;
    if (!alert) return;
    const next = alert.status === 'OPEN' ? 'ACKNOWLEDGED' : alert.status === 'ACKNOWLEDGED' ? 'RESOLVED' : null;
    if (next) void updateWorkflow(`/v1/alerts/${alert.id}`, { status: next });
  };

  const advanceWorkOrder = () => {
    const order = selected?.active_work_order;
    if (!order) return;
    const next = order.status === 'OPEN' ? 'ASSIGNED' : order.status === 'ASSIGNED' ? 'IN_PROGRESS' : order.status === 'IN_PROGRESS' ? 'RESOLVED' : null;
    if (!next) return;
    const body: Record<string, string> = { status: next };
    if (assignee.trim()) body.assigned_to = assignee.trim();
    if (notes.trim()) body.notes = notes.trim();
    void updateWorkflow(`/v1/work-orders/${order.id}`, body);
  };
  const counts = useMemo(() => assets.reduce((acc, asset) => { const severity = asset.latest_prediction?.risk_severity || 'NORMAL'; acc[severity] = (acc[severity] || 0) + 1; return acc; }, {} as Record<string, number>), [assets]);
  const activeAlerts = assets.filter((asset) => asset.active_alert).length;
  const openOrders = assets.filter((asset) => asset.active_work_order && asset.active_work_order.status !== 'RESOLVED').length;

  return <main>
    <header className="topbar"><div className="brand"><span className="brand-mark">AR</span><div><strong>Asset Reliability</strong><span>Operations console</span></div></div><div className="top-actions">{principal && <span className="principal">{principal.subject} · {principal.role}</span>}<span className={`connection ${mode}`}><i />{mode === 'live' ? 'Authenticated API' : mode === 'demo' ? 'Demo snapshot' : 'Connecting'}</span><button className="refresh" onClick={() => void refresh()} aria-label="Refresh operations data">↻ <span>Refresh</span></button></div></header>
    <section className="page-heading"><div><p className="eyebrow">Fleet overview</p><h1>Reliability control room</h1><p>Prioritize maintenance by predicted remaining life and active operational risk.</p></div><div className="timestamp"><span>Last updated</span><strong>{updatedAt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</strong></div></section>
    <section className="metric-grid" aria-label="Fleet health summary">
      <article className="metric primary"><span className="metric-label">Monitored assets</span><strong>{assets.length}</strong><small>{assets.reduce((sum, asset) => sum + asset.telemetry_count, 0).toLocaleString()} telemetry cycles</small></article>
      <article className="metric"><span className="metric-label">Critical</span><strong className="critical-text">{counts.CRITICAL || 0}</strong><small>Immediate intervention</small></article>
      <article className="metric"><span className="metric-label">Active alerts</span><strong>{activeAlerts}</strong><small>{counts.HIGH || 0} high-risk assets</small></article>
      <article className="metric"><span className="metric-label">Open work orders</span><strong>{openOrders}</strong><small>Maintenance response</small></article>
    </section>
    <section className="workspace">
      <article className="panel fleet-panel"><div className="panel-title"><div><p className="eyebrow">Priority queue</p><h2>Asset health</h2></div><span>{assets.length} assets</span></div><div className="asset-list">{assets.map((asset) => { const prediction = asset.latest_prediction; const severity = prediction?.risk_severity || 'NORMAL'; return <button key={asset.asset_id} className={`asset-row ${asset.asset_id === selected?.asset_id ? 'selected' : ''}`} onClick={() => setSelectedId(asset.asset_id)}><span className={severityClass(severity)}>{severity}</span><span className="asset-name"><strong>{asset.name}</strong><small>{asset.asset_id} · Cycle {asset.latest_cycle ?? '—'}</small></span><span className="rul"><strong>{prediction ? prediction.operational_rul.toFixed(1) : '—'}</strong><small>cycles RUL</small></span></button>; })}</div></article>
      <aside className="panel detail-panel">{selected && <><div className="panel-title"><div><p className="eyebrow">Selected asset</p><h2>{selected.name}</h2></div><span className={severityClass(selected.latest_prediction?.risk_severity)}>{selected.latest_prediction?.risk_severity || 'NO DATA'}</span></div><div className="rul-hero"><span>Remaining useful life</span><div><strong>{selected.latest_prediction?.operational_rul.toFixed(1) || '—'}</strong><em>cycles</em></div><p>GRU model v1.0.0 · through cycle {selected.latest_cycle || '—'}</p></div><div className="detail-stats"><div><span>Predictions</span><strong>{selected.prediction_count}</strong></div><div><span>Telemetry</span><strong>{selected.telemetry_count}</strong></div></div><div className="workflow-card"><div className="workflow-heading"><span>Alert workflow</span><strong>{selected.active_alert?.status || 'CLEAR'}</strong></div>{selected.active_alert ? <><p>{selected.active_alert.severity} risk alert #{selected.active_alert.id}{selected.active_alert.last_updated_by ? ` · Updated by ${selected.active_alert.last_updated_by}` : ''}</p><div className="progress"><i /><i className={selected.active_alert.status !== 'OPEN' ? 'done' : ''} /><i className={selected.active_alert.status === 'RESOLVED' ? 'done' : ''} /></div>{canOperate && selected.active_alert.status !== 'RESOLVED' && <button className="workflow-action" disabled={workflowBusy || (selected.active_alert.status === 'ACKNOWLEDGED' && !!selected.active_work_order && selected.active_work_order.status !== 'RESOLVED')} onClick={advanceAlert}>{selected.active_alert.status === 'OPEN' ? 'Acknowledge alert' : 'Resolve alert'}</button>}</> : <p>No active maintenance alert.</p>}</div><div className="workflow-card"><div className="workflow-heading"><span>Work order</span><strong>{selected.active_work_order?.status || 'NOT REQUIRED'}</strong></div><p>{selected.active_work_order ? `Order #${selected.active_work_order.id} · ${selected.active_work_order.assigned_to || 'Unassigned'}${selected.active_work_order.last_updated_by ? ` · Updated by ${selected.active_work_order.last_updated_by}` : ''}` : 'Created automatically at critical risk.'}</p>{canOperate && selected.active_work_order && selected.active_work_order.status !== 'RESOLVED' && <div className="workflow-form">{selected.active_work_order.status === 'OPEN' && <input value={assignee} onChange={(event) => setAssignee(event.target.value)} placeholder="Assignee" aria-label="Work-order assignee" />}<textarea value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Maintenance notes (optional)" aria-label="Maintenance notes" rows={2} /><button className="workflow-action" disabled={workflowBusy || (selected.active_work_order.status === 'OPEN' && !assignee.trim())} onClick={advanceWorkOrder}>{selected.active_work_order.status === 'OPEN' ? 'Assign work order' : selected.active_work_order.status === 'ASSIGNED' ? 'Start work' : 'Resolve work order'}</button></div>}</div>{mode === 'live' && principal?.role === 'viewer' && <p className="read-only-note">Viewer access is read-only. An operator or admin must advance workflows.</p>}{workflowMessage && <p className={`workflow-message ${workflowMessage.includes('successfully') ? 'success' : 'error'}`} role="status">{workflowMessage}</p>}</>}</aside>
    </section>
    {mode === 'demo' && <section className="access-panel"><div><strong>Connect to the operations API</strong><p>Enter an issued bearer credential to view protected fleet data. It is retained only for this browser tab.</p></div><form onSubmit={connect}><input name="token" type="password" defaultValue={token} placeholder="Bearer credential" aria-label="API bearer credential" autoComplete="off" /><button type="submit">Connect</button></form><small>Showing an identified FD001 demonstration snapshot until authenticated.</small></section>}
  </main>;
}
