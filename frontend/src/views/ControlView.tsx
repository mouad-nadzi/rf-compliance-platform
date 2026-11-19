import { useState, useEffect } from 'react';
import { api } from '../api';
import { useAuth } from '../context/AuthContext';
import { Play, Loader, ShieldCheck, Clock, Zap, Plus, Minus, BellRing, Mail, Lock, Copy, Check, Unlink, ChevronDown, ChevronUp } from 'lucide-react';

type Config = {
  enabled: boolean;
  require_approval: boolean;
  interval_hours: number;
  cron_schedule: string;
};

type AlertingConfig = {
  enabled: boolean;
  require_approval: boolean;
  interval_hours: number;
  alert_days_before: number;
  fallback_email: string;
  sender_email: string;
  has_smtp_password: boolean;
  google_client_id: string;
  has_google_client_secret: boolean;
  google_connected: boolean;
  google_connected_email: string;
  in_window_count: number;
};

const ToggleSwitch = ({ checked, onChange, disabled }: { checked: boolean; onChange: () => void; disabled?: boolean }) => (
  <button 
    type="button"
    role="switch"
    aria-checked={checked}
    onClick={onChange}
    disabled={disabled}
    style={{
      width: '46px',
      height: '24px',
      borderRadius: '12px',
      backgroundColor: checked ? 'var(--brand-blue)' : 'transparent',
      border: '1px solid #000000',
      boxShadow: 'none',
      position: 'relative',
      cursor: disabled ? 'not-allowed' : 'pointer',
      transition: 'all 200ms ease',
      padding: 0,
      outline: 'none',
      flexShrink: 0
    }}
  >
    <span 
      style={{
        width: '18px',
        height: '18px',
        borderRadius: '50%',
        backgroundColor: '#ffffff',
        position: 'absolute',
        top: '2px',
        left: checked ? '24px' : '2px',
        transition: 'left 200ms cubic-bezier(0.4, 0, 0.2, 1)',
        boxShadow: 'none',
        border: '1px solid #000000'
      }}
    />
  </button>
);

const ControlView = () => {
  const { isAdmin, isReadOnly } = useAuth();
  const [config, setConfig] = useState<Config | null>(null);
  const [alertConfig, setAlertConfig] = useState<AlertingConfig | null>(null);
  const [fallbackEmailInput, setFallbackEmailInput] = useState('');
  const [senderEmailInput, setSenderEmailInput] = useState('');
  const [smtpPasswordInput, setSmtpPasswordInput] = useState('');
  const [googleClientIdInput, setGoogleClientIdInput] = useState('');
  const [googleClientSecretInput, setGoogleClientSecretInput] = useState('');
  const [showOAuthSetup, setShowOAuthSetup] = useState(false);
  const [showSmtpFallback, setShowSmtpFallback] = useState(false);
  const [copiedRedirectUri, setCopiedRedirectUri] = useState(false);
  const [loading, setLoading] = useState(true);
  const [savingConfig, setSavingConfig] = useState(false);
  const [savingAlertConfig, setSavingAlertConfig] = useState(false);
  const [runningScraper, setRunningScraper] = useState(false);
  const [runningAlertScan, setRunningAlertScan] = useState(false);
  const [forceRescan, setForceRescan] = useState(false);
  const [runStatus, setRunStatus] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [alertStatus, setAlertStatus] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [confRes, alertRes] = await Promise.all([
        api.getSchedulerConfig(),
        api.getAlertingConfig().catch(() => null),
      ]);
      setConfig({
        enabled: confRes.enabled ?? true,
        require_approval: confRes.require_approval ?? true,
        interval_hours: confRes.interval_hours ?? 24,
        cron_schedule: confRes.cron_schedule || '0 0 * * *'
      });
      if (alertRes) {
        setAlertConfig({
          enabled: alertRes.enabled ?? true,
          require_approval: alertRes.require_approval ?? true,
          interval_hours: alertRes.interval_hours ?? 24,
          alert_days_before: alertRes.alert_days_before ?? 30,
          fallback_email: alertRes.fallback_email ?? '',
          sender_email: alertRes.sender_email ?? '',
          has_smtp_password: Boolean(alertRes.has_smtp_password),
          google_client_id: alertRes.google_client_id ?? '',
          has_google_client_secret: Boolean(alertRes.has_google_client_secret),
          google_connected: Boolean(alertRes.google_connected),
          google_connected_email: alertRes.google_connected_email ?? '',
          in_window_count: alertRes.in_window_count ?? 0,
        });
        setFallbackEmailInput(alertRes.fallback_email ?? '');
        setSenderEmailInput(alertRes.sender_email ?? '');
        setGoogleClientIdInput(alertRes.google_client_id ?? '');
        if (!alertRes.google_client_id && !alertRes.google_connected) {
          setShowOAuthSetup(true);
        }
      }
    } catch (err) {
      console.error("Error fetching control data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const handleRef = () => fetchData();
    const handleOAuthMsg = (evt: MessageEvent) => {
      if (evt?.data?.type === 'google-oauth-connected') {
        if (evt.data.ok) {
          setAlertStatus({
            type: 'success',
            text: `Connected Gmail sender via Google OAuth: ${evt.data.email}`
          });
          fetchData();
        } else {
          setAlertStatus({
            type: 'error',
            text: evt.data.message || 'Google OAuth connection failed.'
          });
        }
      }
    };
    window.addEventListener('refresh-proposals', handleRef);
    window.addEventListener('message', handleOAuthMsg);
    return () => {
      window.removeEventListener('refresh-proposals', handleRef);
      window.removeEventListener('message', handleOAuthMsg);
    };
  }, []);

  const updateConfig = async (newCfg: Partial<Config>) => {
    if (!config) return;
    const updated = { ...config, ...newCfg };
    setConfig(updated);
    setSavingConfig(true);
    try {
      await api.updateSchedulerConfig(updated);
    } catch (err: any) {
      console.error(err);
      fetchData();
    } finally {
      setSavingConfig(false);
    }
  };

  const updateAlertConfig = async (
    newCfg: Partial<AlertingConfig> & {
      smtp_password?: string;
      google_client_id?: string;
      google_client_secret?: string;
    }
  ) => {
    if (!alertConfig || isReadOnly) return;
    const updated = { ...alertConfig, ...newCfg };
    setAlertConfig(updated);
    setSavingAlertConfig(true);
    try {
      const res = await api.updateAlertingConfig(newCfg);
      setAlertConfig({
        enabled: res.enabled ?? updated.enabled,
        require_approval: res.require_approval ?? updated.require_approval,
        interval_hours: res.interval_hours ?? updated.interval_hours,
        alert_days_before: res.alert_days_before ?? updated.alert_days_before,
        fallback_email: res.fallback_email ?? updated.fallback_email,
        sender_email: res.sender_email ?? updated.sender_email,
        has_smtp_password: Boolean(res.has_smtp_password),
        google_client_id: res.google_client_id ?? updated.google_client_id,
        has_google_client_secret: Boolean(res.has_google_client_secret),
        google_connected: Boolean(res.google_connected),
        google_connected_email: res.google_connected_email ?? updated.google_connected_email,
        in_window_count: res.in_window_count ?? updated.in_window_count,
      });
      if (newCfg.smtp_password) {
        setSmtpPasswordInput('');
      }
      if (newCfg.google_client_secret) {
        setGoogleClientSecretInput('');
      }
    } catch (err: any) {
      console.error(err);
      setAlertStatus({ type: 'error', text: err?.message || 'Failed to update alerting configuration' });
      fetchData();
    } finally {
      setSavingAlertConfig(false);
    }
  };

  const handleConnectGoogleOAuth = async () => {
    setAlertStatus(null);
    try {
      const { auth_url } = await api.startGoogleOAuth();
      const width = 520;
      const height = 640;
      const left = Math.max(0, window.screenX + (window.outerWidth - width) / 2);
      const top = Math.max(0, window.screenY + (window.outerHeight - height) / 2);
      const popup = window.open(
        auth_url,
        'google_oauth_connect',
        `width=${width},height=${height},left=${left},top=${top},resizable=yes,scrollbars=yes`
      );
      if (!popup) {
        window.location.href = auth_url;
      }
    } catch (err: any) {
      setShowOAuthSetup(true);
      setAlertStatus({ type: 'error', text: err?.message || 'Could not start Google OAuth flow.' });
    }
  };

  const handleDisconnectGoogleOAuth = async () => {
    setAlertStatus(null);
    setSavingAlertConfig(true);
    try {
      await api.disconnectGoogleOAuth();
      await fetchData();
      setAlertStatus({ type: 'success', text: 'Disconnected Gmail sender account.' });
    } catch (err: any) {
      setAlertStatus({ type: 'error', text: err?.message || 'Failed to disconnect Gmail.' });
    } finally {
      setSavingAlertConfig(false);
    }
  };

  const handleRunAlertScanNow = async () => {
    setRunningAlertScan(true);
    setAlertStatus(null);
    try {
      const res = await api.runExpiryAlertScan(forceRescan);
      setAlertStatus({
        type: 'success',
        text: `Scan complete (${res.alert_window_days}d window): ${res.total_in_window} in window, ${res.new_alerts_created} new alert(s), ${res.proposals_staged} email proposal(s) staged, ${res.already_alerted_skipped} already notified.`
      });
      window.dispatchEvent(new Event('refresh-proposals'));
    } catch (err: any) {
      setAlertStatus({
        type: 'error',
        text: `Alert Scan Failed: ${err.message || 'Error running expiry alert scan'}`
      });
    } finally {
      setRunningAlertScan(false);
    }
  };

  const handleRunNow = async () => {
    setRunningScraper(true);
    setRunStatus(null);
    try {
      const res = await api.runAutonomousScraper();
      const runId = res.run_id ? res.run_id.substring(0, 8) : 'active';
      setRunStatus({
        type: 'success',
        text: `Scraper dispatched (Run ${runId}). Staged proposals will appear in Pending Actions.`
      });
      window.dispatchEvent(new Event('refresh-proposals'));
    } catch (err: any) {
      setRunStatus({
        type: 'error',
        text: `Dispatch Failed: ${err.message || 'Error triggering scraper'}`
      });
    } finally {
      setRunningScraper(false);
    }
  };

  const handleAddCardPlaceholder = () => {
    setRunStatus({
      type: 'success',
      text: 'New Workflow builder coming soon (Feature Placeholder).'
    });
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', width: '100%' }}>
      
      {/* Master Automation Cards Container Box */}
      <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', padding: '1.5rem', width: '100%', border: '1px solid #000000' }}>
        
        {/* Header Bar inside Master Box */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%', paddingBottom: '1rem', borderBottom: '1px solid rgba(255, 255, 255, 0.15)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Zap size={20} color="var(--brand-blue)" />
            <h2 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 700, letterSpacing: '0.03em', color: 'var(--text-primary)' }}>
              AUTOMATION CARDS
            </h2>
          </div>

          <button 
            className="btn btn-secondary"
            style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.4rem', 
              padding: '0.4rem 0.85rem', 
              fontSize: '0.82rem', 
              fontWeight: 600,
              border: '1px solid #000000',
              backgroundColor: 'transparent',
              color: 'var(--text-primary)'
            }}
            onClick={handleAddCardPlaceholder}
            title="Add New Workflow"
          >
            <Plus size={16} /> Add Workflow
          </button>
        </div>

        {/* Inner Card Item: Autonomous Scraper */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', padding: '1.25rem', borderRadius: '10px', width: '100%', border: '1px solid #000000' }}>
          
          {/* Sleek Minimal Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h3 style={{ margin: 0, color: 'var(--brand-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.05rem' }}>
              Autonomous Scraper
            </h3>
            {config && (
              <span className={`badge ${config.enabled ? 'badge-green' : 'badge-red'}`} style={{ fontSize: '0.78rem', padding: '0.2rem 0.6rem', borderRadius: '12px' }}>
                {config.enabled ? 'ACTIVE' : 'PAUSED'}
              </span>
            )}
          </div>

          {loading || !config ? (
            <div style={{ padding: '1rem', color: 'var(--text-tertiary)' }}>Loading controls...</div>
          ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            
            {/* Toggles Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              
              {/* Scheduled Ingestion */}
              <div className="glass-panel" style={{ padding: '1rem', borderRadius: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Clock size={16} color="var(--brand-blue)" />
                  <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>Scheduled Ingestion</span>
                </div>
                <ToggleSwitch 
                  checked={config.enabled}
                  onChange={() => !isReadOnly && updateConfig({ enabled: !config.enabled })}
                  disabled={savingConfig || isReadOnly}
                />
              </div>

              {/* Staged Approval */}
              <div className="glass-panel" style={{ padding: '1rem', borderRadius: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldCheck size={16} color="#f59e0b" />
                  <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>Require Approval</span>
                </div>
                <ToggleSwitch 
                  checked={config.require_approval}
                  onChange={() => !isReadOnly && updateConfig({ require_approval: !config.require_approval })}
                  disabled={savingConfig || isReadOnly}
                />
              </div>

            </div>

            {/* Stretched Execution Interval Row with Plus/Minus buttons */}
            <div className="glass-panel" style={{ padding: '1rem 1.25rem', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%', gap: '1rem' }}>
              <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Execution Interval</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, justifyContent: 'flex-end' }}>
                <button 
                  type="button"
                  className="btn btn-secondary"
                  style={{ 
                    padding: '0.35rem 0.65rem', 
                    minWidth: '34px', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                  onClick={() => updateConfig({ interval_hours: Math.max(1, config.interval_hours - 1) })}
                  disabled={savingConfig || config.interval_hours <= 1}
                  title="Decrease interval (-1h)"
                >
                  <Minus size={14} />
                </button>

                <input 
                  type="number" 
                  min={1} 
                  max={168} 
                  value={config.interval_hours}
                  onChange={(e) => updateConfig({ interval_hours: Math.max(1, parseInt(e.target.value) || 1) })}
                  className="input"
                  style={{ 
                    width: '70px', 
                    padding: '0.35rem 0.5rem', 
                    fontSize: '0.9rem', 
                    textAlign: 'center', 
                    fontWeight: 600, 
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                />

                <button 
                  type="button"
                  className="btn btn-secondary"
                  style={{ 
                    padding: '0.35rem 0.65rem', 
                    minWidth: '34px', 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                  onClick={() => updateConfig({ interval_hours: Math.min(168, config.interval_hours + 1) })}
                  disabled={savingConfig || config.interval_hours >= 168}
                  title="Increase interval (+1h)"
                >
                  <Plus size={14} />
                </button>

                <span className="text-secondary" style={{ fontSize: '0.85rem', fontWeight: 500, marginLeft: '0.25rem' }}>hours</span>
              </div>
            </div>

            {/* Minimal Run Button */}
            <button 
              className="btn btn-primary"
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '0.65rem 1.25rem', fontSize: '0.88rem', fontWeight: 600, border: '1px solid #000000', opacity: isReadOnly ? 0.55 : 1 }}
              onClick={handleRunNow}
              disabled={runningScraper || isReadOnly}
            >
              {runningScraper ? <Loader className="spinner" size={16} /> : <Play size={16} />}
              {runningScraper ? 'Dispatching Scraper...' : 'Run Scraper Now'}
            </button>

            {runStatus && (
              <div className={`badge ${runStatus.type === 'success' ? 'badge-green' : 'badge-red'}`} style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem', textAlign: 'center' }}>
                {runStatus.text}
              </div>
            )}

          </div>
        )}

      </div>

      {/* Inner Card Item: Certificate Expiry Alerting System */}
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', padding: '1.25rem', borderRadius: '10px', width: '100%', border: '1px solid #000000' }}>

        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ margin: 0, color: 'var(--brand-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '1.05rem' }}>
            <BellRing size={18} /> Certificate Expiry Alerting System
          </h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {alertConfig && (
              <span className="badge" style={{ fontSize: '0.78rem', padding: '0.2rem 0.6rem', borderRadius: '12px', border: '1px solid #000000', color: 'var(--text-primary)' }}>
                {alertConfig.in_window_count} in {alertConfig.alert_days_before}d window
              </span>
            )}
            {alertConfig && (
              <span className={`badge ${alertConfig.enabled ? 'badge-green' : 'badge-red'}`} style={{ fontSize: '0.78rem', padding: '0.2rem 0.6rem', borderRadius: '12px' }}>
                {alertConfig.enabled ? 'ACTIVE' : 'PAUSED'}
              </span>
            )}
          </div>
        </div>

        {loading || !alertConfig ? (
          <div style={{ padding: '1rem', color: 'var(--text-tertiary)' }}>Loading alerting controls...</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

            {/* Toggles Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>

              {/* Scheduled Alerting */}
              <div className="glass-panel" style={{ padding: '1rem', borderRadius: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Clock size={16} color="var(--brand-blue)" />
                  <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>Scheduled Alerting</span>
                </div>
                <ToggleSwitch
                  checked={alertConfig.enabled}
                  onChange={() => !isReadOnly && updateAlertConfig({ enabled: !alertConfig.enabled })}
                  disabled={savingAlertConfig || isReadOnly}
                />
              </div>

              {/* Require Approval */}
              <div className="glass-panel" style={{ padding: '1rem', borderRadius: '10px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <ShieldCheck size={16} color="#f59e0b" />
                  <span style={{ fontWeight: 600, fontSize: '0.88rem' }}>Require Approval</span>
                </div>
                <ToggleSwitch
                  checked={alertConfig.require_approval}
                  onChange={() => !isReadOnly && updateAlertConfig({ require_approval: !alertConfig.require_approval })}
                  disabled={savingAlertConfig || isReadOnly}
                />
              </div>

            </div>

            {/* Alert Lead Time Adjustment (Preset Pills + Custom Stepper) */}
            <div className="glass-panel" style={{ padding: '1rem 1.25rem', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', width: '100%', gap: '1rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Alert Lead Time (Before Expiration)</span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>Select a preset or set a custom threshold (1 - 365 days)</span>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                {[30, 60, 90].map((presetDays) => {
                  const isActive = alertConfig.alert_days_before === presetDays;
                  return (
                    <button
                      key={presetDays}
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => updateAlertConfig({ alert_days_before: presetDays })}
                      disabled={savingAlertConfig}
                      style={{
                        padding: '0.35rem 0.7rem',
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        border: '1px solid #000000',
                        backgroundColor: isActive ? 'var(--brand-blue)' : 'transparent',
                        color: isActive ? '#ffffff' : 'var(--text-primary)',
                      }}
                    >
                      {presetDays}d
                    </button>
                  );
                })}

                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{
                    padding: '0.35rem 0.65rem',
                    minWidth: '34px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                  onClick={() => updateAlertConfig({ alert_days_before: Math.max(1, alertConfig.alert_days_before - 5) })}
                  disabled={savingAlertConfig || alertConfig.alert_days_before <= 1}
                  title="Decrease lead time (-5 days)"
                >
                  <Minus size={14} />
                </button>

                <input
                  type="number"
                  min={1}
                  max={365}
                  value={alertConfig.alert_days_before}
                  onChange={(e) => {
                    const val = Math.max(1, Math.min(365, parseInt(e.target.value, 10) || 1));
                    updateAlertConfig({ alert_days_before: val });
                  }}
                  className="input"
                  style={{
                    width: '74px',
                    padding: '0.35rem 0.5rem',
                    fontSize: '0.9rem',
                    textAlign: 'center',
                    fontWeight: 600,
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                />

                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{
                    padding: '0.35rem 0.65rem',
                    minWidth: '34px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                  onClick={() => updateAlertConfig({ alert_days_before: Math.min(365, alertConfig.alert_days_before + 5) })}
                  disabled={savingAlertConfig || alertConfig.alert_days_before >= 365}
                  title="Increase lead time (+5 days)"
                >
                  <Plus size={14} />
                </button>

                <span className="text-secondary" style={{ fontSize: '0.85rem', fontWeight: 500, marginLeft: '0.15rem' }}>days</span>
              </div>
            </div>

            {/* Scan Interval + Fallback Email Row */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>

              {/* Scan Interval */}
              <div className="glass-panel" style={{ padding: '1rem 1.25rem', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem' }}>
                <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Scan Interval</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{
                      padding: '0.35rem 0.65rem',
                      minWidth: '34px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: 'transparent',
                      border: '1px solid #000000',
                      color: 'var(--text-primary)'
                    }}
                    onClick={() => updateAlertConfig({ interval_hours: Math.max(1, alertConfig.interval_hours - 1) })}
                    disabled={savingAlertConfig || alertConfig.interval_hours <= 1}
                    title="Decrease scan interval (-1h)"
                  >
                    <Minus size={14} />
                  </button>

                  <input
                    type="number"
                    min={1}
                    max={168}
                    value={alertConfig.interval_hours}
                    onChange={(e) => updateAlertConfig({ interval_hours: Math.max(1, Math.min(168, parseInt(e.target.value, 10) || 1)) })}
                    className="input"
                    style={{
                      width: '66px',
                      padding: '0.35rem 0.5rem',
                      fontSize: '0.9rem',
                      textAlign: 'center',
                      fontWeight: 600,
                      backgroundColor: 'transparent',
                      border: '1px solid #000000',
                      color: 'var(--text-primary)'
                    }}
                  />

                  <button
                    type="button"
                    className="btn btn-secondary"
                    style={{
                      padding: '0.35rem 0.65rem',
                      minWidth: '34px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: 'transparent',
                      border: '1px solid #000000',
                      color: 'var(--text-primary)'
                    }}
                    onClick={() => updateAlertConfig({ interval_hours: Math.min(168, alertConfig.interval_hours + 1) })}
                    disabled={savingAlertConfig || alertConfig.interval_hours >= 168}
                    title="Increase scan interval (+1h)"
                  >
                    <Plus size={14} />
                  </button>

                  <span className="text-secondary" style={{ fontSize: '0.85rem', fontWeight: 500 }}>hours</span>
                </div>
              </div>

              {/* Fallback Admin Email for Unassigned Components */}
              <div className="glass-panel" style={{ padding: '1rem 1.25rem', borderRadius: '10px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.75rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexShrink: 0 }}>
                  <Mail size={15} color="var(--brand-blue)" />
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 600 }}>Fallback Email</span>
                </div>
                <input
                  type="email"
                  value={fallbackEmailInput}
                  placeholder="compliance-admin@stellantis.com"
                  onChange={(e) => setFallbackEmailInput(e.target.value)}
                  onBlur={() => {
                    if (fallbackEmailInput !== alertConfig.fallback_email) {
                      updateAlertConfig({ fallback_email: fallbackEmailInput });
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      updateAlertConfig({ fallback_email: fallbackEmailInput });
                    }
                  }}
                  className="input"
                  style={{
                    flex: 1,
                    minWidth: '160px',
                    padding: '0.35rem 0.6rem',
                    fontSize: '0.85rem',
                    backgroundColor: 'transparent',
                    border: '1px solid #000000',
                    color: 'var(--text-primary)'
                  }}
                />
              </div>

            </div>

            {/* Admin-Only Sender Email (Google OAuth 2.0 + Gmail API & Optional SMTP Fallback) */}
            <div className="glass-panel" style={{ padding: '1rem 1.25rem', borderRadius: '10px', display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                  <Lock size={15} color="var(--brand-blue)" />
                  <span style={{ fontSize: '0.88rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                    Sender Email (Gmail OAuth 2.0 API)
                  </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', flexWrap: 'wrap' }}>
                  <span className={`badge ${alertConfig.google_connected ? 'badge-green' : 'badge-red'}`} style={{ fontSize: '0.7rem', padding: '0.15rem 0.5rem' }}>
                    {alertConfig.google_connected ? `Connected: ${alertConfig.google_connected_email}` : 'Not Connected'}
                  </span>
                  <span className="badge badge-blue" style={{ fontSize: '0.7rem', padding: '0.15rem 0.5rem' }}>
                    Administrator Only
                  </span>
                </div>
              </div>

              {isAdmin ? (
                <>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', flexWrap: 'wrap' }}>
                    <button
                      type="button"
                      className="btn btn-primary"
                      disabled={savingAlertConfig}
                      onClick={handleConnectGoogleOAuth}
                      style={{
                        border: '1px solid #000000',
                        fontSize: '0.84rem',
                        fontWeight: 600,
                        padding: '0.5rem 0.95rem'
                      }}
                    >
                      <Mail size={15} />
                      {alertConfig.google_connected ? 'Switch / Reconnect Gmail Account' : 'Connect Gmail via Google OAuth'}
                    </button>

                    {alertConfig.google_connected && (
                      <button
                        type="button"
                        className="btn btn-secondary"
                        disabled={savingAlertConfig}
                        onClick={handleDisconnectGoogleOAuth}
                        style={{
                          border: '1px solid #ef4444',
                          color: '#ef4444',
                          fontSize: '0.82rem',
                          fontWeight: 600,
                          padding: '0.5rem 0.85rem'
                        }}
                      >
                        <Unlink size={14} /> Disconnect
                      </button>
                    )}

                    <button
                      type="button"
                      className="btn btn-secondary"
                      onClick={() => setShowOAuthSetup((v) => !v)}
                      style={{
                        border: '1px solid #000000',
                        fontSize: '0.78rem',
                        padding: '0.45rem 0.75rem',
                        marginLeft: 'auto'
                      }}
                    >
                      OAuth Client Setup {showOAuthSetup ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                    </button>
                  </div>

                  {showOAuthSetup && (
                    <div style={{ padding: '0.85rem', borderRadius: '8px', backgroundColor: 'rgba(0, 0, 0, 0.04)', border: '1px solid var(--border-color)', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                      <div style={{ fontSize: '0.76rem', color: 'var(--text-secondary)', lineHeight: 1.4 }}>
                        In Google Cloud Console (APIs & Services -&gt; Credentials -&gt; OAuth 2.0 Client ID -&gt; Web application), enable the <strong>Gmail API</strong> and add this Authorized Redirect URI:
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                        {(() => {
                          const host = window.location.hostname;
                          const isIpv4 = /^\d{1,3}(\.\d{1,3}){3}$/.test(host) && host !== '127.0.0.1';
                          const portPart = window.location.port ? `:${window.location.port}` : '';
                          const oauthOrigin = isIpv4
                            ? `${window.location.protocol}//${host}.sslip.io${portPart}`
                            : window.location.origin;
                          const redirectUri = `${oauthOrigin}/api/v1/auth/google/callback`;
                          return (
                            <>
                              <input
                                type="text"
                                readOnly
                                value={redirectUri}
                                className="input"
                                style={{ flex: 1, fontSize: '0.78rem', border: '1px solid #000000', backgroundColor: 'transparent', color: 'var(--text-primary)' }}
                              />
                              <button
                                type="button"
                                className="btn btn-secondary"
                                style={{ border: '1px solid #000000', fontSize: '0.76rem', padding: '0.35rem 0.65rem' }}
                                onClick={() => {
                                  navigator.clipboard?.writeText(redirectUri);
                                  setCopiedRedirectUri(true);
                                  setTimeout(() => setCopiedRedirectUri(false), 2000);
                                }}
                              >
                                {copiedRedirectUri ? <Check size={13} /> : <Copy size={13} />}
                                {copiedRedirectUri ? 'Copied' : 'Copy URI'}
                              </button>
                            </>
                          );
                        })()}
                      </div>

                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.65rem', alignItems: 'center' }}>
                        <input
                          type="text"
                          value={googleClientIdInput}
                          placeholder="Google OAuth Client ID (...apps.googleusercontent.com)"
                          onChange={(e) => setGoogleClientIdInput(e.target.value)}
                          className="input"
                          style={{ padding: '0.4rem 0.6rem', fontSize: '0.82rem', backgroundColor: 'transparent', border: '1px solid #000000', color: 'var(--text-primary)' }}
                        />
                        <input
                          type="password"
                          value={googleClientSecretInput}
                          placeholder={alertConfig.has_google_client_secret ? '•••••••••••• (Client Secret saved)' : 'Google OAuth Client Secret'}
                          onChange={(e) => setGoogleClientSecretInput(e.target.value)}
                          className="input"
                          style={{ padding: '0.4rem 0.6rem', fontSize: '0.82rem', backgroundColor: 'transparent', border: '1px solid #000000', color: 'var(--text-primary)' }}
                        />
                        <button
                          type="button"
                          className="btn btn-secondary"
                          disabled={savingAlertConfig}
                          onClick={() => {
                            const payload: { google_client_id?: string; google_client_secret?: string } = {
                              google_client_id: googleClientIdInput,
                            };
                            if (googleClientSecretInput.trim()) {
                              payload.google_client_secret = googleClientSecretInput.trim();
                            }
                            updateAlertConfig(payload).then(() => {
                              setAlertStatus({ type: 'success', text: 'Google OAuth Client credentials saved. You can now click "Connect Gmail via Google OAuth".' });
                            });
                          }}
                          style={{ border: '1px solid #000000', fontSize: '0.8rem', fontWeight: 600, justifyContent: 'center' }}
                        >
                          Save OAuth Credentials
                        </button>
                      </div>
                    </div>
                  )}

                  <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      type="button"
                      onClick={() => setShowSmtpFallback((v) => !v)}
                      style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', fontSize: '0.74rem', cursor: 'pointer', padding: 0, textDecoration: 'underline' }}
                    >
                      {showSmtpFallback ? 'Hide SMTP Fallback' : 'Or configure SMTP App Password fallback'}
                    </button>
                  </div>

                  {showSmtpFallback && (
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.65rem', alignItems: 'center' }}>
                      <input
                        type="email"
                        value={senderEmailInput}
                        disabled={savingAlertConfig}
                        placeholder="sender@gmail.com"
                        onChange={(e) => setSenderEmailInput(e.target.value)}
                        className="input"
                        style={{ padding: '0.4rem 0.6rem', fontSize: '0.84rem', backgroundColor: 'transparent', border: '1px solid #000000', color: 'var(--text-primary)' }}
                      />
                      <input
                        type="password"
                        value={smtpPasswordInput}
                        disabled={savingAlertConfig}
                        placeholder={alertConfig.has_smtp_password ? '•••••••••••• (App Password saved)' : '16-letter SMTP App Password'}
                        onChange={(e) => setSmtpPasswordInput(e.target.value)}
                        className="input"
                        style={{ padding: '0.4rem 0.6rem', fontSize: '0.84rem', backgroundColor: 'transparent', border: '1px solid #000000', color: 'var(--text-primary)' }}
                      />
                      <button
                        type="button"
                        className="btn btn-secondary"
                        disabled={savingAlertConfig}
                        onClick={() => {
                          const payload: { sender_email?: string; smtp_password?: string } = {
                            sender_email: senderEmailInput,
                          };
                          if (smtpPasswordInput.trim()) {
                            payload.smtp_password = smtpPasswordInput.trim();
                          }
                          updateAlertConfig(payload).then(() => {
                            setAlertStatus({ type: 'success', text: 'SMTP fallback credentials saved.' });
                          });
                        }}
                        style={{ border: '1px solid #000000', fontSize: '0.8rem', fontWeight: 600, justifyContent: 'center' }}
                      >
                        Save SMTP Fallback
                      </button>
                    </div>
                  )}
                </>
              ) : (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Active Sender: <strong>{alertConfig.sender_email || 'Not configured yet'}</strong> (managed exclusively by the Administrator).
                </div>
              )}
            </div>

            {/* Run Alert Scan Row */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
              <button
                className="btn btn-primary"
                style={{ flex: 1, minWidth: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '0.65rem 1.25rem', fontSize: '0.88rem', fontWeight: 600, border: '1px solid #000000', opacity: isReadOnly ? 0.55 : 1 }}
                onClick={handleRunAlertScanNow}
                disabled={runningAlertScan || isReadOnly}
              >
                {runningAlertScan ? <Loader className="spinner" size={16} /> : <Play size={16} />}
                {runningAlertScan ? 'Scanning Certificates...' : 'Run Alert Scan Now'}
              </button>

              <label style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.82rem', color: 'var(--text-secondary)', cursor: isReadOnly ? 'not-allowed' : 'pointer', userSelect: 'none' }}>
                <input
                  type="checkbox"
                  checked={forceRescan}
                  disabled={isReadOnly || runningAlertScan}
                  onChange={(e) => setForceRescan(e.target.checked)}
                />
                <span>Force Rescan (re-alert already notified)</span>
              </label>
            </div>

            {alertStatus && (
              <div className={`badge ${alertStatus.type === 'success' ? 'badge-green' : 'badge-red'}`} style={{ padding: '0.5rem 0.75rem', fontSize: '0.8rem', textAlign: 'center' }}>
                {alertStatus.text}
              </div>
            )}

          </div>
        )}

      </div>
    </div>
  </div>
  );
};

export default ControlView;
