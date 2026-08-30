import { useState, useEffect } from 'react';
import { api } from '../api';
import { Play, Loader, ShieldCheck, Clock, Zap, Plus, Minus } from 'lucide-react';

type Config = {
  enabled: boolean;
  require_approval: boolean;
  interval_hours: number;
  cron_schedule: string;
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
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingConfig, setSavingConfig] = useState(false);
  const [runningScraper, setRunningScraper] = useState(false);
  const [runStatus, setRunStatus] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const confRes = await api.getSchedulerConfig();
      setConfig({
        enabled: confRes.enabled ?? true,
        require_approval: confRes.require_approval ?? true,
        interval_hours: confRes.interval_hours ?? 24,
        cron_schedule: confRes.cron_schedule || '0 0 * * *'
      });
    } catch (err) {
      console.error("Error fetching control data:", err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const handleRef = () => fetchData();
    window.addEventListener('refresh-proposals', handleRef);
    return () => window.removeEventListener('refresh-proposals', handleRef);
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
                  onChange={() => updateConfig({ enabled: !config.enabled })}
                  disabled={savingConfig}
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
                  onChange={() => updateConfig({ require_approval: !config.require_approval })}
                  disabled={savingConfig}
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
              style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', padding: '0.65rem 1.25rem', fontSize: '0.88rem', fontWeight: 600, border: '1px solid #000000' }}
              onClick={handleRunNow}
              disabled={runningScraper}
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
    </div>
  </div>
  );
};

export default ControlView;
