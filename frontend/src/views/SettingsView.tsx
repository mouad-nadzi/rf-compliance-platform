import { useState, useEffect } from 'react';
import { RotateCcw, Trash2, Database, FileText, RefreshCw, X, Cpu } from 'lucide-react';
import { api } from '../api';

interface RecycleItem {
  id: string;
  title: string;
  tableName: string;
  data: any;
  deletedAt: string;
}

const SettingsView = () => {
  const [recycleItems, setRecycleItems] = useState<RecycleItem[]>([]);
  const [loadingBin, setLoadingBin] = useState(false);
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<{ text: string; type: 'success' | 'error' } | null>(null);

  const fetchRecycleBin = async () => {
    setLoadingBin(true);
    try {
      const items = await api.getRecycleBinItems();
      setRecycleItems(items);
      const allIds = items.map(item => item.id);
      localStorage.setItem('seen_recycle_item_ids', JSON.stringify(allIds));
      window.dispatchEvent(new Event('refresh-recycle-bin'));
    } catch (err) {
      console.error("Error fetching recycle bin items:", err);
    } finally {
      setLoadingBin(false);
    }
  };

  useEffect(() => {
    fetchRecycleBin();
  }, []);

  const handleRestoreItem = async (item: RecycleItem) => {
    setRestoringId(item.id);
    setStatusMsg(null);
    try {
      await api.restoreRecycleBinItem(item.id);
      await fetchRecycleBin();
      window.dispatchEvent(new Event('refresh-custom-tables'));
      window.dispatchEvent(new Event('refresh-table-data'));
      window.dispatchEvent(new Event('refresh-recycle-bin'));
      setStatusMsg({ text: `Successfully restored "${item.title}" back to database!`, type: 'success' });
    } catch (err: any) {
      setStatusMsg({ text: `Failed to restore: ${err.message}`, type: 'error' });
    } finally {
      setRestoringId(null);
    }
  };

  const handleDeleteItem = async (itemId: string) => {
    try {
      await api.deleteRecycleBinItem(itemId);
      await fetchRecycleBin();
      window.dispatchEvent(new Event('refresh-recycle-bin'));
    } catch (err: any) {
      console.error("Failed to delete item:", err);
    }
  };

  const handleEmptyBin = async () => {
    if (!window.confirm("Are you sure you want to permanently delete all items and soft-deleted tables in the Recycle Bin?")) return;
    try {
      await api.emptyRecycleBinApi();
      await fetchRecycleBin();
      window.dispatchEvent(new Event('refresh-recycle-bin'));
      setStatusMsg({ text: "Recycle Bin permanently emptied.", type: 'success' });
    } catch (err: any) {
      setStatusMsg({ text: `Failed to empty Recycle Bin: ${err.message}`, type: 'error' });
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      
      {/* Header */}
      <div>
        <h2 style={{ color: 'var(--brand-blue)', margin: '0 0 0.4rem 0', fontSize: '1.4rem' }}>
          Platform Settings & Data Recovery
        </h2>
        <p className="text-secondary" style={{ margin: 0, fontSize: '0.85rem' }}>
          Configure AI engine settings and manage global soft-deleted tables and record recovery.
        </p>
      </div>

      {statusMsg && (
        <div style={{
          padding: '0.65rem 1rem',
          borderRadius: '8px',
          backgroundColor: statusMsg.type === 'success' ? 'rgba(16, 185, 129, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          color: statusMsg.type === 'success' ? '#10b981' : '#ef4444',
          fontSize: '0.82rem',
          fontWeight: 600
        }}>
          {statusMsg.text}
        </div>
      )}

      {/* System Infrastructure Card */}
      <div className="card" style={{ padding: '1.25rem' }}>
        <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.05rem', color: 'var(--brand-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Cpu size={18} /> Active AI Engines & Host Infrastructure
        </h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
          <div style={{ padding: '0.85rem', backgroundColor: 'var(--bg-body)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 700 }}>Language Model</span>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>Qwen3.8-27B GGUF</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>3-bit UD-IQ3_XXS (32k context)</div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'var(--bg-body)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 700 }}>Vision OCR Engine</span>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>GLM-OCR 0.9B</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>Pure PyTorch FP16 HuggingFace</div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'var(--bg-body)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 700 }}>Dense Vector Search</span>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>BAAI/bge-m3 (1024-d)</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>PostgreSQL pgvector engine</div>
          </div>

          <div style={{ padding: '0.85rem', backgroundColor: 'var(--bg-body)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 700 }}>Database Layer</span>
            <div style={{ fontWeight: 700, fontSize: '0.9rem', color: 'var(--text-primary)', marginTop: '0.2rem' }}>PostgreSQL 16</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', marginTop: '0.1rem' }}>Docker container stack</div>
          </div>
        </div>
      </div>

      {/* Global Recycle Bin & Soft-Deleted Tables Section */}
      <div className="card" style={{ padding: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '1.05rem', color: 'var(--brand-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <RotateCcw size={18} /> Global Recycle Bin & Soft-Deleted Tables
            </h3>
            <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              Soft-deleted dynamic tables and individual records are stored here. Restoring a table re-creates its schema and re-populates all records in PostgreSQL.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button 
              className="btn btn-secondary"
              style={{ fontSize: '0.8rem', padding: '0.4rem 0.75rem' }}
              onClick={fetchRecycleBin}
              disabled={loadingBin}
            >
              <RefreshCw size={14} className={loadingBin ? "spin-icon" : ""} /> Refresh
            </button>

            {recycleItems.length > 0 && (
              <button 
                className="btn btn-secondary"
                style={{ fontSize: '0.8rem', padding: '0.4rem 0.75rem', color: '#ef4444', borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.08)' }}
                onClick={handleEmptyBin}
              >
                <Trash2 size={14} /> Empty Recycle Bin
              </button>
            )}
          </div>
        </div>

        {recycleItems.length === 0 ? (
          <div className="glass-panel" style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-tertiary)' }}>
            <RotateCcw size={32} style={{ opacity: 0.3, marginBottom: '0.5rem' }} />
            <p style={{ margin: 0, fontSize: '0.9rem', fontWeight: 600 }}>Recycle Bin is empty.</p>
            <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.78rem' }}>Deleted rows and dropped custom dynamic tables will appear here for one-click recovery.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ backgroundColor: 'var(--bg-body)', borderBottom: '1px solid var(--border-color)', textAlign: 'left' }}>
                  <th style={{ padding: '0.65rem 0.85rem' }}>Type</th>
                  <th style={{ padding: '0.65rem 0.85rem' }}>Title / Object Name</th>
                  <th style={{ padding: '0.65rem 0.85rem' }}>Original Table</th>
                  <th style={{ padding: '0.65rem 0.85rem' }}>Date Deleted</th>
                  <th style={{ padding: '0.65rem 0.85rem', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {recycleItems.map(item => {
                  const isTable = item.data?.is_table || item.tableName?.startsWith('__TABLE__:');
                  const cleanTableName = item.tableName?.replace('__TABLE__:', '');
                  const recCount = item.data?.record_count ?? (Array.isArray(item.data?.records) ? item.data.records.length : 0);

                  return (
                    <tr key={item.id} style={{ borderBottom: '1px solid var(--border-color)' }}>
                      <td style={{ padding: '0.65rem 0.85rem' }}>
                        <span className={`badge ${isTable ? 'badge-blue' : 'badge-gray'}`} style={{ fontSize: '0.68rem' }}>
                          {isTable ? 'TABLE' : 'RECORD'}
                        </span>
                      </td>
                      <td style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                          {isTable ? <Database size={15} color="#3b82f6" /> : <FileText size={15} color="var(--text-secondary)" />}
                          <span>{item.title}</span>
                          {isTable && (
                            <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)', fontWeight: 400 }}>
                              ({recCount} rows)
                            </span>
                          )}
                        </div>
                      </td>
                      <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-secondary)' }}>
                        {cleanTableName}
                      </td>
                      <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-tertiary)' }}>
                        {item.deletedAt ? new Date(item.deletedAt).toLocaleString() : '—'}
                      </td>
                      <td style={{ padding: '0.65rem 0.85rem', textAlign: 'right' }}>
                        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.4rem' }}>
                          <button
                            className="btn btn-primary"
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.6rem', gap: '0.3rem' }}
                            disabled={restoringId === item.id}
                            onClick={() => handleRestoreItem(item)}
                          >
                            {restoringId === item.id ? <RefreshCw size={12} className="spin-icon" /> : <RotateCcw size={12} />}
                            Restore
                          </button>
                          <button
                            className="btn btn-secondary"
                            style={{ fontSize: '0.75rem', padding: '0.25rem 0.45rem', color: '#ef4444', borderColor: '#ef4444' }}
                            onClick={() => handleDeleteItem(item.id)}
                            title="Permanently Delete"
                          >
                            <X size={12} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

    </div>
  );
};

export default SettingsView;
