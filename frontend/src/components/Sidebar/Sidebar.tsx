import { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { FileText, Shield, Users, Layers, HardDrive, Database, Plus, ChevronLeft, ChevronRight, UploadCloud, CheckCircle2, AlertCircle, Loader2, Upload, MessageSquare, PlusCircle, Trash2, ShieldAlert, Check, X } from 'lucide-react';
import * as XLSX from 'xlsx';
import { api, type ChatSession } from '../../api';
import './Sidebar.css';

const BASE_TABLES = [
  { name: "RF Certificates", icon: FileText },
  { name: "Authorities", icon: Shield },
  { name: "Suppliers", icon: Users },
  { name: "Sources", icon: Layers },
  { name: "Agent Memories", icon: HardDrive },
];

const SUPPORTED_EXTENSIONS = ['.pdf', '.png', '.jpg', '.jpeg', '.webp'];
const isSupportedDocument = (filename: string) => {
  const lower = filename.toLowerCase();
  return SUPPORTED_EXTENSIONS.some(ext => lower.endsWith(ext));
};

interface SidebarProps {
  isOpen: boolean;
  onToggle: () => void;
  selectedTable: string;
  setSelectedTable: (tbl: string) => void;
  customTables: string[];
  setCustomTables: React.Dispatch<React.SetStateAction<string[]>>;
  activeSessionId?: string | null;
  setActiveSessionId?: (id: string | null) => void;
}

const Sidebar = ({ 
  isOpen, 
  onToggle, 
  selectedTable, 
  setSelectedTable, 
  customTables, 
  setCustomTables,
  activeSessionId,
  setActiveSessionId
}: SidebarProps) => {
  const location = useLocation();
  const isDatabaseView = location.pathname.includes('/databases');
  const isChatView = location.pathname.includes('/chat');

  // Chat History Sessions State for Assistant View
  const [sessions, setSessions] = useState<ChatSession[]>([]);

  // Agent Proposals State for Automations View
  const [proposals, setProposals] = useState<any[]>([]);

  const fetchChatSessions = async () => {
    try {
      const data = await api.getSessions();
      const list = Array.isArray(data) ? data : [];
      setSessions(list);
    } catch {
      setSessions([]);
    }
  };

  const fetchProposals = async () => {
    try {
      const res = await api.getProposals();
      setProposals(res.proposals || []);
    } catch {
      setProposals([]);
    }
  };

  useEffect(() => {
    if (isChatView) fetchChatSessions();
    if (!isDatabaseView && !isChatView) fetchProposals();

    const handleRefreshChats = () => fetchChatSessions();
    const handleRefreshProps = () => fetchProposals();

    window.addEventListener('refresh-chat-sessions', handleRefreshChats);
    window.addEventListener('refresh-proposals', handleRefreshProps);

    return () => {
      window.removeEventListener('refresh-chat-sessions', handleRefreshChats);
      window.removeEventListener('refresh-proposals', handleRefreshProps);
    };
  }, [location.pathname]);

  // Create Custom Table Form State
  const [showNewTableModal, setShowNewTableModal] = useState(false);
  const [newTableName, setNewTableName] = useState('');
  const [newTableCols, setNewTableCols] = useState('');

  // Ingest Documents State
  const [ingestFiles, setIngestFiles] = useState<File[]>([]);
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  // REAL Server Batch Monitoring State (Persisted in localStorage & synced with server)
  const [activeBatchId, setActiveBatchId] = useState<string | null>(() => localStorage.getItem('active_batch_id'));
  const [batchInfo, setBatchInfo] = useState<any>(null);
  const [displayedPercent, setDisplayedPercent] = useState(() => {
    try {
      const saved = localStorage.getItem('displayed_percent');
      return saved ? parseInt(saved, 10) || 0 : 0;
    } catch {
      return 0;
    }
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Smooth continuous progress percentage & bar width crawler
  useEffect(() => {
    if (!batchInfo) {
      setDisplayedPercent(0);
      localStorage.removeItem('displayed_percent');
      return;
    }

    const phase = batchInfo.phase;
    if (phase === 'done') {
      setDisplayedPercent(100);
      localStorage.setItem('displayed_percent', '100');
      return;
    }
    if (phase === 'error') {
      return;
    }

    const total = batchInfo.total || 1;
    const ocr_done = batchInfo.ocr_done || 0;
    const extract_done = batchInfo.extract_done || 0;
    const skipped = batchInfo.skipped || 0;

    const workUnits = 2 * total;
    const doneUnits = ocr_done + extract_done + (2 * skipped);
    const serverBaseline = Math.min(92, Math.round((doneUnits / workUnits) * 100));
    const targetCeiling = Math.max(serverBaseline, 12);

    // Set immediate baseline on mount/refresh if stored percentage is below current server baseline
    setDisplayedPercent(prev => {
      const next = Math.max(prev, serverBaseline);
      localStorage.setItem('displayed_percent', String(next));
      return next;
    });

    const interval = setInterval(() => {
      setDisplayedPercent(prev => {
        let next = prev;
        if (prev < targetCeiling) {
          next = prev + 1;
        } else if (prev < 94) {
          next = prev + (Math.random() > 0.6 ? 1 : 0);
        }
        localStorage.setItem('displayed_percent', String(next));
        return next;
      });
    }, 220);

    return () => clearInterval(interval);
  }, [batchInfo]);

  const handleImportTableFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const rawName = file.name.replace(/\.[^/.]+$/, "").replace(/[_-]/g, " ");
    const cleanTableName = rawName.charAt(0).toUpperCase() + rawName.slice(1);
    setNewTableName(cleanTableName);

    const reader = new FileReader();
    reader.onload = (evt) => {
      try {
        const bstr = evt.target?.result;
        const wb = XLSX.read(bstr, { type: 'binary' });
        const wsname = wb.SheetNames[0];
        const ws = wb.Sheets[wsname];
        const importedRows: any[] = XLSX.utils.sheet_to_json(ws);

        if (importedRows.length > 0) {
          const keys = Object.keys(importedRows[0]).filter(k => !k.startsWith('_'));
          setNewTableCols(keys.join(', '));
        }
      } catch (err: any) {
        console.error("Failed to parse file for table creation:", err);
      }
    };
    reader.readAsBinaryString(file);
  };

  const handleCreateCustomTable = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTableName.trim()) return;
    const cleanedName = newTableName.trim();
    if (!customTables.includes(cleanedName)) {
      setCustomTables([...customTables, cleanedName]);
    }
    setSelectedTable(cleanedName);
    setNewTableName('');
    setNewTableCols('');
    setShowNewTableModal(false);
  };

  // Check server for active batch on mount and poll REAL server status
  useEffect(() => {
    const checkServerBatch = async () => {
      try {
        const curr = await api.getBatchStatus();
        if (!curr || curr.phase === 'idle') {
          setActiveBatchId(null);
          localStorage.removeItem('active_batch_id');
          localStorage.removeItem('displayed_percent');
          setBatchInfo(null);
          setDisplayedPercent(0);
          return;
        }

        if (curr && curr.batch_id) {
          setActiveBatchId(curr.batch_id);
          localStorage.setItem('active_batch_id', curr.batch_id);
          setBatchInfo(curr);

          // Compute baseline percentage immediately on mount
          const total = curr.total || 1;
          const ocr_done = curr.ocr_done || 0;
          const extract_done = curr.extract_done || 0;
          const skipped = curr.skipped || 0;
          const serverBaseline = Math.min(92, Math.round(((ocr_done + extract_done + (2 * skipped)) / (2 * total)) * 100));

          setDisplayedPercent(prev => {
            const next = Math.max(prev, serverBaseline);
            localStorage.setItem('displayed_percent', String(next));
            return next;
          });

          if (curr.phase === 'done') {
            window.dispatchEvent(new Event('refresh-table-data'));
            setTimeout(() => {
              setActiveBatchId(null);
              localStorage.removeItem('active_batch_id');
              localStorage.removeItem('displayed_percent');
              setBatchInfo(null);
              setDisplayedPercent(0);
            }, 3500);
          }
        }
      } catch (e) {
        // quiet
      }
    };
    checkServerBatch();
  }, []);

  // Poll server for active batch status every 2 seconds
  useEffect(() => {
    if (!activeBatchId) {
      setBatchInfo(null);
      return;
    }

    let intervalId: any = null;

    const pollStatus = async () => {
      try {
        const status = await api.getBatchStatus(activeBatchId);
        setBatchInfo(status);

        const phase = status.phase;
        if (phase === 'done') {
          window.dispatchEvent(new Event('refresh-table-data'));
          setTimeout(() => {
            setActiveBatchId(null);
            localStorage.removeItem('active_batch_id');
            setBatchInfo(null);
          }, 5000);
        } else if (phase === 'error') {
          setTimeout(() => {
            setActiveBatchId(null);
            localStorage.removeItem('active_batch_id');
            setBatchInfo(null);
          }, 6000);
        }
      } catch (err) {
        console.error("Error polling batch status:", err);
      }
    };

    pollStatus();
    intervalId = setInterval(pollStatus, 2000);

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [activeBatchId]);

  // Process selected files or folder files
  const handleFilesAdded = (incomingFiles: FileList | File[]) => {
    const validFiles = Array.from(incomingFiles).filter(f => isSupportedDocument(f.name));
    if (validFiles.length === 0) {
      setIngestStatus({ message: 'No supported certificate documents (PDF, PNG, JPG, WEBP) found.', type: 'error' });
      return;
    }
    setIngestFiles(prev => [...prev, ...validFiles]);
    setIngestStatus(null);
  };

  // Drag & Drop recursive folder scanner & multi-file drop handler
  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);

    const extractedFiles: File[] = [];

    // 1. Direct files array (100% reliable for loose dropped files in all browsers)
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const directFiles = Array.from(e.dataTransfer.files).filter(f => isSupportedDocument(f.name));
      extractedFiles.push(...directFiles);
    }

    // 2. Recursive directory traversal for dropped folders
    const items = e.dataTransfer.items;
    if (items && items.length > 0) {
      const traverseEntry = async (entry: any) => {
        if (entry.isFile) {
          if (isSupportedDocument(entry.name)) {
            await new Promise<void>((resolve) => {
              entry.file((file: File) => {
                if (file && isSupportedDocument(file.name)) {
                  extractedFiles.push(file);
                }
                resolve();
              });
            });
          }
        } else if (entry.isDirectory) {
          const dirReader = entry.createReader();
          const entries = await new Promise<any[]>((resolve) => {
            dirReader.readEntries((results: any[]) => resolve(results));
          });
          for (const child of entries) {
            await traverseEntry(child);
          }
        }
      };

      for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.webkitGetAsEntry) {
          const entry = item.webkitGetAsEntry();
          // Only traverse if it is a directory (loose files are already covered by e.dataTransfer.files above)
          if (entry && entry.isDirectory) {
            await traverseEntry(entry);
          }
        }
      }
    }

    // 3. Append unique files to ingestFiles
    if (extractedFiles.length > 0) {
      setIngestFiles(prev => {
        const existingKeys = new Set(prev.map(f => `${f.name}_${f.size}`));
        const newUnique = extractedFiles.filter(f => !existingKeys.has(`${f.name}_${f.size}`));
        return [...prev, ...newUnique];
      });
      setIngestStatus(null);
    } else {
      setIngestStatus({ message: 'No PDF, PNG, or JPG files found in dropped items.', type: 'error' });
    }
  };

  // Ingestion Execution via Server Background Worker
  const handleIngestPdf = async (e: React.FormEvent) => {
    e.preventDefault();
    if (ingestFiles.length === 0 || ingesting) return;

    setIngesting(true);
    setIngestStatus(null);

    const filesToUpload = [...ingestFiles];

    try {
      // Send all file ingestion requests to server background worker
      const resp = await api.uploadBatchCertificates(filesToUpload);
      if (resp && resp.batch_id) {
        setActiveBatchId(resp.batch_id);
        localStorage.setItem('active_batch_id', resp.batch_id);
        setIngestFiles([]); // Clear selected files ONLY after batch progress UI is ready
      }
    } catch (err: any) {
      setIngestStatus({ message: err.message || 'Ingestion request failed', type: 'error' });
    } finally {
      setIngesting(false);
    }
  };

  // Compute REAL server batch progress metrics directly from server response
  const getBatchMetrics = () => {
    if (!batchInfo) return null;
    const total = batchInfo.total || 0;
    const ocr_done = batchInfo.ocr_done || 0;
    const extract_done = batchInfo.extract_done || 0;
    const skipped = batchInfo.skipped || 0;
    const failed = batchInfo.failed || 0;
    const phase = batchInfo.phase || 'unknown';
    const current_file = batchInfo.current_file || '';

    const workUnits = 2 * Math.max(total, 1);
    const doneUnits = ocr_done + extract_done + (2 * skipped);
    const percent = Math.min(100, Math.round((doneUnits / workUnits) * 100));

    let statusText = `Initializing batch (${total} files)...`;
    let subText = `Preloading OCR & LLM engines`;

    const sub_phase = batchInfo.sub_phase || '';

    if (phase === 'ocr') {
      statusText = `GLM-OCR Vision: Page Scanning`;
      subText = sub_phase || (current_file ? `Scanning ${current_file}` : `Scanning document pages...`);
    } else if (phase === 'extract') {
      statusText = `Qwen LLM Field Extraction`;
      subText = sub_phase || (current_file ? `Extracting ${current_file}` : `Extracting certificate fields...`);
    } else if (phase === 'saving') {
      statusText = `Saving Record to Database`;
      subText = sub_phase || (current_file ? `Saving ${current_file}` : `Persisting certificate...`);
    } else if (phase === 'processing') {
      statusText = `Processing Document Pipeline`;
      subText = sub_phase || (current_file ? `Processing ${current_file}` : `Running OCR & extraction...`);
    } else if (phase === 'done') {
      statusText = `Ingestion Complete! (${extract_done}/${total} saved)`;
      subText = skipped > 0 || failed > 0 ? `${skipped} skipped, ${failed} failed` : `All records saved to database`;
    } else if (phase === 'error') {
      statusText = `Ingestion failed: ${batchInfo.error || 'Unknown error'}`;
      subText = `Check system logs for details`;
    }

    return { percent, statusText, subText, phase, extract_done, ocr_done, total, current_file };
  };

  const batchMetrics = getBatchMetrics();

  return (
    <aside className={`sidebar ${isOpen ? 'open' : 'closed'}`}>
      
      {/* Sidebar Header with Top-Right Toggle Button */}
      <div className="sidebar-header">
        {isOpen && (
          <span className="sidebar-header-title">
            {isDatabaseView ? 'DATABASE TABLES' : isChatView ? 'CHAT HISTORY' : 'PENDING ACTIONS'}
          </span>
        )}
        <button 
          className="sidebar-toggle-btn-internal" 
          onClick={onToggle} 
          title={isOpen ? "Collapse Sidebar" : "Expand Sidebar"}
        >
          {isOpen ? <ChevronLeft size={18} /> : <ChevronRight size={18} />}
        </button>
      </div>

      <div className="sidebar-body">
        {isDatabaseView ? (
          <>
            {/* Table Selector List */}
            <div className="sidebar-tables-list">
              {BASE_TABLES.map(tbl => {
                const Icon = tbl.icon;
                const isActive = selectedTable === tbl.name;
                return (
                  <button
                    key={tbl.name}
                    className={`sidebar-table-btn ${isActive ? 'active' : ''}`}
                    title={tbl.name}
                    onClick={() => setSelectedTable(tbl.name)}
                  >
                    <Icon size={18} className="sidebar-btn-icon" />
                    {isOpen && <span className="sidebar-btn-text">{tbl.name}</span>}
                  </button>
                );
              })}

              {/* Custom Dynamic Tables */}
              {customTables.map(tableName => {
                const isActive = selectedTable === tableName;
                return (
                  <button
                    key={tableName}
                    className={`sidebar-table-btn ${isActive ? 'active' : ''}`}
                    title={tableName}
                    onClick={() => setSelectedTable(tableName)}
                  >
                    <Database size={18} className="sidebar-btn-icon" />
                    {isOpen && <span className="sidebar-btn-text">{tableName}</span>}
                  </button>
                );
              })}
            </div>

            {/* Action Button: + New Table */}
            {isOpen && (
              <div style={{ marginTop: '0.25rem' }}>
                <button 
                  className="btn btn-secondary" 
                  style={{ width: '100%', fontSize: '0.8rem', padding: '0.45rem', justifyContent: 'center' }}
                  onClick={() => setShowNewTableModal(!showNewTableModal)}
                >
                  <Plus size={14} /> New Table
                </button>
              </div>
            )}

            {/* Form Drawer for + New Table */}
            {isOpen && showNewTableModal && (
              <form onSubmit={handleCreateCustomTable} style={{ backgroundColor: 'var(--bg-body)', padding: '0.85rem', borderRadius: '8px', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-secondary)' }}>CREATE CUSTOM TABLE</span>
                
                <label className="btn btn-secondary" style={{ width: '100%', fontSize: '0.75rem', padding: '0.35rem 0.5rem', justifyContent: 'center', cursor: 'pointer', border: '1px dashed var(--brand-blue)', color: 'var(--brand-blue)' }}>
                  <Upload size={12} /> Import CSV / Excel File
                  <input 
                    type="file" 
                    accept=".csv, .xlsx, .xls" 
                    style={{ display: 'none' }} 
                    onChange={handleImportTableFile} 
                  />
                </label>

                <input 
                  type="text" 
                  placeholder="Table Name" 
                  className="input" 
                  style={{ fontSize: '0.8rem', padding: '0.35rem 0.5rem' }}
                  value={newTableName}
                  onChange={e => setNewTableName(e.target.value)}
                  required
                />
                <input 
                  type="text" 
                  placeholder="Columns (comma-separated)" 
                  className="input" 
                  style={{ fontSize: '0.8rem', padding: '0.35rem 0.5rem' }}
                  value={newTableCols}
                  onChange={e => setNewTableCols(e.target.value)}
                />
                <div style={{ display: 'flex', gap: '0.4rem' }}>
                  <button type="submit" className="btn btn-primary" style={{ flex: 1, fontSize: '0.75rem', padding: '0.3rem' }}>
                    Create
                  </button>
                  <button type="button" className="btn btn-secondary" style={{ fontSize: '0.75rem', padding: '0.3rem' }} onClick={() => setShowNewTableModal(false)}>
                    Cancel
                  </button>
                </div>
              </form>
            )}
          </>
        ) : isChatView ? (
          /* ASSISTANT VIEW: CHAT HISTORY */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', flex: 1, overflow: 'hidden' }}>
            {isOpen && (
              <div style={{ marginBottom: '0.25rem' }}>
                <button 
                  className="btn btn-primary" 
                  style={{ width: '100%', fontSize: '0.8rem', padding: '0.45rem', justifyContent: 'center' }}
                  onClick={() => {
                    if (setActiveSessionId) setActiveSessionId(null);
                    window.dispatchEvent(new Event('new-chat-session'));
                  }}
                >
                  <PlusCircle size={14} /> New Chat
                </button>
              </div>
            )}

            <div className="sidebar-tables-list" style={{ flex: 1, overflowY: 'auto' }}>
              {sessions.map(s => {
                const isActive = activeSessionId === s.id;
                return (
                  <div 
                    key={s.id}
                    className={`sidebar-table-btn ${isActive ? 'active' : ''}`}
                    style={{ justifyContent: 'space-between', padding: '0.45rem 0.6rem', cursor: 'pointer' }}
                    onClick={() => setActiveSessionId && setActiveSessionId(s.id)}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', overflow: 'hidden', flex: 1 }}>
                      <MessageSquare size={16} className="sidebar-btn-icon" />
                      {isOpen && <span className="sidebar-btn-text" title={s.title || 'New Chat'}>{s.title || 'New Chat'}</span>}
                    </div>
                    {isOpen && (
                      <button 
                        className="session-delete-btn" 
                        style={{ background: 'none', border: 'none', color: 'var(--text-tertiary)', cursor: 'pointer', padding: '0.2rem', display: 'flex', alignItems: 'center' }}
                        onClick={async (e) => {
                          e.stopPropagation();
                          await api.deleteSession(s.id);
                          if (activeSessionId === s.id && setActiveSessionId) setActiveSessionId(null);
                          fetchChatSessions();
                        }}
                        title="Delete session"
                      >
                        <Trash2 size={13} />
                      </button>
                    )}
                  </div>
                );
              })}
              {sessions.length === 0 && isOpen && (
                <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', textAlign: 'center', paddingTop: '0.5rem' }}>
                  No previous sessions.
                </span>
              )}
            </div>
          </div>
        ) : (
          /* AUTOMATIONS VIEW: PENDING AGENT ACTIONS */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', flex: 1, overflowY: 'auto' }}>
            {proposals.filter((p: any) => p.status === 'PENDING').map((p: any) => (
              <div 
                key={p.id} 
                style={{ 
                  backgroundColor: 'rgba(255, 255, 255, 0.04)', 
                  border: '1px solid rgba(255, 255, 255, 0.1)', 
                  borderRadius: '8px', 
                  padding: '0.65rem', 
                  display: 'flex', 
                  flexDirection: 'column', 
                  gap: '0.4rem' 
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="badge badge-yellow" style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem' }}>{p.type}</span>
                  <span style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)' }}>
                    {new Date(p.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>

                {isOpen && (
                  <>
                    <p style={{ fontSize: '0.72rem', color: 'var(--text-on-brand)', margin: 0, lineHeight: 1.3, whiteSpace: 'normal', wordBreak: 'break-word' }}>
                      {p.reasoning}
                    </p>
                    <div style={{ display: 'flex', gap: '0.35rem', marginTop: '0.25rem' }}>
                      <button 
                        className="btn btn-primary" 
                        style={{ flex: 1, fontSize: '0.7rem', padding: '0.25rem 0.4rem', backgroundColor: '#10b981', justifyContent: 'center' }}
                        onClick={async () => {
                          await api.approveProposal(p.id, 'approved');
                          fetchProposals();
                          window.dispatchEvent(new Event('refresh-proposals'));
                        }}
                      >
                        <Check size={12} /> Approve
                      </button>
                      <button 
                        className="btn btn-secondary" 
                        style={{ fontSize: '0.7rem', padding: '0.25rem 0.4rem', color: '#ef4444', borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.1)', justifyContent: 'center' }}
                        onClick={async () => {
                          await api.approveProposal(p.id, 'rejected');
                          fetchProposals();
                          window.dispatchEvent(new Event('refresh-proposals'));
                        }}
                      >
                        <X size={12} /> Reject
                      </button>
                    </div>
                  </>
                )}
              </div>
            ))}

            {proposals.filter((p: any) => p.status === 'PENDING').length === 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', paddingTop: '1.5rem', gap: '0.5rem', textAlign: 'center' }}>
                <ShieldAlert size={24} style={{ color: 'var(--text-tertiary)', opacity: 0.6 }} />
                {isOpen && (
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', whiteSpace: 'normal' }}>
                    No pending actions requiring review.
                  </span>
                )}
              </div>
            )}
          </div>
        )}

            {/* Ingest Certificate Documents Box at Bottom of Sidebar (Restricted to Home / Databases Page) */}
            {isDatabaseView && (
              isOpen ? (
                <div className="sidebar-uploader-container" style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-tertiary)', fontWeight: 700, fontSize: '0.7rem', letterSpacing: '0.05em' }}>
                    <UploadCloud size={14} color="var(--brand-blue)" />
                    <span>INGEST CERTIFICATE DOCUMENTS</span>
                  </div>

                  {/* REAL SERVER INGESTION MONITORING CARD (100% SYNCHRONIZED WITH SERVER API) */}
                  {activeBatchId && batchMetrics ? (
                    <div style={{ backgroundColor: 'rgba(36, 56, 129, 0.12)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--brand-blue)', display: 'flex', flexDirection: 'column', gap: '0.45rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.75rem', fontWeight: 600 }}>
                        <span style={{ color: 'var(--brand-blue)', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                          {batchMetrics.phase !== 'done' && batchMetrics.phase !== 'error' && <Loader2 size={12} className="spin-icon" />}
                          {batchMetrics.phase === 'done' ? 'Ingestion Complete' : 'Document Processing'}
                        </span>
                        <span style={{ color: '#ffffff', fontWeight: 700 }}>{displayedPercent}%</span>
                      </div>

                      <div style={{ width: '100%', height: '7px', backgroundColor: 'rgba(255, 255, 255, 0.15)', borderRadius: '4px', overflow: 'hidden' }}>
                        <div 
                          style={{ 
                            width: `${Math.max(5, displayedPercent)}%`, 
                            height: '100%', 
                            backgroundColor: batchMetrics.phase === 'error' ? '#ef4444' : batchMetrics.phase === 'done' ? '#10b981' : 'var(--brand-blue)', 
                            backgroundImage: batchMetrics.phase !== 'done' && batchMetrics.phase !== 'error' 
                              ? 'linear-gradient(45deg, rgba(255,255,255,0.2) 25%, transparent 25%, transparent 50%, rgba(255,255,255,0.2) 50%, rgba(255,255,255,0.2) 75%, transparent 75%, transparent)' 
                              : 'none',
                            backgroundSize: '20px 20px',
                            animation: batchMetrics.phase !== 'done' && batchMetrics.phase !== 'error' ? 'moveStripes 1s linear infinite' : 'none',
                            transition: 'width 220ms linear',
                            borderRadius: '4px'
                          }} 
                        />
                      </div>

                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-on-brand)', whiteSpace: 'normal', wordBreak: 'break-word' }}>
                          {batchMetrics.statusText}
                        </span>
                        <span style={{ fontSize: '0.65rem', color: 'var(--text-tertiary)', whiteSpace: 'normal', wordBreak: 'break-all' }}>
                          {batchMetrics.subText}
                        </span>
                      </div>
                    </div>
                  ) : (
                    /* STANDARD UPLOAD DROPZONE FORM */
                    <form 
                      onSubmit={handleIngestPdf} 
                      onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
                      onDragLeave={() => setIsDragOver(false)}
                      onDrop={handleDrop}
                      style={{ 
                        backgroundColor: isDragOver ? 'rgba(36, 56, 129, 0.2)' : 'rgba(255, 255, 255, 0.05)', 
                        padding: '0.75rem', 
                        borderRadius: '8px', 
                        border: `1px dashed ${isDragOver ? 'var(--brand-blue)' : 'rgba(255, 255, 255, 0.2)'}`, 
                        display: 'flex', 
                        flexDirection: 'column', 
                        gap: '0.5rem',
                        transition: 'all 150ms ease'
                      }}
                    >
                      {/* Unified Multi-File Input */}
                      <input 
                        ref={fileInputRef}
                        type="file" 
                        accept=".pdf, .png, .jpg, .jpeg, .webp"
                        multiple
                        id="sidebar-pdf-upload"
                        style={{ display: 'none' }}
                        onChange={e => {
                          if (e.target.files) handleFilesAdded(e.target.files);
                        }}
                      />

                      <label 
                        htmlFor="sidebar-pdf-upload"
                        style={{ 
                          display: 'flex', 
                          flexDirection: 'column', 
                          alignItems: 'center', 
                          justifyContent: 'center', 
                          gap: '0.3rem', 
                          cursor: 'pointer', 
                          padding: '0.5rem 0.25rem',
                          textAlign: 'center'
                        }}
                      >
                        <UploadCloud size={22} color="var(--brand-blue)" />
                        <span style={{ fontSize: '0.75rem', fontWeight: 500, color: 'var(--text-on-brand)', whiteSpace: 'normal', wordBreak: 'break-word' }}>
                          {ingestFiles.length > 0 
                            ? `${ingestFiles.length} document(s) ready` 
                            : 'Click or drop files/folders here'}
                        </span>
                        <span style={{ fontSize: '0.62rem', color: 'var(--text-tertiary)' }}>
                          Supports PDF, PNG, JPG, WEBP
                        </span>
                      </label>

                      {ingestFiles.length > 0 && (
                        <div style={{ display: 'flex', gap: '0.4rem', width: '100%', marginTop: '0.25rem' }}>
                          <button 
                            type="submit" 
                            className="btn btn-primary" 
                            style={{ flex: 1, fontSize: '0.75rem', padding: '0.35rem', justifyContent: 'center', backgroundColor: 'var(--brand-blue)' }}
                            disabled={ingesting}
                          >
                            {ingesting ? (
                              <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                                <Loader2 size={12} className="spin-icon" /> Ingesting...
                              </span>
                            ) : (
                              `Ingest (${ingestFiles.length})`
                            )}
                          </button>
                          <button 
                            type="button" 
                            className="btn btn-secondary" 
                            style={{ fontSize: '0.75rem', padding: '0.35rem 0.6rem', color: '#ef4444', borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.08)' }}
                            onClick={() => {
                              setIngestFiles([]);
                              setIngestStatus(null);
                            }}
                            disabled={ingesting}
                          >
                            Cancel
                          </button>
                        </div>
                      )}

                      {ingestStatus && (
                        <div style={{ 
                          fontSize: '0.7rem', 
                          padding: '0.3rem 0.5rem', 
                          borderRadius: '4px',
                          display: 'flex', 
                          alignItems: 'center', 
                          gap: '0.3rem',
                          backgroundColor: ingestStatus.type === 'success' ? 'rgba(16, 185, 129, 0.2)' : 'rgba(239, 68, 68, 0.2)',
                          color: ingestStatus.type === 'success' ? '#a7f3d0' : '#fecaca'
                        }}>
                          {ingestStatus.type === 'success' ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                          <span>{ingestStatus.message}</span>
                        </div>
                      )}
                    </form>
                  )}
                </div>
              ) : (
                /* COLLAPSED SIDEBAR INGEST ICON BUTTON */
                <div style={{ marginTop: 'auto', paddingTop: '0.75rem', borderTop: '1px solid rgba(255, 255, 255, 0.1)', display: 'flex', justifyContent: 'center', width: '100%' }}>
                  <input 
                    type="file" 
                    accept=".pdf, .png, .jpg, .jpeg, .webp"
                    multiple
                    id="sidebar-pdf-upload-collapsed"
                    style={{ display: 'none' }}
                    onChange={e => {
                      if (e.target.files) handleFilesAdded(e.target.files);
                    }}
                  />
                  <label 
                    htmlFor="sidebar-pdf-upload-collapsed" 
                    className="sidebar-table-btn"
                    title={activeBatchId ? `Ingesting documents (${displayedPercent}%)...` : "Ingest Certificate Documents"}
                    style={{ 
                      justifyContent: 'center', 
                      cursor: 'pointer', 
                      padding: '0.5rem', 
                      borderRadius: '8px', 
                      backgroundColor: activeBatchId ? 'rgba(36, 56, 129, 0.25)' : 'rgba(255, 255, 255, 0.05)',
                      border: `1px solid ${activeBatchId ? 'var(--brand-blue)' : 'rgba(255, 255, 255, 0.1)'}` 
                    }}
                  >
                    {activeBatchId ? (
                      <Loader2 size={18} color="var(--brand-blue)" className="spin-icon" />
                    ) : (
                      <UploadCloud size={18} color="var(--brand-blue)" />
                    )}
                  </label>
                </div>
              )
            )}
      </div>
    </aside>
  );
};

export default Sidebar;
