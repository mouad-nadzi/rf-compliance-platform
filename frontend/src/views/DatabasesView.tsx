import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Search, Download, Plus, Trash2, Save, Upload, CheckCircle2, AlertCircle, XCircle, ArrowUpDown, ArrowUp, ArrowDown, X, FileText, Loader2, MoreVertical, WrapText, Zap, Sparkles } from 'lucide-react';
import * as XLSX from 'xlsx';
import { api } from '../api';
import { useLayoutContext } from '../components/Layout/AppLayout';

type FilterRow = {
  id: string;
  column: string;
  value: string;
};

const DatabasesView = () => {
  const { selectedTable, setSelectedTable } = useLayoutContext();

  const isCoreTable = ["RF Certificates", "Authorities", "Suppliers", "Sources", "Agent Memories"].includes(selectedTable);

  // All tables (including RF Certificates) have full editing & deletion capabilities
  const isExcelGridTable = selectedTable !== "None";

  // Data Grid State
  const [gridData, setGridData] = useState<any[]>([]);
  const [customTableCols, setCustomTableCols] = useState<any[]>([]);
  const [selectedRowIds, setSelectedRowIds] = useState<Set<number>>(new Set());
  const [dirtyRowIndices, setDirtyRowIndices] = useState<Set<number>>(new Set());

  // Drag-to-Select Checkbox State
  const [isMouseDownSelect, setIsMouseDownSelect] = useState(false);
  const [dragStartIdx, setDragStartIdx] = useState<number | null>(null);
  const [dragTargetMode, setDragTargetMode] = useState<boolean>(true);

  // Column Sorting & Resizing State
  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' } | null>(null);
  const [colWidths, setColWidths] = useState<Record<string, number>>({});
  const [resizingCol, setResizingCol] = useState<{ key: string; startX: number; startWidth: number } | null>(null);

  // Wrap Text Mode
  const [isWrapText, setIsWrapText] = useState(false);

  // Loading, Saving, Deleting & Progress Status
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [saveStatus, setSaveStatus] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [progress, setProgress] = useState<{ current: number; total: number; percent: number; statusText: string } | null>(null);

  // 3-dots Menu Dropdown State
  const [showMoreMenu, setShowMoreMenu] = useState(false);
  const moreMenuRef = useRef<HTMLDivElement>(null);

  // Search & Filter state
  const [globalSearch, setGlobalSearch] = useState('');
  const [filterRows, setFilterRows] = useState<FilterRow[]>([]);

  // Import Drawer & Multi-file Queue state
  const [showImportDrawer, setShowImportDrawer] = useState(false);
  const [importedFiles, setImportedFiles] = useState<{ id: string; name: string; count: number }[]>([]);
  const [pendingImportRows, setPendingImportRows] = useState<any[]>([]);
  const [pendingImportFiles, setPendingImportFiles] = useState<File[] | null>(null);
  const [pendingImportUsePipeline, setPendingImportUsePipeline] = useState<boolean>(true);
  const [fileProcessingStatus, setFileProcessingStatus] = useState<string | null>(null);
  const [lastImportStats, setLastImportStats] = useState<{ total: number; newCount: number; skippedCount: number; fileName: string } | null>(null);

  // Add Custom Column Modal state
  const [showAddColumnModal, setShowAddColumnModal] = useState(false);
  const [newColName, setNewColName] = useState('');
  const [newColSemanticType, setNewColSemanticType] = useState('auto');
  const [addingColumn, setAddingColumn] = useState(false);

  // Delete Column Modal state
  const [columnToDelete, setColumnToDelete] = useState<{ key: string; label: string } | null>(null);
  const [deletingColumn, setDeletingColumn] = useState(false);

  const isColumnDeletable = useCallback((colKey: string) => {
    if (colKey === 'id' || colKey === 'created_at') return false;
    if (selectedTable === 'RF Certificates' && colKey === 'certificate_id') return false;
    return true;
  }, [selectedTable]);

  // In-place cell editing state for link and email fields
  const [editingCell, setEditingCell] = useState<{ rowIdx: number; colKey: string } | null>(null);

  const handleAddColumnSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newColName.trim()) return;
    setAddingColumn(true);
    try {
      const sqlDataType = newColSemanticType === 'number' ? 'INTEGER' : (newColSemanticType === 'date' ? 'DATE' : 'VARCHAR(255)');
      await api.addCustomTableColumn(selectedTable, newColName.trim(), sqlDataType, newColSemanticType);
      setSaveStatus({ message: `Successfully added column "${newColName.trim()}" to ${selectedTable}.`, type: 'success' });
      setNewColName('');
      setNewColSemanticType('auto');
      setShowAddColumnModal(false);
      await fetchTableData();
    } catch (err: any) {
      console.error("Failed to add column:", err);
      alert(`Error adding column: ${err.message || 'Server error'}`);
    } finally {
      setAddingColumn(false);
    }
  };

  const handleDeleteColumnConfirm = async () => {
    if (!columnToDelete || !selectedTable) return;
    setDeletingColumn(true);
    try {
      await api.deleteCustomTableColumn(selectedTable, columnToDelete.key);
      setSaveStatus({
        message: `Successfully deleted column "${columnToDelete.label}" from ${selectedTable}.`,
        type: 'success'
      });
      setColumnToDelete(null);
      await fetchTableData();
    } catch (err: any) {
      console.error("Failed to delete column:", err);
      alert(`Error deleting column: ${err.message || 'Server error'}`);
    } finally {
      setDeletingColumn(false);
    }
  };

  // Close 3-dots menu on click outside
  useEffect(() => {
    const handleClickOutsideMore = (event: MouseEvent) => {
      if (moreMenuRef.current && !moreMenuRef.current.contains(event.target as Node)) {
        setShowMoreMenu(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutsideMore);
    return () => document.removeEventListener('mousedown', handleClickOutsideMore);
  }, []);

  // Fetch Table Data based on selected table (resets drawer & filters on table change, restores unsaved draft session)
  const fetchIdRef = useRef(0);

  const fetchTableData = useCallback(async (silentRefresh: boolean = false) => {
    const fetchId = ++fetchIdRef.current;
    
    if (!silentRefresh) {
      setLoading(true);
      setGridData([]);
      setSaveStatus(null);
      setProgress(null);
      setSortConfig(null);
      setShowImportDrawer(false);
      setLastImportStats(null);
      setGlobalSearch('');
      setFilterRows([]);
    }
    setDirtyRowIndices(new Set());
    setSelectedRowIds(new Set());
    setImportedFiles([]);
    try {
      let rows: any[] = [];
      let fetchedCols: any[] = [];
      if (selectedTable === "RF Certificates") {
        const [certs, schemaData] = await Promise.all([
          api.getCertificates(),
          api.getCustomTableRows("certificates").catch(() => null)
        ]);
        if (schemaData && schemaData.columns) {
          fetchedCols = schemaData.columns;
          if (Array.isArray(schemaData.records) && schemaData.records.length > 0) {
            const schemaRecordMap = new Map();
            schemaData.records.forEach((r: any) => {
              if (r.certificate_id) schemaRecordMap.set(r.certificate_id, r);
            });
            rows = (certs || []).map((c: any) => {
              const extra = schemaRecordMap.get(c.certificate_id) || {};
              return { ...extra, ...c };
            });
          } else {
            rows = certs || [];
          }
        } else {
          rows = certs || [];
        }
      } else if (selectedTable === "Authorities") {
        const [auths, schemaData] = await Promise.all([
          api.getAuthorities(),
          api.getCustomTableRows("authority_lookups").catch(() => null)
        ]);
        rows = auths || [];
        if (schemaData && schemaData.columns) fetchedCols = schemaData.columns;
      } else if (selectedTable === "Suppliers") {
        const [supps, schemaData] = await Promise.all([
          api.getSuppliers(),
          api.getCustomTableRows("supplier_lookups").catch(() => null)
        ]);
        rows = supps || [];
        if (schemaData && schemaData.columns) fetchedCols = schemaData.columns;
      } else if (selectedTable === "Sources") {
        const [srcs, schemaData] = await Promise.all([
          api.getSources(),
          api.getCustomTableRows("sources").catch(() => null)
        ]);
        rows = srcs || [];
        if (schemaData && schemaData.columns) fetchedCols = schemaData.columns;
      } else if (selectedTable === "Agent Memories") {
        const [mems, schemaData] = await Promise.all([
          api.getMemories(),
          api.getCustomTableRows("agent_memories").catch(() => null)
        ]);
        rows = mems || [];
        if (schemaData && schemaData.columns) fetchedCols = schemaData.columns;
      } else {
        const data = await api.getCustomTableRows(selectedTable);
        rows = data.records || [];
        fetchedCols = data.columns || [];
      }
      
      // If selectedTable changed while we were fetching, ignore this stale response
      if (fetchId !== fetchIdRef.current) return;

      const fetchedRows = Array.isArray(rows) ? rows : [];

      // Always clear any legacy draft key so page refresh reflects true PostgreSQL database state
      const draftKey = `rf_draft_${selectedTable}`;
      localStorage.removeItem(draftKey);

      setGridData(fetchedRows);
      setCustomTableCols(fetchedCols);
      setImportedFiles([]);
    } catch (err) {
      if (fetchId !== fetchIdRef.current) return;
      console.error("Error fetching table data:", err);
      setGridData([]);
    } finally {
      if (fetchId === fetchIdRef.current) {
        setLoading(false);
      }
    }
  }, [selectedTable]);

  // Unsaved changes calculation (evaluates dirty indices & pending imports)
  const unsavedCount = useMemo(() => {
    return dirtyRowIndices.size + pendingImportRows.length;
  }, [dirtyRowIndices, pendingImportRows]);

  const hasUnsavedChanges = useMemo(() => {
    return saving || unsavedCount > 0;
  }, [saving, unsavedCount]);

  useEffect(() => {
    fetchTableData();
    const handleRefresh = () => fetchTableData();
    window.addEventListener('refresh-table-data', handleRefresh);
    return () => window.removeEventListener('refresh-table-data', handleRefresh);
  }, [fetchTableData]);

  useEffect(() => {
    const handleSidebarImport = (e: any) => {
      const file = e.detail?.file;
      if (file) {
        setPendingImportFiles([file]);
      }
    };
    window.addEventListener('sidebar-import-file', handleSidebarImport);
    return () => window.removeEventListener('sidebar-import-file', handleSidebarImport);
  }, []);

  // Auto-dismiss status toast banner after 4.5 seconds
  useEffect(() => {
    if (!saveStatus) return;
    const timer = setTimeout(() => {
      setSaveStatus(null);
    }, 4500);
    return () => clearTimeout(timer);
  }, [saveStatus]);



  // BeforeUnload protection if there are unsaved rows or active saving
  useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (hasUnsavedChanges || saving) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes and active imports. Are you sure you want to leave?';
        return e.returnValue;
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [hasUnsavedChanges, saving]);

// Global Mouse Up Listener for Drag-to-Select & Column Resizing
useEffect(() => {
  const handleGlobalMouseUp = () => {
    setIsMouseDownSelect(false);
    setDragStartIdx(null);
    setResizingCol(null);
  };
  window.addEventListener('mouseup', handleGlobalMouseUp);
  return () => window.removeEventListener('mouseup', handleGlobalMouseUp);
}, []);

// Handle Column Resizing Dragging
useEffect(() => {
  if (!resizingCol) return;
  const handleMouseMove = (e: MouseEvent) => {
    const diff = e.clientX - resizingCol.startX;
    const newWidth = Math.max(90, resizingCol.startWidth + diff);
    setColWidths(prev => ({ ...prev, [resizingCol.key]: newWidth }));
  };

  window.addEventListener('mousemove', handleMouseMove);
  return () => window.removeEventListener('mousemove', handleMouseMove);
}, [resizingCol]);

  const handleStartResize = (e: React.MouseEvent, key: string, currentWidth: number) => {
    e.preventDefault();
    e.stopPropagation();
    setResizingCol({ key, startX: e.clientX, startWidth: currentWidth });
  };

  // Auto-fit single column width to its content on double click
  const handleAutoFitColumn = (key: string, label: string) => {
    let maxChars = label.length;
    gridData.forEach(row => {
      const val = row[key];
      if (val !== undefined && val !== null) {
        const len = String(val).length;
        if (len > maxChars) maxChars = len;
      }
    });
    const autoWidth = Math.min(650, Math.max(120, Math.round(maxChars * 8.5) + 40));
    setColWidths(prev => ({ ...prev, [key]: autoWidth }));
  };


// Drag-to-Select Checkbox Handlers
const handleCheckboxMouseDown = (e: React.MouseEvent, index: number) => {
  e.preventDefault();
  const willBeSelected = !selectedRowIds.has(index);
  setIsMouseDownSelect(true);
  setDragStartIdx(index);
  setDragTargetMode(willBeSelected);

  setSelectedRowIds(prev => {
    const next = new Set(prev);
    if (willBeSelected) next.add(index);
    else next.delete(index);
    return next;
  });
};

const handleCheckboxMouseEnter = (index: number) => {
  if (!isMouseDownSelect || dragStartIdx === null) return;
  const start = Math.min(dragStartIdx, index);
  const end = Math.max(dragStartIdx, index);

  setSelectedRowIds(prev => {
    const next = new Set(prev);
    for (let i = start; i <= end; i++) {
      if (dragTargetMode) next.add(i);
      else next.delete(i);
    }
    return next;
  });
};

// Toggle Column Sorting on Header Click
const handleSort = (key: string) => {
  setSortConfig(current => {
    if (!current || current.key !== key) return { key, direction: 'asc' };
    if (current.direction === 'asc') return { key, direction: 'desc' };
    return null;
  });
};

// Handle Cell Editing
const handleCellChange = (rowIndex: number, fieldKey: string, newValue: any) => {
  setGridData(prev => {
    const copy = [...prev];
    copy[rowIndex] = { ...copy[rowIndex], [fieldKey]: newValue };
    return copy;
  });
  setDirtyRowIndices(prev => new Set(prev).add(rowIndex));
};

// Add New Blank Row at Top of Table
const handleAddRow = () => {
  let newBlankRow: any = { _isNew: true };
  if (selectedTable === "RF Certificates") {
    newBlankRow = { ...newBlankRow, component: '', supplier: '', country: '', certif_number: '', authority: '', issue_date: '', exp_date: '', cert_link: '', last_update: new Date().toISOString().slice(0, 16).replace('T', ' ') };
  } else if (selectedTable === "Authorities") {
    newBlankRow = { ...newBlankRow, canonical_authority: '', abbreviation: '', country: '', standard_validity_years: 5, aliases: '' };
  } else if (selectedTable === "Suppliers") {
    newBlankRow = { ...newBlankRow, canonical_supplier: '', aliases: '' };
  } else if (selectedTable === "Sources") {
    newBlankRow = { ...newBlankRow, url: '', description: '', active: true };
  } else if (selectedTable === "Agent Memories") {
    newBlankRow = { ...newBlankRow, memory_key: 'preference', fact_text: '' };
  } else {
    newBlankRow = { ...newBlankRow, name: '' };
  }

  setGridData(prev => [newBlankRow, ...prev]);
  setDirtyRowIndices(prev => {
    const next = new Set<number>();
    next.add(0);
    prev.forEach(idx => next.add(idx + 1));
    return next;
  });
};

const toggleSelectAll = () => {
  if (selectedRowIds.size === gridData.length) {
    setSelectedRowIds(new Set());
  } else {
    setSelectedRowIds(new Set(gridData.map((_, i) => i)));
  }
};

  // Delete Selected Rows (Moves to PostgreSQL Recycle Bin)
  const handleDeleteSelectedRows = async () => {
    if (selectedRowIds.size === 0) return;
    if (!window.confirm(`Move ${selectedRowIds.size} selected record(s) to Recycle Bin?`)) return;

    setDeleting(true);
    let deletedCount = 0;

    try {
      const indices = Array.from(selectedRowIds);
      const selectedRows = indices.map(idx => gridData[idx]).filter(Boolean);

      if (selectedTable === "RF Certificates" && selectedRows.length > 5) {
        const idsToDelete = selectedRows.map(r => r.id || r.certificate_id).filter(Boolean);
        if (idsToDelete.length > 0) {
          await api.batchDeleteCertificates(idsToDelete);
          deletedCount = idsToDelete.length;
        }
      } else {
        for (const row of selectedRows) {
          const targetId = row.id || row.certificate_id;
          if (targetId) {
            if (selectedTable === "RF Certificates") await api.deleteCertificate(targetId);
            else if (selectedTable === "Authorities") await api.deleteAuthority(targetId);
            else if (selectedTable === "Suppliers") await api.deleteSupplier(targetId);
            else if (selectedTable === "Sources") await api.deleteSource(targetId);
            else if (selectedTable === "Agent Memories") await api.deleteMemory(targetId);
            else await api.deleteCustomTableRow(selectedTable, targetId);
          }
          deletedCount++;
        }
      }

      setSelectedRowIds(new Set());
      if (selectedTable) localStorage.removeItem(`rf_draft_${selectedTable}`);
      setSaveStatus({ message: `Successfully moved ${deletedCount} record(s) to Recycle Bin!`, type: 'success' });
      window.dispatchEvent(new Event('refresh-recycle-bin'));
      await fetchTableData();
    } catch (err: any) {
      setSaveStatus({ message: `Error deleting records: ${err.message || 'Operation failed'}`, type: 'error' });
    } finally {
      setDeleting(false);
    }
  };

  // Drop Dynamic Custom Table (Soft-deletes table and records to global Recycle Bin)
  const handleDropCustomTable = async () => {
    if (isCoreTable) return;
    if (!window.confirm(`Move custom dynamic table "${selectedTable}" to global Recycle Bin?`)) return;

    try {
      const tableNameToDrop = selectedTable;
      await api.deleteCustomTable(tableNameToDrop);
      if (setSelectedTable) setSelectedTable('RF Certificates');
      window.dispatchEvent(new Event('refresh-custom-tables'));
      window.dispatchEvent(new Event('refresh-recycle-bin'));
    } catch (err: any) {
      alert(`Failed to drop table: ${err.message || 'Error occurred'}`);
    }
  };

  const cancelRequestedRef = useRef(false);

  // Save Changes to Database with Live Progress Bar & Abort Capability
  const handleSaveChangesToDatabase = async (rowsToSaveOverride?: any[], usePipeline: boolean = pendingImportUsePipeline) => {
    cancelRequestedRef.current = false;
    let rowsToSave: any[] = [];
    if (rowsToSaveOverride) {
      rowsToSave = rowsToSaveOverride;
    } else {
      const dirtyGridRows = Array.from(dirtyRowIndices).map(i => gridData[i]).filter(Boolean);
      rowsToSave = [...pendingImportRows, ...dirtyGridRows];
    }

    if (rowsToSave.length === 0) {
      setSaveStatus({ message: "No changes to save.", type: 'success' });
      return;
    }

    const total = rowsToSave.length;

    setSaving(true);
    setSaveStatus(null);
    let savedCount = 0;
    let failedCount = 0;

    setProgress({
      current: 0,
      total,
      percent: 0,
      statusText: `Preparing to save ${total} record(s) to ${selectedTable}...`
    });

    try {
      if (selectedTable === "RF Certificates") {
        const CHUNK_SIZE = 100;
        for (let i = 0; i < total; i += CHUNK_SIZE) {
          if (cancelRequestedRef.current) {
            setSaveStatus({ message: `Import / Save operation cancelled by user. Saved ${savedCount} of ${total} records.`, type: 'error' });
            break;
          }

          const chunk = rowsToSave.slice(i, i + CHUNK_SIZE);
          const chunkEndIndex = Math.min(i + CHUNK_SIZE, total);
          const pct = Math.round((chunkEndIndex / total) * 100);

          setProgress({
            current: chunkEndIndex,
            total,
            percent: pct,
            statusText: `Batch saving records ${i + 1} to ${chunkEndIndex} of ${total}... (${pct}%)`
          });

          try {
            await api.batchSaveCertificates(chunk);
            savedCount += chunk.length;
          } catch (e: any) {
            console.error("Error batch saving certificates:", e);
            failedCount += chunk.length;
          }
        }
      } else if (!isCoreTable) {
        // Batch saving for Custom Dynamic Tables
        const CHUNK_SIZE = usePipeline ? 10 : 100;
        for (let i = 0; i < total; i += CHUNK_SIZE) {
          if (cancelRequestedRef.current) {
            setSaveStatus({ message: `Import / Save operation cancelled by user. Saved ${savedCount} of ${total} records.`, type: 'error' });
            break;
          }

          const chunk = rowsToSave.slice(i, i + CHUNK_SIZE);
          const chunkEndIndex = Math.min(i + CHUNK_SIZE, total);
          const pct = Math.round((chunkEndIndex / total) * 100);

          setProgress({
            current: chunkEndIndex,
            total,
            percent: pct,
            statusText: `Processing file... (${pct}%)`
          });

          try {
            await api.batchSaveCustomTableRows(selectedTable, chunk, usePipeline);
            savedCount += chunk.length;
          } catch (e: any) {
            console.error("Error batch saving custom table rows:", e);
            failedCount += chunk.length;
          }
        }
      } else {
        // Individual saving for core lookup tables
        for (let i = 0; i < total; i++) {
          if (cancelRequestedRef.current) {
            setSaveStatus({ message: `Import / Save operation cancelled by user. Saved ${savedCount} of ${total} records.`, type: 'error' });
            break;
          }

          const row = rowsToSave[i];
          try {
            if (selectedTable === "Authorities") {
              const aliasesArr = Array.isArray(row.aliases) ? row.aliases : (String(row.aliases || '').split(',').map(s => s.trim()).filter(Boolean));
              await api.saveAuthorityRow({ ...row, aliases: aliasesArr });
            } else if (selectedTable === "Suppliers") {
              const aliasesArr = Array.isArray(row.aliases) ? row.aliases : (String(row.aliases || '').split(',').map(s => s.trim()).filter(Boolean));
              await api.saveSupplierRow({ ...row, aliases: aliasesArr });
            } else if (selectedTable === "Sources") {
              await api.saveSourceRow(row);
            } else if (selectedTable === "Agent Memories") {
              await api.saveMemoryRow(row);
            }
            savedCount++;
          } catch (e: any) {
            console.error("Error saving row:", e);
            failedCount++;
          }

          const pct = Math.round(((i + 1) / total) * 100);
          setProgress({
            current: i + 1,
            total,
            percent: pct,
            statusText: `Saving record ${i + 1} of ${total} to ${selectedTable}... (${pct}%)`
          });
        }
      }

      if (!cancelRequestedRef.current) {
        if (failedCount === 0) {
          if (selectedTable) localStorage.removeItem(`rf_draft_${selectedTable}`);
          setImportedFiles([]);
          setPendingImportRows([]);
          setDirtyRowIndices(new Set());
          setShowImportDrawer(false);
          setSaveStatus({ message: `Successfully committed ${savedCount} record(s) to Database!`, type: 'success' });
        } else {
          setSaveStatus({ message: `Saved ${savedCount} record(s), ${failedCount} failed.`, type: 'error' });
        }
      }
    } catch (err: any) {
      setSaveStatus({ message: `Error saving changes: ${err.message}`, type: 'error' });
    } finally {
      setSaving(false);
      setTimeout(() => setProgress(null), 1000);
      await fetchTableData(true);
    }
  };

  // Cancel Unsaved Changes or Stop Active Import/Save
  const handleCancelChanges = () => {
    cancelRequestedRef.current = true;
    setSaving(false);
    setProgress(null);
    setImportedFiles([]);
    setPendingImportRows([]);
    if (selectedTable) localStorage.removeItem(`rf_draft_${selectedTable}`);
    setSaveStatus({ message: "Import / Save cancelled. Unsaved changes discarded.", type: 'error' });
    fetchTableData();
  };

  // Remove a specific imported file from queue
  const handleRemoveImportedFile = (fileId: string) => {
    const targetFile = importedFiles.find(f => f.id === fileId);
    setPendingImportRows(prev => prev.filter(r => r._fileId !== fileId));
    setImportedFiles(prev => prev.filter(f => f.id !== fileId));
    
    if (targetFile) {
      setSaveStatus({ message: `Removed ${targetFile.name} (${targetFile.count} rows) from import queue.`, type: 'success' });
    }
  };

// Helper to parse Excel numeric serial dates (e.g. 44654 -> "2022-03-31")
const formatExcelDate = (val: any): string => {
  if (val === undefined || val === null || val === '') return '';
  const str = String(val).trim();

  // Check if value is an Excel numeric serial integer (e.g., 44654 or "44654")
  if (/^\d{4,5}(\.\d+)?$/.test(str)) {
    const num = parseFloat(str);
    if (num > 1000 && num < 100000) {
      const d = new Date((num - (25567 + 2)) * 86400 * 1000);
      if (!isNaN(d.getTime())) {
        return d.toISOString().split('T')[0];
      }
    }
  }

  // Parse text dates like "3-Apr-22", "19-Sep-24", "18-Nov-21", "14-Aug-23"
  const parsedTimestamp = Date.parse(str);
  if (!isNaN(parsedTimestamp)) {
    const d = new Date(parsedTimestamp);
    const year = d.getFullYear();
    const month = String(d.getMonth() + 1).padStart(2, '0');
    const day = String(d.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  return str;
};

  // File selection triggers the Middle Phase Import Mode Modal
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    // Filter out files that already exist in importedFiles queue SILENTLY
    const existingFileNames = new Set(importedFiles.map(f => f.name.toLowerCase()));
    const validFiles = files.filter(f => !existingFileNames.has(f.name.toLowerCase()));

    if (validFiles.length === 0) {
      setSaveStatus({ message: `Selected file(s) are already in the import queue.`, type: 'success' });
      e.target.value = '';
      return;
    }

    setPendingImportFiles(validFiles);
    e.target.value = '';
  };

  // Execute Import (either Fast/As-Is or AI Standardization Pipeline)
  const executeImport = async (validFiles: File[], usePipeline: boolean) => {
    setPendingImportFiles(null);
    if (!validFiles || validFiles.length === 0) return;

    let newlyAddedFilesCount = 0;
    let newlyAddedRowsCount = 0;
    let totalSkippedDuplicatesCount = 0;
    let totalParsedRowsCount = 0;
    let lastFileName = '';

    const newFileEntries: { id: string; name: string; count: number }[] = [];
    const allNewRows: any[] = [];

    const totalFilesCount = validFiles.length;
    let fileIdx = 0;

    for (const file of validFiles) {
      fileIdx++;
      lastFileName = file.name;
      setFileProcessingStatus(`Processing file ${fileIdx} of ${totalFilesCount} (${file.name})...`);
      const fileId = 'file_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);

      try {
        const bstr = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = (evt) => resolve(evt.target?.result as string);
          reader.onerror = (err) => reject(err);
          reader.readAsBinaryString(file);
        });

        const wb = XLSX.read(bstr, { type: 'binary', cellDates: true, dateNF: 'yyyy-mm-dd' });
        const wsname = wb.SheetNames[0];
        const ws = wb.Sheets[wsname];
        const rawImportedRows: any[] = XLSX.utils.sheet_to_json(ws, { raw: false, dateNF: 'yyyy-mm-dd' });

        if (rawImportedRows.length > 0) {
          totalParsedRowsCount += rawImportedRows.length;
          const sampleRow = rawImportedRows[0] || {};
          const headers = Object.keys(sampleRow);
          let targetFields = columnsMeta.map((c: any) => String(c.key));

          if (!isCoreTable) {
            try {
              const tableData = await api.getCustomTableRows(selectedTable);
              if (tableData && Array.isArray(tableData.columns) && tableData.columns.length > 0) {
                const dbCols = tableData.columns
                  .filter((c: any) => c.column_name !== 'id' && c.column_name !== 'created_at')
                  .map((c: any) => c.column_name);
                if (dbCols.length > 0) {
                  targetFields = dbCols;
                  setCustomTableCols(tableData.columns);
                }
              }
            } catch (e) {
              console.warn("Could not fetch latest custom table columns for import", e);
            }
          }

          let llmMapping: Record<string, string> = {};
          let suppMap: Record<string, string> = {};

          if (usePipeline) {
            // Request dynamic LLM header mapping & supplier lookups from backend
            try {
              const mapRes = await api.mapSpreadsheetHeaders(headers, sampleRow, selectedTable, targetFields);
              if (mapRes && mapRes.mapping) {
                llmMapping = mapRes.mapping;
              }
            } catch (e) {
              console.warn("LLM header mapping request failed, using fallback mapper", e);
            }

            try {
              const suppList = await api.getSuppliers();
              if (Array.isArray(suppList)) {
                for (const s of suppList) {
                  if (s.canonical_supplier) {
                    const canon = String(s.canonical_supplier).trim();
                    suppMap[canon.toLowerCase()] = canon;
                    if (Array.isArray(s.aliases)) {
                      for (const alias of s.aliases) {
                        if (alias) suppMap[String(alias).trim().toLowerCase()] = canon;
                      }
                    }
                  }
                }
              }
            } catch (e) {
              console.warn("Could not load supplier lookups for client canonicalization", e);
            }
          }

          const normalizeKey = (k: string) => String(k || '').toLowerCase().replace(/[^a-z0-9]/g, '');

          const HEADER_ALIASES: Record<string, string[]> = {
            component: ['component', 'componentmodel', 'componentname', 'model', 'modelname', 'equipment', 'name'],
            supplier: ['supplier', 'suppliermanufacturer', 'canonicalsupplier', 'manufacturer', 'vendor', 'brand'],
            country: ['country', 'market', 'nation'],
            certif_number: ['certifnumber', 'certificateno', 'certificatenumber', 'homologationnumber', 'certno', 'certnumber', 'approvalnumber'],
            authority: ['authority', 'canonicalauthority', 'regulatoryauthority', 'agency', 'body'],
            issue_date: ['issuedate', 'approvaldate', 'startdate', 'validfrom'],
            exp_date: ['expdate', 'expirationdate', 'expirydate', 'validuntil', 'validto', 'enddate'],
            cert_link: ['certlink', 'pdfdocumentlink', 'doclink', 'docinfolink', 'url', 'link', 'documenturl'],
          };

          const importedRows = rawImportedRows.map(r => {
            if (!r || typeof r !== 'object') return r;

            const getLlmVal = (targetKey: string) => {
              if (usePipeline) {
                const srcHeader = llmMapping[targetKey];
                if (srcHeader && r[srcHeader] !== undefined && r[srcHeader] !== null) {
                  return r[srcHeader];
                }
                const reverseEntry = Object.entries(llmMapping).find(
                  ([k, v]) => v === targetKey && r[k] !== undefined && r[k] !== null
                );
                if (reverseEntry) {
                  return r[reverseEntry[0]];
                }
              }
              if (r[targetKey] !== undefined) return r[targetKey];
              const normTarget = normalizeKey(targetKey);
              const aliases = HEADER_ALIASES[targetKey] || [normTarget];
              const matchKey = Object.keys(r).find(k => {
                const nk = normalizeKey(k);
                return nk === normTarget || aliases.includes(nk);
              });
              if (matchKey && r[matchKey] !== undefined) return r[matchKey];
              return undefined;
            };

            if (selectedTable === 'RF Certificates' || selectedTable === 'certificates') {
              const comp = getLlmVal('component') ?? r.component ?? r.Component ?? r.name;
              const rawSuppVal = getLlmVal('supplier') ?? r.supplier ?? r.Supplier ?? r.canonical_supplier;
              const ctry = getLlmVal('country') ?? r.country ?? r.Country;
              const certNo = getLlmVal('certif_number') ?? r.certif_number ?? r['Certif Number'] ?? r['Certificate No.'];
              const auth = getLlmVal('authority') ?? r.authority ?? r.Authority ?? r.canonical_authority;
              const issueD = formatExcelDate(getLlmVal('issue_date') ?? r.issue_date ?? r['Issue Date']);
              const expD = formatExcelDate(getLlmVal('exp_date') ?? r.exp_date ?? r.expiration_date ?? r['Expiration Date'] ?? r['Exp Date']);
              const certL = getLlmVal('cert_link') ?? r.cert_link ?? r['PDF Document Link'] ?? r.url;

              let suppStr = rawSuppVal !== undefined ? String(rawSuppVal).trim() : '';
              if (usePipeline) {
                if (suppStr && suppMap[suppStr.toLowerCase()]) {
                  suppStr = suppMap[suppStr.toLowerCase()];
                } else if (suppStr) {
                  for (const [aliasKey, canonTarget] of Object.entries(suppMap)) {
                    if (aliasKey.length >= 3 && suppStr.toLowerCase().includes(aliasKey)) {
                      suppStr = canonTarget;
                      break;
                    }
                  }
                }
              }

              return {
                ...r,
                component: comp !== undefined ? String(comp).trim() : '',
                supplier: suppStr,
                country: ctry !== undefined ? String(ctry).trim() : '',
                certif_number: certNo !== undefined ? String(certNo).trim() : '',
                authority: auth !== undefined ? String(auth).trim() : '',
                issue_date: issueD,
                exp_date: expD,
                cert_link: certL !== undefined ? String(certL).trim() : ''
              };
            } else {
              // Custom Dynamic Tables mapping
              const dynamicRow: any = {};
              for (const tf of targetFields) {
                const mappedVal = getLlmVal(tf);
                if (mappedVal !== undefined) {
                  dynamicRow[tf] = mappedVal;
                } else if (r[tf] !== undefined) {
                  dynamicRow[tf] = r[tf];
                } else {
                  dynamicRow[tf] = '';
                }
              }
              return dynamicRow;
            }
          });

          const getSig = (r: any) => {
            if (selectedTable === 'RF Certificates' || selectedTable === 'certificates') {
              const fields = [
                r.component,
                r.supplier,
                r.country,
                r.certif_number,
                r.authority,
                r.issue_date,
                r.exp_date,
              ].map(v => String(v || '').trim().toLowerCase());
              return fields.join('::');
            } else {
              const vals = targetFields.map(tf => (r[tf] || '').toString().trim().toLowerCase());
              return vals.join('::');
            }
          };

          const existingSignatures = new Set(
            [...gridData, ...pendingImportRows, ...allNewRows].map(getSig).filter(sig => sig.replace(/:/g, '') !== '')
          );

          const uniqueRows = importedRows.filter(r => {
            const sig = getSig(r);
            return sig.replace(/:/g, '') === '' || !existingSignatures.has(sig);
          });

          const fileDuplicatesCount = rawImportedRows.length - uniqueRows.length;
          totalSkippedDuplicatesCount += fileDuplicatesCount;

          if (uniqueRows.length > 0) {
            const rowsWithFlags = uniqueRows.map(r => ({
              ...r,
              _fileId: fileId,
              _fileName: file.name,
              _isNew: true,
              _isDirty: true
            }));

            allNewRows.push(...rowsWithFlags);
            newFileEntries.push({ id: fileId, name: file.name, count: uniqueRows.length });
            newlyAddedFilesCount++;
            newlyAddedRowsCount += uniqueRows.length;
          }
        }
      } catch (err) {
        console.error(`Error reading file ${file.name}:`, err);
      }
    }

    setFileProcessingStatus(null);

    setLastImportStats({
      total: totalParsedRowsCount,
      newCount: newlyAddedRowsCount,
      skippedCount: totalSkippedDuplicatesCount,
      fileName: validFiles.length === 1 ? lastFileName : `${validFiles.length} files`
    });

    if (allNewRows.length > 0) {
      setPendingImportUsePipeline(usePipeline);
      setPendingImportRows(prev => [...allNewRows, ...prev]);
      setImportedFiles(prev => [...prev, ...newFileEntries]);
      setShowImportDrawer(true);

      const duplicateNote = totalSkippedDuplicatesCount > 0 ? ` (${totalSkippedDuplicatesCount} duplicate row(s) skipped)` : '';
      const modeNote = usePipeline ? ' [Standardized]' : ' [As-Is]';
      setSaveStatus({
        message: `${newlyAddedFilesCount} file(s) staged${modeNote}: ${newlyAddedRowsCount} new row(s) ready in Preview${duplicateNote}. Review below and click "Confirm & Import to Database".`,
        type: 'success'
      });
    } else {
      setShowImportDrawer(true);
      setSaveStatus({
        message: `0 new records to import (all duplicate or empty).`,
        type: 'error'
      });
    }

    setProgress({
      current: totalFilesCount,
      total: totalFilesCount,
      percent: 100,
      statusText: `Processed ${totalFilesCount} file(s) successfully! (100%)`
    });
    setTimeout(() => setProgress(null), 1200);
  };

// Export Excel (.xlsx)
const handleExportExcel = () => {
  if (gridData.length === 0) return;

  // Export selected rows if checkboxes checked, otherwise export full table
  let rowsToExport = gridData;
  if (selectedRowIds.size > 0) {
    const selectedIndices = Array.from(selectedRowIds);
    rowsToExport = selectedIndices.map(idx => gridData[idx]).filter(Boolean);
  }

  const cleanData = rowsToExport.map(({ _isNew, ...rest }) => rest);
  const worksheet = XLSX.utils.json_to_sheet(cleanData);
  const workbook = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(workbook, worksheet, selectedTable.slice(0, 31));
  const filename = selectedRowIds.size > 0
    ? `${selectedTable.toLowerCase().replace(/\s+/g, '_')}_selected_${selectedRowIds.size}.xlsx`
    : `${selectedTable.toLowerCase().replace(/\s+/g, '_')}_all.xlsx`;
  XLSX.writeFile(workbook, filename);
};

// Column definitions per table
const columnsMeta = useMemo(() => {
  let baseCols: any[] = [];
  if (selectedTable === "Authorities") {
    baseCols = [
      { key: 'canonical_authority', label: 'Canonical Authority', semantic_type: 'text', defaultWidth: 220 },
      { key: 'abbreviation', label: 'Abbreviation', semantic_type: 'badge', defaultWidth: 130 },
      { key: 'country', label: 'Country', semantic_type: 'text', defaultWidth: 140 },
      { key: 'standard_validity_years', label: 'Validity (Years)', semantic_type: 'number', defaultWidth: 140 },
      { key: 'aliases', label: 'Aliases', semantic_type: 'text', defaultWidth: 260 },
    ];
  } else if (selectedTable === "Suppliers") {
    baseCols = [
      { key: 'canonical_supplier', label: 'Canonical Supplier', semantic_type: 'supplier', defaultWidth: 240 },
      { key: 'aliases', label: 'Aliases', semantic_type: 'text', defaultWidth: 320 },
    ];
  } else if (selectedTable === "Sources") {
    baseCols = [
      { key: 'url', label: 'Source URL', semantic_type: 'url', defaultWidth: 300 },
      { key: 'description', label: 'Description', semantic_type: 'text', defaultWidth: 260 },
      { key: 'active', label: 'Active Status', semantic_type: 'badge', defaultWidth: 130 },
    ];
  } else if (selectedTable === "Agent Memories") {
    baseCols = [
      { key: 'memory_key', label: 'Category / Key', semantic_type: 'badge', defaultWidth: 180 },
      { key: 'fact_text', label: 'Memory Fact / Directive', semantic_type: 'text', defaultWidth: 380 },
      { key: 'created_at', label: 'Date Created', semantic_type: 'date', defaultWidth: 160 },
    ];
  } else if (selectedTable === "RF Certificates") {
    baseCols = [
      { key: 'component', label: 'Component / Model', semantic_type: 'text', defaultWidth: 200 },
      { key: 'supplier', label: 'Supplier / Manufacturer', semantic_type: 'supplier', defaultWidth: 200 },
      { key: 'country', label: 'Country', semantic_type: 'text', defaultWidth: 120 },
      { key: 'certif_number', label: 'Certificate No.', semantic_type: 'text', defaultWidth: 180 },
      { key: 'authority', label: 'Authority', semantic_type: 'text', defaultWidth: 160 },
      { key: 'issue_date', label: 'Issue Date', semantic_type: 'date', defaultWidth: 130 },
      { key: 'exp_date', label: 'Exp Date', semantic_type: 'date', defaultWidth: 130 },
      { key: 'cert_link', label: 'PDF Document Link', semantic_type: 'url', defaultWidth: 200 },
      { key: 'last_update', label: 'Last Update', semantic_type: 'date', defaultWidth: 150 },
    ];
  }

  if (baseCols.length > 0) {
    const existingKeys = new Set(baseCols.map(c => c.key));
    const extraCols = (customTableCols || [])
      .filter(c => c.column_name !== 'id' && c.column_name !== 'created_at' && c.column_name !== 'certificate_id' && c.column_name !== 'file_name' && !existingKeys.has(c.column_name))
      .map(c => ({
        key: c.column_name,
        label: c.label || c.column_name.replace(/_/g, ' ').toUpperCase(),
        semantic_type: c.semantic_type || (c.column_name.toLowerCase().includes('email') ? 'email' : (c.column_name.toLowerCase().includes('url') || c.column_name.toLowerCase().includes('link') ? 'url' : (c.column_name.toLowerCase().includes('date') ? 'date' : 'text'))),
        defaultWidth: 180
      }));
    return [...baseCols, ...extraCols];
  }

  // Custom Dynamic Tables
  if (customTableCols.length > 0) {
    return customTableCols
      .filter(c => c.column_name !== 'id' && c.column_name !== 'created_at')
      .map(c => ({ 
        key: c.column_name, 
        label: c.label || c.column_name.replace(/_/g, ' ').toUpperCase(),
        semantic_type: c.semantic_type || (c.column_name.toLowerCase().includes('email') ? 'email' : (c.column_name.toLowerCase().includes('url') || c.column_name.toLowerCase().includes('link') ? 'url' : (c.column_name.toLowerCase().includes('date') ? 'date' : 'text'))),
        defaultWidth: 180 
      }));
  }
  return [{ key: 'name', label: 'Name', semantic_type: 'text', defaultWidth: 200 }];
}, [selectedTable, customTableCols]);

// Filtered & Sorted Grid Data
const filteredGridData = useMemo(() => {
  let result = gridData.map((item, originalIndex) => ({ item, originalIndex })).filter(({ item }) => {
    if (!item || typeof item !== 'object') return false;
    if (globalSearch.trim()) {
      const query = globalSearch.toLowerCase();
      const matches = Object.values(item).some(val =>
        val && String(val).toLowerCase().includes(query)
      );
      if (!matches) return false;
    }
    for (const row of filterRows) {
      if (row.column && row.value) {
        const itemVal = item[row.column];
        if (itemVal === undefined || itemVal === null) return false;
        if (!String(itemVal).toLowerCase().includes(row.value.toLowerCase())) {
          return false;
        }
      }
    }
    return true;
  });

  if (sortConfig) {
    result.sort((a, b) => {
      const valA = a.item[sortConfig.key] ?? '';
      const valB = b.item[sortConfig.key] ?? '';
      if (valA < valB) return sortConfig.direction === 'asc' ? -1 : 1;
      if (valA > valB) return sortConfig.direction === 'asc' ? 1 : -1;
      return 0;
    });
  }

  return result;
}, [gridData, globalSearch, filterRows, sortConfig]);

return (
  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', height: '100%' }}>

    {/* ────────────────────────────────────────────────────────────────────── */}
    {/* TOP ACTION TOOLBAR */}
    {/* ────────────────────────────────────────────────────────────────────── */}
    <div className="card" style={{ padding: '1rem 1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ color: 'var(--brand-blue)', margin: 0, fontSize: '1.35rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {selectedTable}
            {unsavedCount > 0 && (
              <span className="badge badge-yellow" style={{ fontSize: '0.75rem', fontWeight: 600 }}>
                {unsavedCount} Unsaved Change(s)
              </span>
            )}
          </h2>
          <p className="text-secondary" style={{ fontSize: '0.85rem', margin: 0 }}>
            Showing {gridData.length} records
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'flex-end', marginLeft: 'auto' }}>
          {/* Save & Cancel Buttons (Left of the button group) */}
          {isExcelGridTable && hasUnsavedChanges && (
            <>
              <button
                className="btn btn-primary"
                style={{ fontSize: '0.85rem', backgroundColor: '#10b981', borderColor: '#059669' }}
                onClick={() => handleSaveChangesToDatabase()}
                disabled={saving}
              >
                <Save size={16} /> {saving ? 'Saving...' : 'Save Changes to Database'}
              </button>
              <button
                className="btn btn-secondary"
                style={{ fontSize: '0.85rem', color: '#ef4444', borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.08)', cursor: 'pointer' }}
                onClick={handleCancelChanges}
                title="Cancel unsaved changes & stop active save/import"
              >
                <XCircle size={16} /> {saving ? 'Stop / Cancel' : 'Cancel'}
              </button>
            </>
          )}

          {/* Selection Action Buttons (Delete Selected, Export Selected, Clear Selection) */}
          {selectedRowIds.size > 0 && (
            <>
              <button
                className="btn btn-secondary"
                style={{ fontSize: '0.85rem', color: 'var(--error)', borderColor: 'var(--error)' }}
                onClick={handleDeleteSelectedRows}
                disabled={saving || deleting}
              >
                <Trash2 size={14} /> {deleting ? 'Deleting...' : `Delete Selected (${selectedRowIds.size})`}
              </button>

              <button
                className="btn btn-secondary"
                style={{ fontSize: '0.85rem' }}
                onClick={handleExportExcel}
                disabled={saving || deleting}
              >
                <Upload size={14} /> Export Selected ({selectedRowIds.size})
              </button>

              <button
                className="btn btn-secondary"
                style={{ fontSize: '0.85rem' }}
                onClick={() => setSelectedRowIds(new Set())}
                disabled={saving || deleting}
                title="Clear all selected rows"
              >
                <X size={14} /> Clear Selection
              </button>
            </>
          )}

          {/* 3-Dots Action Menu Dropdown (Add Row, Import, Export) */}
          <div ref={moreMenuRef} style={{ position: 'relative' }}>
            <button
              className="btn btn-secondary"
              style={{
                fontSize: '0.85rem',
                padding: '0.45rem 0.65rem',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: showMoreMenu ? 'rgba(36, 56, 129, 0.12)' : 'var(--bg-surface)',
                borderColor: showMoreMenu ? 'var(--brand-blue)' : 'var(--border-color)'
              }}
              onClick={() => setShowMoreMenu(prev => !prev)}
              title="Table Actions Menu"
            >
              <MoreVertical size={18} />
            </button>

            {showMoreMenu && (
              <div
                style={{
                  position: 'absolute',
                  top: '115%',
                  right: 0,
                  backgroundColor: 'var(--bg-surface)',
                  border: '1px solid var(--border-color)',
                  borderRadius: '8px',
                  boxShadow: '0 10px 25px rgba(0, 0, 0, 0.18)',
                  zIndex: 999,
                  minWidth: '200px',
                  padding: '0.35rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.25rem'
                }}
              >
                {isExcelGridTable && (
                  <button
                    className="btn btn-secondary"
                    style={{
                      border: 'none',
                      justifyContent: 'flex-start',
                      fontSize: '0.82rem',
                      padding: '0.45rem 0.75rem',
                      width: '100%',
                      backgroundColor: 'transparent'
                    }}
                    onClick={() => {
                      setShowMoreMenu(false);
                      handleAddRow();
                    }}
                  >
                    <Plus size={14} /> Add Row
                  </button>
                )}

                <button
                  className="btn btn-secondary"
                  style={{
                    border: 'none',
                    justifyContent: 'flex-start',
                    fontSize: '0.82rem',
                    padding: '0.45rem 0.75rem',
                    width: '100%',
                    backgroundColor: 'transparent'
                  }}
                  onClick={() => {
                    setShowMoreMenu(false);
                    setShowAddColumnModal(true);
                  }}
                >
                  <Plus size={14} /> Add Column
                </button>

                <button
                  className="btn btn-secondary"
                  style={{
                    border: 'none',
                    justifyContent: 'flex-start',
                    fontSize: '0.82rem',
                    padding: '0.45rem 0.75rem',
                    width: '100%',
                    backgroundColor: isWrapText ? 'rgba(36, 56, 129, 0.08)' : 'transparent',
                    color: isWrapText ? 'var(--brand-blue)' : 'inherit',
                    fontWeight: isWrapText ? 600 : 400
                  }}
                  onClick={() => {
                    setIsWrapText(prev => !prev);
                    setShowMoreMenu(false);
                  }}
                >
                  <WrapText size={14} /> {isWrapText ? 'Disable Wrap Text' : 'Wrap Text'}
                </button>

                <button
                  className="btn btn-secondary"
                  style={{
                    border: 'none',
                    justifyContent: 'flex-start',
                    fontSize: '0.82rem',
                    padding: '0.45rem 0.75rem',
                    width: '100%',
                    backgroundColor: 'transparent'
                  }}
                  onClick={() => {
                    setShowMoreMenu(false);
                    setShowImportDrawer(prev => !prev);
                  }}
                >
                  <Download size={14} /> Import Excel / CSV
                </button>

                <button
                  className="btn btn-secondary"
                  style={{
                    border: 'none',
                    justifyContent: 'flex-start',
                    fontSize: '0.82rem',
                    padding: '0.45rem 0.75rem',
                    width: '100%',
                    backgroundColor: 'transparent'
                  }}
                  disabled={gridData.length === 0}
                  onClick={() => {
                    setShowMoreMenu(false);
                    handleExportExcel();
                  }}
                >
                  <Upload size={14} /> {selectedRowIds.size > 0 ? `Export Selected (${selectedRowIds.size})` : 'Export Excel'}
                </button>

                {!isCoreTable && (
                  <>
                    <div style={{ height: '1px', backgroundColor: 'var(--border-color)', margin: '0.25rem 0' }} />
                    <button
                      className="btn btn-secondary"
                      style={{
                        border: 'none',
                        justifyContent: 'flex-start',
                        fontSize: '0.82rem',
                        padding: '0.45rem 0.75rem',
                        width: '100%',
                        backgroundColor: 'transparent',
                        color: 'var(--error)'
                      }}
                      onClick={() => {
                        setShowMoreMenu(false);
                        handleDropCustomTable();
                      }}
                    >
                      <Trash2 size={14} color="var(--error)" /> Drop Table
                    </button>
                  </>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* File Processing Banner with Spinning Loader Animation */}
      {fileProcessingStatus && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.6rem',
          padding: '0.6rem 0.85rem',
          borderRadius: '6px',
          fontSize: '0.85rem',
          backgroundColor: 'rgba(36, 56, 129, 0.08)',
          color: 'var(--brand-blue)',
          border: '1px solid var(--brand-blue)',
          fontWeight: 600
        }}>
          <Loader2 size={16} style={{ animation: 'spin 1s linear infinite' }} />
          <span>{fileProcessingStatus}</span>
        </div>
      )}

      {/* Save / Batch Operation Progress Bar */}
      {progress && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '0.45rem',
          padding: '0.75rem 1rem',
          borderRadius: '8px',
          backgroundColor: 'rgba(36, 56, 129, 0.05)',
          border: '1px solid var(--brand-blue)',
          width: '100%'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', fontWeight: 600, color: 'var(--brand-blue)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              {progress.percent < 100 && (
                <Loader2 size={16} className="spin-icon" />
              )}
              <span>{progress.statusText}</span>
            </div>
            <span>{progress.percent}%</span>
          </div>
          <div style={{ width: '100%', height: '8px', backgroundColor: 'rgba(148, 163, 184, 0.2)', borderRadius: '4px', overflow: 'hidden' }}>
            <div style={{
              width: `${progress.percent}%`,
              height: '100%',
              backgroundColor: '#10b981',
              borderRadius: '4px',
              transition: 'width 150ms ease-in-out'
            }} />
          </div>
        </div>
      )}

      {/* Status Toast Banner */}
      {saveStatus && (
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '0.5rem',
          padding: '0.6rem 0.85rem',
          borderRadius: '6px',
          fontSize: '0.85rem',
          backgroundColor: saveStatus.type === 'success' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
          color: saveStatus.type === 'success' ? '#065f46' : '#991b1b',
          border: `1px solid ${saveStatus.type === 'success' ? '#a7f3d0' : '#fecaca'}`
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            {saveStatus.type === 'success' ? <CheckCircle2 size={16} /> : <AlertCircle size={16} />}
            <span>{saveStatus.message}</span>
          </div>
          <X size={14} style={{ cursor: 'pointer', opacity: 0.7, flexShrink: 0 }} onClick={() => setSaveStatus(null)} />
        </div>
      )}

      {/* Import File Drawer */}
      {showImportDrawer && (
        <div style={{ backgroundColor: 'var(--bg-body)', padding: '1rem', borderRadius: '8px', border: '1.5px dashed var(--brand-blue)', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--brand-blue)' }}>
              IMPORT SPREADSHEETS (.XLSX / .CSV) INTO: <span style={{ textDecoration: 'underline' }}>{selectedTable.toUpperCase()}</span>
            </span>
            <button
              type="button"
              className="btn btn-secondary"
              style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
              onClick={() => setShowImportDrawer(false)}
            >
              Close
            </button>
          </div>
          <input
            type="file"
            multiple
            accept=".csv, .xlsx, .xls"
            onChange={handleFileUpload}
            style={{ fontSize: '0.85rem' }}
          />

          {/* Alert Banner for Duplicate Statistics */}
          {lastImportStats && lastImportStats.skippedCount > 0 && (
            <div style={{
              backgroundColor: lastImportStats.newCount === 0 ? 'rgba(239, 68, 68, 0.08)' : 'rgba(234, 179, 8, 0.1)',
              border: `1px solid ${lastImportStats.newCount === 0 ? '#ef4444' : '#eab308'}`,
              color: lastImportStats.newCount === 0 ? '#b91c1c' : '#a16207',
              padding: '0.55rem 0.85rem',
              borderRadius: '6px',
              fontSize: '0.82rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '0.5rem',
              marginTop: '0.25rem'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <AlertCircle size={16} />
                <span>
                  {lastImportStats.newCount === 0
                    ? `All ${lastImportStats.total} row(s) in "${lastImportStats.fileName}" already exist in PostgreSQL and were skipped (0 new rows queued).`
                    : `Duplicate Notice: ${lastImportStats.skippedCount} duplicate row(s) out of ${lastImportStats.total} in "${lastImportStats.fileName}" were automatically skipped.`}
                </span>
              </div>
              <X
                size={14}
                style={{ cursor: 'pointer', opacity: 0.7 }}
                onClick={() => setLastImportStats(null)}
              />
            </div>
          )}

          {/* Queued Imported File Badges with X Remove Icon */}
          {importedFiles.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.25rem', paddingTop: '0.5rem', borderTop: '1px dashed var(--border-color)' }}>
              <span style={{ fontSize: '0.78rem', fontWeight: 700, color: 'var(--brand-blue)' }}>
                QUEUED FILES ({importedFiles.length}):
              </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                {importedFiles.map(f => (
                  <span
                    key={f.id}
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '0.45rem',
                      padding: '0.35rem 0.65rem',
                      borderRadius: '16px',
                      backgroundColor: 'rgba(36, 56, 129, 0.08)',
                      border: '1px solid var(--brand-blue)',
                      color: 'var(--brand-blue)',
                      fontSize: '0.8rem',
                      fontWeight: 600
                    }}
                  >
                    <FileText size={14} color="var(--brand-blue)" />
                    <span>{f.name} ({f.count} rows)</span>
                    <X
                      size={14}
                      style={{ cursor: 'pointer', borderRadius: '50%', padding: '1px', backgroundColor: 'rgba(239, 68, 68, 0.15)', color: '#ef4444' }}
                      onClick={() => handleRemoveImportedFile(f.id)}
                    />
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Snippet Preview Box before clicking save */}
          {pendingImportRows.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', marginTop: '0.5rem', paddingTop: '0.5rem', borderTop: '1px dashed var(--border-color)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.78rem', fontWeight: 700, color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                  SNIPPET PREVIEW (FIRST 5 QUEUED RECORDS BEFORE SAVING):
                </span>
                <span style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>
                  Total Queued: {pendingImportRows.length} rows
                </span>
              </div>
              <div style={{ overflowX: 'auto', border: '1px solid var(--border-color)', borderRadius: '6px', backgroundColor: 'var(--bg-surface)' }}>
                <table style={{ width: '100%', fontSize: '0.78rem', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr style={{ backgroundColor: 'rgba(36, 56, 129, 0.05)', color: 'var(--brand-blue)', textAlign: 'left' }}>
                      {columnsMeta.map(col => (
                        <th key={col.key} style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>{col.label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {pendingImportRows.slice(0, 5).map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        {columnsMeta.map(col => {
                          const val = row[col.key] || row[col.label] || row[col.key.replace(/_/g, ' ')] || row[col.key.replace(/ /g, '_')] || '—';
                          const isLink = col.key.toLowerCase().includes('link') || col.key.toLowerCase().includes('url');
                          
                          if (isLink && val !== '—') {
                            return (
                              <td key={col.key} style={{ padding: '0.35rem 0.6rem', maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                <a href={String(val)} target="_blank" rel="noreferrer" style={{ color: 'var(--brand-blue)', textDecoration: 'underline' }}>{String(val)}</a>
                              </td>
                            );
                          }
                          
                          return (
                            <td key={col.key} style={{ padding: '0.35rem 0.6rem' }}>
                              {String(val)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.6rem', marginTop: '0.35rem' }}>
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', color: '#ef4444', borderColor: '#ef4444', backgroundColor: 'rgba(239, 68, 68, 0.06)' }}
                  onClick={handleCancelChanges}
                  disabled={saving}
                >
                  <XCircle size={14} /> Discard Queue
                </button>
                <button
                  type="button"
                  className="btn btn-primary"
                  style={{ fontSize: '0.8rem', backgroundColor: '#10b981', borderColor: '#059669' }}
                  onClick={() => handleSaveChangesToDatabase(undefined, pendingImportUsePipeline)}
                  disabled={saving}
                >
                  <Save size={14} /> {saving ? 'Importing...' : `Confirm & Import ${pendingImportRows.length} Row(s) to Database`}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Search & Filter Controls */}
      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
        <div style={{ position: 'relative', flex: 1 }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: 10, color: 'var(--text-tertiary)' }} />
          <input
            type="text"
            placeholder={`Quick search across all ${selectedTable} records...`}
            className="input"
            style={{ paddingLeft: '2rem', fontSize: '0.85rem' }}
            value={globalSearch}
            onChange={e => setGlobalSearch(e.target.value)}
          />
        </div>

        <button
          className="btn btn-secondary"
          style={{ fontSize: '0.8rem' }}
          onClick={() => setFilterRows([...filterRows, { id: String(Date.now()), column: columnsMeta[0]?.key || '', value: '' }])}
        >
          Filter
        </button>
      </div>

      {/* Active Filter Rows */}
      {filterRows.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', backgroundColor: 'var(--bg-body)', padding: '0.6rem', borderRadius: '6px' }}>
          {filterRows.map((fRow, idx) => {
            const activeColKey = fRow.column || columnsMeta[0]?.key || '';
            const uniqueSuggestions = Array.from(
              new Set(
                gridData
                  .map(item => item[activeColKey])
                  .filter(val => val !== undefined && val !== null && String(val).trim() !== '')
              )
            ).slice(0, 50);

            const datalistId = `datalist-filter-${fRow.id}`;

            return (
              <div key={fRow.id} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                {/* Column Select Dropdown */}
                <select
                  className="input"
                  style={{ width: '180px', padding: '0.25rem 0.5rem', fontSize: '0.8rem', cursor: 'pointer' }}
                  value={activeColKey}
                  onChange={e => {
                    const updated = [...filterRows];
                    updated[idx].column = e.target.value;
                    updated[idx].value = '';
                    setFilterRows(updated);
                  }}
                >
                  {columnsMeta.map(col => (
                    <option key={col.key} value={col.key}>
                      {col.label}
                    </option>
                  ))}
                </select>

                {/* Autocomplete Datalist Suggestions */}
                <datalist id={datalistId}>
                  {uniqueSuggestions.map((sugVal: any, sIdx) => (
                    <option key={sIdx} value={String(sugVal)} />
                  ))}
                </datalist>

                {/* Query Text Box */}
                <input
                  type="text"
                  list={datalistId}
                  placeholder={`Filter or select ${columnsMeta.find(c => c.key === activeColKey)?.label || activeColKey}...`}
                  className="input"
                  style={{ flex: 1, padding: '0.25rem 0.5rem', fontSize: '0.8rem' }}
                  value={fRow.value}
                  onChange={e => {
                    const updated = [...filterRows];
                    updated[idx].value = e.target.value;
                    setFilterRows(updated);
                  }}
                />
                <button
                  className="btn btn-secondary"
                  style={{ padding: '0.25rem 0.4rem', color: 'var(--error)' }}
                  onClick={() => setFilterRows(filterRows.filter((_, i) => i !== idx))}
                  title="Remove filter"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            );
          })}
        </div>
      )}

    </div>

    {/* ────────────────────────────────────────────────────────────────────── */}
    {/* DATA GRID DISPLAY (INTERACTIVE SPREADSHEET WITH EDIT & DELETE ON ALL TABLES) */}
    {/* ────────────────────────────────────────────────────────────────────── */}
    <div className="card" style={{ flex: 1, minHeight: '500px', overflow: 'hidden', padding: 0, display: 'flex', flexDirection: 'column' }}>
      <div style={{ flex: 1, overflow: 'auto', maxHeight: 'calc(100vh - 230px)', userSelect: isMouseDownSelect ? 'none' : 'auto' }}>
        {loading ? (
          <p className="text-secondary" style={{ padding: '1.5rem' }}>Loading database data...</p>
        ) : (
          <table style={{ width: 'max-content', minWidth: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', tableLayout: 'fixed' }}>
            <thead>
              <tr style={{ backgroundColor: 'var(--bg-body)', borderBottom: '2px solid var(--border-color)', position: 'sticky', top: 0, zIndex: 2 }}>
                <th style={{ width: '45px', padding: '0.65rem 0.5rem', textAlign: 'center', borderRight: '1px solid var(--border-color)' }}>
                  <input
                    type="checkbox"
                    style={{ cursor: 'pointer', accentColor: 'var(--brand-blue)' }}
                    checked={selectedRowIds.size === gridData.length && gridData.length > 0}
                    onChange={toggleSelectAll}
                    title="Select All Rows"
                  />
                </th>
                {columnsMeta.map(col => {
                  const currentWidth = colWidths[col.key] || col.defaultWidth || 180;
                  const isSorted = sortConfig?.key === col.key;

                  return (
                    <th
                      key={col.key}
                      style={{
                        width: `${currentWidth}px`,
                        padding: '0.65rem 0.75rem',
                        color: isSorted ? '#000' : 'var(--brand-blue)',
                        fontWeight: 600,
                        borderRight: '1px solid var(--border-color)',
                        position: 'relative',
                        userSelect: 'none',
                        cursor: 'pointer'
                      }}
                      onClick={() => handleSort(col.key)}
                      title={`Click to sort by ${col.label}`}
                    >
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '0.4rem' }}>
                        <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                          {col.label}
                        </span>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', flexShrink: 0 }}>
                          {/* Sort Indicator Arrow */}
                          {sortConfig && sortConfig.key === col.key ? (
                            sortConfig.direction === 'asc' ? <ArrowUp size={14} color="var(--brand-blue)" /> : <ArrowDown size={14} color="var(--brand-blue)" />
                          ) : (
                            <ArrowUpDown size={12} color="var(--text-tertiary)" style={{ opacity: 0.5 }} />
                          )}

                          {/* Delete Column Button */}
                          {isColumnDeletable(col.key) && (
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setColumnToDelete({ key: col.key, label: col.label });
                              }}
                              title={`Delete column "${col.label}"`}
                              style={{
                                background: 'transparent',
                                border: 'none',
                                cursor: 'pointer',
                                padding: '2px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                color: 'var(--text-tertiary)',
                                opacity: 0.6,
                                borderRadius: '4px',
                                transition: 'all 0.15s ease'
                              }}
                              onMouseEnter={(e) => {
                                e.currentTarget.style.color = '#dc2626';
                                e.currentTarget.style.opacity = '1';
                                e.currentTarget.style.backgroundColor = 'rgba(220, 38, 38, 0.1)';
                              }}
                              onMouseLeave={(e) => {
                                e.currentTarget.style.color = 'var(--text-tertiary)';
                                e.currentTarget.style.opacity = '0.6';
                                e.currentTarget.style.backgroundColor = 'transparent';
                              }}
                            >
                              <Trash2 size={12} />
                            </button>
                          )}
                        </div>
                      </div>

                      {/* Column Resizer Handle */}
                      <div
                        onMouseDown={(e) => handleStartResize(e, col.key, currentWidth)}
                        onDoubleClick={(e) => {
                          e.stopPropagation();
                          handleAutoFitColumn(col.key, col.label);
                        }}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          position: 'absolute',
                          right: 0,
                          top: 0,
                          bottom: 0,
                          width: '8px',
                          cursor: 'col-resize',
                          backgroundColor: resizingCol?.key === col.key ? 'var(--brand-blue)' : 'transparent',
                          zIndex: 3
                        }}
                        title="Drag to resize, double-click to auto-fit column"
                      />
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {filteredGridData.map(({ item, originalIndex }) => {
                const isDirty = dirtyRowIndices.has(originalIndex);
                const isSelected = selectedRowIds.has(originalIndex);

                return (
                  <tr
                    key={originalIndex}
                    style={{
                      borderBottom: '1px solid var(--border-color)',
                      backgroundColor: isSelected ? 'rgba(36, 56, 129, 0.12)' : isDirty ? 'rgba(251, 191, 36, 0.08)' : 'transparent',
                      transition: 'background-color 150ms ease'
                    }}
                    onMouseEnter={(e) => {
                      if (!isSelected && !isDirty) e.currentTarget.style.backgroundColor = 'rgba(36, 56, 129, 0.03)';
                    }}
                    onMouseLeave={(e) => {
                      if (!isSelected && !isDirty) e.currentTarget.style.backgroundColor = 'transparent';
                    }}
                  >
                    {/* Drag-to-Select Checkbox Cell */}
                    <td
                      style={{
                        textAlign: 'center',
                        padding: '0.4rem 0.5rem',
                        borderRight: '1px solid var(--border-color)',
                        backgroundColor: isSelected ? 'rgba(36, 56, 129, 0.08)' : 'transparent',
                        cursor: 'pointer',
                        userSelect: 'none'
                      }}
                      onMouseDown={(e) => handleCheckboxMouseDown(e, originalIndex)}
                      onMouseEnter={() => handleCheckboxMouseEnter(originalIndex)}
                    >
                      <input
                        type="checkbox"
                        style={{ cursor: 'pointer', accentColor: 'var(--brand-blue)', pointerEvents: 'none' }}
                        checked={isSelected}
                        readOnly
                      />
                    </td>

                    {/* Cell Content: Deterministically rendered based on AI Schema semantic_type */}
                    {columnsMeta.map(col => {
                      const stype = (col as any).semantic_type || 'text';
                      const cellVal = item[col.key] ?? '';
                      const rawStr = typeof cellVal === 'string' ? cellVal.trim() : String(cellVal || '').trim();
                      const cleanEmail = rawStr.replace(/#mailto:[^#]*#?/i, '').replace(/^mailto:/i, '').trim();
                      const isEditingThisCell = editingCell?.rowIdx === originalIndex && editingCell?.colKey === col.key;

                      if (stype === 'email') {
                        if (isEditingThisCell || !cleanEmail) {
                          return (
                            <td
                              key={col.key}
                              style={{
                                padding: 0,
                                borderRight: '1px solid var(--border-color)',
                                verticalAlign: isWrapText ? 'top' : 'middle'
                              }}
                            >
                              {isWrapText ? (
                                <textarea
                                  className="excel-cell-input"
                                  value={cellVal}
                                  title={String(cellVal)}
                                  rows={Math.min(6, Math.max(1, Math.ceil(String(cellVal).length / 25)))}
                                  autoFocus={isEditingThisCell}
                                  onFocus={() => setEditingCell({ rowIdx: originalIndex, colKey: col.key })}
                                  onBlur={() => setEditingCell(null)}
                                  onChange={e => handleCellChange(originalIndex, col.key, e.target.value)}
                                  style={{
                                    width: '100%',
                                    padding: '0.55rem 0.75rem',
                                    border: 'none',
                                    outline: 'none',
                                    background: 'transparent',
                                    color: 'var(--text-primary)',
                                    fontFamily: 'inherit',
                                    fontSize: '0.85rem',
                                    resize: 'none',
                                    whiteSpace: 'normal',
                                    wordBreak: 'break-word',
                                    display: 'block'
                                  }}
                                />
                              ) : (
                                <input
                                  type="text"
                                  className="excel-cell-input"
                                  value={cellVal}
                                  title={String(cellVal)}
                                  autoFocus={isEditingThisCell}
                                  onFocus={() => setEditingCell({ rowIdx: originalIndex, colKey: col.key })}
                                  onBlur={() => setEditingCell(null)}
                                  onChange={e => handleCellChange(originalIndex, col.key, e.target.value)}
                                  onKeyDown={e => {
                                    if (e.key === 'Enter' || e.key === 'Escape') setEditingCell(null);
                                  }}
                                  style={{
                                    width: '100%',
                                    height: '100%',
                                    padding: '0.55rem 0.75rem',
                                    border: 'none',
                                    outline: 'none',
                                    background: 'transparent',
                                    color: 'var(--text-primary)',
                                    fontFamily: 'inherit',
                                    fontSize: '0.85rem'
                                  }}
                                />
                              )}
                            </td>
                          );
                        }

                        return (
                          <td
                            key={col.key}
                            style={{
                              padding: '0.55rem 0.75rem',
                              borderRight: '1px solid var(--border-color)',
                              overflow: 'hidden',
                              textOverflow: isWrapText ? 'clip' : 'ellipsis',
                              whiteSpace: isWrapText ? 'normal' : 'nowrap',
                              wordBreak: isWrapText ? 'break-word' : 'normal',
                              verticalAlign: isWrapText ? 'top' : 'middle',
                              cursor: 'pointer'
                            }}
                            title={`${cleanEmail} (Double-click to edit)`}
                            onDoubleClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setEditingCell({ rowIdx: originalIndex, colKey: col.key });
                            }}
                            onClick={(e) => {
                              if (e.target === e.currentTarget) {
                                setEditingCell({ rowIdx: originalIndex, colKey: col.key });
                              }
                            }}
                          >
                            <a
                              href={`mailto:${cleanEmail}`}
                              title={`${cleanEmail} (Double-click to edit)`}
                              onDoubleClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                setEditingCell({ rowIdx: originalIndex, colKey: col.key });
                              }}
                              style={{
                                color: 'var(--brand-blue)',
                                fontWeight: 500,
                                display: isWrapText ? 'inline' : 'inline-block',
                                textDecoration: 'underline',
                                maxWidth: '100%',
                                overflow: 'hidden',
                                textOverflow: isWrapText ? 'clip' : 'ellipsis',
                                whiteSpace: isWrapText ? 'normal' : 'nowrap',
                                wordBreak: isWrapText ? 'break-word' : 'normal'
                              }}
                            >
                              {cleanEmail}
                            </a>
                          </td>
                        );
                      }

                      if (stype === 'url') {
                        if (isEditingThisCell || !rawStr) {
                          return (
                            <td
                              key={col.key}
                              style={{
                                padding: 0,
                                borderRight: '1px solid var(--border-color)',
                                verticalAlign: isWrapText ? 'top' : 'middle'
                              }}
                            >
                              {isWrapText ? (
                                <textarea
                                  className="excel-cell-input"
                                  value={cellVal}
                                  title={String(cellVal)}
                                  rows={Math.min(6, Math.max(1, Math.ceil(String(cellVal).length / 25)))}
                                  autoFocus={isEditingThisCell}
                                  onFocus={() => setEditingCell({ rowIdx: originalIndex, colKey: col.key })}
                                  onBlur={() => setEditingCell(null)}
                                  onChange={e => handleCellChange(originalIndex, col.key, e.target.value)}
                                  style={{
                                    width: '100%',
                                    padding: '0.55rem 0.75rem',
                                    border: 'none',
                                    outline: 'none',
                                    background: 'transparent',
                                    color: 'var(--text-primary)',
                                    fontFamily: 'inherit',
                                    fontSize: '0.85rem',
                                    resize: 'none',
                                    whiteSpace: 'normal',
                                    wordBreak: 'break-word',
                                    display: 'block'
                                  }}
                                />
                              ) : (
                                <input
                                  type="text"
                                  className="excel-cell-input"
                                  value={cellVal}
                                  title={String(cellVal)}
                                  autoFocus={isEditingThisCell}
                                  onFocus={() => setEditingCell({ rowIdx: originalIndex, colKey: col.key })}
                                  onBlur={() => setEditingCell(null)}
                                  onChange={e => handleCellChange(originalIndex, col.key, e.target.value)}
                                  onKeyDown={e => {
                                    if (e.key === 'Enter' || e.key === 'Escape') setEditingCell(null);
                                  }}
                                  style={{
                                    width: '100%',
                                    height: '100%',
                                    padding: '0.55rem 0.75rem',
                                    border: 'none',
                                    outline: 'none',
                                    background: 'transparent',
                                    color: 'var(--text-primary)',
                                    fontFamily: 'inherit',
                                    fontSize: '0.85rem'
                                  }}
                                />
                              )}
                            </td>
                          );
                        }

                        const href = rawStr.startsWith('http://') || rawStr.startsWith('https://') ? rawStr : `https://${rawStr}`;
                        const displayLabel = selectedTable === "RF Certificates" && col.key === 'cert_link' ? "Open Link" : rawStr;

                        return (
                          <td
                            key={col.key}
                            style={{
                              padding: '0.55rem 0.75rem',
                              borderRight: '1px solid var(--border-color)',
                              overflow: 'hidden',
                              textOverflow: isWrapText ? 'clip' : 'ellipsis',
                              whiteSpace: isWrapText ? 'normal' : 'nowrap',
                              wordBreak: isWrapText ? 'break-word' : 'normal',
                              verticalAlign: isWrapText ? 'top' : 'middle',
                              cursor: 'pointer'
                            }}
                            title={`${rawStr} (Double-click to edit)`}
                            onDoubleClick={(e) => {
                              e.preventDefault();
                              e.stopPropagation();
                              setEditingCell({ rowIdx: originalIndex, colKey: col.key });
                            }}
                            onClick={(e) => {
                              if (e.target === e.currentTarget) {
                                setEditingCell({ rowIdx: originalIndex, colKey: col.key });
                              }
                            }}
                          >
                            <a
                              href={href}
                              target="_blank"
                              rel="noreferrer"
                              title={`${rawStr} (Double-click to edit)`}
                              onDoubleClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                setEditingCell({ rowIdx: originalIndex, colKey: col.key });
                              }}
                              style={{
                                color: 'var(--brand-blue)',
                                fontWeight: 500,
                                display: isWrapText ? 'inline' : 'inline-block',
                                textDecoration: 'underline',
                                maxWidth: '100%',
                                overflow: 'hidden',
                                textOverflow: isWrapText ? 'clip' : 'ellipsis',
                                whiteSpace: isWrapText ? 'normal' : 'nowrap',
                                wordBreak: isWrapText ? 'break-word' : 'normal'
                              }}
                            >
                              {displayLabel}
                            </a>
                          </td>
                        );
                      }

                      return (
                        <td
                          key={col.key}
                          style={{
                            padding: 0,
                            borderRight: '1px solid var(--border-color)',
                            verticalAlign: isWrapText ? 'top' : 'middle'
                          }}
                        >
                          {isWrapText ? (
                            <textarea
                              className="excel-cell-input"
                              value={cellVal}
                              title={String(cellVal)}
                              rows={Math.min(6, Math.max(1, Math.ceil(String(cellVal).length / 25)))}
                              onChange={e => handleCellChange(originalIndex, col.key, e.target.value)}
                              style={{
                                width: '100%',
                                padding: '0.55rem 0.75rem',
                                border: 'none',
                                outline: 'none',
                                background: 'transparent',
                                color: 'var(--text-primary)',
                                fontFamily: 'inherit',
                                fontSize: '0.85rem',
                                resize: 'none',
                                whiteSpace: 'normal',
                                wordBreak: 'break-word',
                                display: 'block'
                              }}
                            />
                          ) : (
                            <input
                              type="text"
                              className="excel-cell-input"
                              value={cellVal}
                              title={String(cellVal)}
                              onChange={e => handleCellChange(originalIndex, col.key, e.target.value)}
                              style={{
                                width: '100%',
                                height: '100%',
                                padding: '0.55rem 0.75rem',
                                border: 'none',
                                outline: 'none',
                                background: 'transparent',
                                color: 'var(--text-primary)',
                                fontFamily: 'inherit',
                                fontSize: '0.85rem'
                              }}
                            />
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}

              {filteredGridData.length === 0 && (
                <tr>
                  <td colSpan={columnsMeta.length + 1} style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-tertiary)' }}>
                    No records found in {selectedTable}.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>

    {/* ────────────────────────────────────────────────────────────────────── */}
    {/* MIDDLE PHASE: IMPORT MODE SELECTION MODAL (FAST AS-IS VS AI PIPELINE) */}
    {/* ────────────────────────────────────────────────────────────────────── */}
    {pendingImportFiles && (
      <div
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(15, 23, 42, 0.45)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000
        }}
      >
        <div
          className="card"
          style={{
            width: '540px',
            maxWidth: '92vw',
            padding: '1.75rem',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.25rem',
            backgroundColor: 'var(--bg-surface)',
            boxShadow: 'var(--shadow-lg)'
          }}
          onClick={e => e.stopPropagation()}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <h3 style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                Import into <span style={{ color: 'var(--brand-blue)' }}>{selectedTable}</span>
              </h3>
              <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                {pendingImportFiles.length === 1
                  ? `File: "${pendingImportFiles[0].name}"`
                  : `${pendingImportFiles.length} files queued`} — Choose your import mode:
              </p>
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.25rem', border: 'none' }}
              onClick={() => setPendingImportFiles(null)}
            >
              <X size={18} />
            </button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
            {/* Option 1: Fast As-Is Import */}
            <div
              onClick={() => executeImport(pendingImportFiles, false)}
              style={{
                border: '1.5px solid var(--border-color)',
                borderRadius: '10px',
                padding: '1rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease-in-out',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.85rem',
                backgroundColor: 'var(--bg-body)'
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = 'var(--brand-blue)';
                e.currentTarget.style.backgroundColor = 'rgba(36, 56, 129, 0.04)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'var(--border-color)';
                e.currentTarget.style.backgroundColor = 'var(--bg-body)';
              }}
            >
              <div style={{ padding: '0.5rem', borderRadius: '8px', backgroundColor: 'rgba(16, 185, 129, 0.12)', color: 'var(--success)', marginTop: '0.1rem' }}>
                <Zap size={20} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                    Import As-Is (Fast)
                  </span>
                  <span className="badge badge-green" style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem' }}>
                    Instant
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.25rem', lineHeight: 1.4 }}>
                  Imports records directly with exact raw cell values without AI processing. Instant and preserves exact source spreadsheet text.
                </p>
              </div>
            </div>

            {/* Option 2: AI Standardization Pipeline */}
            <div
              onClick={() => executeImport(pendingImportFiles, true)}
              style={{
                border: '1.5px solid var(--border-color)',
                borderRadius: '10px',
                padding: '1rem',
                cursor: 'pointer',
                transition: 'all 0.15s ease-in-out',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.85rem',
                backgroundColor: 'var(--bg-body)'
              }}
              onMouseEnter={e => {
                e.currentTarget.style.borderColor = 'var(--brand-blue)';
                e.currentTarget.style.backgroundColor = 'rgba(36, 56, 129, 0.04)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.borderColor = 'var(--border-color)';
                e.currentTarget.style.backgroundColor = 'var(--bg-body)';
              }}
            >
              <div style={{ padding: '0.5rem', borderRadius: '8px', backgroundColor: 'rgba(36, 56, 129, 0.1)', color: 'var(--brand-blue)', marginTop: '0.1rem' }}>
                <Sparkles size={20} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                    AI Standardization Pipeline
                  </span>
                  <span className="badge badge-blue" style={{ fontSize: '0.7rem', padding: '0.15rem 0.45rem' }}>
                    Recommended
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.25rem', lineHeight: 1.4 }}>
                  Passes data through semantic header mapping and AI Diffing to normalize dates (ISO 8601), phone numbers (E.164), official country titles, and rescue misplaced cells.
                </p>
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.25rem' }}>
            <button
              className="btn btn-secondary"
              onClick={() => setPendingImportFiles(null)}
              style={{ fontSize: '0.8rem' }}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    )}

    {/* Add Custom Column Modal */}
    {showAddColumnModal && (
      <div
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.55)',
          backdropFilter: 'blur(3px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1100,
          padding: '1rem'
        }}
        onClick={() => {
          if (!addingColumn) setShowAddColumnModal(false);
        }}
      >
        <div
          style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            width: '100%',
            maxWidth: '460px',
            padding: '1.4rem',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.25)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.1rem'
          }}
          onClick={e => e.stopPropagation()}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <div style={{ padding: '0.45rem', borderRadius: '8px', backgroundColor: 'rgba(36, 56, 129, 0.1)', color: 'var(--brand-blue)' }}>
                <Plus size={18} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Add Column
                </h3>
                <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  Add a new field to table "{selectedTable}"
                </p>
              </div>
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.35rem', border: 'none', backgroundColor: 'transparent' }}
              onClick={() => setShowAddColumnModal(false)}
              disabled={addingColumn}
            >
              <X size={16} />
            </button>
          </div>

          <form onSubmit={handleAddColumnSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem' }}>
            <div>
              <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                Column Identifier / Name *
              </label>
              <input
                type="text"
                required
                placeholder="e.g. lead_engineer, datasheet_link, priority"
                value={newColName}
                onChange={e => setNewColName(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-body)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                  outline: 'none'
                }}
                autoFocus
              />
            </div>

            <div>
              <label style={{ display: 'block', fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.35rem' }}>
                Semantic Type / Behavior
              </label>
              <select
                value={newColSemanticType}
                onChange={e => setNewColSemanticType(e.target.value)}
                style={{
                  width: '100%',
                  padding: '0.55rem 0.75rem',
                  borderRadius: '8px',
                  border: '1px solid var(--border-color)',
                  backgroundColor: 'var(--bg-body)',
                  color: 'var(--text-primary)',
                  fontSize: '0.85rem',
                  outline: 'none',
                  cursor: 'pointer'
                }}
              >
                <option value="auto">Auto / AI Inferred (Recommended)</option>
                <option value="text">Text (General string / description)</option>
                <option value="email">Email (Clickable mailto: link)</option>
                <option value="url">URL / Document Link (Clickable external link)</option>
                <option value="date">Date (ISO 8601 formatting)</option>
                <option value="phone">Phone Number (E.164 standardized)</option>
                <option value="supplier">Supplier / Manufacturer</option>
                <option value="number">Number (Integer / Decimal metric)</option>
                <option value="badge">Status / Badge (Category tag)</option>
              </select>
            </div>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.6rem', marginTop: '0.4rem' }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setShowAddColumnModal(false)}
                disabled={addingColumn}
                style={{ fontSize: '0.82rem' }}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="btn btn-primary"
                disabled={addingColumn || !newColName.trim()}
                style={{ fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}
              >
                {addingColumn ? (
                  <>
                    <Loader2 size={14} className="spin" /> Adding...
                  </>
                ) : (
                  <>
                    <Plus size={14} /> Add Column
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    )}

    {/* Delete Custom Column Confirmation Modal */}
    {columnToDelete && (
      <div
        style={{
          position: 'fixed',
          inset: 0,
          backgroundColor: 'rgba(0, 0, 0, 0.55)',
          backdropFilter: 'blur(3px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1200,
          padding: '1rem'
        }}
        onClick={() => {
          if (!deletingColumn) setColumnToDelete(null);
        }}
      >
        <div
          style={{
            backgroundColor: 'var(--bg-surface)',
            border: '1px solid var(--border-color)',
            borderRadius: '12px',
            width: '100%',
            maxWidth: '440px',
            padding: '1.4rem',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.25)',
            display: 'flex',
            flexDirection: 'column',
            gap: '1.1rem'
          }}
          onClick={e => e.stopPropagation()}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <div style={{ padding: '0.45rem', borderRadius: '8px', backgroundColor: 'rgba(220, 38, 38, 0.1)', color: '#dc2626' }}>
                <Trash2 size={18} />
              </div>
              <div>
                <h3 style={{ margin: 0, fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
                  Delete Column
                </h3>
                <p style={{ margin: 0, fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  Table: "{selectedTable}"
                </p>
              </div>
            </div>
            <button
              className="btn btn-secondary"
              style={{ padding: '0.35rem', border: 'none', backgroundColor: 'transparent' }}
              onClick={() => setColumnToDelete(null)}
              disabled={deletingColumn}
            >
              <X size={16} />
            </button>
          </div>

          <p style={{ margin: 0, fontSize: '0.85rem', color: 'var(--text-primary)', lineHeight: 1.5 }}>
            Are you sure you want to permanently delete the column <strong>"{columnToDelete.label}"</strong> (<code>{columnToDelete.key}</code>)?
          </p>
          <div style={{ padding: '0.65rem 0.8rem', borderRadius: '8px', backgroundColor: 'rgba(220, 38, 38, 0.08)', border: '1px solid rgba(220, 38, 38, 0.2)', fontSize: '0.78rem', color: '#dc2626', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertCircle size={15} style={{ flexShrink: 0 }} />
            <span>This action cannot be undone. All stored data in this column will be permanently dropped from the database.</span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.6rem', marginTop: '0.3rem' }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => setColumnToDelete(null)}
              disabled={deletingColumn}
              style={{ fontSize: '0.82rem' }}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn"
              onClick={handleDeleteColumnConfirm}
              disabled={deletingColumn}
              style={{
                fontSize: '0.82rem',
                backgroundColor: '#dc2626',
                color: '#fff',
                border: 'none',
                display: 'flex',
                alignItems: 'center',
                gap: '0.4rem',
                cursor: deletingColumn ? 'not-allowed' : 'pointer'
              }}
            >
              {deletingColumn ? (
                <>
                  <Loader2 size={14} className="spin" /> Deleting...
                </>
              ) : (
                <>
                  <Trash2 size={14} /> Delete Column
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    )}

  </div>
);
};

export default DatabasesView;
