import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { Search, Download, Plus, Trash2, Save, Upload, CheckCircle2, AlertCircle, ExternalLink, XCircle, ArrowUpDown, ArrowUp, ArrowDown, X, FileText, RotateCcw, Loader2 } from 'lucide-react';
import * as XLSX from 'xlsx';
import { api } from '../api';
import { useLayoutContext } from '../components/Layout/AppLayout';

type FilterRow = {
  id: string;
  column: string;
  value: string;
};

type RecycleItem = {
  id: string;
  title: string;
  tableName: string;
  data: any;
  deletedAt: string;
};

const DatabasesView = () => {
  const { selectedTable } = useLayoutContext();

  // All tables (including RF Certificates) have full editing & deletion capabilities
  const isExcelGridTable = selectedTable !== "None";

  // Data Grid State
  const [gridData, setGridData] = useState<any[]>([]);
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

  // Loading, Saving, Deleting & Progress Status
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [restoringId, setRestoringId] = useState<string | null>(null);
  const [saveStatus, setSaveStatus] = useState<{ message: string; type: 'success' | 'error' } | null>(null);
  const [progress, setProgress] = useState<{ current: number; total: number; percent: number; statusText: string } | null>(null);

  // Search & Filter state
  const [globalSearch, setGlobalSearch] = useState('');
  const [filterRows, setFilterRows] = useState<FilterRow[]>([]);

  // Import Drawer & Multi-file Queue state
  const [showImportDrawer, setShowImportDrawer] = useState(false);
  const [importedFiles, setImportedFiles] = useState<{ id: string; name: string; count: number }[]>([]);
  const [pendingImportRows, setPendingImportRows] = useState<any[]>([]);
  const [fileProcessingStatus, setFileProcessingStatus] = useState<string | null>(null);

  // Recycle Bin State (persisted in localStorage)
  const [showRecycleDrawer, setShowRecycleDrawer] = useState(false);
  const [recycleBin, setRecycleBin] = useState<RecycleItem[]>(() => {
    try {
      const stored = localStorage.getItem('rf_compliance_recycle_bin');
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });

  // Fetch Table Data based on selected table (resets drawer & filters on table change, restores unsaved draft session)
  const fetchTableData = useCallback(async () => {
    setLoading(true);
    setSaveStatus(null);
    setProgress(null);
    setDirtyRowIndices(new Set());
    setSelectedRowIds(new Set());
    setSortConfig(null);
    setShowImportDrawer(false);
    setImportedFiles([]);
    setGlobalSearch('');
    setFilterRows([]);
    try {
      let rows: any[] = [];
      if (selectedTable === "RF Certificates") {
        rows = await api.getCertificates();
      } else if (selectedTable === "Authorities") {
        rows = await api.getAuthorities();
      } else if (selectedTable === "Suppliers") {
        rows = await api.getSuppliers();
      } else if (selectedTable === "Sources") {
        rows = await api.getSources();
      } else if (selectedTable === "Agent Memories") {
        rows = await api.getMemories();
      } else {
        rows = await api.getCustomTableRows(selectedTable);
      }
      const fetchedRows = Array.isArray(rows) ? rows : [];

      // Always clear any legacy draft key so page refresh reflects true PostgreSQL database state
      const draftKey = `rf_draft_${selectedTable}`;
      localStorage.removeItem(draftKey);

      setGridData(fetchedRows);
      setImportedFiles([]);
    } catch (err) {
      console.error("Error fetching table data:", err);
      setGridData([]);
    } finally {
      setLoading(false);
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

  // Fetch Recycle Bin items from PostgreSQL backend
  const fetchRecycleBin = useCallback(async () => {
    try {
      const items = await api.getRecycleBinItems();
      setRecycleBin(items);
    } catch (e) {
      console.warn("Could not fetch recycle bin items from DB", e);
    }
  }, []);

  useEffect(() => {
    fetchRecycleBin();
  }, [fetchRecycleBin]);

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
          }
          deletedCount++;
        }
      }

      setSelectedRowIds(new Set());
      if (selectedTable) localStorage.removeItem(`rf_draft_${selectedTable}`);
      setSaveStatus({ message: `Successfully moved ${deletedCount} record(s) to Recycle Bin!`, type: 'success' });
      await fetchRecycleBin();
      await fetchTableData();
    } catch (err: any) {
      setSaveStatus({ message: `Error deleting records: ${err.message || 'Operation failed'}`, type: 'error' });
    } finally {
      setDeleting(false);
    }
  };

  // Restore Item from PostgreSQL Recycle Bin
  const handleRestoreRecycleItem = async (item: RecycleItem) => {
    setRestoringId(item.id);
    try {
      await api.restoreRecycleBinItem(item.id);
      await fetchRecycleBin();
      setSaveStatus({ message: `Restored "${item.title}" back to ${item.tableName}!`, type: 'success' });
      if (selectedTable === item.tableName) await fetchTableData();
    } catch (err: any) {
      setSaveStatus({ message: `Error restoring record: ${err.message}`, type: 'error' });
    } finally {
      setRestoringId(null);
    }
  };

  // Permanently Delete Item from PostgreSQL Recycle Bin
  const handlePermanentDeleteRecycleItem = async (itemId: string) => {
    try {
      await api.deleteRecycleBinItem(itemId);
      await fetchRecycleBin();
    } catch (e) {
      console.error("Error deleting recycle item", e);
    }
  };

  // Empty All Items in PostgreSQL Recycle Bin
  const handleEmptyRecycleBin = async () => {
    if (!window.confirm("Are you sure you want to permanently delete all items in the Recycle Bin?")) return;
    try {
      await api.emptyRecycleBinApi();
      await fetchRecycleBin();
      setSaveStatus({ message: "Recycle Bin emptied.", type: 'success' });
    } catch (e: any) {
      setSaveStatus({ message: `Error emptying Recycle Bin: ${e.message}`, type: 'error' });
    }
  };

  const cancelRequestedRef = useRef(false);

  // Save Changes to Database with Live Progress Bar & Abort Capability
  const handleSaveChangesToDatabase = async () => {
    cancelRequestedRef.current = false;
    const dirtyGridRows = Array.from(dirtyRowIndices).map(i => gridData[i]).filter(Boolean);
    const rowsToSave = [...pendingImportRows, ...dirtyGridRows];

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
      if (selectedTable === "RF Certificates" && rowsToSave.length > 5) {
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

          await api.batchSaveCertificates(chunk);
          savedCount += chunk.length;
        }
      } else {
        for (let i = 0; i < total; i++) {
          if (cancelRequestedRef.current) {
            setSaveStatus({ message: `Import / Save operation cancelled by user. Saved ${savedCount} of ${total} records.`, type: 'error' });
            break;
          }

          const row = rowsToSave[i];
          try {
            if (selectedTable === "RF Certificates") {
              await api.saveCertificateRow(row);
            } else if (selectedTable === "Authorities") {
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
      await fetchTableData();
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

  // Import Excel / CSV Files (Multi-file Support + Silent Duplicate Rejection)
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
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

    let newlyAddedFilesCount = 0;
    let newlyAddedRowsCount = 0;
    const newFileEntries: { id: string; name: string; count: number }[] = [];
    const allNewRows: any[] = [];

    const totalFilesCount = validFiles.length;
    let fileIdx = 0;

    for (const file of validFiles) {
      fileIdx++;
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
          const sampleRow = rawImportedRows[0] || {};
          const headers = Object.keys(sampleRow);
          
          // Request dynamic LLM header mapping from backend for selected table
          let llmMapping: Record<string, string> = {};
          try {
            const targetFields = columnsMeta.map((c: any) => String(c.key));
            const mapRes = await api.mapSpreadsheetHeaders(headers, sampleRow, selectedTable, targetFields);
            if (mapRes && mapRes.mapping) {
              llmMapping = mapRes.mapping;
            }
          } catch (e) {
            console.warn("LLM header mapping request failed, using fallback mapper", e);
          }

          const importedRows = rawImportedRows.map(r => {
            if (!r || typeof r !== 'object') return r;
            
            const getLlmVal = (targetKey: string) => {
              const srcHeader = llmMapping[targetKey];
              if (srcHeader && r[srcHeader] !== undefined && r[srcHeader] !== null) {
                return r[srcHeader];
              }
              return undefined;
            };

            const comp = getLlmVal('component') ?? r.component ?? r.Component ?? r.name;
            const supp = getLlmVal('supplier') ?? r.supplier ?? r.Supplier ?? r.canonical_supplier;
            const ctry = getLlmVal('country') ?? r.country ?? r.Country;
            const certNo = getLlmVal('certif_number') ?? r.certif_number ?? r['Certif Number'];
            const auth = getLlmVal('authority') ?? r.authority ?? r.Authority ?? r.canonical_authority;
            const issueD = formatExcelDate(getLlmVal('issue_date') ?? r.issue_date ?? r['Issue Date']);
            const expD = formatExcelDate(getLlmVal('exp_date') ?? r.exp_date ?? r.expiration_date ?? r['Expiration Date'] ?? r['Exp Date']);
            const certL = getLlmVal('cert_link') ?? r.cert_link ?? r['PDF Document Link'] ?? r.url;

            return {
              ...r,
              component: comp !== undefined ? String(comp).trim() : '',
              supplier: supp !== undefined ? String(supp).trim() : '',
              country: ctry !== undefined ? String(ctry).trim() : '',
              certif_number: certNo !== undefined ? String(certNo).trim() : '',
              authority: auth !== undefined ? String(auth).trim() : '',
              issue_date: issueD,
              exp_date: expD,
              cert_link: certL !== undefined ? String(certL).trim() : ''
            };
          });

          const existingSignatures = new Set(
            [...gridData, ...allNewRows].map(r => {
              const primary = (r.certif_number || r.component || r.canonical_authority || r.canonical_supplier || r.url || r.memory_key || r.name || '').toString().trim().toLowerCase();
              const secondary = (r.country || r.supplier || r.authority || '').toString().trim().toLowerCase();
              return `${primary}::${secondary}`;
            }).filter(sig => sig !== '::')
          );

          const uniqueRows = importedRows.filter(r => {
            const primary = (r.certif_number || r.component || r.canonical_authority || r.canonical_supplier || r.url || r.memory_key || r.name || '').toString().trim().toLowerCase();
            const secondary = (r.country || r.supplier || r.authority || '').toString().trim().toLowerCase();
            const sig = `${primary}::${secondary}`;
            return sig === '::' || !existingSignatures.has(sig);
          });

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

    if (allNewRows.length > 0) {
      setPendingImportRows(prev => [...allNewRows, ...prev]);
      setImportedFiles(prev => [...prev, ...newFileEntries]);
      setSaveStatus({ message: `${newlyAddedFilesCount} file(s) queued for import (${newlyAddedRowsCount} rows). Review preview in Import Drawer and click "Save Changes to Database" to commit to PostgreSQL.`, type: 'success' });
    } else {
      setSaveStatus({ message: `No new unique rows found in selected file(s).`, type: 'success' });
    }

    setProgress({
      current: totalFilesCount,
      total: totalFilesCount,
      percent: 100,
      statusText: `Processed ${totalFilesCount} file(s) successfully! (100%)`
    });
    setTimeout(() => setProgress(null), 1200);

    e.target.value = '';
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
  if (selectedTable === "Authorities") {
    return [
      { key: 'canonical_authority', label: 'Canonical Authority', defaultWidth: 220 },
      { key: 'abbreviation', label: 'Abbreviation', defaultWidth: 130 },
      { key: 'country', label: 'Country', defaultWidth: 140 },
      { key: 'standard_validity_years', label: 'Validity (Years)', defaultWidth: 140 },
      { key: 'aliases', label: 'Aliases', defaultWidth: 260 },
    ];
  }
  if (selectedTable === "Suppliers") {
    return [
      { key: 'canonical_supplier', label: 'Canonical Supplier', defaultWidth: 240 },
      { key: 'aliases', label: 'Aliases', defaultWidth: 320 },
    ];
  }
  if (selectedTable === "Sources") {
    return [
      { key: 'url', label: 'Source URL', defaultWidth: 300 },
      { key: 'description', label: 'Description', defaultWidth: 260 },
      { key: 'active', label: 'Active Status', defaultWidth: 130 },
    ];
  }
  if (selectedTable === "Agent Memories") {
    return [
      { key: 'memory_key', label: 'Category / Key', defaultWidth: 180 },
      { key: 'fact_text', label: 'Memory Fact / Directive', defaultWidth: 380 },
      { key: 'created_at', label: 'Date Created', defaultWidth: 160 },
    ];
  }
  if (selectedTable === "RF Certificates") {
    return [
      { key: 'component', label: 'Component / Model', defaultWidth: 200 },
      { key: 'supplier', label: 'Supplier / Manufacturer', defaultWidth: 200 },
      { key: 'country', label: 'Country', defaultWidth: 120 },
      { key: 'certif_number', label: 'Certificate No.', defaultWidth: 180 },
      { key: 'authority', label: 'Authority', defaultWidth: 160 },
      { key: 'issue_date', label: 'Issue Date', defaultWidth: 130 },
      { key: 'exp_date', label: 'Exp Date', defaultWidth: 130 },
      { key: 'cert_link', label: 'PDF Document Link', defaultWidth: 200 },
      { key: 'last_update', label: 'Last Update', defaultWidth: 150 },
    ];
  }
  // Custom Tables
  if (gridData.length > 0) {
    const keys = Object.keys(gridData[0]).filter(k => k !== '_isNew' && k !== 'id');
    return keys.map(k => ({ key: k, label: k.replace(/_/g, ' ').toUpperCase(), defaultWidth: 180 }));
  }
  return [{ key: 'name', label: 'Name', defaultWidth: 200 }];
}, [selectedTable, gridData]);

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
                onClick={handleSaveChangesToDatabase}
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

          {/* Delete Selected Rows */}
          {selectedRowIds.size > 0 && (
            <button
              className="btn btn-secondary"
              style={{ fontSize: '0.85rem', color: 'var(--error)', borderColor: 'var(--error)' }}
              onClick={handleDeleteSelectedRows}
              disabled={saving || deleting}
            >
              <Trash2 size={14} /> {deleting ? 'Deleting...' : `Delete Selected (${selectedRowIds.size})`}
            </button>
          )}

          {/* + Add Row */}
          {isExcelGridTable && (
            <button
              className="btn btn-secondary"
              style={{ fontSize: '0.85rem' }}
              onClick={handleAddRow}
            >
              <Plus size={14} /> Add Row
            </button>
          )}

          {/* Import Excel / CSV Button */}
          <button
            className="btn btn-secondary"
            style={{ fontSize: '0.85rem' }}
            onClick={() => setShowImportDrawer(!showImportDrawer)}
          >
            <Download size={14} /> Import Excel / CSV
          </button>

          {/* Export Excel (.xlsx) */}
          <button
            className="btn btn-secondary"
            style={{ fontSize: '0.85rem' }}
            onClick={handleExportExcel}
            disabled={gridData.length === 0}
          >
            <Upload size={14} /> {selectedRowIds.size > 0 ? `Export Selected (${selectedRowIds.size})` : 'Export Excel'}
          </button>

          {/* Recycle Bin Button (Always visible) */}
          <button
            className="btn btn-secondary"
            style={{ fontSize: '0.85rem' }}
            onClick={() => setShowRecycleDrawer(!showRecycleDrawer)}
          >
            <Trash2 size={14} /> Recycle Bin ({recycleBin.length})
          </button>
        </div>
      </div>

      {/* Recycle Bin Drawer */}
      {showRecycleDrawer && (
        <div style={{ backgroundColor: 'var(--bg-body)', padding: '1rem', borderRadius: '8px', border: '1.5px dashed #f59e0b', display: 'flex', flexDirection: 'column', gap: '0.65rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#b45309', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Trash2 size={16} /> RECYCLE BIN ({recycleBin.length} DELETED RECORDS)
            </span>
            <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              {recycleBin.length > 0 && (
                <button
                  type="button"
                  className="btn btn-secondary"
                  style={{ padding: '0.2rem 0.55rem', fontSize: '0.75rem', color: '#ef4444', borderColor: '#ef4444' }}
                  onClick={handleEmptyRecycleBin}
                >
                  Empty Recycle Bin
                </button>
              )}
              <button
                type="button"
                className="btn btn-secondary"
                style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                onClick={() => setShowRecycleDrawer(false)}
              >
                Close
              </button>
            </div>
          </div>

          {recycleBin.length === 0 ? (
            <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', margin: '0.5rem 0' }}>Recycle Bin is empty.</p>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', maxHeight: '220px', overflowY: 'auto' }}>
              {recycleBin.map(item => (
                <div
                  key={item.id}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.5rem 0.75rem',
                    backgroundColor: 'var(--bg-surface)',
                    border: '1px solid var(--border-color)',
                    borderRadius: '6px',
                    fontSize: '0.82rem'
                  }}
                >
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{item.title}</span>
                    <span style={{ fontSize: '0.74rem', color: 'var(--text-tertiary)' }}>
                      Table: {item.tableName} • Deleted: {item.deletedAt}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem', color: '#10b981', borderColor: '#10b981' }}
                      onClick={() => handleRestoreRecycleItem(item)}
                      disabled={restoringId === item.id}
                      title="Restore record back to database table"
                    >
                      <RotateCcw size={12} /> {restoringId === item.id ? 'Restoring...' : 'Restore'}
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem', color: '#ef4444', borderColor: '#ef4444' }}
                      onClick={() => handlePermanentDeleteRecycleItem(item.id)}
                      title="Permanently delete from Recycle Bin"
                    >
                      <X size={12} /> Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

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
            <span>{progress.statusText}</span>
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
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Component</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Supplier</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Country</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Certif #</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Authority</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Issue Date</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>Exp Date</th>
                      <th style={{ padding: '0.4rem 0.6rem', borderBottom: '1px solid var(--border-color)' }}>PDF Link</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pendingImportRows.slice(0, 5).map((row, idx) => (
                      <tr key={idx} style={{ borderBottom: '1px solid var(--border-color)' }}>
                        <td style={{ padding: '0.35rem 0.6rem', fontWeight: 600 }}>{row.component || row.Component || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem' }}>{row.supplier || row.Supplier || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem' }}>{row.country || row.Country || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem' }}>{row.certif_number || row['Certif Number'] || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem' }}>{row.authority || row.Authority || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem', color: row.issue_date ? '#10b981' : '#94a3b8' }}>{row.issue_date || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem', color: row.exp_date ? '#10b981' : '#94a3b8' }}>{row.exp_date || '—'}</td>
                        <td style={{ padding: '0.35rem 0.6rem', maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {row.cert_link ? (
                            <a href={row.cert_link} target="_blank" rel="noreferrer" style={{ color: 'var(--brand-blue)', textDecoration: 'underline' }}>
                              {row.cert_link}
                            </a>
                          ) : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
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
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem', tableLayout: 'fixed' }}>
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

                        {/* Sort Indicator Arrow */}
                        {isSorted ? (
                          sortConfig.direction === 'asc' ? <ArrowUp size={14} color="var(--brand-blue)" /> : <ArrowDown size={14} color="var(--brand-blue)" />
                        ) : (
                          <ArrowUpDown size={12} color="var(--text-tertiary)" style={{ opacity: 0.5 }} />
                        )}
                      </div>

                      {/* Column Resizer Handle */}
                      <div
                        onMouseDown={(e) => handleStartResize(e, col.key, currentWidth)}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          position: 'absolute',
                          right: 0,
                          top: 0,
                          bottom: 0,
                          width: '6px',
                          cursor: 'col-resize',
                          backgroundColor: resizingCol?.key === col.key ? 'var(--brand-blue)' : 'transparent',
                          zIndex: 3
                        }}
                        title="Drag to resize column"
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

                    {/* Cell Content: Clickable Link for cert_link, Interactive Input for all other cells */}
                    {columnsMeta.map(col => {
                      const cellVal = item[col.key] ?? '';

                      if (col.key === 'cert_link') {
                        return (
                          <td key={col.key} style={{ padding: '0.55rem 0.75rem', borderRight: '1px solid var(--border-color)' }}>
                            {cellVal ? (
                              <a
                                href={String(cellVal)}
                                target="_blank"
                                rel="noreferrer"
                                title="Open Document PDF"
                                style={{
                                  color: 'var(--brand-blue)',
                                  fontWeight: 600,
                                  display: 'inline-flex',
                                  alignItems: 'center',
                                  gap: '0.35rem',
                                  textDecoration: 'none'
                                }}
                              >
                                <ExternalLink size={14} /> View Document
                              </a>
                            ) : (
                              <span className="text-tertiary">—</span>
                            )}
                          </td>
                        );
                      }

                      return (
                        <td key={col.key} style={{ padding: 0, borderRight: '1px solid var(--border-color)' }}>
                          <input
                            type="text"
                            className="excel-cell-input"
                            value={cellVal}
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

  </div>
);
};

export default DatabasesView;
