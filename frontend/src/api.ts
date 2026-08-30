export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  frozen: boolean;
}

export interface ChatMessage {
  id?: string;
  session_id: string;
  role: 'user' | 'assistant';
  content: string;
  intent?: string;
  metadata?: any;
  created_at?: string;
}

const API_BASE = '/api/v1';

export const api = {
  // Chat Sessions
  getSessions: async (): Promise<ChatSession[]> => {
    const res = await fetch(`${API_BASE}/chat/sessions`);
    if (!res.ok) throw new Error('Failed to fetch sessions');
    const json = await res.json();
    return Array.isArray(json) ? json : (json.sessions || []);
  },

  getSessionMessages: async (sessionId: string): Promise<ChatMessage[]> => {
    const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages`);
    if (!res.ok) throw new Error('Failed to fetch messages');
    const json = await res.json();
    return Array.isArray(json) ? json : (json.messages || []);
  },

  deleteSession: async (sessionId: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete session');
  },

  startChatStream: async (query: string, sessionId?: string): Promise<{ job_id: string | null; session_id: string; frozen: boolean; answer?: string; intent?: string }> => {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, session_id: sessionId }),
    });
    if (!res.ok) throw new Error('Failed to start chat stream');
    return res.json();
  },

  // Certificates & Batch Ingestion
  getCertificates: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/certificates`);
    if (!res.ok) throw new Error('Failed to fetch certificates');
    const json = await res.json();
    return Array.isArray(json) ? json : (json.certificates || []);
  },
  
  ingestFiles: async (files: File[]): Promise<any> => {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    const res = await fetch(`${API_BASE}/batch/ingest`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error('Failed to ingest files');
    return res.json();
  },

  // Agent Control & Scheduler
  getProposals: async (): Promise<{ proposals: any[] }> => {
    const res = await fetch(`${API_BASE}/agent/proposals`);
    if (!res.ok) throw new Error('Failed to fetch proposals');
    return res.json();
  },

  approveProposal: async (id: string, decision: 'approved' | 'rejected'): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/proposals/${id}/approve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision })
    });
    if (!res.ok) throw new Error(`Failed to ${decision} proposal`);
    return res.json();
  },

  getSchedulerConfig: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/autonomous/config`);
    if (!res.ok) throw new Error('Failed to fetch scheduler config');
    return res.json();
  },

  updateSchedulerConfig: async (config: any): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/autonomous/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!res.ok) throw new Error('Failed to update scheduler config');
    return res.json();
  },

  runAutonomousScraper: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/autonomous/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    if (!res.ok) throw new Error('Failed to run autonomous scraper');
    return res.json();
  },

  // URL Ingestion
  ingestUrl: async (url: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/ingest/url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) throw new Error('Failed to ingest URL');
    return res.json();
  },

  // Lookups & Tables Management
  getAuthorities: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/lookups/authorities`);
    if (!res.ok) return [];
    const json = await res.json();
    return Array.isArray(json) ? json : (json.authorities || []);
  },

  getSuppliers: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/lookups/suppliers`);
    if (!res.ok) return [];
    const json = await res.json();
    return Array.isArray(json) ? json : (json.suppliers || []);
  },

  getSources: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/sources`);
    if (!res.ok) return [];
    const json = await res.json();
    return Array.isArray(json) ? json : (json.sources || []);
  },

  getMemories: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/memories`);
    if (!res.ok) return [];
    const json = await res.json();
    return Array.isArray(json) ? json : (json.memories || []);
  },

  getCustomTableRows: async (tableName: string): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/rows`);
    if (!res.ok) return [];
    const json = await res.json();
    return Array.isArray(json) ? json : (json.rows || []);
  },

  deleteAuthority: async (id: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/lookups/authorities/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete authority');
  },

  deleteSupplier: async (id: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/lookups/suppliers/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete supplier');
  },

  deleteSource: async (id: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/sources/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete source');
  },

  deleteMemory: async (id: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/memories/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete memory');
  },

  deleteCertificate: async (id: string | number): Promise<void> => {
    const res = await fetch(`${API_BASE}/certificates/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete certificate');
  },

  saveCertificateRow: async (payload: any): Promise<any> => {
    const isUpdate = !!payload.id || !!payload.certificate_id;
    const url = isUpdate ? `${API_BASE}/certificates/${payload.id || payload.certificate_id}` : `${API_BASE}/certificates`;
    const method = isUpdate ? 'PUT' : 'POST';
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Failed to save certificate row');
    return res.json();
  },

  batchSaveCertificates: async (rows: any[]): Promise<any> => {
    const res = await fetch(`${API_BASE}/certificates/batch`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(rows)
    });
    if (!res.ok) throw new Error('Failed to batch save certificates');
    return res.json();
  },

  mapSpreadsheetHeaders: async (headers: string[], sampleRow: any, tableName?: string, targetFields?: string[]): Promise<{ mapping: Record<string, string>; standardized_sample?: any }> => {
    const res = await fetch(`${API_BASE}/databases/map-headers`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ headers, sample_row: sampleRow, table_name: tableName || 'RF Certificates', target_fields: targetFields || [] })
    });
    if (!res.ok) return { mapping: {} };
    return res.json();
  },

  checkAgentBusy: async (): Promise<{ is_busy: boolean }> => {
    const res = await fetch(`${API_BASE}/agent/busy`);
    if (!res.ok) return { is_busy: false };
    return res.json();
  },

  saveAuthorityRow: async (payload: any): Promise<any> => {
    const isUpdate = !!payload.id;
    const url = isUpdate ? `${API_BASE}/lookups/authorities/${payload.id}` : `${API_BASE}/lookups/authorities`;
    const method = isUpdate ? 'PUT' : 'POST';
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return res.json();
  },

  saveSupplierRow: async (payload: any): Promise<any> => {
    const isUpdate = !!payload.id;
    const url = isUpdate ? `${API_BASE}/lookups/suppliers/${payload.id}` : `${API_BASE}/lookups/suppliers`;
    const method = isUpdate ? 'PUT' : 'POST';
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return res.json();
  },

  saveSourceRow: async (payload: any): Promise<any> => {
    const isUpdate = !!payload.id;
    const url = isUpdate ? `${API_BASE}/sources/${payload.id}` : `${API_BASE}/sources`;
    const method = isUpdate ? 'PUT' : 'POST';
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return res.json();
  },

  saveMemoryRow: async (payload: any): Promise<any> => {
    const isUpdate = !!payload.id;
    const url = isUpdate ? `${API_BASE}/memories/${payload.id}` : `${API_BASE}/memories`;
    const method = isUpdate ? 'PUT' : 'POST';
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    return res.json();
  },

  uploadCertificatePdf: async (file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/parse`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      throw new Error(`Parse error (${res.status}): ${errText || 'Document extraction failed'}`);
    }
    return res.json();
  },

  uploadBatchCertificates: async (files: File[]): Promise<any> => {
    const formData = new FormData();
    for (const f of files) {
      formData.append('files', f);
    }
    const res = await fetch(`${API_BASE}/batch/ingest`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const errText = await res.text().catch(() => '');
      throw new Error(`Batch ingest failed (${res.status}): ${errText || 'Error starting batch'}`);
    }
    return res.json();
  },

  getBatchStatus: async (batchId?: string): Promise<any> => {
    const url = batchId ? `${API_BASE}/batch/status/${batchId}` : `${API_BASE}/batch/status`;
    const res = await fetch(url);
    if (!res.ok) throw new Error('Failed to fetch batch status');
    return res.json();
  },

  // System Diagnostics
  getSystemCheck: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/system/check`);
    if (!res.ok) throw new Error('Failed to fetch system readiness check');
    return res.json();
  },

  getHealth: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error('Failed to fetch system health');
    return res.json();
  },

  batchDeleteCertificates: async (ids: string[]): Promise<any> => {
    const res = await fetch(`${API_BASE}/certificates/batch-delete`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ids })
    });
    if (!res.ok) throw new Error('Failed to batch delete certificates');
    return res.json();
  },

  getRecycleBinItems: async (): Promise<any[]> => {
    const res = await fetch(`${API_BASE}/recycle-bin`);
    if (!res.ok) throw new Error('Failed to fetch recycle bin items');
    const json = await res.json();
    return json.items || [];
  },

  restoreRecycleBinItem: async (id: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/recycle-bin/restore/${id}`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to restore item');
    return res.json();
  },

  deleteRecycleBinItem: async (id: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/recycle-bin/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete recycle item');
    return res.json();
  },

  emptyRecycleBinApi: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/recycle-bin`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to empty recycle bin');
    return res.json();
  },

  // Notifications API
  getNotifications: async (): Promise<{ unread_count: number; notifications: any[] }> => {
    const res = await fetch(`${API_BASE}/notifications`);
    if (!res.ok) return { unread_count: 0, notifications: [] };
    return res.json();
  },

  markNotificationsRead: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/notifications/mark-read`, { method: 'POST' });
    if (!res.ok) throw new Error('Failed to mark notifications read');
    return res.json();
  },

  clearNotifications: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/notifications/clear`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to clear notifications');
    return res.json();
  }
};
