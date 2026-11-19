export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: 'admin' | 'editor' | 'viewer';
  status: string;
}

export interface ManagedUser extends UserProfile {
  invite_token?: string | null;
  created_at?: string | null;
  last_login_at?: string | null;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at?: string;
  frozen: boolean;
  is_streaming?: boolean;
}

export interface ActiveChatJob {
  job_id: string;
  session_id: string;
  user_query?: string;
  status: string;
  thinking: string;
  partial_answer: string;
  intent: string;
  event_count: number;
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

  getSessionDetails: async (sessionId: string): Promise<{ messages: ChatMessage[]; active_job: ActiveChatJob | null; frozen: boolean }> => {
    const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages`);
    if (!res.ok) throw new Error('Failed to fetch session details');
    const json = await res.json();
    if (Array.isArray(json)) {
      return { messages: json, active_job: null, frozen: false };
    }
    return {
      messages: Array.isArray(json.messages) ? json.messages : [],
      active_job: json.active_job || null,
      frozen: Boolean(json.frozen),
    };
  },

  deleteSession: async (sessionId: string): Promise<void> => {
    const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error('Failed to delete session');
  },

  startChatStream: async (
    query: string,
    sessionId?: string,
    deepThinking?: boolean
  ): Promise<{ job_id: string | null; session_id: string; frozen: boolean; answer?: string; intent?: string }> => {
    const res = await fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, session_id: sessionId, deep_thinking: Boolean(deepThinking) }),
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
    const actionPath = decision === 'rejected' ? 'reject' : 'approve';
    const res = await fetch(`${API_BASE}/agent/proposals/${encodeURIComponent(id)}/${actionPath}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision })
    });
    if (!res.ok) {
      let detail = `Failed to ${decision} proposal`;
      try {
        const errJson = await res.json();
        if (errJson?.detail) detail = errJson.detail;
      } catch {
        // ignore JSON parse error
      }
      throw new Error(detail);
    }
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

  getAlertingConfig: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/alerts/config`);
    if (!res.ok) throw new Error('Failed to fetch alerting config');
    return res.json();
  },

  updateAlertingConfig: async (config: any): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/alerts/config`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    if (!res.ok) throw new Error('Failed to update alerting config');
    return res.json();
  },

  runExpiryAlertScan: async (forceRescan: boolean = false): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/alerts/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ force_rescan: forceRescan })
    });
    if (!res.ok) throw new Error('Failed to run expiry alert scan');
    return res.json();
  },

  startGoogleOAuth: async (): Promise<{ auth_url: string; redirect_uri: string }> => {
    const res = await fetch(`${API_BASE}/agent/alerts/oauth/start`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to start Google OAuth flow');
    }
    return res.json();
  },

  disconnectGoogleOAuth: async (): Promise<any> => {
    const res = await fetch(`${API_BASE}/agent/alerts/oauth/disconnect`, {
      method: 'POST',
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to disconnect Google OAuth');
    }
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

  getCustomTables: async (): Promise<string[]> => {
    const res = await fetch(`${API_BASE}/schema/tables`);
    if (!res.ok) return [];
    const json = await res.json();
    const tables: any[] = json.tables || [];
    return tables.filter((t: any) => t.is_custom).map((t: any) => t.table_name);
  },

  createCustomTable: async (tableName: string, columns: { name: string; type?: string }[]): Promise<any> => {
    const res = await fetch(`${API_BASE}/schema/tables`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ table_name: tableName, columns })
    });
    if (!res.ok) {
      const err = await res.text().catch(() => '');
      throw new Error(`Failed to create custom table (${res.status}): ${err}`);
    }
    return res.json();
  },

  deleteCustomTable: async (tableName: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Failed to delete custom table');
    return res.json();
  },

  addCustomTableColumn: async (tableName: string, columnName: string, columnType: string = 'VARCHAR', semanticType?: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/columns`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ column_name: columnName, column_type: columnType, semantic_type: semanticType })
    });
    if (!res.ok) {
      const err = await res.text().catch(() => '');
      throw new Error(`Failed to add column (${res.status}): ${err}`);
    }
    return res.json();
  },

  deleteCustomTableColumn: async (tableName: string, columnName: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/columns/${encodeURIComponent(columnName)}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.text().catch(() => '');
      throw new Error(`Failed to delete column (${res.status}): ${err}`);
    }
    return res.json();
  },

  getCustomTableRows: async (tableName: string): Promise<{ columns?: any[], records?: any[] }> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/data`);
    if (!res.ok) return { columns: [], records: [] };
    const json = await res.json();
    return json;
  },

  importCustomTableFile: async (tableName: string, file: File): Promise<any> => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/import`, {
      method: 'POST',
      body: formData
    });
    if (!res.ok) {
      const err = await res.text().catch(() => '');
      throw new Error(`Failed to import file into custom table (${res.status}): ${err}`);
    }
    return res.json();
  },

  saveCustomTableRow: async (tableName: string, payload: any): Promise<any> => {
    const isUpdate = !!payload.id;
    const url = isUpdate 
      ? `${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/data/${payload.id}`
      : `${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/data`;
    const method = isUpdate ? 'PUT' : 'POST';
    const res = await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Failed to save custom table row');
    return res.json();
  },

  batchSaveCustomTableRows: async (tableName: string, rows: any[], standardize: boolean = true): Promise<any> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/data/batch?standardize=${standardize}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(rows)
    });
    if (!res.ok) throw new Error('Failed to batch save custom table rows');
    return res.json();
  },

  deleteCustomTableRow: async (tableName: string, recordId: number | string): Promise<any> => {
    const res = await fetch(`${API_BASE}/schema/tables/${encodeURIComponent(tableName)}/data/${recordId}`, {
      method: 'DELETE'
    });
    if (!res.ok) throw new Error('Failed to delete custom table row');
    return res.json();
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
  },

  // Authentication & User Management
  login: async (email: string, password: string): Promise<{ token: string; user: UserProfile }> => {
    const res = await fetch(`${API_BASE}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Invalid email or password');
    }
    return res.json();
  },

  logout: async (): Promise<void> => {
    await fetch(`${API_BASE}/auth/logout`, { method: 'POST' });
  },

  getMe: async (): Promise<{ user: UserProfile } | null> => {
    const res = await fetch(`${API_BASE}/auth/me`);
    if (!res.ok) return null;
    return res.json();
  },

  updateMe: async (payload: { full_name?: string; email?: string; current_password?: string; new_password?: string }): Promise<{ user: UserProfile }> => {
    const res = await fetch(`${API_BASE}/auth/me`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to update account');
    }
    return res.json();
  },

  validateInvite: async (token: string): Promise<{ email: string; full_name: string; role: 'editor' | 'viewer' }> => {
    const res = await fetch(`${API_BASE}/auth/invite/${encodeURIComponent(token)}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Invalid or expired invitation link');
    }
    return res.json();
  },

  acceptInvite: async (token: string, password: string, full_name?: string): Promise<{ token: string; user: UserProfile }> => {
    const res = await fetch(`${API_BASE}/auth/accept-invite`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token, password, full_name })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to activate account');
    }
    return res.json();
  },

  getUsers: async (): Promise<{ users: ManagedUser[] }> => {
    const res = await fetch(`${API_BASE}/users`);
    if (!res.ok) throw new Error('Failed to fetch users');
    return res.json();
  },

  inviteUser: async (email: string, full_name: string, role: 'editor' | 'viewer'): Promise<{ user: ManagedUser; invite_url: string; email_sent: boolean; email_error?: string | null }> => {
    const res = await fetch(`${API_BASE}/users/invite`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, full_name, role })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to invite user');
    }
    return res.json();
  },

  updateUserRole: async (userId: string, role: 'editor' | 'viewer'): Promise<any> => {
    const res = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}/role`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ role })
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to update user role');
    }
    return res.json();
  },

  deleteUser: async (userId: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/users/${encodeURIComponent(userId)}`, {
      method: 'DELETE'
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.detail || 'Failed to remove user');
    }
    return res.json();
  }
};
