import { useState, useEffect, useRef, useCallback } from 'react';
import { Outlet, useOutletContext } from 'react-router-dom';
import Navbar from '../Navbar/Navbar';
import Sidebar from '../Sidebar/Sidebar';
import { api, type ActiveChatJob } from '../../api';

export interface SessionStreamState {
  jobId: string;
  sessionId: string;
  isStreaming: boolean;
  answer: string;
  thinking: string;
  status: string;
  intent: string;
}

export type LayoutContextType = {
  selectedTable: string;
  setSelectedTable: (tbl: string) => void;
  customTables: string[];
  setCustomTables: React.Dispatch<React.SetStateAction<string[]>>;
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  sessionStreams: Record<string, SessionStreamState>;
  deepThinking: boolean;
  setDeepThinking: (val: boolean) => void;
  sendChatQuery: (query: string, targetSessionId?: string | null) => Promise<void>;
  attachToActiveJob: (job: ActiveChatJob) => void;
};

export function useLayoutContext() {
  return useOutletContext<LayoutContextType>();
}

const AppLayout = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [selectedTable, setSelectedTable] = useState('RF Certificates');
  const [customTables, setCustomTables] = useState<string[]>(() => {
    try {
      const cached = localStorage.getItem('rf_custom_tables');
      return cached ? JSON.parse(cached) : [];
    } catch {
      return [];
    }
  });
  const [activeSessionId, setActiveSessionIdState] = useState<string | null>(() => {
    try {
      return sessionStorage.getItem('rf_active_chat_session_id') || null;
    } catch {
      return null;
    }
  });
  const [deepThinking, setDeepThinking] = useState<boolean>(false);
  const [sessionStreams, setSessionStreams] = useState<Record<string, SessionStreamState>>({});
  const eventSourcesRef = useRef<Record<string, EventSource>>({});

  const setActiveSessionId = useCallback((id: string | null) => {
    setActiveSessionIdState(id);
    try {
      if (id) {
        sessionStorage.setItem('rf_active_chat_session_id', id);
      } else {
        sessionStorage.removeItem('rf_active_chat_session_id');
      }
    } catch {
      // Ignore storage errors
    }
  }, []);

  useEffect(() => {
    const loadCustomTables = async () => {
      try {
        const tables = await api.getCustomTables();
        const cleanTables = Array.isArray(tables) ? tables : [];
        setCustomTables(cleanTables);
        localStorage.setItem('rf_custom_tables', JSON.stringify(cleanTables));
      } catch (err) {
        console.error("Failed to load custom tables on mount:", err);
      }
    };
    loadCustomTables();

    const handleRefreshTables = () => loadCustomTables();
    window.addEventListener('refresh-custom-tables', handleRefreshTables);
    return () => window.removeEventListener('refresh-custom-tables', handleRefreshTables);
  }, []);

  // Cleanup EventSources only when the entire app unmounts
  useEffect(() => {
    return () => {
      Object.values(eventSourcesRef.current).forEach(es => {
        try {
          es.close();
        } catch {
          // ignore
        }
      });
      eventSourcesRef.current = {};
    };
  }, []);

  const connectEventSource = useCallback((
    sessionId: string,
    jobId: string,
    initial?: Partial<SessionStreamState>,
    fromIndex: number = 0
  ) => {
    // If already connected to this exact job, do nothing
    const existingEs = eventSourcesRef.current[sessionId];
    if (existingEs && existingEs.readyState !== EventSource.CLOSED) {
      return;
    }

    setSessionStreams(prev => ({
      ...prev,
      [sessionId]: {
        jobId,
        sessionId,
        isStreaming: true,
        answer: initial?.answer ?? '',
        thinking: initial?.thinking ?? '',
        status: initial?.status ?? 'Understanding your request...',
        intent: initial?.intent ?? '',
      },
    }));

    const es = new EventSource(`/api/v1/chat/stream/${jobId}?from_index=${fromIndex}`);
    eventSourcesRef.current[sessionId] = es;

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'status') {
          const statusMsg = data.message || data.status || 'Analyzing compliance records...';
          setSessionStreams(prev => {
            const cur = prev[sessionId];
            if (!cur) return prev;
            return {
              ...prev,
              [sessionId]: {
                ...cur,
                status: statusMsg,
                intent: data.intent || cur.intent,
              },
            };
          });
        } else if (data.type === 'thinking') {
          const thinkChunk = data.text || data.chunk || '';
          setSessionStreams(prev => {
            const cur = prev[sessionId];
            if (!cur) return prev;
            return {
              ...prev,
              [sessionId]: {
                ...cur,
                thinking: cur.thinking + thinkChunk,
              },
            };
          });
        } else if (data.type === 'token') {
          const tokenChunk = data.text || data.chunk || '';
          setSessionStreams(prev => {
            const cur = prev[sessionId];
            if (!cur) return prev;
            return {
              ...prev,
              [sessionId]: {
                ...cur,
                answer: cur.answer + tokenChunk,
                status: '',
              },
            };
          });
        } else if (data.type === 'done') {
          es.close();
          delete eventSourcesRef.current[sessionId];
          setSessionStreams(prev => {
            const next = { ...prev };
            delete next[sessionId];
            return next;
          });
          window.dispatchEvent(new CustomEvent('chat-session-updated', { detail: { sessionId } }));
          window.dispatchEvent(new Event('refresh-chat-sessions'));
        } else if (data.type === 'error') {
          es.close();
          delete eventSourcesRef.current[sessionId];
          setSessionStreams(prev => {
            const next = { ...prev };
            delete next[sessionId];
            return next;
          });
          window.dispatchEvent(new CustomEvent('chat-session-updated', { detail: { sessionId } }));
        }
      } catch (err) {
        console.error('Error parsing SSE event payload', err);
      }
    };

    es.onerror = () => {
      es.close();
      delete eventSourcesRef.current[sessionId];
      setSessionStreams(prev => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });
      window.dispatchEvent(new CustomEvent('chat-session-updated', { detail: { sessionId } }));
    };
  }, []);

  const attachToActiveJob = useCallback((job: ActiveChatJob) => {
    if (!job || !job.session_id || !job.job_id) return;
    connectEventSource(
      job.session_id,
      job.job_id,
      {
        answer: job.partial_answer || '',
        thinking: job.thinking || '',
        status: job.status || 'Processing...',
        intent: job.intent || '',
      },
      job.event_count || 0
    );
  }, [connectEventSource]);

  const sendChatQuery = useCallback(async (query: string, targetSessionId?: string | null) => {
    const clean = query.trim();
    if (!clean) return;
    if (targetSessionId && sessionStreams[targetSessionId]?.isStreaming) {
      return;
    }

    const res = await api.startChatStream(clean, targetSessionId || undefined, deepThinking);
    const resolvedSessionId = res.session_id;

    if (!targetSessionId && resolvedSessionId) {
      setActiveSessionId(resolvedSessionId);
    }

    // Notify ChatView & Sidebar immediately so the persisted user message and session title render
    window.dispatchEvent(new CustomEvent('chat-session-updated', { detail: { sessionId: resolvedSessionId } }));
    window.dispatchEvent(new Event('refresh-chat-sessions'));

    if (res.frozen || !res.job_id) {
      window.dispatchEvent(new CustomEvent('chat-session-updated', { detail: { sessionId: resolvedSessionId } }));
      return;
    }

    connectEventSource(resolvedSessionId, res.job_id, {
      status: 'Understanding your request...',
    }, 0);
  }, [sessionStreams, deepThinking, setActiveSessionId, connectEventSource]);

  return (
    <div className="app-container" style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <Navbar />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <Sidebar 
          isOpen={sidebarOpen} 
          onToggle={() => setSidebarOpen(prev => !prev)} 
          selectedTable={selectedTable}
          setSelectedTable={setSelectedTable}
          customTables={customTables}
          setCustomTables={setCustomTables}
          activeSessionId={activeSessionId}
          setActiveSessionId={setActiveSessionId}
          sessionStreams={sessionStreams}
        />
        <main className="page-content" style={{ flex: 1, padding: '1.5rem', overflowY: 'auto' }}>
          <Outlet context={{
            selectedTable,
            setSelectedTable,
            customTables,
            setCustomTables,
            activeSessionId,
            setActiveSessionId,
            sessionStreams,
            deepThinking,
            setDeepThinking,
            sendChatQuery,
            attachToActiveJob,
          }} />
        </main>
      </div>
    </div>
  );
};

export default AppLayout;
