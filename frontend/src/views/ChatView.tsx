import { useState, useEffect, useRef, useCallback } from 'react';
import { FileText, ShieldAlert, Building2, BrainCircuit, Sparkles, ArrowUp } from 'lucide-react';
import { api, type ChatMessage as ChatMessageType } from '../api';
import { useLayoutContext } from '../components/Layout/AppLayout';
import { ChatMessage } from '../components/Chat/ChatMessage';
import './ChatView.css';

const ChatView = () => {
  const {
    activeSessionId,
    sessionStreams,
    deepThinking,
    setDeepThinking,
    sendChatQuery,
    attachToActiveJob,
  } = useLayoutContext();

  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);

  const activeSessionIdRef = useRef<string | null>(activeSessionId);
  activeSessionIdRef.current = activeSessionId;

  const currentStream = activeSessionId ? sessionStreams[activeSessionId] : undefined;
  const isStreaming = Boolean(currentStream?.isStreaming);
  const streamingAnswer = currentStream?.answer || '';
  const streamingThinking = currentStream?.thinking || '';
  const streamingStatus = currentStream?.status || '';
  const streamingIntent = currentStream?.intent || '';

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const loadSessionMessages = useCallback(async (sessionId: string | null) => {
    if (!sessionId) {
      setMessages([]);
      return;
    }
    try {
      const details = await api.getSessionDetails(sessionId);
      if (activeSessionIdRef.current !== sessionId) return;
      setMessages(Array.isArray(details.messages) ? details.messages : []);
      if (details.active_job) {
        attachToActiveJob(details.active_job);
      }
    } catch {
      if (activeSessionIdRef.current === sessionId) {
        setMessages([]);
      }
    }
  }, [attachToActiveJob]);

  // Always load messages when activeSessionId changes or ChatView mounts
  useEffect(() => {
    loadSessionMessages(activeSessionId);
  }, [activeSessionId, loadSessionMessages]);

  // Listen for session updates (user turn persisted or stream completed) and new-chat-session
  useEffect(() => {
    const handleNew = () => setMessages([]);
    const handleSessionUpdated = (e: Event) => {
      const customEvt = e as CustomEvent<{ sessionId?: string }>;
      const updatedId = customEvt.detail?.sessionId;
      if (updatedId && updatedId === activeSessionIdRef.current) {
        loadSessionMessages(updatedId);
      }
    };

    window.addEventListener('new-chat-session', handleNew);
    window.addEventListener('chat-session-updated', handleSessionUpdated);
    return () => {
      window.removeEventListener('new-chat-session', handleNew);
      window.removeEventListener('chat-session-updated', handleSessionUpdated);
    };
  }, [loadSessionMessages]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingAnswer, streamingThinking, streamingStatus]);

  const executeSend = async (overrideQuery?: string) => {
    const query = (overrideQuery || input).trim();
    if (!query || isStreaming || sending) return;

    if (!overrideQuery) setInput('');
    setSending(true);

    // Optimistic user message for instant visual feedback before the round-trip completes
    setMessages(prev => [
      ...prev,
      {
        session_id: activeSessionId || 'temp',
        role: 'user',
        content: query,
      },
    ]);

    try {
      await sendChatQuery(query, activeSessionId);
    } catch (err) {
      console.error('Failed to start chat stream:', err);
    } finally {
      setSending(false);
    }
  };

  const handleSend = () => executeSend();

  const handleSuggestionClick = (promptText: string) => {
    setInput('');
    executeSend(promptText);
  };

  return (
    <div className="chat-container">
      <main className="chat-main">
        <div className="chat-messages-area">
          {!activeSessionId && (!Array.isArray(messages) || messages.length === 0) && !isStreaming ? (
            <div className="gemini-landing">
              <div className="gemini-hero">
                <h1 className="gemini-title">
                  <span className="gemini-gradient-text">Compliance</span> Assistant
                </h1>
                <p className="gemini-subtitle">
                  Ask anything about certificates, suppliers, authorities and expirations.
                </p>
              </div>

              <div className="gemini-suggestions-grid">
                <div 
                  className="gemini-card"
                  onClick={() => handleSuggestionClick("List all active ANATEL certificates from Brazil")}
                >
                  <div className="gemini-card-icon"><FileText size={22} color="#3b82f6" /></div>
                  <div className="gemini-card-text">List all active ANATEL certificates from Brazil</div>
                  <div className="gemini-card-arrow"><Sparkles size={15} /></div>
                </div>

                <div 
                  className="gemini-card"
                  onClick={() => handleSuggestionClick("Which certificates are expiring in the next 90 days?")}
                >
                  <div className="gemini-card-icon"><ShieldAlert size={22} color="#f59e0b" /></div>
                  <div className="gemini-card-text">Which certificates are expiring in the next 90 days?</div>
                  <div className="gemini-card-arrow"><Sparkles size={15} /></div>
                </div>

                <div 
                  className="gemini-card"
                  onClick={() => handleSuggestionClick("Find suppliers with missing FCC compliance docs")}
                >
                  <div className="gemini-card-icon"><Building2 size={22} color="#10b981" /></div>
                  <div className="gemini-card-text">Find suppliers with missing FCC compliance docs</div>
                  <div className="gemini-card-arrow"><Sparkles size={15} /></div>
                </div>

                <div 
                  className="gemini-card"
                  onClick={() => handleSuggestionClick("Summarize memory insights from recent audit logs")}
                >
                  <div className="gemini-card-icon"><BrainCircuit size={22} color="#8b5cf6" /></div>
                  <div className="gemini-card-text">Summarize memory insights from recent audit logs</div>
                  <div className="gemini-card-arrow"><Sparkles size={15} /></div>
                </div>
              </div>
            </div>
          ) : (
            <>
              {Array.isArray(messages) && messages.map((msg, i) => (
                <ChatMessage key={msg.id || i} message={msg} />
              ))}

              {isStreaming && !streamingAnswer && (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '0.45rem', color: 'var(--brand-blue)', marginTop: '0.5rem', marginBottom: '1rem' }}>
                  {streamingThinking && (
                    <div className="thinking-block" style={{ width: '100%' }}>
                      {streamingThinking}
                    </div>
                  )}
                  <span style={{ fontSize: '0.88rem', fontWeight: 500 }}>
                    {streamingStatus || 'Analyzing query & searching compliance records...'}
                  </span>
                  <div className="dot-pulse-container" style={{ marginTop: '0.1rem' }}>
                    <span className="dot-pulse"></span>
                    <span className="dot-pulse"></span>
                    <span className="dot-pulse"></span>
                  </div>
                </div>
              )}

              {isStreaming && streamingAnswer && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', width: '100%', marginTop: '0.5rem', marginBottom: '1rem' }}>
                  {streamingThinking && (
                    <div className="thinking-block">
                      {streamingThinking}
                    </div>
                  )}
                  <ChatMessage 
                    message={{ role: 'assistant', content: streamingAnswer, intent: streamingIntent, session_id: activeSessionId || 'temp' }} 
                  />
                </div>
              )}
            </>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="chat-input-area">
          <div className="chat-input-wrapper">
            <textarea 
              className="chat-input"
              placeholder={isStreaming ? "Agent is responding in this session (switch sessions to ask in parallel)..." : "Ask anything about certificates, suppliers, or regulations..."}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={isStreaming || sending}
              rows={1}
            />
            <button
              type="button"
              onClick={() => setDeepThinking(!deepThinking)}
              title={deepThinking ? "Deep Reasoning Enabled (click for fast mode)" : "Fast Mode Active (click to enable deep reasoning)"}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.3rem',
                padding: '0.32rem 0.6rem',
                borderRadius: '999px',
                fontSize: '0.72rem',
                fontWeight: 600,
                cursor: 'pointer',
                border: `1px solid ${deepThinking ? 'var(--brand-blue)' : 'var(--border-color)'}`,
                backgroundColor: deepThinking ? 'rgba(36, 56, 129, 0.12)' : 'transparent',
                color: deepThinking ? 'var(--brand-blue)' : 'var(--text-secondary)',
                whiteSpace: 'nowrap',
                transition: 'all 150ms ease',
              }}
            >
              <BrainCircuit size={14} />
              <span>Deep Think</span>
            </button>
            <button 
              className={`chat-circle-send-btn ${input.trim() && !isStreaming && !sending ? 'active' : ''}`}
              onClick={handleSend}
              disabled={isStreaming || sending || !input.trim()}
              title="Send message"
            >
              <ArrowUp size={18} />
            </button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ChatView;
