import { useState, useEffect, useRef } from 'react';
import { FileText, ShieldAlert, Building2, BrainCircuit, Sparkles, ArrowUp } from 'lucide-react';
import { api, type ChatMessage as ChatMessageType } from '../api';
import { useLayoutContext } from '../components/Layout/AppLayout';
import { ChatMessage } from '../components/Chat/ChatMessage';
import './ChatView.css';

const ChatView = () => {
  const { activeSessionId, setActiveSessionId } = useLayoutContext();
  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [input, setInput] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Streaming state
  const [streamingAnswer, setStreamingAnswer] = useState('');
  const [streamingThinking, setStreamingThinking] = useState('');
  const [streamingStatus, setStreamingStatus] = useState('');
  const [streamingIntent, setStreamingIntent] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Sync messages when activeSessionId changes (guarded against overwriting active stream)
  useEffect(() => {
    if (isStreaming) return;
    if (activeSessionId) {
      api.getSessionMessages(activeSessionId).then(data => {
        setMessages(Array.isArray(data) ? data : []);
      }).catch(() => setMessages([]));
    } else {
      setMessages([]);
    }
  }, [activeSessionId, isStreaming]);

  // Listen for new-chat-session event from Sidebar
  useEffect(() => {
    const handleNew = () => setMessages([]);
    window.addEventListener('new-chat-session', handleNew);
    return () => window.removeEventListener('new-chat-session', handleNew);
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streamingAnswer, streamingThinking, streamingStatus]);

  const executeSend = async (overrideQuery?: string) => {
    const query = (overrideQuery || input).trim();
    if (!query || isStreaming) return;

    if (!overrideQuery) setInput('');

    try {
      const busyCheck = await api.checkAgentBusy();
      if (busyCheck && busyCheck.is_busy) {
        setMessages(prev => [
          ...prev,
          { session_id: activeSessionId || 'temp', role: 'user', content: query },
          { session_id: activeSessionId || 'temp', role: 'assistant', content: 'The agent is currently processing an ongoing background task. Please wait until the task completes before sending a new query.' }
        ]);
        return;
      }
    } catch (e) {
      console.warn('Error checking agent busy status', e);
    }
    
    // Optimistic user message
    const optimisticUserMsg: ChatMessageType = {
      session_id: activeSessionId || 'temp',
      role: 'user',
      content: query,
    };
    setMessages(prev => [...prev, optimisticUserMsg]);
    setIsStreaming(true);
    setStreamingAnswer('');
    setStreamingThinking('');
    setStreamingStatus('Analyzing query...');
    setStreamingIntent('');

    try {
      const res = await api.startChatStream(query, activeSessionId || undefined);
      
      if (!activeSessionId && res.session_id) {
        setActiveSessionId(res.session_id);
        window.dispatchEvent(new Event('refresh-chat-sessions'));
      }

      if (res.frozen || !res.job_id) {
        setMessages(prev => [...prev, {
          session_id: res.session_id,
          role: 'assistant',
          content: res.answer || 'Context window full or missing job_id.',
          intent: res.intent
        }]);
        setIsStreaming(false);
        return;
      }

      // Start SSE onmessage stream handler
      const eventSource = new EventSource(`/api/v1/chat/stream/${res.job_id}`);

      eventSource.onmessage = async (e) => {
        try {
          const data = JSON.parse(e.data);
          
          if (data.type === 'status') {
            const statusMsg = data.message || data.status || 'Analyzing compliance records...';
            setStreamingStatus(statusMsg);
            if (data.intent) setStreamingIntent(data.intent);
          } else if (data.type === 'thinking') {
            const thinkChunk = data.text || data.chunk || '';
            setStreamingThinking(prev => prev + thinkChunk);
          } else if (data.type === 'token') {
            const tokenChunk = data.text || data.chunk || '';
            setStreamingAnswer(prev => prev + tokenChunk);
            setStreamingStatus('');
          } else if (data.type === 'done') {
            eventSource.close();
            const finalAnswer = data.answer || streamingAnswer;
            const finalIntent = data.intent || streamingIntent;
            setMessages(prev => [
              ...prev,
              { role: 'assistant', content: finalAnswer, intent: finalIntent, session_id: res.session_id }
            ]);
            setIsStreaming(false);
            setStreamingAnswer('');
            setStreamingThinking('');
            setStreamingStatus('');
            setStreamingIntent('');
            window.dispatchEvent(new Event('refresh-chat-sessions'));
          } else if (data.type === 'error') {
            eventSource.close();
            setIsStreaming(false);
            setStreamingStatus(`Error: ${data.message || 'Stream failed'}`);
          }
        } catch (err) {
          console.error('Error parsing SSE event payload', err);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setIsStreaming(false);
      };

    } catch (err) {
      console.error(err);
      setIsStreaming(false);
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
          {!activeSessionId && (!Array.isArray(messages) || messages.length === 0) ? (
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
                    message={{ role: 'assistant', content: streamingAnswer, intent: streamingIntent, session_id: 'temp' }} 
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
              placeholder="Ask anything about certificates, suppliers, or regulations..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={isStreaming}
              rows={1}
            />
            <button 
              className={`chat-circle-send-btn ${input.trim() && !isStreaming ? 'active' : ''}`}
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
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
