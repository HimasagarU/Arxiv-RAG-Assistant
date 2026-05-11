import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { getMessages, sendQuery, listConversations, createConversation, deleteConversation, getSimilarPapers } from '../api';
import ThemeToggle from '../components/ThemeToggle';

const MAX_QUERIES = 20;

export default function ChatView() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const [messages, setMessages] = useState([]);
  const [conversations, setConversations] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [expandedSource, setExpandedSource] = useState(null);
  const [similarPapers, setSimilarPapers] = useState({});
  const [loadingSimilar, setLoadingSimilar] = useState({});
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const queryCount = Math.floor(messages.filter((m) => m.role === 'user').length);

  useEffect(() => {
    async function fetchConversations() {
      try {
        const convs = await listConversations();
        setConversations(convs);
      } catch (err) {
        console.error('Failed to load conversations', err);
      }
    }

    fetchConversations();
  }, []);

  useEffect(() => {
    async function fetchMessages() {
      setLoading(true);
      setError('');
      try {
        const msgs = await getMessages(conversationId);
        setMessages(msgs);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    fetchMessages();
  }, [conversationId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend(e) {
    e.preventDefault();
    if (!query.trim() || sending) return;

    const userMsg = { role: 'user', content: query, id: `temp-${Date.now()}` };
    setMessages((prev) => [...prev, userMsg]);
    const currentQuery = query;
    setQuery('');
    setSending(true);
    setError('');

    try {
      const res = await sendQuery(conversationId, currentQuery);
      const assistantMsg = {
        role: 'assistant',
        content: res.answer,
        sources_json: JSON.stringify(res.sources),
        id: `res-${Date.now()}`,
      };
      setMessages((prev) => [...prev, assistantMsg]);
      const convs = await listConversations();
      setConversations(convs);
    } catch (err) {
      setError(err.message);
      // Remove the optimistic user message on failure
      setMessages((prev) => prev.filter((m) => m.id !== userMsg.id));
      setQuery(currentQuery);
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  async function handleNewChat() {
    try {
      const conv = await createConversation();
      const convs = await listConversations();
      setConversations(convs);
      navigate(`/chat/${conv.id}`);
    } catch (err) {
      console.error('Failed to create conversation', err);
    }
  }

  async function handleDeleteConv(id) {
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
      if (id === conversationId) navigate('/');
    } catch (err) {
      console.error('Failed to delete conversation', err);
    }
  }

  function parseSources(sourcesJson) {
    if (!sourcesJson) return [];
    try {
      return JSON.parse(sourcesJson);
    } catch {
      return [];
    }
  }

  async function handleLoadSimilar(paperId, key) {
    if (similarPapers[key] !== undefined) return; // already loaded or loading
    setLoadingSimilar((prev) => ({ ...prev, [key]: true }));
    try {
      const data = await getSimilarPapers(paperId);
      setSimilarPapers((prev) => ({ ...prev, [key]: data.similar_papers || [] }));
    } catch (err) {
      console.error(err);
      setSimilarPapers((prev) => ({ ...prev, [key]: null }));
    } finally {
      setLoadingSimilar((prev) => ({ ...prev, [key]: false }));
    }
  }

  function renderMarkdown(text) {
    // Simple markdown rendering
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br/>');
  }

  return (
    <div className="h-screen flex overflow-hidden bg-[var(--color-bg-primary)] text-[var(--color-text-primary)]">
      {/* Sidebar */}
      <aside
        className="flex-shrink-0 border-r flex flex-col transition-all duration-300 bg-[var(--color-bg-secondary)]"
        style={{
          width: sidebarOpen ? '280px' : '0px',
          borderColor: 'var(--color-border)',
          overflow: 'hidden',
        }}
      >
        <div className="p-4 border-b" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-3">
            <Link to="/dashboard" className="min-w-0 flex-1 text-lg font-bold leading-none" style={{ fontFamily: 'var(--font-heading)', color: 'var(--color-text-primary)' }}>
              ArXiv RAG
            </Link>
            <button
              id="sidebar-new-chat"
              onClick={handleNewChat}
              className="btn-primary shrink-0 px-3 py-2 text-sm"
            >
              + New Conversation
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {conversations.map((conv) => (
            <div
              key={conv.id}
              className={`p-3 rounded-xl cursor-pointer flex items-start justify-between gap-3 group transition-colors border ${conv.id === conversationId ? 'bg-[var(--color-bg-hover)] border-[var(--color-border)]' : 'bg-transparent border-transparent hover:bg-[var(--color-bg-hover)] hover:border-[var(--color-border)]'}`}
              style={{
                color: conv.id === conversationId ? 'var(--color-text-primary)' : 'var(--color-text-secondary)',
              }}
              onClick={() => navigate(`/chat/${conv.id}`)}
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate leading-snug">{conv.title}</p>
                <p className="text-xs mt-0.5" style={{ color: 'var(--color-text-muted)' }}>
                  {conv.message_count} msgs
                  {conv.paper_id && ` · 📄`}
                </p>
              </div>
              <button
                onClick={(e) => { e.stopPropagation(); handleDeleteConv(conv.id); }}
                className="icon-btn opacity-0 group-hover:opacity-100 transition-all text-xs"
                style={{ color: 'var(--color-error)' }}
                title="Delete conversation"
                aria-label="Delete conversation"
              >
                <span className="text-base leading-none">×</span>
              </button>
            </div>
          ))}
        </div>

        <div className="p-4 border-t" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-3">
            <div className="min-w-0 flex-1 rounded-xl border px-3 py-2" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-card)' }}>
              <p className="text-[11px] uppercase tracking-[0.18em]" style={{ color: 'var(--color-text-muted)' }}>
                Signed in as
              </p>
              <p className="truncate text-sm font-medium" style={{ color: 'var(--color-text-primary)' }}>
                {user?.display_name}
              </p>
            </div>
            <button onClick={logout} className="btn-soft shrink-0 px-3 py-2 text-sm">
              Sign out
            </button>
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top bar */}
        <div className="flex items-center gap-3 px-4 py-3 border-b bg-[var(--color-bg-card)]/90 backdrop-blur"
             style={{ borderColor: 'var(--color-border)' }}>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="icon-btn text-lg" aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}>
            {sidebarOpen ? '◀' : '▶'}
          </button>
          
          <button onClick={() => navigate('/dashboard')} className="btn-ghost text-sm flex items-center gap-1">
            <span>←</span> Back to Dashboard
          </button>
          
          <div className="flex-1" />
          <ThemeToggle />
          {/* Query counter */}
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-24 rounded-full overflow-hidden" style={{ background: 'var(--color-border)' }}>
              <div
                className="h-full rounded-full transition-all duration-300"
                style={{
                  width: `${Math.min((queryCount / MAX_QUERIES) * 100, 100)}%`,
                  background: queryCount >= MAX_QUERIES
                    ? 'var(--color-error)'
                    : queryCount >= MAX_QUERIES * 0.8
                      ? 'var(--color-warning)'
                      : 'var(--color-accent)',
                }}
              />
            </div>
            <span className="text-xs whitespace-nowrap" style={{ color: 'var(--color-text-muted)' }}>
              {queryCount}/{MAX_QUERIES} queries
            </span>
          </div>
          <Link to="/how-it-works" className="btn-ghost text-xs px-3 py-2" style={{ color: 'var(--color-text-secondary)' }}>
            How It Works
          </Link>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          {loading ? (
            <div className="text-center py-12">
              <div className="w-8 h-8 border-2 border-t-transparent rounded-full mx-auto mb-3"
                   style={{ borderColor: 'var(--color-accent)', borderTopColor: 'transparent', animation: 'spin-slow 1s linear infinite' }} />
            </div>
          ) : messages.length === 0 ? (
            <div className="text-center py-20 animate-fade-in">
              <p className="text-5xl mb-4">🔬</p>
              <h2 className="text-2xl font-bold mb-2" style={{ fontFamily: 'var(--font-heading)' }}>
                Start Researching
              </h2>
              <p className="max-w-md mx-auto" style={{ color: 'var(--color-text-secondary)' }}>
                Ask questions about AI research papers. Your queries will be answered
                using a hybrid retrieval pipeline with source citations.
              </p>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((msg, i) => (
                <div
                  key={msg.id || i}
                  className={`animate-fade-in ${msg.role === 'user' ? 'flex justify-end' : ''}`}
                  style={{ animationDelay: `${i * 30}ms` }}
                >
                  {msg.role === 'user' ? (
                    <div className="max-w-lg px-4 py-3 rounded-2xl rounded-br-sm"
                         style={{ background: 'linear-gradient(135deg, #6366f1, #7c3aed)', color: '#fff' }}>
                      <p className="text-sm leading-relaxed">{msg.content}</p>
                    </div>
                  ) : (
                    <div className="w-full">
                      <div className="glass-card p-5">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                               style={{ background: 'var(--color-accent-glow)', color: 'var(--color-accent)' }}>
                            AI
                          </div>
                          <span className="text-xs font-medium" style={{ color: 'var(--color-text-muted)' }}>
                            Research Assistant
                          </span>
                        </div>
                        <div
                          className="chat-markdown text-sm leading-relaxed"
                          style={{ color: 'var(--color-text-primary)' }}
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
                        />

                        {/* Sources */}
                        {(() => {
                          const sources = parseSources(msg.sources_json);
                          if (sources.length === 0) return null;
                          return (
                            <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--color-border)' }}>
                              <p className="text-xs font-medium mb-2" style={{ color: 'var(--color-text-muted)' }}>
                                📚 Sources ({sources.length})
                              </p>
                              <div className="space-y-2">
                                {sources.map((src, si) => (
                                  <div key={si}>
                                    <button
                                      className="text-xs text-left w-full p-2 rounded-lg transition-colors border"
                                      style={{
                                        background: expandedSource === `${i}-${si}` ? 'var(--color-bg-hover)' : 'transparent',
                                        color: 'var(--color-accent)',
                                        borderColor: 'transparent',
                                      }}
                                      onClick={() => setExpandedSource(expandedSource === `${i}-${si}` ? null : `${i}-${si}`)}
                                    >
                                      [{si + 1}] {src.title}
                                      {src.rerank_score > 0 && (
                                        <span className="ml-2" style={{ color: 'var(--color-text-muted)' }}>
                                          (score: {src.rerank_score.toFixed(3)})
                                        </span>
                                      )}
                                    </button>
                                    {expandedSource === `${i}-${si}` && (
                                      <div className="ml-4 mt-1 p-3 rounded-lg text-xs animate-fade-in"
                                           style={{ background: 'var(--color-bg-primary)', color: 'var(--color-text-secondary)', border: '1px solid var(--color-border)' }}>
                                        <div className="flex justify-between items-start mb-2">
                                          <div>
                                            <p className="mb-1"><strong>Paper:</strong> {src.paper_id}</p>
                                            {src.authors && <p className="mb-1"><strong>Authors:</strong> {src.authors}</p>}
                                          </div>
                                          {src.paper_id && (
                                            <a href={`https://arxiv.org/abs/${src.paper_id}`} target="_blank" rel="noopener noreferrer"
                                               className="text-xs hover:underline flex items-center gap-1" style={{ color: 'var(--color-accent)' }}>
                                              View on ArXiv ↗
                                            </a>
                                          )}
                                        </div>
                                        <p className="mt-2 leading-relaxed" style={{ color: 'var(--color-text-secondary)' }}>
                                          {src.chunk_text?.slice(0, 500)}
                                          {src.chunk_text?.length > 500 && '...'}
                                        </p>
                                        
                                        {/* Similar Papers */}
                                        {src.paper_id && (
                                          <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--color-border)' }}>
                                            {!similarPapers[`${i}-${si}`] && !loadingSimilar[`${i}-${si}`] ? (
                                              <button onClick={() => handleLoadSimilar(src.paper_id, `${i}-${si}`)}
                                                      className="text-[11px] px-2 py-1 rounded transition-colors"
                                                      style={{ background: 'var(--color-warning)', color: '#fff' }}>
                                                Find Similar Papers
                                              </button>
                                            ) : loadingSimilar[`${i}-${si}`] ? (
                                              <span className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>Loading...</span>
                                            ) : similarPapers[`${i}-${si}`] === null ? (
                                              <span className="text-[11px]" style={{ color: 'var(--color-error)' }}>Failed to load</span>
                                            ) : (
                                              <div className="space-y-1">
                                                <p className="text-[11px] font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>Similar Papers:</p>
                                                {similarPapers[`${i}-${si}`].map((p, idx) => (
                                                  <div key={idx} className="flex justify-between items-center bg-[var(--color-bg-secondary)] p-1.5 rounded">
                                                     <a href={`https://arxiv.org/abs/${p.paper_id}`} target="_blank" rel="noopener noreferrer"
                                                       className="truncate flex-1 text-[11px] hover:underline" style={{ color: 'var(--color-accent)' }}>
                                                      {p.title}
                                                    </a>
                                                    <span className="text-[10px] ml-2" style={{ color: 'var(--color-text-muted)' }}>
                                                      {(p.similarity_score * 100).toFixed(1)}%
                                                    </span>
                                                  </div>
                                                ))}
                                              </div>
                                            )}
                                          </div>
                                        )}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            </div>
                          );
                        })()}
                      </div>
                    </div>
                  )}
                </div>
              ))}

              {sending && (
                <div className="animate-fade-in">
                  <div className="glass-card p-5">
                    <div className="flex items-center gap-2">
                      <div className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold"
                           style={{ background: 'var(--color-accent-glow)', color: 'var(--color-accent)' }}>
                        AI
                      </div>
                      <div className="flex gap-1">
                        <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-accent)', animation: 'pulse-glow 1s ease-in-out infinite' }} />
                        <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-accent)', animation: 'pulse-glow 1s ease-in-out 0.2s infinite' }} />
                        <div className="w-2 h-2 rounded-full" style={{ background: 'var(--color-accent)', animation: 'pulse-glow 1s ease-in-out 0.4s infinite' }} />
                      </div>
                      <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>Searching papers & generating answer...</span>
                    </div>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t px-4 py-4 bg-[var(--color-bg-card)]/90 backdrop-blur" style={{ borderColor: 'var(--color-border)' }}>
          {error && (
            <div className="max-w-3xl mx-auto mb-3 p-3 rounded-lg text-sm"
                 style={{ background: 'rgba(248, 113, 113, 0.1)', color: 'var(--color-error)', border: '1px solid rgba(248, 113, 113, 0.2)' }}>
              {error}
            </div>
          )}
          <form onSubmit={handleSend} className="max-w-3xl mx-auto flex gap-3">
            <input
              ref={inputRef}
              id="chat-input"
              type="text"
              className="input-field flex-1"
              placeholder={queryCount >= MAX_QUERIES ? 'Query limit reached — start a new conversation' : 'Ask about AI research papers...'}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={sending || queryCount >= MAX_QUERIES}
              autoFocus
            />
            <button
              id="chat-send"
              type="submit"
              className="btn-primary"
              disabled={sending || !query.trim() || queryCount >= MAX_QUERIES}
            >
              {sending ? '...' : 'Send'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
