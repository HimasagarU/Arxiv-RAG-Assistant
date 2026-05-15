import { useState, useEffect, useRef, memo } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import {
  getMessages, sendQuery, streamConversationQuery,
  listConversations, createConversation, deleteConversation, getSimilarPapers,
} from '../api';
import ThemeToggle from '../components/ThemeToggle';
import { renderMarkdown } from '../utils/renderMarkdown';

const MAX_QUERIES = 20;

// ── Memoized message components ─────────────────────────────

const UserMessage = memo(function UserMessage({ content }) {
  return (
    <div className="max-w-lg px-4 py-3 rounded-2xl rounded-br-sm"
         style={{ background: 'linear-gradient(135deg, #6366f1, #7c3aed)', color: '#fff' }}>
      <p className="text-sm leading-relaxed">{content}</p>
    </div>
  );
});

const SourceItem = memo(function SourceItem({ src, id, expanded, onToggle }) {
  const [similarPapers, setSimilarPapers] = useState(undefined);
  const [loadingSimilar, setLoadingSimilar] = useState(false);

  async function handleLoadSimilar() {
    if (similarPapers !== undefined) return;
    setLoadingSimilar(true);
    try {
      const data = await getSimilarPapers(src.paper_id);
      setSimilarPapers(data.similar_papers || []);
    } catch {
      setSimilarPapers(null);
    } finally {
      setLoadingSimilar(false);
    }
  }

  return (
    <div>
      <button
        className="text-xs text-left w-full p-2 rounded-lg transition-colors border"
        style={{
          background: expanded ? 'var(--color-bg-hover)' : 'transparent',
          color: 'var(--color-accent)',
          borderColor: 'transparent',
        }}
        onClick={onToggle}
      >
        [{id}] {src.title}
        {src.rerank_score !== 0 && (
          <span className="ml-2" style={{ color: 'var(--color-text-muted)' }}>
            (reranker score: {src.rerank_score.toFixed(3)})
          </span>
        )}
      </button>
      {expanded && (
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
            {src.chunk_text?.length > 500 && '…'}
          </p>

          {/* Similar Papers */}
          {src.paper_id && (
            <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--color-border)' }}>
              {similarPapers === undefined && !loadingSimilar ? (
                <button onClick={handleLoadSimilar}
                        className="text-[11px] px-2 py-1 rounded transition-colors"
                        style={{ background: 'var(--color-warning)', color: '#fff' }}>
                  Find Similar Papers
                </button>
              ) : loadingSimilar ? (
                <span className="text-[11px]" style={{ color: 'var(--color-text-muted)' }}>Loading…</span>
              ) : similarPapers === null ? (
                <span className="text-[11px]" style={{ color: 'var(--color-error)' }}>Failed to load</span>
              ) : (
                <div className="space-y-1">
                  <p className="text-[11px] font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>Similar Papers:</p>
                  {similarPapers.map((p, idx) => (
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
  );
});

const AssistantMessage = memo(function AssistantMessage({ msg, msgIdx, streaming }) {
  const [expandedSource, setExpandedSource] = useState(null);

  function parseSources(sourcesJson) {
    if (!sourcesJson) return [];
    try { return JSON.parse(sourcesJson); } catch { return []; }
  }

  const sources = parseSources(msg.sources_json);

  return (
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
          {streaming && (
            <span className="inline-block w-2 h-3 rounded-sm animate-pulse ml-1"
                  style={{ background: 'var(--color-accent)' }} />
          )}
        </div>

        {msg.streaming && (
          <div className="mb-4 space-y-2">
            <div className="flex items-center justify-between text-[10px] uppercase tracking-[0.16em]" style={{ color: 'var(--color-text-muted)' }}>
              <span>Live backend stages</span>
              <span>{msg.currentStage || 'Waiting for first stage…'}</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {(msg.progressStages || []).length === 0 ? (
                <div className="flex items-center gap-2 px-2 py-1 rounded-md text-[10px] font-medium border" style={{ color: 'var(--color-text-muted)', borderColor: 'var(--color-border)' }}>
                  <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-border)]" />
                  Waiting for retrieval progress…
                </div>
              ) : (
                (msg.progressStages || []).map((stage, idx) => {
                  const currentIdx = (msg.progressStages || []).lastIndexOf(msg.currentStage);
                  const isActive = stage === msg.currentStage;
                  const isDone = currentIdx !== -1 && idx < currentIdx;

                  return (
                    <div
                      key={`${stage}-${idx}`}
                      className="flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-medium transition-all border"
                      style={{
                        background: isDone ? 'rgba(16, 185, 129, 0.1)' : isActive ? 'var(--color-bg-hover)' : 'transparent',
                        color: isDone ? 'var(--color-success)' : isActive ? 'var(--color-accent)' : 'var(--color-text-muted)',
                        borderColor: isDone ? 'rgba(16, 185, 129, 0.2)' : isActive ? 'var(--color-accent)' : 'var(--color-border)',
                        opacity: isDone || isActive ? 1 : 0.7,
                      }}
                    >
                      <div className={`w-1.5 h-1.5 rounded-full ${isDone ? 'bg-[var(--color-success)]' : isActive ? 'bg-[var(--color-accent)] animate-pulse' : 'bg-[var(--color-text-muted)]'}`} />
                      {stage}
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {msg.retrievalStatus === 'done' && !msg.streaming && (
          <div className="mb-3 px-3 py-1.5 rounded-lg text-[10px] inline-flex items-center gap-2"
               style={{ background: 'rgba(16, 185, 129, 0.05)', color: 'var(--color-success)', border: '1px solid rgba(16, 185, 129, 0.1)' }}>
            <div className="w-1.5 h-1.5 rounded-full bg-[var(--color-success)]" />
            Analysis complete · {msg.retrievalChunks ?? ''} evidence blocks processed
          </div>
        )}

        <div
          className="chat-markdown text-sm leading-relaxed"
          style={{ color: 'var(--color-text-primary)' }}
          dangerouslySetInnerHTML={{ __html: renderMarkdown(msg.content) }}
        />

        {/* Sources */}
        {sources.length > 0 && (
          <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--color-border)' }}>
            <p className="text-xs font-medium mb-2" style={{ color: 'var(--color-text-muted)' }}>
              📚 Sources ({sources.length})
            </p>
            <div className="space-y-2">
              {sources.map((src, si) => (
                <SourceItem
                  key={si}
                  src={src}
                  id={si + 1}
                  expanded={expandedSource === si}
                  onToggle={() => setExpandedSource(expandedSource === si ? null : si)}
                />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

// ── Main component ───────────────────────────────────────────

export default function ChatView() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const [messages, setMessages]         = useState([]);
  const [conversations, setConversations] = useState([]);
  const [query, setQuery]               = useState('');
  const [loading, setLoading]           = useState(false);
  const [sending, setSending]           = useState(false);
  const [error, setError]               = useState('');
  const [sidebarOpen, setSidebarOpen]   = useState(true);
  const [topK, setTopK]                 = useState('10');

  const messagesEndRef = useRef(null);
  const inputRef       = useRef(null);
  // Stable ID for the currently streaming assistant message
  const streamingMsgId = useRef(null);

  const queryCount = Math.floor(messages.filter((m) => m.role === 'user').length);

  useEffect(() => {
    listConversations().then(setConversations).catch(console.error);
  }, []);

  useEffect(() => {
    setLoading(true);
    setError('');
    getMessages(conversationId)
      .then(setMessages)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [conversationId]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  async function handleSend(e) {
    e.preventDefault();
    if (!query.trim() || sending) return;

    const currentQuery = query;
    const userMsgId    = `user-${Date.now()}`;
    const asstMsgId    = `asst-${Date.now()}`;
    streamingMsgId.current = asstMsgId;

    // Optimistic user message
    setMessages((prev) => [...prev, { role: 'user', content: currentQuery, id: userMsgId }]);
    setQuery('');
    setSending(true);
    setError('');

    // Optimistic assistant placeholder
    setMessages((prev) => [...prev, {
      role: 'assistant',
      content: '',
      sources_json: null,
      id: asstMsgId,
      retrievalStatus: 'searching',
      currentStage: '',
      progressStages: [],
      retrievalChunks: null,
      streaming: true,
    }]);

    let firstToken = false;

    await streamConversationQuery(conversationId, currentQuery, {
      topK: parseInt(topK, 10),

      onRetrievalStart: () => {
        setMessages((prev) => prev.map((m) =>
          m.id === asstMsgId ? { ...m, retrievalStatus: 'searching' } : m
        ));
      },

      onRetrievalDone: (payload) => {
        setMessages((prev) => prev.map((m) =>
          m.id === asstMsgId
            ? { ...m, retrievalStatus: 'done', retrievalChunks: payload.num_chunks, currentStage: null }
            : m
        ));
      },

      onStatus: (stage) => {
        setMessages((prev) => prev.map((m) =>
          m.id === asstMsgId ? {
            ...m,
            currentStage: stage,
            progressStages: m.progressStages?.includes(stage) ? m.progressStages : [...(m.progressStages || []), stage],
          } : m
        ));
      },

      onToken: (token) => {
        firstToken = true;
        setMessages((prev) => prev.map((m) =>
          m.id === asstMsgId ? { ...m, content: m.content + token } : m
        ));
      },

      onError: (msg) => {
        setError(msg);
        if (!firstToken) {
          // Remove optimistic messages on pre-token failure
          setMessages((prev) => prev.filter((m) => m.id !== userMsgId && m.id !== asstMsgId));
          setQuery(currentQuery);
        } else {
          // Mark streaming done even on error
          setMessages((prev) => prev.map((m) =>
            m.id === asstMsgId ? { ...m, streaming: false } : m
          ));
        }
      },

      onDone: (payload) => {
        const sources     = payload.sources || [];
        const sourcesJson = JSON.stringify(sources);
        setMessages((prev) => prev.map((m) =>
          m.id === asstMsgId
            ? { ...m, sources_json: sourcesJson, streaming: false, retrievalStatus: 'done', currentStage: null }
            : m
        ));
        // Refresh conversation list (title auto-set after first message)
        listConversations().then(setConversations).catch(() => {});
      },
    });

    setSending(false);
    inputRef.current?.focus();
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
      if (id === conversationId) navigate('/dashboard');
    } catch (err) {
      console.error('Failed to delete conversation', err);
    }
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
        <div className="px-4 h-14 flex items-center border-b" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-3 w-full">
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
              style={{ color: conv.id === conversationId ? 'var(--color-text-primary)' : 'var(--color-text-secondary)' }}
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

        <div className="px-4 h-20 flex items-center border-t" style={{ borderColor: 'var(--color-border)' }}>
          <div className="flex items-center gap-3 w-full">
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
        <div className="flex items-center gap-3 px-4 h-14 border-b bg-[var(--color-bg-card)]/90 backdrop-blur"
             style={{ borderColor: 'var(--color-border)' }}>
          <button onClick={() => setSidebarOpen(!sidebarOpen)} className="icon-btn text-lg" aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}>
            {sidebarOpen ? '◀' : '▶'}
          </button>

          <button onClick={() => navigate('/dashboard')} className="btn-soft text-sm flex items-center gap-1">
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
                  style={{ animationDelay: `${Math.min(i, 10) * 30}ms` }}
                >
                  {msg.role === 'user' ? (
                    <UserMessage content={msg.content} />
                  ) : (
                    <AssistantMessage
                      msg={msg}
                      msgIdx={i}
                      streaming={msg.streaming}
                    />
                  )}
                </div>
              ))}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t px-4 min-h-[5rem] flex items-center bg-[var(--color-bg-card)]/90 backdrop-blur" style={{ borderColor: 'var(--color-border)' }}>
          <div className="w-full">
            {error && (
              <div className="max-w-3xl mx-auto mb-3 p-3 rounded-lg text-sm"
                   style={{ background: 'rgba(248, 113, 113, 0.1)', color: 'var(--color-error)', border: '1px solid rgba(248, 113, 113, 0.2)' }}>
                {error}
              </div>
            )}
            <form onSubmit={handleSend} className="max-w-3xl mx-auto flex gap-3">
              <select
                className="bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded-lg px-2 py-2 text-sm outline-none focus:border-[var(--color-accent)]"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
                disabled={sending || loading}
                title="Chunks to retrieve"
              >
                <option value="5">5 Chunks</option>
                <option value="10">10 Chunks</option>
              </select>
              <input
                ref={inputRef}
                id="chat-input"
                type="text"
                className="input-field flex-1"
                placeholder={queryCount >= MAX_QUERIES ? 'Query limit reached — start a new conversation' : 'Ask about AI research papers…'}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                disabled={sending || loading || queryCount >= MAX_QUERIES}
                autoFocus
              />
              <button
                id="chat-send"
                type="submit"
                className="btn-primary"
                disabled={sending || loading || !query.trim() || queryCount >= MAX_QUERIES}
              >
                {sending ? '…' : 'Send'}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
