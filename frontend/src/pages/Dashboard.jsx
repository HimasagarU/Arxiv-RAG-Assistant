import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { listConversations, createConversation, deleteConversation, listDocuments, addDocument, getDocumentStatus } from '../api';
import ThemeToggle from '../components/ThemeToggle';
import { PageHeader, PageShell } from '../components/PageShell';

export default function Dashboard() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [conversations, setConversations] = useState([]);
  const [documents, setDocuments] = useState([]);
  const [showAddDoc, setShowAddDoc] = useState(false);
  const [arxivId, setArxivId] = useState('');
  const [addingDoc, setAddingDoc] = useState(false);
  const [addError, setAddError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadData();
  }, []);

  async function loadData() {
    setLoading(true);
    try {
      const [convs, docs] = await Promise.all([
        listConversations().catch(() => []),
        listDocuments().catch(() => []),
      ]);
      setConversations(convs);
      setDocuments(docs);
    } finally {
      setLoading(false);
    }
  }

  async function handleNewChat() {
    try {
      const conv = await createConversation();
      navigate(`/chat/${conv.id}`);
    } catch (err) {
      console.error('Failed to create conversation', err);
    }
  }

  async function handleDeleteConv(id) {
    try {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
    } catch (err) {
      console.error('Failed to delete', err);
    }
  }

  async function handleAddDocument(e) {
    e.preventDefault();
    setAddError('');
    setAddingDoc(true);
    try {
      const job = await addDocument(arxivId.trim());
      setDocuments((prev) => [job, ...prev]);
      setArxivId('');
      setShowAddDoc(false);
      // Start polling
      pollJobStatus(job.id);
    } catch (err) {
      setAddError(err.message);
    } finally {
      setAddingDoc(false);
    }
  }

  function pollJobStatus(jobId) {
    const interval = setInterval(async () => {
      try {
        const status = await getDocumentStatus(jobId);
        setDocuments((prev) =>
          prev.map((d) => (d.id === jobId ? status : d))
        );
        if (['done', 'failed'].includes(status.status)) {
          clearInterval(interval);
        }
      } catch {
        clearInterval(interval);
      }
    }, 3000);
  }

  // Resume polling for in-progress docs on load
  useEffect(() => {
    documents.forEach((doc) => {
      if (!['done', 'failed'].includes(doc.status)) {
        pollJobStatus(doc.id);
      }
    });
  }, [documents.length]);

  const statusConfig = {
    queued: { label: 'Queued', color: 'var(--color-text-muted)', icon: '⏳' },
    downloading: { label: 'Downloading PDF...', color: 'var(--color-warning)', icon: '📥' },
    chunking: { label: 'Extracting & Chunking...', color: 'var(--color-warning)', icon: '✂️' },
    embedding: { label: 'Embedding Vectors...', color: 'var(--color-accent)', icon: '🧬' },
    done: { label: 'Ready', color: 'var(--color-success)', icon: '✅' },
    failed: { label: 'Failed', color: 'var(--color-error)', icon: '❌' },
  };

  return (
    <PageShell>
      <PageHeader
        eyebrow="Workspace"
        title="ArXiv RAG Assistant"
        subtitle="Manage conversations, add papers, and keep your research workflow in one place."
        actions={(
          <>
            <Link to="/how-it-works" className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              How It Works
            </Link>
            <span className="text-sm" style={{ color: 'var(--color-text-secondary)' }}>
              {user?.display_name}
            </span>
            <ThemeToggle />
            <button onClick={logout} className="btn-ghost text-xs">
              Sign Out
            </button>
          </>
        )}
      />

      <main className="mx-auto max-w-6xl px-4 sm:px-6 py-8">
        {/* Quick Actions */}
        <div className="flex gap-4 mb-8">
          <button id="new-chat-btn" onClick={handleNewChat} className="btn-primary flex items-center gap-2">
            <span>+</span> New Conversation
          </button>
          <button id="add-doc-btn" onClick={() => setShowAddDoc(true)} className="btn-ghost flex items-center gap-2">
            <span>📄</span> Add Document
          </button>
        </div>

        {/* Add Document Modal */}
        {showAddDoc && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4"
               style={{ background: 'rgba(0,0,0,0.6)' }}>
            <div className="glass-card p-6 w-full max-w-md animate-fade-in">
              <h3 className="text-lg font-bold mb-4" style={{ fontFamily: 'var(--font-heading)' }}>
                Add ArXiv Paper
              </h3>
              <form onSubmit={handleAddDocument} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1.5"
                         style={{ color: 'var(--color-text-secondary)' }}>
                    ArXiv Paper ID
                  </label>
                  <input
                    id="arxiv-id-input"
                    type="text"
                    className="input-field"
                    placeholder="e.g., 2301.12345"
                    value={arxivId}
                    onChange={(e) => setArxivId(e.target.value)}
                    required
                  />
                  <p className="text-xs mt-1" style={{ color: 'var(--color-text-muted)' }}>
                    The paper will be downloaded, chunked, and embedded for you to chat with.
                  </p>
                </div>
                {addError && (
                  <p className="text-sm" style={{ color: 'var(--color-error)' }}>{addError}</p>
                )}
                <div className="flex gap-3 justify-end">
                  <button type="button" onClick={() => setShowAddDoc(false)} className="btn-ghost">
                    Cancel
                  </button>
                  <button type="submit" className="btn-primary" disabled={addingDoc}>
                    {addingDoc ? 'Submitting...' : 'Add Paper'}
                  </button>
                </div>
              </form>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Conversations */}
          <div className="lg:col-span-2">
            <h2 className="text-lg font-bold mb-4" style={{ fontFamily: 'var(--font-heading)', color: 'var(--color-text-primary)' }}>
              Your Conversations
            </h2>
            {loading ? (
              <div className="glass-card p-8 text-center">
                <div className="w-8 h-8 border-2 border-t-transparent rounded-full mx-auto mb-3"
                     style={{ borderColor: 'var(--color-accent)', borderTopColor: 'transparent', animation: 'spin-slow 1s linear infinite' }} />
                <p style={{ color: 'var(--color-text-muted)' }}>Loading...</p>
              </div>
            ) : conversations.length === 0 ? (
              <div className="glass-card p-8 text-center">
                <p className="text-4xl mb-3">🔬</p>
                <p className="font-medium mb-1" style={{ color: 'var(--color-text-primary)' }}>No conversations yet</p>
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                  Start a new conversation to explore AI research papers
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {conversations.map((conv, i) => (
                  <div
                    key={conv.id}
                    className="glass-card p-4 flex items-center justify-between cursor-pointer hover:border-indigo-500/30 transition-all animate-fade-in"
                    style={{ animationDelay: `${i * 50}ms` }}
                    onClick={() => navigate(`/chat/${conv.id}`)}
                  >
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate" style={{ color: 'var(--color-text-primary)' }}>
                        {conv.title}
                      </p>
                      <div className="flex items-center gap-3 mt-1">
                        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                          {conv.message_count} messages
                        </span>
                        {conv.paper_id && (
                          <span className="text-xs px-2 py-0.5 rounded-full"
                                style={{ background: 'var(--color-accent-glow)', color: 'var(--color-accent)' }}>
                            📄 {conv.paper_id}
                          </span>
                        )}
                        <span className="text-xs" style={{ color: 'var(--color-text-muted)' }}>
                          {new Date(conv.updated_at).toLocaleDateString()}
                        </span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleDeleteConv(conv.id); }}
                      className="text-sm px-2 py-1 rounded opacity-50 hover:opacity-100 transition-opacity"
                      style={{ color: 'var(--color-error)' }}
                      title="Delete"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Documents Sidebar */}
          <div>
            <h2 className="text-lg font-bold mb-4" style={{ fontFamily: 'var(--font-heading)', color: 'var(--color-text-primary)' }}>
              Your Documents
            </h2>
            {documents.length === 0 ? (
              <div className="glass-card p-6 text-center">
                <p className="text-3xl mb-2">📚</p>
                <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
                  No documents added yet. Add an ArXiv paper to chat with it!
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {documents.map((doc, i) => {
                  const cfg = statusConfig[doc.status] || statusConfig.queued;
                  const isProcessing = !['done', 'failed'].includes(doc.status);
                  return (
                    <div
                      key={doc.id}
                      className="glass-card p-4 animate-fade-in"
                      style={{ animationDelay: `${i * 50}ms` }}
                    >
                      <div className="flex items-start justify-between mb-2">
                        <p className="font-medium text-sm truncate flex-1"
                           style={{ color: 'var(--color-text-primary)' }}>
                          {doc.title || doc.arxiv_id}
                        </p>
                        <span className="text-lg ml-2">{cfg.icon}</span>
                      </div>
                      <p className="text-xs mb-2" style={{ color: 'var(--color-text-muted)' }}>
                        {doc.arxiv_id}
                      </p>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium" style={{ color: cfg.color }}>
                          {cfg.label}
                        </span>
                        {isProcessing && (
                          <div className="w-3 h-3 border border-t-transparent rounded-full"
                               style={{ borderColor: cfg.color, borderTopColor: 'transparent', animation: 'spin-slow 1s linear infinite' }} />
                        )}
                      </div>
                      {doc.status === 'done' && doc.chunks_created > 0 && (
                        <p className="text-xs mt-1" style={{ color: 'var(--color-success)' }}>
                          {doc.chunks_created} chunks indexed
                        </p>
                      )}
                      {doc.status === 'failed' && doc.error_message && (
                        <p className="text-xs mt-1" style={{ color: 'var(--color-error)' }}>
                          {doc.error_message}
                        </p>
                      )}
                      {doc.status === 'done' && (
                        <button
                          className="btn-ghost text-xs mt-2 w-full"
                          onClick={async () => {
                            const conv = await createConversation(`Chat: ${doc.title || doc.arxiv_id}`, doc.arxiv_id);
                            navigate(`/chat/${conv.id}`);
                          }}
                        >
                          💬 Chat with this paper
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </main>
    </PageShell>
  );
}
