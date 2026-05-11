import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import { useAuth } from '../context/AuthContext';
import { PageHeader, PageShell } from '../components/PageShell';

function LandingPage() {
  const { user } = useAuth();
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [filterYear, setFilterYear] = useState('');
  const [filterAuthor, setFilterAuthor] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const steps = [
    "Analyzing query intent...",
    "Retrieving from Qdrant (dense search)...",
    "Retrieving from PostgreSQL (full-text search)...",
    "Reranking candidates with cross-encoder...",
    "Generating answer with Llama 3..."
  ];
  const [answer, setAnswer] = useState('');
  const [sources, setSources] = useState([]);
  const [status, setStatus] = useState('Connecting…');
  const [papersCount, setPapersCount] = useState(0);
  const [chunksCount, setChunksCount] = useState(0);
  const [error, setError] = useState('');

  const API_BASE = (import.meta.env.VITE_API_URL || 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space').replace(/\/$/, '');

  useEffect(() => {
    // Health check on mount
    async function checkHealth() {
      try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        const collections = data.collections || {};
        const textCount = collections['arxiv_text'] || 0;
        setChunksCount(textCount);
        setPapersCount(data.db_papers || 0);
        setStatus('connected');
      } catch {
        setStatus('offline');
      }
    }
    checkHealth();
  }, [API_BASE]);

  const submitQuery = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setActiveStep(0);
    setError('');
    setAnswer('');
    setSources([]);

    // Simulate steps
    const interval = setInterval(() => {
      setActiveStep((prev) => {
        if (prev < steps.length - 1) return prev + 1;
        clearInterval(interval);
        return prev;
      });
    }, 800);

    try {
      const resp = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          top_k: parseInt(topK),
          start_year: filterYear ? parseInt(filterYear) : null,
          author: filterAuthor || null
        })
      });

      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `API error ${resp.status}`);
      }

      const data = await resp.json();
      setAnswer(data.answer);
      setSources(data.sources || []);
    } catch (e) {
      setError(e.message || 'Failed to get answer. Please try again.');
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  };

  const applySample = (text) => {
    setQuery(text);
  };

  function renderMarkdown(text) {
    if (!text) return '';
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br/>');
  }

  return (
    <PageShell>
      <PageHeader
        eyebrow="Hybrid RAG"
        title="ArXiv Research Assistant"
        actions={(
          <>
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border ${
              status === 'connected'
                ? 'bg-[var(--color-accent-glow)] text-[var(--color-accent)] border-[var(--color-accent)]'
                : 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)] border-[var(--color-border)]'
            }`}>
              <div className={`w-2 h-2 rounded-full ${status === 'connected' ? 'bg-[var(--color-accent)] animate-pulse' : 'bg-[var(--color-text-muted)]'}`} />
              {status === 'connected'
                ? `${chunksCount.toLocaleString()} chunks · ${papersCount.toLocaleString()} papers`
                : status === 'offline' ? 'API Offline' : 'Connecting…'}
            </div>

            <ThemeToggle />
            <Link to="/how-it-works" className="btn-ghost text-sm">
              How It Works
            </Link>

            <div className="flex items-center gap-2">
              {user ? (
                <Link to="/dashboard" className="btn-primary">
                  Go to App
                </Link>
              ) : (
                <>
                  <Link to="/login" className="btn-ghost">
                    Sign in
                  </Link>
                  <Link to="/register" className="btn-primary">
                    Create account
                  </Link>
                </>
              )}
            </div>
          </>
        )}
      />

      <main className="mx-auto max-w-4xl px-4 sm:px-6 py-12">
        {/* Hero */}
        <section className="text-center mb-10">
          <h2 className="font-heading text-4xl font-bold mb-3 text-[var(--color-text-primary)]">Mechanistic Interpretability Research</h2>
          <p className="text-[var(--color-text-secondary)] max-w-2xl mx-auto">
            Ask questions about transformer circuits, sparse autoencoders, activation patching, and more.
            Powered by hybrid dense + BM25 retrieval with cross-encoder reranking.
          </p>
        </section>

        {/* Search Box */}
        <section className="glass-card p-6 mb-8 shadow-sm">
          <div className="flex gap-3 mb-4 flex-col sm:flex-row">
            <input
              type="text"
              className="input-field flex-1"
              placeholder="e.g. How do sparse autoencoders find interpretable features?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && submitQuery()}
            />
            <button 
              className="btn-primary flex items-center justify-center gap-2"
              onClick={submitQuery}
              disabled={loading}
            >
              {loading ? (
                <div className="w-4 h-4 border-2 border-t-transparent border-white rounded-full animate-spin"></div>
              ) : (
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              )}
              Search
            </button>
          </div>

          <div className="flex items-center gap-4 flex-wrap text-sm text-[var(--color-text-secondary)]">
            <div className="flex items-center gap-2">
              <label className="font-medium uppercase text-xs text-[var(--color-text-muted)]">Results:</label>
              <select 
                className="bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded px-2 py-1 text-sm outline-none focus:border-[var(--color-accent)]"
                value={topK}
                onChange={(e) => setTopK(e.target.value)}
              >
                <option value="3">3</option>
                <option value="5">5</option>
                <option value="10">10</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <label className="font-medium uppercase text-xs text-[var(--color-text-muted)]">From Year:</label>
              <input 
                type="number" 
                className="bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded px-2 py-1 text-sm w-24 outline-none focus:border-[var(--color-accent)]"
                placeholder="e.g. 2023"
                value={filterYear}
                onChange={(e) => setFilterYear(e.target.value)}
              />
            </div>

            <div className="flex items-center gap-2">
              <label className="font-medium uppercase text-xs text-[var(--color-text-muted)]">Author:</label>
              <input 
                type="text" 
                className="bg-[var(--color-bg-primary)] border border-[var(--color-border)] rounded px-2 py-1 text-sm w-32 outline-none focus:border-[var(--color-accent)]"
                placeholder="e.g. Hinton"
                value={filterAuthor}
                onChange={(e) => setFilterAuthor(e.target.value)}
              />
            </div>
          </div>
        </section>

        {/* Loading Steps */}
        {loading && (
          <div className="flex flex-col gap-2 mb-6 text-sm max-w-md mx-auto bg-[var(--color-bg-card)] p-4 rounded-lg border border-[var(--color-border)] shadow-sm">
            {steps.map((step, i) => (
              <div key={i} className={`flex items-center gap-3 transition-colors ${
                i === activeStep ? 'text-[var(--color-accent)] font-medium' :
                i < activeStep ? 'text-[var(--color-success)]' :
                'text-[var(--color-text-muted)]'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  i === activeStep ? 'bg-[var(--color-accent)] animate-pulse' :
                  i < activeStep ? 'bg-[var(--color-success)]' :
                  'bg-[var(--color-border)]'
                }`}></div>
                {step}
                {i === activeStep && (
                  <div className="w-3 h-3 border-2 border-t-transparent border-[var(--color-accent)] rounded-full animate-spin ml-auto"></div>
                )}
                {i < activeStep && (
                  <svg className="w-4 h-4 text-[var(--color-success)] ml-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Sample Queries */}
        <section className="flex flex-wrap gap-2 justify-center mb-10">
          {[
            "What are induction heads?",
            "How does activation patching work?",
            "Explain sparse autoencoders for interpretability",
            "What is superposition in neural networks?",
            "Circuit discovery vs probing classifiers",
            "Latest advances in mechanistic interpretability"
          ].map((text, i) => (
            <button
              key={i}
              onClick={() => applySample(text)}
              className="px-4 py-2 text-sm bg-[var(--color-bg-card)] border border-[var(--color-border)] rounded-full text-[var(--color-text-secondary)] hover:border-[var(--color-accent)] hover:text-[var(--color-accent)] transition-colors"
            >
              {text}
            </button>
          ))}
        </section>

        {/* Error State */}
        {error && (
          <div className="bg-[var(--color-error)] text-white p-4 rounded-lg mb-6 text-sm">
            {error}
          </div>
        )}

        {/* Results Section */}
        {(answer || sources.length > 0) && (
          <section className="animate-fade-in">
            {/* Answer Card */}
            {answer && (
              <div className="bg-[var(--color-bg-card)] border border-[var(--color-border)] border-l-4 border-l-[var(--color-accent)] rounded-lg p-6 mb-8 shadow-sm">
                <div className="font-heading text-sm font-semibold text-[var(--color-accent)] mb-3 uppercase tracking-wider">
                  Generated Answer
                </div>
                <div 
                  className="chat-markdown text-[var(--color-text-primary)] text-lg leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(answer) }}
                />
              </div>
            )}

            {/* Sources */}
            {sources.length > 0 && (
              <div>
                <h3 className="font-heading text-xl font-semibold mb-4 text-[var(--color-text-primary)]">
                  Sources
                </h3>
                <div className="space-y-4">
                  {sources.map((source, i) => (
                    <div key={i} className="bg-[var(--color-bg-card)] border border-[var(--color-border)] rounded-lg p-5 hover:border-[var(--color-accent)] transition-colors shadow-sm">
                      <div className="flex items-start gap-3">
                        <div className="w-6 h-6 rounded-full bg-[var(--color-accent)] text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                          {i + 1}
                        </div>
                        <div className="flex-1">
                          <h4 className="font-heading font-semibold text-[var(--color-text-primary)] mb-1">
                            {source.title}
                          </h4>
                          <p className="text-sm text-[var(--color-text-secondary)] line-clamp-3">
                            {source.chunk_text}
                          </p>
                          <div className="flex items-center gap-3 mt-3 text-xs text-[var(--color-text-muted)] flex-wrap">
                            {source.authors && <span>By {source.authors}</span>}
                            {source.categories && <span className="bg-[var(--color-bg-hover)] px-2 py-0.5 rounded">{source.categories}</span>}
                            {source.rerank_score > 0 && (
                              <span className="text-[var(--color-success)] font-medium">Score: {source.rerank_score.toFixed(3)}</span>
                            )}
                            <a href={`https://arxiv.org/abs/${source.paper_id}`} target="_blank" rel="noopener noreferrer" className="text-[var(--color-accent)] hover:underline ml-auto">
                              View on ArXiv
                            </a>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="border-top border-[var(--color-border)] bg-[var(--color-bg-card)] text-center py-6 text-sm text-[var(--color-text-muted)] mt-12">
        ArXiv Research Assistant · Hybrid Dense + BM25 + Cross-Encoder Reranking ·
        <a href="https://groq.com" target="_blank" rel="noopener noreferrer" className="text-[var(--color-accent)] hover:underline mx-1">Groq</a> &amp;
        <a href="https://arxiv.org" target="_blank" rel="noopener noreferrer" className="text-[var(--color-accent)] hover:underline mx-1">ArXiv API</a>
      </footer>
    </PageShell>
  );
}

export default LandingPage;
