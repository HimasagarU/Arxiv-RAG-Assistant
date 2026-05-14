import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import { useAuth } from '../context/AuthContext';
import { PageHeader, PageShell } from '../components/PageShell';
import { renderMarkdown } from '../utils/renderMarkdown';
import { streamPublicQuery } from '../api';

// Real retrieval steps that match the backend pipeline order.
// Shown in sequence during loading; the stream metadata updates them.
const PIPELINE_STEPS = [
  { id: 'intent',      label: 'Classifying query intent…'          },
  { id: 'hyde',        label: 'Generating HyDE passage…'            },
  { id: 'expand',      label: 'Expanding query variants…'           },
  { id: 'dense',       label: 'Dense vector search (Qdrant)…'       },
  { id: 'lexical',     label: 'BM25 lexical retrieval…'             },
  { id: 'rrf',         label: 'Fusing results with RRF…'            },
  { id: 'parent',      label: 'Parent-child chunk expansion…'       },
  { id: 'citation',    label: 'Citation-graph boost…'               },
  { id: 'rerank',      label: 'BGE-Reranker-v2-m3…'                 },
  { id: 'mmr',         label: 'MMR diversity filter…'               },
  { id: 'compress',    label: 'Compressing context…'                },
  { id: 'generate',    label: 'Generating answer with Groq…'        },
];

function LandingPage() {
  const { user } = useAuth();
  const [query, setQuery]           = useState('');
  const [topK, setTopK]             = useState('6');
  const [filterYear, setFilterYear] = useState('');
  const [filterAuthor, setFilterAuthor] = useState('');

  const [loading, setLoading]   = useState(false);
  const [activeStep, setActiveStep] = useState(-1);
  const [doneSteps, setDoneSteps]   = useState(new Set());

  const [streamingText, setStreamingText]   = useState('');
  const [sources, setSources]               = useState([]);
  const [status, setStatus]                 = useState('Connecting…');
  const [papersCount, setPapersCount]       = useState(0);
  const [chunksCount, setChunksCount]       = useState(0);
  const [error, setError]                   = useState('');

  // Streaming cursor blink
  const [streaming, setStreaming] = useState(false);

  const answerRef = useRef(null);
  const stepTimerRef = useRef(null);

  const API_BASE = (import.meta.env.VITE_API_URL || 'https://himasagaru-arxiv-rag-mechanistic-interpretability.hf.space').replace(/\/$/, '');

  useEffect(() => {
    async function checkHealth() {
      try {
        const resp = await fetch(`${API_BASE}/health`);
        const data = await resp.json();
        const collections = data.collections || {};
        setChunksCount(collections['arxiv_text'] || 0);
        setPapersCount(data.db_papers || 0);
        setStatus('connected');
      } catch {
        setStatus('offline');
      }
    }
    checkHealth();
  }, [API_BASE]);

  // Advance a fake step ticker while waiting for the real stream metadata
  function startStepTicker() {
    let idx = 0;
    setActiveStep(0);
    setDoneSteps(new Set());
    stepTimerRef.current = setInterval(() => {
      idx += 1;
      if (idx < PIPELINE_STEPS.length - 1) {
        setDoneSteps((prev) => new Set([...prev, PIPELINE_STEPS[idx - 1].id]));
        setActiveStep(idx);
      } else {
        clearInterval(stepTimerRef.current);
      }
    }, 900);
  }

  function stopStepTicker(completedUpTo) {
    clearInterval(stepTimerRef.current);
    // Mark all steps up to generate as done
    const allDone = new Set(PIPELINE_STEPS.slice(0, completedUpTo ?? PIPELINE_STEPS.length - 1).map((s) => s.id));
    setDoneSteps(allDone);
    setActiveStep(PIPELINE_STEPS.length - 1); // "generating" step
  }

  const submitQuery = async () => {
    if (!query.trim() || loading) return;
    setLoading(true);
    setStreaming(false);
    setError('');
    setStreamingText('');
    setSources([]);
    startStepTicker();

    let metaReceived = false;

    await streamPublicQuery(query.trim(), {
      topK: parseInt(topK),
      startYear: filterYear ? parseInt(filterYear) : null,
      author: filterAuthor || null,

      onMetadata: (meta) => {
        metaReceived = true;
        setSources(meta.sources || []);
        // Stop ticker and advance to generate step
        stopStepTicker(PIPELINE_STEPS.length - 1);
        setStreaming(true);
      },

      onToken: (token) => {
        setStreaming(true);
        setStreamingText((prev) => prev + token);
        // Auto-scroll answer into view
        answerRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      },

      onError: (msg) => {
        setError(msg || 'Failed to get answer. Please try again.');
        clearInterval(stepTimerRef.current);
      },

      onDone: (finalSources) => {
        setStreaming(false);
        if (finalSources?.length) setSources(finalSources);
        setLoading(false);
        setActiveStep(-1);
      },
    });

    setLoading(false);
    setStreaming(false);
  };

  const applySample = (text) => {
    setQuery(text);
  };

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
            Powered by hybrid dense + BM25 retrieval, HyDE, query expansion, and cross-encoder reranking.
          </p>
        </section>

        {/* Search Box */}
        <section className="glass-card p-6 mb-8 shadow-sm">
          <div className="flex gap-3 mb-4 flex-col sm:flex-row">
            <input
              id="landing-search-input"
              type="text"
              className="input-field flex-1"
              placeholder="e.g. How do sparse autoencoders find interpretable features?"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && submitQuery()}
            />
            <button
              id="landing-search-btn"
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
                <option value="6">6</option>
                <option value="12">12</option>
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
                placeholder="e.g. Elhage"
                value={filterAuthor}
                onChange={(e) => setFilterAuthor(e.target.value)}
              />
            </div>
          </div>
        </section>

        {/* Pipeline Progress */}
        {loading && (
          <div className="flex flex-col gap-2 mb-6 text-sm max-w-md mx-auto bg-[var(--color-bg-card)] p-4 rounded-lg border border-[var(--color-border)] shadow-sm">
            {PIPELINE_STEPS.map((step, i) => {
              const done = doneSteps.has(step.id);
              const active = i === activeStep;
              return (
                <div key={step.id} className={`flex items-center gap-3 transition-colors ${
                  active ? 'text-[var(--color-accent)] font-medium' :
                  done   ? 'text-[var(--color-success)]' :
                           'text-[var(--color-text-muted)]'
                }`}>
                  <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                    active ? 'bg-[var(--color-accent)] animate-pulse' :
                    done   ? 'bg-[var(--color-success)]' :
                             'bg-[var(--color-border)]'
                  }`} />
                  <span className="flex-1 text-xs">{step.label}</span>
                  {active && (
                    <div className="w-3 h-3 border-2 border-t-transparent border-[var(--color-accent)] rounded-full animate-spin ml-auto flex-shrink-0" />
                  )}
                  {done && (
                    <svg className="w-3 h-3 text-[var(--color-success)] ml-auto flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                    </svg>
                  )}
                </div>
              );
            })}
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
        {(streamingText || sources.length > 0) && (
          <section className="animate-fade-in" ref={answerRef}>
            {/* Streaming Answer Card */}
            {streamingText && (
              <div className="bg-[var(--color-bg-card)] border border-[var(--color-border)] border-l-4 border-l-[var(--color-accent)] rounded-lg p-6 mb-8 shadow-sm">
                <div className="font-heading text-sm font-semibold text-[var(--color-accent)] mb-3 uppercase tracking-wider flex items-center gap-2">
                  Generated Answer
                  {streaming && (
                    <span className="inline-block w-2 h-4 bg-[var(--color-accent)] rounded-sm animate-pulse ml-1" />
                  )}
                </div>
                <div
                  className="chat-markdown text-[var(--color-text-primary)] text-lg leading-relaxed"
                  dangerouslySetInnerHTML={{ __html: renderMarkdown(streamingText) }}
                />
              </div>
            )}

            {/* Sources */}
            {sources.length > 0 && !streaming && (
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
                            {source.rerank_score !== 0 && (
                              <span className="text-[var(--color-success)] font-medium">Reranker Score: {source.rerank_score.toFixed(3)}</span>
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
        Public queries powered by{' '}
        <a href="https://groq.com" target="_blank" rel="noopener noreferrer" className="text-[var(--color-accent)] hover:underline mx-1">Groq</a> ·
        <a href="https://arxiv.org" target="_blank" rel="noopener noreferrer" className="text-[var(--color-accent)] hover:underline mx-1">ArXiv API</a>
      </footer>
    </PageShell>
  );
}

export default LandingPage;
