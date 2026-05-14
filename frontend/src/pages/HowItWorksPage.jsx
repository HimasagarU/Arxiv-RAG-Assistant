import { memo, useState } from 'react';
import { Link } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import { PageHeader, PageShell } from '../components/PageShell';

// Full 13-step pipeline matching the actual backend order in retrieval.py / chat.py / app.py
const steps = [
  {
    icon: '🎯',
    title: 'Intent Classification',
    description:
      'The pipeline initiates with heuristic intent detection, categorizing queries into four archetypes: explanatory, comparative, discovery, or evidence. This classification modulates the downstream retrieval budget—prioritizing precision for technical queries—and selects the optimal generative prompt template to maintain scientific rigor.',
    details: ['Explanatory / Comparative / Discovery / Evidence', 'Adjusts merge_top_m budget', 'Selects prompt template'],
    color: '#3b82f6',
  },
  {
    icon: '🧬',
    title: 'Query Expansion',
    description:
      'To maximize recall, the system generates semantically distinct paraphrases of the query. These variants are retrieved independently to bridge the gap between user phrasing and specialized academic terminology, ensuring relevant evidence is surfaced even when exact keywords differ.',
    details: ['Up to 3 Groq paraphrases', 'Independent retrieval per variant', 'Merged before RRF fusion'],
    color: '#6366f1',
  },
  {
    icon: '💡',
    title: 'HyDE — Hypothetical Document',
    description:
      'For discovery-focused queries, the system generates a hypothetical research excerpt. This "pseudo-document" is embedded to create an auxiliary dense vector, shifting the search from question-answer matching to document-to-document similarity, which often yields more relevant technical context.',
    details: ['2-4 sentence paper excerpt', 'Groq Llama 3.3 (≤220 tokens)', 'Discovery/Explanatory only'],
    color: '#8b5cf6',
  },
  {
    icon: '🔵',
    title: 'Dense Vector Search',
    description:
      'BGE-Large-EN-v1.5 encodes the query into 1024-dimensional vectors. The system performs a high-performance vector search against Qdrant Cloud using HNSW indexing (M=32, ef=200), retrieving the top 60 candidates based on deep semantic alignment rather than simple word overlap.',
    details: ['BGE-Large-EN-v1.5 · 1024-dim', 'HNSW M=32 ef=200 on Qdrant Cloud', 'K_DENSE = 60 candidates'],
    color: '#0ea5e9',
  },
  {
    icon: '🟡',
    title: 'BM25 Lexical Retrieval',
    description:
      'In parallel, an in-memory BM25 index executes a keyword-based search to capture exact technical terms, model names, and specific identifiers. This ensures that precise academic jargon and unique symbols—which can be "smoothed" over by dense embeddings—remain highly retrievable.',
    details: ['rank-bm25 in-process index', 'Exact term matching', 'K_LEX = 60 candidates'],
    color: '#f59e0b',
  },
  {
    icon: '🔀',
    title: 'Reciprocal Rank Fusion (RRF)',
    description:
      'The dense and lexical candidates are integrated via Reciprocal Rank Fusion (RRF). By calculating a consensus rank based on position rather than raw scores, the system robustly merges results from different search paradigms (RRF_K=50) to produce a unified, high-confidence candidate list.',
    details: ['RRF_K = 50 constant', 'Up to MERGE_TOP_M = 50 candidates', 'Intent-weighted merge budget'],
    color: '#10b981',
  },
  {
    icon: '👨‍👧',
    title: 'Parent-Child Expansion',
    description:
      'To prevent the loss of critical surrounding context, the system utilizes a parent-child relationship. If a specific chunk is retrieved, the pipeline can "expand" it to include the broader paper context or adjacent sentences, ensuring the LLM receives a coherent narrative rather than isolated fragments.',
    details: ['arxiv_docs parent collection', 'PARENT_TOP_DOCS = 6', 'PARENT_CHUNKS_PER_DOC = 4'],
    color: '#14b8a6',
  },
  {
    icon: '📊',
    title: 'Section & Recency Boosts',
    description:
      'Heuristic scoring adjustments are applied to favor high-signal sections like Methods or Results and to reward recency. This bias ensures that the latest breakthroughs and the most technically robust sections of a paper are prioritized during the final selection process.',
    details: ['Section-type bias', 'Recency decay bonus', 'Citation-graph boost (optional)'],
    color: '#f97316',
  },
  {
    icon: '⚖️',
    title: 'BGE Reranker v2-m3',
    description:
      'The unified candidate list undergoes rigorous re-evaluation using BAAI/bge-reranker-v2-m3. This cross-encoder model analyzes the full interaction between the query and each passage, generating a definitive relevance score that selects the most precise 6 evidence blocks for generation.',
    details: ['BGE-Reranker-v2-m3', 'Joint query-passage scoring', 'FINAL_TOP_N = 6 passages'],
    color: '#ec4899',
  },
  {
    icon: '🌈',
    title: 'MMR Diversity Filter',
    description:
      'Maximum Marginal Relevance (MMR) is applied to minimize information redundancy. By penalizing candidates that are semantically too similar to those already selected, the system ensures a diverse set of evidence that covers multiple facets of the query.',
    details: ['MMR λ = 0.5 (tunable)', 'Removes near-duplicate passages', 'Improves answer breadth'],
    color: '#a855f7',
  },
  {
    icon: '✂️',
    title: 'Context Compression',
    description:
      'To optimize the LLM context window, an intelligent compression pass identifies and extracts only the most salient sentences from the retrieved blocks. This process preserves essential citation markers while stripping away peripheral noise that could lead to hallucination.',
    details: ['Groq compression (optional)', 'Triggered at >12k chars', 'Preserves citation markers'],
    color: '#d946ef',
  },
  {
    icon: '🤖',
    title: 'Answer Generation',
    description:
      'The final, evidence-grounded context is synthesized into a technical response. The system enforces strict attribution requirements and utilizes Gemini 1.5 / 3.x Flash to generate answers that clearly distinguish between empirical data, hypotheses, and speculative gaps.',
    details: [
      'Public: Groq Llama 3.3 70B',
      'Chat: Gemini 1.5 / 3.x Flash → Groq fallback',
      'GENERATION_CONTEXT_TOP_N = 6',
    ],
    color: '#9333ea',
  },
];

// Extracted component to prevent remounting and properly track the working image source (.png vs .jpg)
const ImageExpandable = memo(function ImageExpandable({ src, alt, onExpand }) {
  const [currentSrc, setCurrentSrc] = useState(src);
  const [hasError, setHasError] = useState(false);

  return (
    <div
      className="relative group cursor-pointer rounded-lg overflow-hidden border bg-white shadow-sm hover:shadow-md transition-shadow"
      style={{ borderColor: 'var(--color-border)', height: '200px' }}
      onClick={() => onExpand(hasError ? 'https://via.placeholder.com/1200x800/f8f6f1/333333?text=' + encodeURIComponent('Missing: ' + src) : currentSrc)}
    >
      {hasError ? (
        <div className="w-full h-full flex flex-col items-center justify-center text-gray-400 bg-gray-50 p-6 text-center">
          <svg className="w-12 h-12 mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
          <span className="text-sm font-medium">Image not found: <code className="bg-gray-200 px-1 rounded">{src}</code></span>
        </div>
      ) : (
        <img
          src={currentSrc}
          alt={alt}
          className="w-full h-full object-cover object-top transition-transform duration-700 group-hover:scale-105"
          onError={() => {
            if (currentSrc.endsWith('.png')) {
              setCurrentSrc(src.replace('.png', '.jpg'));
            } else {
              setHasError(true);
            }
          }}
        />
      )}
      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center opacity-0 group-hover:opacity-100">
        <div className="bg-white/90 backdrop-blur text-gray-900 px-6 py-3 rounded-full font-bold shadow-xl flex items-center gap-2 transform translate-y-4 group-hover:translate-y-0 transition-transform duration-300">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v3m0 0v3m0-3h3m-3 0H7" /></svg>
          View Full Diagram
        </div>
      </div>
    </div>
  );
});

export default function HowItWorksPage() {
  const [selectedImg, setSelectedImg] = useState(null);

  return (
    <PageShell>
      {/* Modal */}
      {selectedImg && (
        <div
          className="fixed inset-0 z-[9999] flex items-center justify-center bg-black/90 p-4 sm:p-8 cursor-zoom-out animate-fade-in backdrop-blur-sm"
          onClick={() => setSelectedImg(null)}
        >
          <img
            src={selectedImg}
            className="max-w-full max-h-full object-contain rounded-lg shadow-2xl animate-scale-up"
            alt="Expanded view"
            onClick={(e) => e.stopPropagation()}
            style={{ cursor: 'default' }}
          />
          <button
            className="absolute top-4 right-4 sm:top-8 sm:right-8 bg-white/10 hover:bg-white/20 text-white rounded-full p-2 transition-colors cursor-pointer"
            onClick={() => setSelectedImg(null)}
            aria-label="Close modal"
          >
            <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      )}

      <PageHeader
        eyebrow="Product Tour"
        title="How It Works"
        subtitle="A 12-stage AI pipeline that retrieves, re-ranks, and synthesises answers from thousands of research papers on neural network internals."
        containerClassName="max-w-4xl"
        leading={(
          <Link to="/" className="text-xl font-bold text-[var(--color-text-primary)]" style={{ fontFamily: 'var(--font-heading)' }}>
            ArXiv RAG Assistant
          </Link>
        )}
        actions={(
          <>
            <ThemeToggle />
            <Link to="/" className="btn-ghost text-sm">
              ← Back to App
            </Link>
          </>
        )}
      />

      <main className="mx-auto max-w-4xl px-4 sm:px-6 py-12">

        {/* Architecture summary */}
        <div className="mt-12 glass-card p-8 text-center animate-fade-in" style={{ animationDelay: '300ms' }}>
          <h2 className="text-2xl font-bold mb-6" style={{ fontFamily: 'var(--font-heading)' }}>
            System Architecture
          </h2>
          <ImageExpandable src="/architecture.png" alt="System Architecture Diagram" onExpand={setSelectedImg} />
          <p className="text-sm mt-6" style={{ color: 'var(--color-text-muted)' }}>
            Built with FastAPI · Qdrant Cloud · Groq · Gemini · React
          </p>
        </div>

        {/* Model routing callout */}
        <div className="mt-10 glass-card p-6 border-l-4 animate-fade-in" style={{ borderLeftColor: '#9333ea', animationDelay: '400ms' }}>
          <h3 className="text-lg font-bold mb-2" style={{ fontFamily: 'var(--font-heading)', color: '#9333ea' }}>
            Model Routing
          </h3>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm" style={{ color: 'var(--color-text-secondary)' }}>
            <div className="flex items-start gap-3">
              <span className="text-xl flex-shrink-0">🌐</span>
              <div>
                <p className="font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>Public (Landing Page)</p>
                <p>Groq — Llama 3.3 70B Versatile<br/>Ultra-fast inference, no auth required.</p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-xl flex-shrink-0">🔐</span>
              <div>
                <p className="font-semibold mb-1" style={{ color: 'var(--color-text-primary)' }}>Authenticated Chat</p>
                <p>Gemini 2.5 Flash (primary)<br/>Groq Llama 3.3 70B (auto-fallback on rate limits)</p>
              </div>
            </div>
          </div>
        </div>

        {/* Pipeline steps */}
        <div className="relative mt-20">
          {/* Vertical connecting line */}
          <div className="absolute left-8 top-0 bottom-0 w-px" style={{ background: 'var(--color-border)' }} />

          <div className="space-y-10">
            {steps.map((step, i) => (
              <div key={i} className="relative pl-20 animate-fade-in" style={{ animationDelay: `${i * 80 + 500}ms` }}>
                {/* Step number circle */}
                <div
                  className="absolute left-4 w-8 h-8 rounded-full flex items-center justify-center text-lg z-10"
                  style={{
                    background: `linear-gradient(135deg, ${step.color}, ${step.color}99)`,
                    boxShadow: `0 0 20px ${step.color}33`,
                  }}
                >
                  {step.icon}
                </div>

                {/* Step number badge */}
                <div
                  className="absolute left-[3.4rem] top-0 w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white z-10"
                  style={{ background: step.color }}
                >
                  {i + 1}
                </div>

                <div className="glass-card p-6 hover:border-opacity-50 transition-all"
                     style={{ borderColor: `${step.color}33` }}>
                  <h3 className="text-xl font-bold mb-3" style={{ fontFamily: 'var(--font-heading)', color: step.color }}>
                    {step.title}
                  </h3>
                  <p className="text-sm leading-relaxed mb-4" style={{ color: 'var(--color-text-secondary)' }}>
                    {step.description}
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {step.details.map((detail, di) => (
                      <span
                        key={di}
                        className="text-xs px-3 py-1 rounded-full font-medium text-white"
                        style={{ background: step.color, border: `1px solid ${step.color}33` }}
                      >
                        {detail}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Internal Pipelines */}
        <div className="mt-20 animate-fade-in" style={{ animationDelay: '500ms' }}>
          <h2 className="text-2xl font-bold mb-8 text-center" style={{ fontFamily: 'var(--font-heading)' }}>
            System Internals
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="glass-card p-6" style={{ borderColor: 'var(--color-border)' }}>
              <h3 className="text-xl font-bold mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Offline Ingestion</h3>
              <p className="text-sm mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                The high-throughput pipeline that processes the Mechanistic Interpretability corpus. It extracts structured text, generates hierarchical chunks, and synchronises vector (Qdrant) and lexical (BM25) indexes.
              </p>
              <ImageExpandable src="/flow-offline-ingestion.jpg" alt="Offline Ingestion Pipeline" onExpand={setSelectedImg} />
            </div>
            <div className="glass-card p-6" style={{ borderColor: 'var(--color-border)' }}>
              <h3 className="text-xl font-bold mb-4" style={{ fontFamily: 'var(--font-heading)' }}>Hybrid Retrieval</h3>
              <p className="text-sm mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Our multi-stage retrieval logic combining semantic dense search with exact lexical matching. It features intent-aware gating, RRF fusion, and precision reranking to ensure maximum grounding.
              </p>
              <ImageExpandable src="/flow-retrieval.jpg" alt="Hybrid Retrieval Pipeline" onExpand={setSelectedImg} />
            </div>
          </div>
        </div>

        {/* Feature Workflows */}
        <div className="mt-20 animate-fade-in" style={{ animationDelay: '500ms' }}>
          <h2 className="text-2xl font-bold mb-8 text-center" style={{ fontFamily: 'var(--font-heading)' }}>
            Core Features &amp; Workflows
          </h2>
          <div className="space-y-12">
            <div className="glass-card p-8" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center gap-4 mb-6">
                <div className="text-4xl">💬</div>
                <h3 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-heading)' }}>General Chat</h3>
              </div>
              <p className="text-lg leading-relaxed mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Explore the depths of neural network internals. Ask broad or specific questions about transformer circuits, superposition, or feature visualization. The pipeline synthesises an answer across the entire corpus of 3,000+ papers with real-time token streaming.
              </p>
              <ImageExpandable src="/flow-chat.png" alt="General Chat Flow Diagram" onExpand={setSelectedImg} />
            </div>

            <div className="glass-card p-8" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center gap-4 mb-6">
                <div className="text-4xl">📥</div>
                <h3 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-heading)' }}>Add Document</h3>
              </div>
              <p className="text-lg leading-relaxed mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Found a brand new paper on ArXiv? Simply drop the ID here. The system will pull the PDF, run it through the extraction and embedding pipeline, and make it searchable in seconds — expanding the collective knowledge base.
              </p>
              <ImageExpandable src="/flow-add-doc.png" alt="Add Document Flow Diagram" onExpand={setSelectedImg} />
            </div>

            <div className="glass-card p-8" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center gap-4 mb-6">
                <div className="text-4xl">📄</div>
                <h3 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-heading)' }}>Chat with Document</h3>
              </div>
              <p className="text-lg leading-relaxed mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Deep dive into a specific study. By locking the context to a single paper, the AI becomes a dedicated expert on that specific methodology, helping you extract figures, understand proofs, or summarize findings without outside noise.
              </p>
              <ImageExpandable src="/flow-chat-doc.png" alt="Chat with Document Flow Diagram" onExpand={setSelectedImg} />
            </div>

            <div className="glass-card p-8" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center gap-4 mb-6">
                <div className="text-4xl">🔗</div>
                <h3 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-heading)' }}>Similar Papers</h3>
              </div>
              <p className="text-lg leading-relaxed mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Accelerate your literature review. By computing the mean embedding of a paper's chunks (preferring the arxiv_docs parent vector when available), we instantly locate other studies exploring similar phenomena or related techniques.
              </p>
              <ImageExpandable src="/flow-similar.png" alt="Similar Papers Flow Diagram" onExpand={setSelectedImg} />
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="text-center mt-12 animate-fade-in" style={{ animationDelay: '600ms' }}>
          <Link to="/" className="btn-primary inline-block">
            Start Exploring →
          </Link>
        </div>
      </main>
    </PageShell>
  );
}
