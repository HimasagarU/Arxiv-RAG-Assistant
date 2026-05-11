import { useState } from 'react';
import { Link } from 'react-router-dom';
import ThemeToggle from '../components/ThemeToggle';
import { PageHeader, PageShell } from '../components/PageShell';

const steps = [
  {
    icon: '📥',
    title: 'Paper Ingestion',
    description:
      'We continuously monitor ArXiv for the latest research on transformer circuits and sparse autoencoders. ' +
      'PDFs are parsed with high precision, extracting the core text while preserving section hierarchy ' +
      '(Abstract, Methods, Results) for optimal context retrieval.',
    details: ['~3,000+ papers indexed', 'Section-aware chunking', 'PDF storage on Cloudflare R2'],
    color: '#4f46e5',
  },
  {
    icon: '🧬',
    title: 'Vector Embedding',
    description:
      'Text chunks are transformed into 1024-dimensional semantic vectors using the advanced BGE-Large model. ' +
      'This allows the system to understand complex concepts like "activation patching" or "induction heads" ' +
      'even if you use different words.',
    details: ['BGE-Large-EN-v1.5 embeddings', '1024-dimensional vectors', 'Qdrant Cloud vector DB'],
    color: '#6366f1',
  },
  {
    icon: '🔍',
    title: 'Hybrid Retrieval',
    description:
      'To ensure no insight is missed, we run a dual-track retrieval pipeline. Dense search finds conceptually ' +
      'related passages in Qdrant, while precision BM25 search catches exact technical terms and specific equations. ' +
      'Reciprocal Rank Fusion (RRF) then merges them into a perfect candidate list.',
    details: ['Dense + Lexical fusion (RRF)', 'Intent-classified weighting', 'BM25 lexical retrieval'],
    color: '#7c3aed',
  },
  {
    icon: '⚖️',
    title: 'Cross-Encoder Reranking',
    description:
      'The combined list is passed through a deep Cross-Encoder model. Unlike standard search, it evaluates ' +
      'the precise relationship between your specific question and each passage, re-ordering them to put ' +
      'the most directly relevant insights at the very top.',
    details: ['ms-marco-MiniLM-L-6-v2', 'Direct query-passage scoring', 'Top-5 passage selection'],
    color: '#8b5cf6',
    color: '#8b5cf6',
  },
  {
    icon: '🤖',
    title: 'Answer Generation',
    description:
      'Finally, the highly curated passages are provided to Llama 3.3 (70B) via Groq\'s ultra-fast inference engine. ' +
      'The model synthesizes a comprehensive answer, complete with inline citations, code snippets, and a clear ' +
      'admission of uncertainty if the papers don\'t contain the answer.',
    details: ['Llama 3.3 70B Versatile', 'Grounded answers with citations', 'Intent-specific prompts'],
    color: '#9333ea',
  },
];

// Extracted component to prevent remounting and properly track the working image source (.png vs .jpg)
function ImageExpandable({ src, alt, onExpand }) {
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
}

export default function HowItWorksPage() {
  const [selectedImg, setSelectedImg] = useState(null);

  return (
    <PageShell>
      {/* Modal - fixed with very high z-index and portal-like behavior */}
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
        subtitle="A multi-stage AI pipeline that retrieves, ranks, and synthesizes answers from thousands of research papers on neural network internals."
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
        {/* Hero */}
        <div className="text-center mb-16 animate-fade-in">
          <h1 className="text-4xl font-bold mb-4" style={{ fontFamily: 'var(--font-heading)' }}>
            How It Works
          </h1>
          <p className="text-lg max-w-2xl mx-auto" style={{ color: 'var(--color-text-secondary)' }}>
            A multi-stage AI pipeline that retrieves, ranks, and synthesizes answers
            from thousands of research papers on neural network internals.
          </p>
        </div>

        {/* Pipeline steps */}
        <div className="relative">
          {/* Vertical connecting line */}
          <div className="absolute left-8 top-0 bottom-0 w-px" style={{ background: 'var(--color-border)' }} />

          <div className="space-y-12">
            {steps.map((step, i) => (
              <div key={i} className="relative pl-20 animate-fade-in" style={{ animationDelay: `${i * 100}ms` }}>
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

        {/* Architecture summary */}
        <div className="mt-16 glass-card p-8 text-center animate-fade-in" style={{ animationDelay: '400ms' }}>
          <h2 className="text-2xl font-bold mb-6" style={{ fontFamily: 'var(--font-heading)' }}>
            System Architecture
          </h2>
          <ImageExpandable src="/architecture.png" alt="System Architecture Diagram" onExpand={setSelectedImg} />
          <p className="text-sm mt-6" style={{ color: 'var(--color-text-muted)' }}>
            Built with FastAPI · Qdrant Cloud · Groq · React
          </p>
        </div>

        {/* Feature Workflows */}
        <div className="mt-20 animate-fade-in" style={{ animationDelay: '500ms' }}>
          <h2 className="text-2xl font-bold mb-8 text-center" style={{ fontFamily: 'var(--font-heading)' }}>
            Core Features & Workflows
          </h2>
          <div className="space-y-12">
            <div className="glass-card p-8" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center gap-4 mb-6">
                <div className="text-4xl">💬</div>
                <h3 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-heading)' }}>General Chat</h3>
              </div>
              <p className="text-lg leading-relaxed mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Explore the depths of neural network internals. Ask broad or specific questions about transformer circuits, superposition, or feature visualization. The pipeline will synthesize an answer across the entire corpus of 3,000+ papers.
              </p>
              <ImageExpandable src="/flow-chat.png" alt="General Chat Flow Diagram" onExpand={setSelectedImg} />
            </div>
            
            <div className="glass-card p-8" style={{ borderColor: 'var(--color-border)' }}>
              <div className="flex items-center gap-4 mb-6">
                <div className="text-4xl">📥</div>
                <h3 className="text-2xl font-bold" style={{ fontFamily: 'var(--font-heading)' }}>Add Document</h3>
              </div>
              <p className="text-lg leading-relaxed mb-6" style={{ color: 'var(--color-text-secondary)' }}>
                Found a brand new paper on ArXiv? Simply drop the ID here. The system will pull the PDF, run it through the extraction and embedding pipeline, and make it searchable in seconds—expanding the collective knowledge base.
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
                Accelerate your literature review. By computing the average vector space of a paper's chunks, we can instantly locate other studies in the corpus that explore similar phenomena or use related techniques.
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
