import { Link } from 'react-router-dom';

const steps = [
  {
    icon: '📥',
    title: 'Paper Ingestion',
    description:
      'Papers are fetched from ArXiv, their PDFs are downloaded and stored in Cloudflare R2. ' +
      'Full text is extracted using PyMuPDF and intelligently chunked into semantic passages ' +
      'with section-aware splitting (abstract, methods, results, etc.).',
    details: ['~3,000+ papers indexed', 'Section-aware chunking', 'PDF storage on Cloudflare R2'],
    color: '#6366f1',
  },
  {
    icon: '🧬',
    title: 'Vector Embedding',
    description:
      'Each text chunk is embedded using BGE-Large-EN-v1.5, a state-of-the-art sentence transformer. ' +
      'The 1024-dimensional vectors are stored in Qdrant Cloud for fast approximate nearest neighbor search.',
    details: ['BGE-Large-EN-v1.5 embeddings', '1024-dimensional vectors', 'Qdrant Cloud vector DB'],
    color: '#7c3aed',
  },
  {
    icon: '🔍',
    title: 'Hybrid Retrieval',
    description:
      'When you ask a question, two retrieval paths run simultaneously: dense retrieval (semantic similarity ' +
      'via Qdrant) and lexical retrieval (keyword matching via BM25 lexical retrieval). ' +
      'Results are fused using Reciprocal Rank Fusion (RRF) with intent-aware weights.',
    details: ['Dense + Lexical fusion (RRF)', 'Intent-classified weighting', 'BM25 lexical retrieval'],
    color: '#8b5cf6',
  },
  {
    icon: '⚖️',
    title: 'Cross-Encoder Reranking',
    description:
      'The fused candidates are scored by a cross-encoder (MiniLM-L6) that evaluates the direct ' +
      'relevance of each passage to your query. This dramatically improves precision over raw retrieval scores.',
    details: ['ms-marco-MiniLM-L-6-v2', 'Direct query-passage scoring', 'Top-5 passage selection'],
    color: '#a78bfa',
  },
  {
    icon: '🤖',
    title: 'Answer Generation',
    description:
      'The top passages are compressed and sent to Llama 3.3 70B (via Groq) with a carefully engineered ' +
      'prompt that enforces source citations, structured formatting, and honest uncertainty when information ' +
      'is insufficient.',
    details: ['Llama 3.3 70B Versatile', 'Grounded answers with citations', 'Intent-specific prompts'],
    color: '#c4b5fd',
  },
];

export default function HowItWorksPage() {
  return (
    <div className="min-h-screen" style={{ background: 'var(--color-bg-primary)' }}>
      {/* Header */}
      <header className="border-b" style={{ borderColor: 'var(--color-border)', background: 'var(--color-bg-secondary)' }}>
        <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="text-xl font-bold" style={{ fontFamily: 'var(--font-heading)', color: 'var(--color-text-primary)' }}>
            ArXiv RAG Assistant
          </Link>
          <Link to="/" className="btn-ghost text-sm">
            ← Back to App
          </Link>
        </div>
      </header>

      <main className="max-w-4xl mx-auto px-6 py-12">
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
                        className="text-xs px-3 py-1 rounded-full font-medium"
                        style={{ background: `${step.color}15`, color: step.color, border: `1px solid ${step.color}33` }}
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
        <div className="mt-16 glass-card p-8 text-center animate-fade-in" style={{ animationDelay: '500ms' }}>
          <h2 className="text-2xl font-bold mb-6" style={{ fontFamily: 'var(--font-heading)' }}>
            System Architecture
          </h2>
          <div className="mb-6 rounded-lg overflow-hidden border" style={{ borderColor: 'var(--color-border)' }}>
            <img 
              src="/architecture.png" 
              alt="System Architecture Diagram" 
              className="w-full h-auto object-cover"
              onError={(e) => {
                e.target.onerror = null;
                // Fallback to .jpg if .png doesn't exist
                if (e.target.src.endsWith('.png')) {
                  e.target.src = '/architecture.jpg';
                } else {
                  e.target.style.display = 'none';
                }
              }}
            />
          </div>
          <p className="text-sm" style={{ color: 'var(--color-text-muted)' }}>
            Built with FastAPI · Qdrant Cloud · Groq · React
          </p>
        </div>

        {/* CTA */}
        <div className="text-center mt-12 animate-fade-in" style={{ animationDelay: '600ms' }}>
          <Link to="/" className="btn-primary inline-block">
            Start Exploring →
          </Link>
        </div>
      </main>
    </div>
  );
}
