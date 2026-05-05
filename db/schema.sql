-- ============================================================
-- ArXiv RAG Assistant — PostgreSQL Schema
-- Mechanistic Interpretability Corpus
-- ============================================================

-- Papers table: one row per unique paper
CREATE TABLE IF NOT EXISTS papers (
    paper_id            TEXT PRIMARY KEY,
    title               TEXT NOT NULL,
    abstract            TEXT NOT NULL DEFAULT '',
    authors             TEXT DEFAULT '',
    categories          TEXT DEFAULT '',
    pdf_url             TEXT DEFAULT '',
    published           TIMESTAMPTZ,
    updated             TIMESTAMPTZ,
    full_text           TEXT DEFAULT '',
    download_status     TEXT DEFAULT 'pending',   -- pending, downloaded, failed
    parse_status        TEXT DEFAULT 'pending',   -- pending, parsed, failed
    local_pdf_path      TEXT DEFAULT '',
    quality_score       REAL DEFAULT 0.0,
    is_seed             BOOLEAN DEFAULT FALSE,
    layer               TEXT DEFAULT 'core',      -- prerequisite, foundation, core, latest
    source              TEXT DEFAULT 'arxiv',     -- arxiv, semantic_scholar, transformer_circuits
    semantic_scholar_id TEXT DEFAULT '',
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Chunks table: retrieval units built from artifacts / full text
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    paper_id        TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    chunk_type      TEXT NOT NULL DEFAULT 'text',  -- text only
    modality        TEXT NOT NULL DEFAULT 'text',   -- text, structured, visual
    title           TEXT DEFAULT '',
    authors         TEXT DEFAULT '',
    categories      TEXT DEFAULT '',
    chunk_text      TEXT NOT NULL,
    section_hint    TEXT DEFAULT 'other',
    page_start      INTEGER,
    page_end        INTEGER,
    token_count     INTEGER DEFAULT 0,
    chunk_index     INTEGER DEFAULT 0,
    total_chunks    INTEGER DEFAULT 1,
    chunk_source    TEXT DEFAULT 'full_text',
    layer           TEXT DEFAULT 'core',
    artifact_meta   JSONB DEFAULT '{}',
    search_tsv      tsvector GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(categories, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(authors, '')), 'B') ||
        setweight(to_tsvector('english', coalesce(section_hint, '')), 'C') ||
        setweight(to_tsvector('english', coalesce(chunk_text, '')), 'D')
    ) STORED,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Citation graph edges (seed-driven expansion tracking)
CREATE TABLE IF NOT EXISTS citation_edges (
    source_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    target_paper_id TEXT NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    direction       TEXT NOT NULL,  -- reference (backward) or citation (forward)
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_paper_id, target_paper_id, direction)
);

-- Indexes for fast lookups
CREATE INDEX IF NOT EXISTS idx_papers_layer       ON papers(layer);
CREATE INDEX IF NOT EXISTS idx_papers_is_seed     ON papers(is_seed);
CREATE INDEX IF NOT EXISTS idx_papers_source      ON papers(source);
CREATE INDEX IF NOT EXISTS idx_papers_published   ON papers(published);
CREATE INDEX IF NOT EXISTS idx_chunks_paper        ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_chunks_type         ON chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_chunks_modality     ON chunks(modality);
CREATE INDEX IF NOT EXISTS idx_chunks_section      ON chunks(section_hint);
CREATE INDEX IF NOT EXISTS idx_chunks_layer        ON chunks(layer);
CREATE INDEX IF NOT EXISTS idx_chunks_search_tsv   ON chunks USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS idx_citation_source     ON citation_edges(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_citation_target     ON citation_edges(target_paper_id);
