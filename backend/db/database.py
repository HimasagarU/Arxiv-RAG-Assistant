"""
database.py — PostgreSQL connection manager and CRUD helpers.

Provides a single Database class that handles:
  - Connection pooling via psycopg
  - Schema migration (runs schema.sql on first connect)
    - CRUD helpers for papers, chunks, citation edges
  - Layer assignment and timeline balance queries

Usage:
    from db.database import get_db
    db = get_db()
    db.upsert_paper({...})
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/arxiv_rag"
)
SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_db_instance: Optional["Database"] = None


def get_db(database_url: str = None) -> "Database":
    """Return the singleton Database instance, creating it on first call."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database(database_url or DEFAULT_DATABASE_URL)
    return _db_instance


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """PostgreSQL connection manager with CRUD helpers."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._conn: Optional[psycopg.Connection] = None
        log.info(f"Database URL: {database_url.split('@')[-1]}")  # log host only

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> psycopg.Connection:
        """Get or create the database connection."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(
                self.database_url,
                row_factory=dict_row,
                autocommit=False,
            )
            log.info("PostgreSQL connection established.")
        return self._conn

    @property
    def conn(self) -> psycopg.Connection:
        return self.connect()

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
            log.info("PostgreSQL connection closed.")
        self._conn = None

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    # ------------------------------------------------------------------
    # Schema migration
    # ------------------------------------------------------------------

    def run_migrations(self):
        """Execute schema.sql to create tables if they don't exist."""
        if not SCHEMA_PATH.exists():
            log.warning(f"Schema file not found: {SCHEMA_PATH}")
            return
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.conn.cursor() as cur:
            cur.execute(sql)
        self.conn.commit()
        log.info("Database schema applied successfully.")

    # ------------------------------------------------------------------
    # Papers CRUD
    # ------------------------------------------------------------------

    def upsert_paper(self, paper: dict):
        """Insert or update a paper record (idempotent by paper_id)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO papers (
                    paper_id, title, abstract, authors, categories,
                    pdf_url, published, updated, full_text,
                    download_status, parse_status, local_pdf_path,
                    quality_score, is_seed, layer, source, semantic_scholar_id
                ) VALUES (
                    %(paper_id)s, %(title)s, %(abstract)s, %(authors)s, %(categories)s,
                    %(pdf_url)s, %(published)s, %(updated)s, %(full_text)s,
                    %(download_status)s, %(parse_status)s, %(local_pdf_path)s,
                    %(quality_score)s, %(is_seed)s, %(layer)s, %(source)s, %(semantic_scholar_id)s
                )
                ON CONFLICT (paper_id) DO UPDATE SET
                    title               = COALESCE(NULLIF(EXCLUDED.title, ''), papers.title),
                    abstract            = COALESCE(NULLIF(EXCLUDED.abstract, ''), papers.abstract),
                    authors             = COALESCE(NULLIF(EXCLUDED.authors, ''), papers.authors),
                    categories          = COALESCE(NULLIF(EXCLUDED.categories, ''), papers.categories),
                    pdf_url             = COALESCE(NULLIF(EXCLUDED.pdf_url, ''), papers.pdf_url),
                    published           = COALESCE(EXCLUDED.published, papers.published),
                    updated             = COALESCE(EXCLUDED.updated, papers.updated),
                    full_text           = COALESCE(NULLIF(EXCLUDED.full_text, ''), papers.full_text),
                    download_status     = EXCLUDED.download_status,
                    parse_status        = EXCLUDED.parse_status,
                    local_pdf_path      = COALESCE(NULLIF(EXCLUDED.local_pdf_path, ''), papers.local_pdf_path),
                    quality_score       = GREATEST(EXCLUDED.quality_score, papers.quality_score),
                    is_seed             = EXCLUDED.is_seed OR papers.is_seed,
                    layer               = EXCLUDED.layer,
                    source              = EXCLUDED.source,
                    semantic_scholar_id = COALESCE(NULLIF(EXCLUDED.semantic_scholar_id, ''), papers.semantic_scholar_id)
                """,
                _paper_defaults(paper),
            )

    def get_paper(self, paper_id: str) -> Optional[dict]:
        """Fetch a single paper by ID."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE paper_id = %s", (paper_id,))
            return cur.fetchone()

    def paper_exists(self, paper_id: str) -> bool:
        """Check if a paper exists."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM papers WHERE paper_id = %s", (paper_id,)
            )
            return cur.fetchone() is not None

    def count_papers(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM papers")
            return cur.fetchone()["cnt"]

    def get_all_papers(self, limit: int = 0) -> list[dict]:
        """Fetch all papers, optionally limited."""
        query = "SELECT * FROM papers ORDER BY published DESC"
        if limit > 0:
            query += f" LIMIT {limit}"
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    def get_seed_papers(self) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT * FROM papers WHERE is_seed = TRUE ORDER BY published")
            return cur.fetchall()

    def get_papers_missing_full_text(self, limit: int = 0, include_failed: bool = False) -> list[dict]:
        query = """
            SELECT * FROM papers
                        WHERE (full_text IS NULL OR TRIM(full_text) = '')
                            AND paper_id ~ '^\\d{4}\\.\\d{4,5}$'
        """
        if not include_failed:
            query += " AND COALESCE(download_status, '') NOT IN ('failed', 'skipped') AND COALESCE(parse_status, '') != 'failed'"
        query += " ORDER BY published DESC"
        if limit > 0:
            query += f" LIMIT {limit}"
        with self.conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()

    def update_paper_field(self, paper_id: str, field: str, value):
        """Update a single field on a paper."""
        # Whitelist allowed fields to prevent SQL injection
        allowed = {
            "full_text", "download_status", "parse_status", "local_pdf_path",
            "quality_score", "layer", "source", "semantic_scholar_id",
        }
        if field not in allowed:
            raise ValueError(f"Field '{field}' is not updatable.")
        # Sanitize NUL bytes for PostgreSQL
        if isinstance(value, str):
            value = value.replace('\x00', '')

        with self.conn.cursor() as cur:
            cur.execute(
                f"UPDATE papers SET {field} = %s WHERE paper_id = %s",
                (value, paper_id),
            )

    # ------------------------------------------------------------------
    # Chunks CRUD
    # ------------------------------------------------------------------

    def insert_chunk(self, chunk: dict):
        """Insert a single chunk."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO chunks (
                    chunk_id, paper_id, chunk_type, modality, title, authors, categories, chunk_text,
                    section_hint, page_start, page_end, token_count,
                    chunk_index, total_chunks, chunk_source, layer, artifact_meta
                ) VALUES (
                    %(chunk_id)s, %(paper_id)s, %(chunk_type)s, %(modality)s, %(title)s, %(authors)s, %(categories)s, %(chunk_text)s,
                    %(section_hint)s, %(page_start)s, %(page_end)s, %(token_count)s,
                    %(chunk_index)s, %(total_chunks)s, %(chunk_source)s, %(layer)s, %(artifact_meta)s
                )
                ON CONFLICT (chunk_id) DO UPDATE SET
                    chunk_text   = EXCLUDED.chunk_text,
                    section_hint = EXCLUDED.section_hint,
                    token_count  = EXCLUDED.token_count,
                    layer        = EXCLUDED.layer
                """,
                {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "chunk_type": chunk.get("chunk_type", "text"),
                    "modality": chunk.get("modality", "text"),
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", ""),
                    "categories": chunk.get("categories", ""),
                    "chunk_text": chunk["chunk_text"],
                    "section_hint": chunk.get("section_hint", "other"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "token_count": chunk.get("token_count", 0),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "chunk_source": chunk.get("chunk_source", "full_text"),
                    "layer": chunk.get("layer", "core"),
                    "artifact_meta": json.dumps(chunk.get("artifact_meta", {})),
                },
            )

    def search_chunks_fts(
        self,
        query: str,
        limit: int = 50,
        category: Optional[str] = None,
        author: Optional[str] = None,
        start_year: Optional[int] = None,
    ) -> list[dict]:
        """Search chunks with PostgreSQL full-text search.

        This is the primary lexical retrieval path for the hybrid
        retrieval pipeline (dense + FTS).
        """
        if not query.strip():
            return []

        conditions = ["c.search_tsv @@ websearch_to_tsquery('english', %s)"]
        params: list = [query]

        if category:
            conditions.append("c.categories ILIKE %s")
            params.append(f"%{category.strip()}%")
        if author:
            conditions.append("c.authors ILIKE %s")
            params.append(f"%{author.strip()}%")
        if start_year:
            conditions.append("EXTRACT(YEAR FROM p.published) >= %s")
            params.append(start_year)

        params.append(limit)
        sql = f"""
            SELECT
                c.chunk_id,
                c.paper_id,
                c.chunk_text,
                c.chunk_type,
                c.modality,
                c.section_hint,
                c.layer,
                c.chunk_source,
                c.token_count,
                c.chunk_index,
                c.total_chunks,
                c.title,
                c.authors,
                c.categories,
                p.title AS paper_title,
                p.authors AS paper_authors,
                p.categories AS paper_categories,
                p.published,
                ts_rank_cd(c.search_tsv, websearch_to_tsquery('english', %s)) AS lexical_score
            FROM chunks c
            JOIN papers p ON p.paper_id = c.paper_id
            WHERE {' AND '.join(conditions)}
            ORDER BY lexical_score DESC, p.published DESC NULLS LAST, c.chunk_index ASC
            LIMIT %s
        """
        rank_query = query

        with self.conn.cursor() as cur:
            cur.execute(sql, [rank_query, *params])
            rows = cur.fetchall()

        return rows

    def insert_chunks_batch(self, chunks: list[dict]):
        """Insert multiple chunks in a single transaction."""
        if not chunks:
            return
            
        sql = """
            INSERT INTO chunks (
                chunk_id, paper_id, chunk_type, modality, title, authors, categories, chunk_text,
                section_hint, page_start, page_end, token_count,
                chunk_index, total_chunks, chunk_source, layer, artifact_meta
            ) VALUES (
                %(chunk_id)s, %(paper_id)s, %(chunk_type)s, %(modality)s, %(title)s, %(authors)s, %(categories)s, %(chunk_text)s,
                %(section_hint)s, %(page_start)s, %(page_end)s, %(token_count)s,
                %(chunk_index)s, %(total_chunks)s, %(chunk_source)s, %(layer)s, %(artifact_meta)s
            )
            ON CONFLICT (chunk_id) DO UPDATE SET
                chunk_text   = EXCLUDED.chunk_text,
                section_hint = EXCLUDED.section_hint,
                token_count  = EXCLUDED.token_count,
                layer        = EXCLUDED.layer
        """
        
        params = [
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "chunk_type": chunk.get("chunk_type", "text"),
                "modality": chunk.get("modality", "text"),
                "title": chunk.get("title", ""),
                "authors": chunk.get("authors", ""),
                "categories": chunk.get("categories", ""),
                "chunk_text": chunk["chunk_text"],
                "section_hint": chunk.get("section_hint", "other"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "token_count": chunk.get("token_count", 0),
                "chunk_index": chunk.get("chunk_index", 0),
                "total_chunks": chunk.get("total_chunks", 1),
                "chunk_source": chunk.get("chunk_source", "full_text"),
                "layer": chunk.get("layer", "core"),
                "artifact_meta": json.dumps(chunk.get("artifact_meta", {})),
            }
            for chunk in chunks
        ]
        
        with self.conn.cursor() as cur:
            cur.executemany(sql, params)

    def get_chunks(self, paper_id: str) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM chunks WHERE paper_id = %s ORDER BY chunk_index",
                (paper_id,),
            )
            return cur.fetchall()

    def count_chunks(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM chunks")
            return cur.fetchone()["cnt"]

    def count_chunks_by_type(self) -> dict:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_type, COUNT(*) AS cnt FROM chunks GROUP BY chunk_type"
            )
            return {row["chunk_type"]: row["cnt"] for row in cur.fetchall()}

    def delete_all_chunks(self):
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")

    # ------------------------------------------------------------------
    # Citation edges
    # ------------------------------------------------------------------

    def insert_citation_edge(self, source_id: str, target_id: str, direction: str):
        """Insert a citation edge (idempotent)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO citation_edges (source_paper_id, target_paper_id, direction)
                VALUES (%s, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (source_id, target_id, direction),
            )

    def get_references(self, paper_id: str) -> list[dict]:
        """Papers referenced BY this paper (backward)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.* FROM papers p
                JOIN citation_edges ce ON ce.target_paper_id = p.paper_id
                WHERE ce.source_paper_id = %s AND ce.direction = 'reference'
                """,
                (paper_id,),
            )
            return cur.fetchall()

    def get_citations(self, paper_id: str) -> list[dict]:
        """Papers that CITE this paper (forward)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.* FROM papers p
                JOIN citation_edges ce ON ce.target_paper_id = p.paper_id
                WHERE ce.source_paper_id = %s AND ce.direction = 'citation'
                """,
                (paper_id,),
            )
            return cur.fetchall()

    # ------------------------------------------------------------------
    # Timeline & layer analytics
    # ------------------------------------------------------------------

    def get_layer_distribution(self) -> dict:
        """Get count of papers per layer."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT layer, COUNT(*) AS cnt FROM papers GROUP BY layer ORDER BY layer"
            )
            return {row["layer"]: row["cnt"] for row in cur.fetchall()}

    def get_era_distribution(self) -> dict:
        """Get paper counts by era (early/middle/recent)."""
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    CASE
                        WHEN EXTRACT(YEAR FROM published) <= 2016 THEN 'pre_2017'
                        WHEN EXTRACT(YEAR FROM published) <= 2020 THEN 'early_2017_2020'
                        WHEN EXTRACT(YEAR FROM published) <= 2023 THEN 'middle_2021_2023'
                        ELSE 'recent_2024_plus'
                    END AS era,
                    COUNT(*) AS cnt
                FROM papers
                WHERE published IS NOT NULL
                GROUP BY era
                ORDER BY era
                """
            )
            return {row["era"]: row["cnt"] for row in cur.fetchall()}

    def get_corpus_health(self) -> dict:
        """Return a corpus health summary."""
        total = self.count_papers()
        layers = self.get_layer_distribution()
        eras = self.get_era_distribution()

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM papers WHERE is_seed = TRUE")
            seed_count = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM papers WHERE full_text IS NOT NULL AND TRIM(full_text) != ''"
            )
            with_text = cur.fetchone()["cnt"]

        era_total = sum(eras.values()) or 1
        return {
            "total_papers": total,
            "seed_papers": seed_count,
            "with_full_text": with_text,
            "without_full_text": total - with_text,
            "layers": layers,
            "eras": eras,
            "era_percentages": {k: round(v / era_total * 100, 1) for k, v in eras.items()},
        }

    # ------------------------------------------------------------------
    # Reset / truncate
    # ------------------------------------------------------------------

    def truncate_all(self):
        """Truncate all tables (correct FK order). Use for full corpus reset."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM chunks")
            cur.execute("DELETE FROM citation_edges")
            cur.execute("DELETE FROM papers")
        self.conn.commit()
        log.info("All tables truncated.")

    def truncate_chunks_table(self) -> None:
        """Remove all chunk rows (faster than DELETE for large tables)."""
        with self.conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE chunks RESTART IDENTITY CASCADE")
        self.conn.commit()
        log.info("Truncated chunks table.")

    def clear_full_text_for_paper_ids(self, paper_ids: list[str]) -> int:
        """Set full_text empty and parse_status pending for re-extraction from PDFs."""
        if not paper_ids:
            return 0
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE papers
                SET full_text = '', parse_status = 'pending'
                WHERE paper_id = ANY(%s)
                """,
                (paper_ids,),
            )
            n = cur.rowcount or 0
        self.conn.commit()
        return n

    def neon_metadata_report(self, data_dir: Optional[Path] = None) -> dict:
        """Summarize Postgres paper/chunk coverage vs optional local artifacts."""
        base = data_dir or Path(os.getenv("DATA_DIR", "data"))
        report: dict = {}

        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS cnt FROM papers")
            report["papers_rows"] = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM chunks")
            report["chunks_rows"] = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM papers WHERE title IS NULL OR TRIM(title) = ''"
            )
            report["papers_missing_title"] = cur.fetchone()["cnt"]

            cur.execute(
                "SELECT COUNT(*) AS cnt FROM papers WHERE abstract IS NULL OR TRIM(abstract) = ''"
            )
            report["papers_missing_abstract"] = cur.fetchone()["cnt"]

            cur.execute(
                """
                SELECT download_status, COUNT(*) AS cnt
                FROM papers
                GROUP BY download_status
                ORDER BY cnt DESC
                """
            )
            report["papers_by_download_status"] = {
                row["download_status"]: row["cnt"] for row in cur.fetchall()
            }

            cur.execute(
                """
                SELECT parse_status, COUNT(*) AS cnt
                FROM papers
                GROUP BY parse_status
                ORDER BY cnt DESC
                """
            )
            report["papers_by_parse_status"] = {
                row["parse_status"]: row["cnt"] for row in cur.fetchall()
            }

        papers_meta = base / "papers_meta.json"
        chunks_jsonl = base / "chunks.jsonl"
        if papers_meta.exists():
            raw = json.loads(papers_meta.read_text(encoding="utf-8"))
            report["papers_meta_keys"] = len(raw) if isinstance(raw, dict) else 0
        else:
            report["papers_meta_keys"] = None

        if chunks_jsonl.exists():
            n = 0
            with open(chunks_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        n += 1
            report["chunks_jsonl_lines"] = n
        else:
            report["chunks_jsonl_lines"] = None

        report["corpus_health"] = self.get_corpus_health()
        return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper_defaults(paper: dict) -> dict:
    """Fill missing paper fields with safe defaults."""
    defaults = {
        "paper_id": "",
        "title": "",
        "abstract": "",
        "authors": "",
        "categories": "",
        "pdf_url": "",
        "published": None,
        "updated": None,
        "full_text": "",
        "download_status": "pending",
        "parse_status": "pending",
        "local_pdf_path": "",
        "quality_score": 0.0,
        "is_seed": False,
        "layer": "core",
        "source": "arxiv",
        "semantic_scholar_id": "",
    }
    result = {**defaults, **paper}
    # Ensure None for empty timestamps
    for ts_field in ("published", "updated"):
        if result[ts_field] == "":
            result[ts_field] = None
    return result
