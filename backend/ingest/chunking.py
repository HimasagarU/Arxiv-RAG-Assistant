"""
chunking.py — Full-text chunking for the mechanistic interpretability corpus.

Creates retrieval chunks from paper full text only.

Usage:
    conda run -n pytorch python ingest/chunking.py [--source auto] [--reset]
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

import tiktoken
import nltk
from dotenv import load_dotenv

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        raise RuntimeError("NLTK punkt tokenizer is missing. Please run `conda run -n pytorch python -m nltk.downloader punkt punkt_tab` to install it.")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db
from utils.metadata_normalize import normalize_published
from utils.section_labels import normalize_section_label

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CHUNKS_PATH = os.getenv("CHUNKS_PATH", "data/chunks.jsonl")
DEFAULT_CHUNK_SIZE = 450
DEFAULT_OVERLAP_FRAC = 0.15


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def get_tokenizer():
    """Get the tiktoken tokenizer for chunk sizing."""
    return tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Text chunk helpers
# ---------------------------------------------------------------------------


SECTION_HEADER_RE = re.compile(
    r"^(\d+(?:\.\d+)*)\s*\.?\s*(Introduction|Related Work|Background|Method|Methods|"
    r"Methodology|Experiments|Results|Discussion|Conclusion|Conclusions|"
    r"Evaluation|Analysis|Ablation|Abstract|Appendix|Preliminaries)\b",
    re.IGNORECASE | re.MULTILINE,
)

MULTI_SECTION_RE = [
    re.compile(r"^(?:\d+(?:\.\d+)*|M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})|[A-Z])\s*\.?\s*(Introduction|Related Work|Background|Method|Methods|Methodology|Experiments|Results|Discussion|Conclusion|Conclusions|Evaluation|Analysis|Ablation|Abstract|Appendix|Preliminaries)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(Abstract|Acknowledgements?|References|Bibliography)\b", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(?:\d+(?:\.\d+)*)\s+[A-Z][a-zA-Z0-9 ]+", re.MULTILINE), # Generic numbered heading
]

def detect_section_profile(section_name: str) -> str:
    name = section_name.lower()
    if any(x in name for x in ["abstract", "conclusion", "conclusions"]):
        return "small"
    if any(x in name for x in ["method", "methods", "methodology", "experiment", "experiments", "result", "results", "evaluation", "analysis"]):
        return "medium"
    if any(x in name for x in ["related work", "background", "preliminaries", "appendix"]):
        return "large"
    return "default"

def split_into_sections(text: str) -> list[dict]:
    headings = []
    for pattern in MULTI_SECTION_RE:
        for match in pattern.finditer(text):
            heading_text = match.group(0).strip()
            headings.append((match.start(), heading_text))
            
    headings.sort(key=lambda x: x[0])
    valid_headings = []
    last_end = -1
    for start, htext in headings:
        if start >= last_end:
            valid_headings.append((start, htext))
            last_end = start + len(htext)
            
    if not valid_headings:
        return [{"heading": "unknown", "profile": "default", "text": text.strip()}]
        
    sections = []
    if valid_headings[0][0] > 0:
        pre_text = text[:valid_headings[0][0]].strip()
        if pre_text:
            sections.append({"heading": "preface", "profile": "small", "text": pre_text})
            
    for i in range(len(valid_headings)):
        start_idx = valid_headings[i][0]
        end_idx = valid_headings[i+1][0] if i + 1 < len(valid_headings) else len(text)
        h_text = valid_headings[i][1]
        s_text = text[start_idx:end_idx].strip()
        if s_text.startswith(h_text):
            s_text = s_text[len(h_text):].strip()
        sections.append({"heading": h_text, "profile": detect_section_profile(h_text), "text": s_text})
        
    return sections

def _is_special_block(text: str) -> bool:
    text_strip = text.strip()
    if re.match(r"^(?:[-*•]|\d+\.\s|\[\d+\]|\([a-z]\)\s)", text_strip):
        return True
    if "={" in text or "}_{" in text or "\\begin{" in text or "$$" in text:
        return True
    if re.match(r"^(?:Algorithm|Table|Figure|Theorem|Lemma|Proposition|Proof)\b", text_strip, re.IGNORECASE):
        return True
    if text_strip.startswith("def ") or text_strip.startswith("class "):
        return True
    return False

def validate_chunk(chunk: dict) -> tuple[bool, list[str]]:
    """Warn on unusual chunks; only hard-reject clearly broken oversized chunks."""
    warnings = []
    is_valid = True
    if chunk["token_count"] < 50:
        warnings.append(f"Chunk short ({chunk['token_count']} tokens); may be abstract or math-heavy")
    if len(chunk["chunk_text"]) < 0.5 * len(chunk.get("contextual_text", "")):
        warnings.append("Chunk text is less than 50% of contextual text (soft warning)")
        # Note: We no longer reject these, as short but important abstracts/equations 
        # shouldn't be penalized just because contextual metadata is long.
    if chunk["token_count"] > 800:
        warnings.append(f"Chunk too long ({chunk['token_count']} tokens)")
        is_valid = False
    return is_valid, warnings


def detect_section_hint(text: str) -> str:
    """Detect the section a text chunk belongs to."""
    match = SECTION_HEADER_RE.search(text)
    if match:
        name = match.group(2).lower()
        mapping = {
            "introduction": "introduction",
            "related work": "related_work",
            "background": "background",
            "preliminaries": "background",
            "method": "method", "methods": "method", "methodology": "method",
            "experiments": "experiments", "experiment": "experiments",
            "evaluation": "experiments",
            "results": "results", "analysis": "results", "ablation": "results",
            "discussion": "discussion",
            "conclusion": "conclusion", "conclusions": "conclusion",
            "abstract": "abstract",
            "appendix": "appendix",
        }
        return normalize_section_label(mapping.get(name, "other"))
    return normalize_section_label("other")


def _sentence_is_formula_or_table_atomic(sentence: str) -> bool:
    """Avoid splitting technical spans across chunk boundaries."""
    s = sentence.strip()
    if not s:
        return False
    if _is_special_block(s):
        return True
    if len(s) > 120 and (
        "$" in s
        or "\\begin{" in s
        or "}_{" in s
        or ":=" in s
        or "\\[" in s
        or "\\]" in s
    ):
        return True
    if re.search(r"\b(?:Table|Figure|Algorithm)\s+\d", s, re.I):
        return True
    return False


def _extract_local_summary(text: str, max_chars: int = 220) -> str:
    """Create a lightweight local summary from the first one or two sentences."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    summary = " ".join(sentences[:2]).strip()
    if len(summary) > max_chars:
        summary = summary[: max_chars - 3].rstrip() + "..."
    return summary


def _published_json(value):
    return normalize_published(value)


def build_contextual_text(
    *,
    title: str,
    authors: str,
    categories: str,
    section_hint: str,
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
) -> str:
    """Build chunk-specific retrieval text with local context."""
    section_label = (section_hint or "other").replace("_", " ").title()
    local_summary = _extract_local_summary(chunk_text)
    
    prefix = f"Paper: {title} | Authors: {authors} | Categories: {categories} | Section: {section_label} | Chunk {chunk_index + 1}/{total_chunks}"
    if local_summary:
        prefix += f"\nLocal summary: {local_summary}"
        
    return f"{prefix}\n\n{chunk_text}".strip()


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
) -> list[dict]:
    """Split text into overlapping token-based chunks with section hints."""
    if not text or not text.strip():
        return []
    if not 0 <= overlap_frac < 1:
        raise ValueError("overlap_frac must be in the range [0, 1).")

    tokens = tokenizer.encode(text, disallowed_special=())
    if not tokens:
        return []

    overlap_tokens = max(0, int(chunk_size * overlap_frac))
    if overlap_tokens >= chunk_size:
        overlap_tokens = max(0, chunk_size - 1)
    chunks = []
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_str = tokenizer.decode(chunk_tokens)

        section_hint = normalize_section_label(detect_section_hint(chunk_text_str))

        chunks.append({
            "chunk_text": chunk_text_str.strip(),
            "token_count": len(chunk_tokens),
            "section_hint": section_hint,
        })

        if end >= len(tokens):
            break
        start = end - overlap_tokens

    return chunks


def chunk_text_section_sentence(
    text: str,
    tokenizer,
) -> list[dict]:
    """Structure-aware chunking with sentence packing."""
    if not text or not text.strip():
        return []
        
    PROFILE_TARGETS = {"small": 250, "medium": 350, "large": 500, "default": 350}
    MIN_SIZE = 120
    MAX_SIZE = 650
    MAX_OVERLAP_TOKENS = 80
    
    sections = split_into_sections(text)
    chunks = []
    
    for sec in sections:
        sec_text = sec["text"]
        if not sec_text:
            continue

        sec_label = normalize_section_label(sec.get("heading", "other"))
        target_size = PROFILE_TARGETS.get(sec["profile"], 350)
        paragraphs = re.split(r"\n\s*\n", sec_text)
        
        current_chunk_sentences = []
        current_chunk_tokens = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            if _is_special_block(para):
                sentences = [para]
            else:
                sentences = nltk.sent_tokenize(para)
                
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue

                atomic = _sentence_is_formula_or_table_atomic(sent)
                    
                sent_tokens = tokenizer.encode(sent, disallowed_special=())
                sent_len = len(sent_tokens)
                
                if atomic and current_chunk_sentences and (
                    current_chunk_tokens + sent_len > MAX_SIZE
                    or (current_chunk_tokens + sent_len > target_size and current_chunk_tokens >= MIN_SIZE)
                ):
                    chunk_text_str = " ".join(current_chunk_sentences)
                    chunks.append({
                        "chunk_text": chunk_text_str.strip(),
                        "token_count": current_chunk_tokens,
                        "section_hint": sec_label,
                    })
                    current_chunk_sentences = []
                    current_chunk_tokens = 0

                if (current_chunk_tokens + sent_len > target_size and current_chunk_tokens >= MIN_SIZE) or \
                   (current_chunk_tokens + sent_len > MAX_SIZE and current_chunk_sentences):
                    chunk_text_str = " ".join(current_chunk_sentences)
                    chunks.append({
                        "chunk_text": chunk_text_str.strip(),
                        "token_count": current_chunk_tokens,
                        "section_hint": sec_label,
                    })
                    
                    overlap_sentences = []
                    overlap_tokens = 0
                    for prev_sent in reversed(current_chunk_sentences):
                        prev_tok_len = len(tokenizer.encode(prev_sent, disallowed_special=()))
                        if overlap_tokens + prev_tok_len > MAX_OVERLAP_TOKENS or len(overlap_sentences) >= 2:
                            break
                        overlap_sentences.insert(0, prev_sent)
                        overlap_tokens += prev_tok_len
                        
                    current_chunk_sentences = overlap_sentences
                    current_chunk_tokens = overlap_tokens
                    
                current_chunk_sentences.append(sent)
                current_chunk_tokens += sent_len
                
        if current_chunk_sentences:
            chunk_text_str = " ".join(current_chunk_sentences)
            if current_chunk_tokens >= 50 or not chunks:
                chunks.append({
                    "chunk_text": chunk_text_str.strip(),
                    "token_count": current_chunk_tokens,
                    "section_hint": sec_label,
                })
                
    return chunks


# ---------------------------------------------------------------------------
# Source text builder
# ---------------------------------------------------------------------------


def _strip_non_retrieval_sections(text: str) -> str:
    """Remove references, bibliography, acknowledgements, and appendix sections.
    
    These sections don't contribute useful retrieval signal and add noise.
    """
    # Patterns that mark the start of non-retrieval sections
    cut_patterns = [
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?(?:References|Bibliography|Works Cited)\s*\n",
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?Acknowledg(?:e)?ments?\s*\n",
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?Appendix\s*(?:[A-Z])?\s*\n",
        r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\s*\.?\s*)?Supplementary\s+Materials?\s*\n",
    ]
    
    earliest_cut = len(text)
    for pattern in cut_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() < earliest_cut:
            earliest_cut = match.start()
    
    if earliest_cut < len(text):
        text = text[:earliest_cut].rstrip()
    
    return text


def build_chunk_source_text(paper: dict, source_mode: str = "auto") -> str:
    """Build the text to chunk from a paper record."""
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    full_text = (paper.get("full_text") or "").strip()

    if source_mode == "abstract":
        base = abstract
    elif source_mode == "full_text":
        if not full_text:
            return ""
        base = full_text
    elif source_mode == "auto":
        base = full_text if full_text else abstract
    else:
        base = full_text if full_text else abstract

    if not base:
        return ""

    # Strip references, acknowledgements, appendix before chunking
    if len(base) > 500:  # Only for full text, not abstracts
        base = _strip_non_retrieval_sections(base)

    return f"{title}. {base}" if title else base


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------


def chunk_paper(
    paper: dict,
    tokenizer,
    source_mode: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    strategy: str = "section-sentence",
) -> list[dict]:
    """Chunk a single paper into text chunks."""
    paper_id = paper["paper_id"]
    title = paper.get("title", "")
    authors = paper.get("authors", "")
    categories = paper.get("categories", "")
    layer = paper.get("layer", "core")
    published = _published_json(paper.get("published"))
    abstract_text = (paper.get("abstract") or "").strip()

    all_chunks = []

    # Text chunks from full text / abstract
    source_text = build_chunk_source_text(paper, source_mode)
    if source_text:
        if strategy == "section-sentence":
            text_chunks = chunk_text_section_sentence(source_text, tokenizer)
        else:
            text_chunks = chunk_text(source_text, tokenizer, chunk_size, overlap_frac)
            
        chunk_source = source_mode
        if source_mode == "auto":
            chunk_source = "full_text" if paper.get("full_text", "").strip() else "abstract"

        for idx, tc in enumerate(text_chunks):
            chunk_id = f"{paper_id}_text_{idx}"
            sec_hint = normalize_section_label(tc.get("section_hint", "other"))
            contextual_text = build_contextual_text(
                title=title,
                authors=authors,
                categories=categories,
                section_hint=sec_hint,
                chunk_text=tc["chunk_text"],
                chunk_index=idx,
                total_chunks=len(text_chunks),
            )
            
            chunk_data = {
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "chunk_type": "text",
                "modality": "text",
                "chunk_text": tc["chunk_text"],
                "contextual_text": contextual_text,
                "section_hint": sec_hint,
                "page_start": None,
                "page_end": None,
                "token_count": tc["token_count"],
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "chunk_source": chunk_source,
                "layer": layer,
                "artifact_meta": {},
                # Extra metadata for JSONL compat
                "title": title,
                "authors": authors,
                "categories": categories,
                "published": published,
            }
            if idx == 0 and abstract_text:
                chunk_data["paper_abstract"] = abstract_text

            is_valid, warnings = validate_chunk(chunk_data)
            if warnings:
                log.debug(f"Chunk warning for {chunk_id}: {', '.join(warnings)}")
                
            if is_valid:
                all_chunks.append(chunk_data)

    return all_chunks


def run_chunking(
    source_mode: str = "auto",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap_frac: float = DEFAULT_OVERLAP_FRAC,
    limit: int = 0,
    reset: bool = False,
    strategy: str = "section-sentence",
    offline: bool = False,
    papers_file: str = None,
):
    """Run chunking for all papers in the corpus."""
    tokenizer = get_tokenizer()
    db = None

    if papers_file:
        log.info(f"Loading papers from local file: {papers_file}")
        papers = []
        with open(papers_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    papers.append(json.loads(line))
        if limit > 0:
            papers = papers[:limit]
    else:
        db = get_db()
        db.run_migrations()
        papers = db.get_all_papers(limit=limit)

    log.info(f"Chunking {len(papers)} papers (strategy={strategy}, source={source_mode})")

    if reset:
        if offline:
            log.info("Reset flag set (offline): Database not truncated. Run without --offline or clear JSONL manually.")
        else:
            if not db:
                db = get_db()
            log.info("Reset flag set: clearing existing chunks before rebuild.")
            db.delete_all_chunks()
            db.commit()

    chunks_path = Path(CHUNKS_PATH)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    type_counts = {"text": 0}

    with open(chunks_path, "w", encoding="utf-8") as f:
        for idx, paper in enumerate(papers):
            paper_chunks = chunk_paper(paper, tokenizer, source_mode, chunk_size, overlap_frac, strategy)

            if not paper_chunks:
                continue

            for chunk in paper_chunks:
                # Write to JSONL
                jsonl_record = {
                    "chunk_id": chunk["chunk_id"],
                    "paper_id": chunk["paper_id"],
                    "chunk_type": chunk["chunk_type"],
                    "modality": chunk["modality"],
                    "chunk_text": chunk["chunk_text"],
                    "contextual_text": chunk.get("contextual_text", chunk["chunk_text"]),
                    "title": chunk.get("title", ""),
                    "authors": chunk.get("authors", ""),
                    "categories": chunk.get("categories", ""),
                    "section_hint": chunk.get("section_hint", "other"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "token_count": chunk["token_count"],
                    "chunk_index": chunk.get("chunk_index", 0),
                    "total_chunks": chunk.get("total_chunks", 1),
                    "chunk_source": chunk.get("chunk_source", "full_text"),
                    "layer": chunk.get("layer", "core"),
                    "published": chunk.get("published"),
                    "paper_abstract": chunk.get("paper_abstract"),
                }
                f.write(json.dumps(jsonl_record, ensure_ascii=False, default=str) + "\n")

                # Insert to PostgreSQL (if not offline)
                if not offline:
                    if not db:
                        db = get_db()
                    db.insert_chunk(chunk)

                type_counts[chunk["chunk_type"]] = type_counts.get(chunk["chunk_type"], 0) + 1
                total_chunks += 1

            if (idx + 1) % 50 == 0:
                if not offline:
                    db.commit()
                mode_str = " [Offline]" if offline else ""
                log.info(f"  Chunked {idx + 1}/{len(papers)} papers ({total_chunks} chunks){mode_str}")

    if not offline:
        if not db:
            db = get_db()
        db.commit()

    log.info("\nChunking complete:")
    log.info(f"  Total chunks:    {total_chunks}")
    log.info(f"  By type:         {type_counts}")
    log.info(f"  JSONL:           {chunks_path}")
    if not offline:
        log.info(f"  PostgreSQL:      {db.count_chunks()} rows")

    if db:
        db.close()
    return total_chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Full-text chunking")
    parser.add_argument("--source", choices=["abstract", "full_text", "auto"], default="auto",
                        help="Text source mode")
    parser.add_argument("--strategy", choices=["token", "section-sentence"], default="section-sentence",
                        help="Chunking strategy (default: section-sentence)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help="Target chunk size in tokens (for token strategy)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP_FRAC,
                        help="Overlap fraction between chunks (for token strategy)")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to chunk (0=all)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete all existing chunks before rebuilding")
    parser.add_argument("--offline", action="store_true",
                        help="Chunk offline to JSONL without inserting to PostgreSQL")
    parser.add_argument("--papers-file", help="Path to local papers.jsonl to read from instead of DB")
    args = parser.parse_args()

    run_chunking(
        source_mode=args.source,
        chunk_size=args.chunk_size,
        overlap_frac=args.overlap,
        limit=args.limit,
        reset=args.reset,
        strategy=args.strategy,
        offline=args.offline,
        papers_file=args.papers_file,
    )


if __name__ == "__main__":
    main()
