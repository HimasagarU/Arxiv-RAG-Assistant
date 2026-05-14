"""
fetch_papers.py — Download paper metadata and full text to a local JSONL file.

This allows for truly offline chunking by caching paper data locally.

Usage:
    conda run -n pytorch python ingest/fetch_papers.py [--output data/papers.jsonl] [--limit 0]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_PAPERS_PATH = "data/papers.jsonl"

def fetch_papers(output_path: str, limit: int = 0):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    db = get_db()
    log.info(f"Fetching papers from database (limit={limit})...")
    
    papers = db.get_all_papers(limit=limit)
    log.info(f"Retrieved {len(papers)} papers. Saving to {path}...")
    
    # Custom encoder for datetime objects if any (though get_all_papers usually returns strings/None for dates depending on row_factory)
    def json_serial(obj):
        from datetime import datetime
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    with open(path, "w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, default=json_serial, ensure_ascii=False) + "\n")
            
    db.close()
    log.info(f"Successfully cached {len(papers)} papers locally.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch papers from DB to local JSONL")
    parser.add_argument("--output", default=DEFAULT_PAPERS_PATH, help="Output JSONL path")
    parser.add_argument("--limit", type=int, default=0, help="Limit papers to fetch")
    args = parser.parse_args()
    
    fetch_papers(args.output, args.limit)
