"""
build_bm25.py — Legacy stub.

Lexical retrieval is handled by PostgreSQL full-text search (FTS).
No separate index file is built or loaded.
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main():
    log.info("PostgreSQL full-text search is the lexical retrieval layer.")
    log.info("No separate lexical index is built or loaded.")


if __name__ == "__main__":
    main()
