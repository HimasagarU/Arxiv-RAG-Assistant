"""Quick smoke test for core modules."""
import sys
sys.path.insert(0, ".")

# Test database
from db.database import get_db
db = get_db()
db.run_migrations()
print(f"DB OK — papers: {db.count_papers()}")

# Test ingestion imports
from ingest.ingest_arxiv import SEED_PAPERS, build_keyword_query
print(f"Seeds defined: {len(SEED_PAPERS)}")
print(f"Keyword query built: {len(build_keyword_query())} chars")

# Test relevance filter
from ingest.citation_expander import is_relevant, assign_layer
assert is_relevant("mechanistic interpretability of transformers", "")
assert is_relevant("Sparse autoencoders find features", "")
assert not is_relevant("Image classification with CNNs", "generic deep learning paper")
print("Relevance filter OK")

# Test layer assignment
assert assign_layer(2017) == "prerequisite"
assert assign_layer(2021) == "foundation"
assert assign_layer(2023) == "core"
assert assign_layer(2025) == "latest"
print("Layer assignment OK")

# Test chunking imports
from ingest.chunking import get_tokenizer, chunk_text
tok = get_tokenizer()
chunks = chunk_text("This is a test. Mechanistic interpretability studies circuits.", tok, chunk_size=50)
print(f"Chunking OK — {len(chunks)} chunks from test text")

# Test timeline balancer
from ingest.timeline_balancer import check_balance
report = check_balance(db)
print(f"Timeline balance OK — total: {report['total_papers']}")

print("\n=== ALL SMOKE TESTS PASSED ===")
db.close()
