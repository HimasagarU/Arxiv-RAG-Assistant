"""
reset_corpus.py — Archive old data and reset everything for clean ingestion.

1. Renames data/ → data_archive_legacy/
2. Clears PostgreSQL tables
3. Deletes all objects from R2 bucket
4. Creates fresh data/ directory
"""
import os
import sys
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db.database import get_db
from ingest.r2_storage import R2Storage

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    data_dir = PROJECT_ROOT / "data"
    archive_dir = PROJECT_ROOT / "data_archive_legacy"

    # Step 1: Archive old data/
    if data_dir.exists():
        if archive_dir.exists():
            print(f"Archive dir already exists: {archive_dir}")
            print("Removing old archive first...")
            shutil.rmtree(archive_dir)
        print(f"Archiving {data_dir} → {archive_dir}")
        data_dir.rename(archive_dir)
        print(f"  Done. Old data preserved in {archive_dir}")
    else:
        print("No data/ directory to archive.")

    # Step 2: Create fresh data/
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "pdfs").mkdir(exist_ok=True)
    print(f"Created fresh {data_dir}")

    # Step 3: Clear PostgreSQL
    print("\nClearing PostgreSQL tables...")
    db = get_db()
    db.run_migrations()
    db.truncate_all()
    print(f"  Papers: {db.count_papers()}")
    print(f"  Chunks: {db.count_chunks()}")
    db.close()

    # Step 4: Clear R2 bucket
    print("\nClearing Cloudflare R2 bucket...")
    r2 = R2Storage()
    if r2.is_available:
        deleted = r2.delete_all_objects()
        print(f"  Deleted {deleted} objects from R2")
    else:
        print("  R2 not configured, skipping.")

    print("\n" + "=" * 50)
    print("CORPUS RESET COMPLETE")
    print("=" * 50)
    print("Run the pipeline: scripts\\run_mech_interp_pipeline.bat")


if __name__ == "__main__":
    main()
