import hashlib
import logging
import os
import zipfile
from pathlib import Path

import boto3
from botocore.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _bm25_doc_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import joblib

        bm25 = joblib.load(path)
        corpus_size = getattr(bm25, "corpus_size", None)
        if isinstance(corpus_size, int):
            return corpus_size
        doc_len = getattr(bm25, "doc_len", None)
        return len(doc_len) if doc_len is not None else 0
    except Exception as exc:
        log.warning("Could not inspect local BM25 artifact: %s", exc)
        return 0


def _local_artifacts_consistent(dest_dir: Path) -> bool:
    bm25_docs = _bm25_doc_count(dest_dir / "bm25_v1.pkl")
    meta_rows = _count_jsonl(dest_dir / "chunks_meta.jsonl")
    if not bm25_docs or not meta_rows:
        return False
    if bm25_docs != meta_rows:
        log.warning(
            "Local artifact mismatch: bm25 has %s docs but chunks_meta has %s rows. "
            "Will refresh from R2.",
            bm25_docs,
            meta_rows,
        )
        return False
    return True


def fetch_and_extract():
    # --- Config from Environment ---
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
    R2_ENDPOINT = os.getenv("R2_ENDPOINT")

    ZIP_FILENAME = os.getenv("ARTIFACT_ZIP_NAME", "artifacts_v1.zip")
    SHA256_FILENAME = f"{ZIP_FILENAME}.sha256"
    DEST_DIR = Path(os.getenv("DATA_DIR", "data"))
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    local_sha_path = DEST_DIR / SHA256_FILENAME
    local_zip_path = DEST_DIR / ZIP_FILENAME

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        log.warning("Missing R2 environment variables. Skipping data fetch.")
        return

    # S3 Client for R2
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

    try:
        log.info("Checking remote checksum %s...", SHA256_FILENAME)
        remote_sha_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=SHA256_FILENAME)
        remote_sha = remote_sha_obj["Body"].read().decode("utf-8").strip().split()[0]

        if local_sha_path.exists():
            local_sha = local_sha_path.read_text(encoding="utf-8").strip().split()[0]
            if (
                local_sha == remote_sha
                and (DEST_DIR / "bm25_v1.pkl").exists()
                and _local_artifacts_consistent(DEST_DIR)
            ):
                log.info("Local artifacts match remote checksum. Skipping download.")
                return

        log.info("Checksum mismatch or missing artifacts. Downloading %s from R2...", ZIP_FILENAME)
        s3_client.download_file(R2_BUCKET_NAME, ZIP_FILENAME, str(local_zip_path))

        zip_digest = _sha256_file(local_zip_path)
        if zip_digest != remote_sha:
            log.error(
                "Downloaded zip SHA256 mismatch (expected %s, got %s). Removing zip.",
                remote_sha[:16],
                zip_digest[:16],
            )
            local_zip_path.unlink(missing_ok=True)
            raise RuntimeError("Artifact zip failed checksum validation.")

        log.info("Download verified. Extracting to %s ...", DEST_DIR)
        with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
            names = zip_ref.namelist()
            if not names:
                raise RuntimeError("Artifact zip is empty.")
            zip_ref.extractall(DEST_DIR)

        if not (DEST_DIR / "bm25_v1.pkl").exists():
            raise RuntimeError("Extraction incomplete: bm25_v1.pkl missing after unzip.")

        local_sha_path.write_text(remote_sha + "\n", encoding="utf-8")
        log.info("Extraction complete. Data ready in %s/", DEST_DIR)

        local_zip_path.unlink(missing_ok=True)
    except Exception as e:
        log.error("Data fetch failed: %s", e)
        if not (DEST_DIR / "bm25_v1.pkl").exists():
            log.warning("Fallback: trying to download %s without checksum verification...", ZIP_FILENAME)
            try:
                s3_client.download_file(R2_BUCKET_NAME, ZIP_FILENAME, str(local_zip_path))
                with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                    names = zip_ref.namelist()
                    if not names:
                        raise RuntimeError("Fallback zip empty.")
                    zip_ref.extractall(DEST_DIR)
                if not (DEST_DIR / "bm25_v1.pkl").exists():
                    raise RuntimeError("Fallback extraction incomplete.")
                local_zip_path.unlink(missing_ok=True)
                log.warning("Fallback extraction complete (checksum file missing or mismatch on remote).")
            except Exception as e2:
                log.error("Fallback download failed: %s", e2)


if __name__ == "__main__":
    fetch_and_extract()
