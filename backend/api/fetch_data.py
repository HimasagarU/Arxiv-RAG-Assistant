import os
import zipfile
import boto3
from botocore.config import Config
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

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
        log.info(f"Checking remote checksum {SHA256_FILENAME}...")
        # Download the sha256 file to memory
        remote_sha_obj = s3_client.get_object(Bucket=R2_BUCKET_NAME, Key=SHA256_FILENAME)
        remote_sha = remote_sha_obj['Body'].read().decode('utf-8').strip()
        
        # Check if local matches
        if local_sha_path.exists():
            local_sha = local_sha_path.read_text(encoding="utf-8").strip()
            if local_sha == remote_sha:
                log.info("Local artifacts match remote checksum. Skipping download.")
                return
        
        log.info(f"Checksum mismatch or missing. Downloading {ZIP_FILENAME} from R2...")
        s3_client.download_file(R2_BUCKET_NAME, ZIP_FILENAME, str(local_zip_path))
        log.info("Download complete. Extracting...")
        
        with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
            zip_ref.extractall(DEST_DIR)
        
        # Save the new checksum locally
        local_sha_path.write_text(remote_sha, encoding="utf-8")
        
        log.info(f"Extraction complete. Data ready in {DEST_DIR}/")
        
        # Cleanup zip to save space
        os.remove(local_zip_path)
    except Exception as e:
        log.error(f"Data fetch failed: {e}")
        # If the checksum file doesn't exist on R2, fallback to checking if artifacts exist locally
        if not (DEST_DIR / "bm25_v1.pkl").exists():
            log.info(f"Fallback: trying to download {ZIP_FILENAME} without checksum...")
            try:
                s3_client.download_file(R2_BUCKET_NAME, ZIP_FILENAME, str(local_zip_path))
                with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(DEST_DIR)
                os.remove(local_zip_path)
                log.info(f"Fallback extraction complete.")
            except Exception as e2:
                log.error(f"Fallback download failed: {e2}")

if __name__ == "__main__":
    fetch_and_extract()
