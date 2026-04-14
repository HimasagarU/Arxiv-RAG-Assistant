import os
import zipfile
import boto3
from botocore.config import Config
from pathlib import Path

def fetch_and_extract():
    # --- Config from Environment ---
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
    R2_ENDPOINT = os.getenv("R2_ENDPOINT")
    
    ZIP_FILENAME = "arxiv_rag_data_20k.zip"
    DEST_DIR = Path("/app/data")
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        print("Error: Missing R2 environment variables. Skipping data fetch.")
        return

    # S3 Client for R2
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

    print(f"Downloading {ZIP_FILENAME} from R2...")
    try:
        s3_client.download_file(R2_BUCKET_NAME, ZIP_FILENAME, ZIP_FILENAME)
        print("Download complete. Extracting...")
        
        with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
            zip_ref.extractall(DEST_DIR)
        
        print(f"Extraction complete. Data ready in {DEST_DIR}/")
        # Cleanup
        os.remove(ZIP_FILENAME)
    except Exception as e:
        print(f"Data fetch failed: {e}")

if __name__ == "__main__":
    fetch_and_extract()
