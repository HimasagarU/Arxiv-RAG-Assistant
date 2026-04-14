import os
import zipfile
import boto3
from botocore.config import Config
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- Config ---
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
R2_ENDPOINT = os.getenv("R2_ENDPOINT")

DATA_DIR = Path("data")
FILES_TO_ZIP = [
    "arxiv_papers.db",
    "bm25_index.pkl",
    "chroma_db"  # Directory
]
ZIP_FILENAME = "arxiv_rag_data_20k.zip"

def zip_data():
    print(f"Zipping data files into {ZIP_FILENAME}...")
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for item in FILES_TO_ZIP:
            path = DATA_DIR / item
            if not path.exists():
                print(f"Warning: {path} not found, skipping.")
                continue
            
            if path.is_file():
                zipf.write(path, arcname=item)
            else:
                for root, dirs, files in os.walk(path):
                    for file in files:
                        full_path = Path(root) / file
                        arcname = os.path.join(item, os.path.relpath(full_path, path))
                        zipf.write(full_path, arcname=arcname)
    print("Zip complete.")

def upload_to_r2():
    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME]):
        print("Error: Missing R2 configuration in .env")
        return

    # S3 Client for R2
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        region_name="auto",  # Highly recommended for R2
        config=Config(signature_version="s3v4"),
    )

    # Optional: Enable debug logging if errors persist
    # boto3.set_stream_logger('botocore')

    print(f"Uploading {ZIP_FILENAME} to R2 bucket '{R2_BUCKET_NAME}'...")
    try:
        s3_client.upload_file(
            ZIP_FILENAME, 
            R2_BUCKET_NAME, 
            ZIP_FILENAME
        )
        print(f"Upload Successful!")
        print(f"Your public data URL should be approximately: https://pub-<your-id>.r2.dev/{ZIP_FILENAME}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    zip_data()
    upload_to_r2()
    # Clean up local zip after upload
    # os.remove(ZIP_FILENAME)
