import os
import zipfile
import hashlib
import boto3
from botocore.config import Config
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("data")
ZIP_FILENAME = "artifacts_v1.zip"
SHA256_FILENAME = f"{ZIP_FILENAME}.sha256"

FILES_TO_ZIP = [
    "bm25_v1.pkl",
    "chunks_meta.jsonl",
    "chunks_text.jsonl",
    "papers_meta.json"
]

def generate_sha256(filepath):
    """Generate SHA256 checksum for a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    # 1. Check if all files exist
    for f in FILES_TO_ZIP:
        if not (DATA_DIR / f).exists():
            print(f"Error: {f} not found in {DATA_DIR}/. Did you run build_bm25.py?")
            return

    # 2. Create ZIP archive
    print(f"Zipping artifacts into {ZIP_FILENAME}...")
    with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for f in FILES_TO_ZIP:
            file_path = DATA_DIR / f
            print(f"  Adding {f} ({file_path.stat().st_size / (1024*1024):.2f} MB)")
            # Add file to root of zip (arcname=f)
            zipf.write(file_path, arcname=f)
    print("Zip created successfully.")

    # 3. Generate SHA256 checksum
    print("Generating SHA256 checksum...")
    checksum = generate_sha256(ZIP_FILENAME)
    with open(SHA256_FILENAME, "w", encoding="utf-8") as f:
        f.write(checksum)
    print(f"Checksum: {checksum}")
    print(f"Saved to {SHA256_FILENAME}")

    # 4. Upload to Cloudflare R2
    R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
    R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    R2_BUCKET_NAME = os.getenv("R2_BUCKET_NAME")
    R2_ENDPOINT = os.getenv("R2_ENDPOINT")

    if not all([R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET_NAME, R2_ENDPOINT]):
        print("Missing R2 credentials in .env! Skipping upload.")
        print(f"You will need to manually upload {ZIP_FILENAME} and {SHA256_FILENAME} to your R2 bucket.")
        return

    print(f"\nConnecting to Cloudflare R2 bucket: {R2_BUCKET_NAME}...")
    s3_client = boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )

    try:
        print(f"Uploading {ZIP_FILENAME}...")
        s3_client.upload_file(ZIP_FILENAME, R2_BUCKET_NAME, ZIP_FILENAME)
        
        print(f"Uploading {SHA256_FILENAME}...")
        s3_client.upload_file(SHA256_FILENAME, R2_BUCKET_NAME, SHA256_FILENAME)
        
        print("Upload complete!")
        
        # Clean up local zip files (optional)
        # os.remove(ZIP_FILENAME)
        # os.remove(SHA256_FILENAME)
        
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
