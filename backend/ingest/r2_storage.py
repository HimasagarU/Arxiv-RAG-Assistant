"""
r2_storage.py — Cloudflare R2 upload/download helpers.

Uses boto3 with S3-compatible API to interact with Cloudflare R2.
Handles PDFs, figure images, and artifact JSON uploads.

Usage:
    from ingest.r2_storage import R2Storage
    r2 = R2Storage()
    key = r2.upload_pdf("2301.12345", pdf_bytes)
"""

import io
import json
import logging
import os
from typing import Optional

import boto3
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class R2Storage:
    """Cloudflare R2 object storage client."""

    def __init__(self):
        self.account_id = os.getenv("R2_ACCOUNT_ID", "")
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "")
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "")
        self.bucket = os.getenv("R2_BUCKET_NAME", "arxiv-rag-assist")
        self.endpoint = os.getenv("R2_ENDPOINT", "")

        if not all([self.account_id, self.access_key, self.secret_key, self.bucket]):
            log.warning("R2 credentials not fully configured. Uploads will be skipped.")
            self._client = None
            return

        self._client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version="s3v4"),
        )
        log.info(f"R2 storage initialized (bucket={self.bucket})")

    @property
    def is_available(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------
    # Upload helpers
    # ------------------------------------------------------------------

    def upload_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        """Upload raw bytes to R2. Returns the R2 key."""
        if not self.is_available:
            log.debug(f"R2 not available, skipping upload: {key}")
            return ""
        try:
            self._client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            log.debug(f"Uploaded to R2: {key} ({len(data)} bytes)")
            return key
        except Exception as e:
            log.warning(f"R2 upload failed for {key}: {e}")
            return ""

    def upload_pdf(self, paper_id: str, pdf_bytes: bytes) -> str:
        """Upload a paper PDF to R2."""
        key = f"pdfs/{paper_id}.pdf"
        return self.upload_bytes(key, pdf_bytes, content_type="application/pdf")

    def upload_figure(self, paper_id: str, figure_index: int, image_bytes: bytes, ext: str = "png") -> str:
        """Upload a figure image to R2."""
        key = f"figures/{paper_id}/fig_{figure_index}.{ext}"
        content_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
        return self.upload_bytes(key, image_bytes, content_type=content_type)

    def upload_artifact_json(self, paper_id: str, artifact_dict: dict) -> str:
        """Upload an artifact JSON to R2."""
        key = f"artifacts/{paper_id}.json"
        data = json.dumps(artifact_dict, ensure_ascii=False, indent=2).encode("utf-8")
        return self.upload_bytes(key, data, content_type="application/json")

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def download_bytes(self, key: str) -> Optional[bytes]:
        """Download raw bytes from R2."""
        if not self.is_available:
            return None
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return response["Body"].read()
        except Exception as e:
            log.warning(f"R2 download failed for {key}: {e}")
            return None

    def download_pdf(self, paper_id: str) -> Optional[bytes]:
        key = f"pdfs/{paper_id}.pdf"
        return self.download_bytes(key)

    def download_artifact_json(self, paper_id: str) -> Optional[dict]:
        data = self.download_bytes(f"artifacts/{paper_id}.json")
        if data:
            return json.loads(data.decode("utf-8"))
        return None

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in R2."""
        if not self.is_available:
            return False
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def pdf_exists(self, paper_id: str) -> bool:
        return self.file_exists(f"pdfs/{paper_id}.pdf")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def delete_all_objects(self, prefix: str = "") -> int:
        """Delete all objects in the bucket (or under a prefix). Returns count deleted."""
        if not self.is_available:
            log.warning("R2 not available, skipping delete.")
            return 0
        deleted = 0
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)
            for page in pages:
                objects = page.get("Contents", [])
                if not objects:
                    continue
                delete_keys = [{"Key": obj["Key"]} for obj in objects]
                self._client.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": delete_keys, "Quiet": True},
                )
                deleted += len(delete_keys)
            log.info(f"R2: deleted {deleted} objects (prefix='{prefix}')")
        except Exception as e:
            log.warning(f"R2 delete_all_objects failed: {e}")
        return deleted
