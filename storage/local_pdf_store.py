import os
import time
import requests
import logging
from pathlib import Path
import re

log = logging.getLogger(__name__)

MAX_PDF_DOWNLOAD_RETRIES = 2

class LocalPDFStore:
    def __init__(self, base_dir: str = "data/arxiv_pdfs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.legacy_dir = Path("data/pdfs")

    def _is_valid_arxiv_id(self, paper_id: str) -> bool:
        return bool(re.match(r"^\d{4}\.\d{4,5}$", paper_id or ""))

    def get_pdf_path(self, paper_id: str, year: str = None) -> Path:
        """Get the local path for a PDF, optionally sharded by year."""
        if year:
            shard_dir = self.base_dir / str(year)
        else:
            # Fallback to prefix (e.g., '1706' from '1706.03762')
            shard_dir = self.base_dir / paper_id[:4]
            
        shard_dir.mkdir(parents=True, exist_ok=True)
        return shard_dir / f"{paper_id}.pdf"

    def download_pdf(self, paper_id: str, pdf_url: str, year: str = None, timeout: int = 30) -> str:
        """Download a PDF to local disk and return the path."""
        if not self._is_valid_arxiv_id(paper_id):
            log.warning(f"Skipping invalid arXiv ID: {paper_id}")
            return ""

        target_path = self.get_pdf_path(paper_id, year)
        
        if target_path.exists() and target_path.stat().st_size > 1024:
            log.info(f"PDF {paper_id} already exists locally.")
            return str(target_path)
            
        # Check legacy directory first and reuse it without re-downloading.
        legacy_path = self.legacy_dir / f"{paper_id}.pdf"
        if legacy_path.exists() and legacy_path.stat().st_size > 1024:
            log.info(f"Using legacy PDF for {paper_id} from {legacy_path}.")
            return str(legacy_path)
            
        if not pdf_url:
            if len(paper_id) == 40 and "." not in paper_id:
                log.warning(f"Skipping download for {paper_id}: No open access PDF URL provided by Semantic Scholar.")
                return ""
            pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"
            
        log.info(f"Downloading PDF {paper_id} to {target_path}...")
        for attempt in range(MAX_PDF_DOWNLOAD_RETRIES):
            try:
                resp = requests.get(pdf_url, stream=True, timeout=timeout)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    with open(target_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    if target_path.stat().st_size > 1024:
                        return str(target_path)
                    else:
                        log.warning(f"Downloaded PDF {paper_id} is suspiciously small.")
                elif resp.status_code == 404:
                    log.warning(f"PDF not ready yet: {paper_id}")
                    return ""
                elif resp.status_code == 403:
                    log.warning(f"PDF forbidden for {paper_id}")
                    return ""
                elif resp.status_code == 429:
                    log.warning(f"arXiv rate limited (429) downloading {paper_id}. Retry {attempt+1}/{MAX_PDF_DOWNLOAD_RETRIES}...")
                    time.sleep(5 * (attempt + 1))
                else:
                    log.warning(f"Failed to download {paper_id}. HTTP {resp.status_code}")
            except Exception as e:
                log.warning(f"Error downloading {paper_id}: {e}")
                
            time.sleep(2)
            
        return ""
        
    def read_pdf(self, paper_id: str, year: str = None) -> bytes:
        """Read a PDF from disk as bytes."""
        path = self.get_pdf_path(paper_id, year)
        if path.exists():
            return path.read_bytes()
        legacy_path = self.legacy_dir / f"{paper_id}.pdf"
        if legacy_path.exists():
            return legacy_path.read_bytes()
        return None
