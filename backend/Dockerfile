FROM python:3.11-slim

# Create a non-root user "user" with UID 1000 (Required by HF Spaces)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Install minimal system dependencies needed for PyTorch and PDF tools as root
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	ca-certificates \
	libgl1 \
	libglib2.0-0 \
	libsm6 \
	libxrender1 \
	libxext6 \
	wget \
 && rm -rf /var/lib/apt/lists/*

# Fix permissions before switching to the non-root user
RUN chown -R user:user $HOME/app

# Switch to the non-root user
USER user

# Install PyTorch (CPU) wheels first, then other Python requirements
COPY --chown=user requirements.txt .
RUN pip install --user --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
	"torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0" \
 && pip install --user --no-cache-dir -r requirements.txt

# Pre-download NLTK tokenizer data needed by production text processing
RUN python -m nltk.downloader punkt punkt_tab

# Copy application code
COPY --chown=user api/ api/
COPY --chown=user ingest/ ingest/
COPY --chown=user storage/ storage/
COPY --chown=user rerank/ rerank/
COPY --chown=user index/ index/
COPY --chown=user db/ db/
COPY --chown=user utils/ utils/

# Create data directory
RUN mkdir -p $HOME/app/data

# Make entrypoint executable
RUN chmod +x api/entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["./api/entrypoint.sh"]
