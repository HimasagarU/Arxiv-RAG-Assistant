FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies needed for PyTorch and PDF tools
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

# Install PyTorch (CPU) wheels first, then Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
	"torch>=2.0.0" "torchvision>=0.15.0" "torchaudio>=2.0.0" \
 && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY ingest/ ingest/
COPY storage/ storage/
COPY db/ db/
COPY rerank/ rerank/
COPY frontend/ frontend/
COPY index/params.yaml index/params.yaml

# Create data directory
RUN mkdir -p /app/data

# Make entrypoint executable
RUN chmod +x api/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/api/entrypoint.sh"]
