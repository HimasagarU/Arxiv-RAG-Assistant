FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY rerank/ rerank/
COPY frontend/ frontend/
COPY index/params.yaml index/params.yaml

# Create data directory
RUN mkdir -p /app/data

# Make entrypoint executable
RUN chmod +x api/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/api/entrypoint.sh"]
