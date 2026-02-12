FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY rerank/ rerank/
COPY index/params.yaml index/params.yaml
COPY .env .env

# Data will be mounted as volumes
# VOLUME /app/data

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
