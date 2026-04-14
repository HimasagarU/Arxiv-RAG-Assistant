FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/
COPY rerank/ rerank/
COPY index/params.yaml index/params.yaml

# Copy frontend UI
COPY frontend/ frontend/

# Copy pre-computed index data (since it is small ~80MB, we can bake it into the image)
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
