# Base Python image
FROM python:3.11-slim

# Working directory set karein
WORKDIR /app

# System level dependencies install karein
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements & install dependencies with network retries
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 10 --timeout 60 -r requirements.txt

# Project code copy karein
COPY . .

# Environment port expose karein
EXPOSE 8000

# FastAPI Server start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]