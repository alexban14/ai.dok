#!/bin/bash

export PATH="/root/.local/bin:$PATH"

# Check the value of ENV variable
if [ "$ENV" = "development" ]; then
  echo "Running in development mode with hot-reloading..."
  exec uvicorn app.main:api --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 300 --timeout-graceful-shutdown 30
else
  echo "Running in production mode..."
  # Increased timeout to 7200s (2 hours) for processing 6000 files
  # Single worker to avoid memory issues with PDF processing and large models
  # Keep-alive increased for long-running background tasks
  exec gunicorn -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --workers 1 \
    --timeout 7200 \
    --graceful-timeout 120 \
    --keep-alive 600 \
    --max-requests 50 \
    --max-requests-jitter 5 \
    --worker-tmp-dir /dev/shm \
    app.main:api
fi