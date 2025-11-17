#!/bin/bash

export PATH="/root/.local/bin:$PATH"

# Check the value of ENV variable
if [ "$ENV" = "development" ]; then
  echo "Running in development mode with hot-reloading..."
  exec uvicorn app.main:api --host 0.0.0.0 --port 8000 --reload --timeout-keep-alive 300 --timeout-graceful-shutdown 30
else
  echo "Running in production mode..."
  # Increased timeout to 300s (5 minutes) for file processing
  # Single worker to avoid memory issues with PDF processing
  # Grace period for graceful shutdown
  exec gunicorn -k uvicorn.workers.UvicornWorker \
    -b 0.0.0.0:8000 \
    --workers 1 \
    --timeout 300 \
    --graceful-timeout 30 \
    --max-requests 100 \
    --max-requests-jitter 10 \
    app.main:api
fi