#!/bin/bash

export PATH="/root/.local/bin:$PATH"

# Check the value of ENV variable
if [ "$ENV" = "development" ]; then
  echo "Running in development mode with hot-reloading..."
  exec uvicorn app.main:api --host 0.0.0.0 --port 8000 --reload
else
  echo "Running in production mode..."
  exec gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 app.main:api
fi