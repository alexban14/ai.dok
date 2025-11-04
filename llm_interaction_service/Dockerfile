FROM python:3.12-slim AS builder

ARG ENV=development

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    # Image processing build deps
    libjpeg-dev \
    zlib1g-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libopenjp2-7-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    export PATH="/root/.local/bin:$PATH" && \
    poetry config virtualenvs.create false

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy Poetry files
COPY ./pyproject.toml poetry.lock* ./

# Install dependencies with Poetry
RUN if [ "$ENV" = "development" ] ; then \
        poetry install --no-root --with dev; \
    else \
        poetry install --no-root --without dev; \
    fi

# Production stage
FROM python:3.12-slim AS production

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Image processing runtime
    libjpeg62-turbo \
    zlib1g \
    libfreetype6 \
    liblcms2-2 \
    libopenjp2-7 \
    # libtiff5 \
    # Graphics libraries
    # libgl1-mesa-glx \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Utils\
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create user
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app/logs /app/tmp && \
    chown -R app:app /app && \
    chmod -R 755 /app

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install web servers
RUN pip install --no-cache-dir uvicorn gunicorn

# Copy application
COPY --chown=app:app ./app /app/app
COPY --chown=app:app ./.docker/entrypoint.sh /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

USER app

ENV PYTHONPATH=/app \
    VARIABLE_NAME=api \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000 || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
