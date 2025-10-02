# Multi-stage Dockerfile for both backend and frontend

# Stage 1: Backend builder
FROM python:3.11-slim AS backend-builder

WORKDIR /app

# Install system dependencies for backend (OpenCV headless compatible)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Copy backend dependency files
COPY pyproject.toml ./

# Generate poetry.lock and install dependencies
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry lock && \
    poetry install --only main --no-root

# Copy model download script
COPY scripts/download_models.py ./scripts/

# Download models from Huggingface
RUN mkdir -p /app/checkpoints && \
    export HF_HOME="/app/checkpoints" && \
    export TRANSFORMERS_CACHE="/app/checkpoints" && \
    python scripts/download_models.py --cache-dir /app/checkpoints || echo "Model download failed, continuing..."

# Stage 2: Frontend builder
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend

# Copy frontend dependency files
COPY frontend/package*.json ./

# Install frontend dependencies
RUN npm ci --only=production

# Copy frontend source code
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 3: Backend runtime
FROM python:3.11-slim AS backend

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Install system runtime dependencies (OpenCV headless compatible)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy backend virtual environment from builder
COPY --from=backend-builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy downloaded models from builder
COPY --from=backend-builder --chown=appuser:appuser /app/checkpoints /app/checkpoints

ENV PATH="/app/.venv/bin:$PATH"

# Copy backend application code
COPY --chown=appuser:appuser ct_clf_backend ./ct_clf_backend
COPY --chown=appuser:appuser ct_clf_hack ./ct_clf_hack
COPY --chown=appuser:appuser configs ./configs

# Create checkpoints directory
RUN mkdir -p /app/checkpoints && \
    chown -R appuser:appuser /app/checkpoints

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R appuser:appuser /app/data /app/logs /app/cache

USER appuser

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    QT_QPA_PLATFORM=offscreen \
    DEBIAN_FRONTEND=noninteractive

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "ct_clf_backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Stage 4: Frontend runtime (nginx)
FROM nginx:alpine AS frontend

# Copy built frontend from builder
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]

# Stage 5: Full application (both backend and frontend)
FROM python:3.11-slim AS fullstack

# Install system dependencies (OpenCV headless compatible)
RUN apt-get update && apt-get install -y \
    curl \
    nginx \
    supervisor \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libfontconfig1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy backend from backend stage
COPY --from=backend-builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=backend-builder --chown=appuser:appuser /app/checkpoints /app/checkpoints
COPY --chown=appuser:appuser ct_clf_backend ./ct_clf_backend
COPY --chown=appuser:appuser ct_clf_hack ./ct_clf_hack
COPY --chown=appuser:appuser configs ./configs

# Create checkpoints directory
RUN mkdir -p /app/checkpoints && \
    chown -R appuser:appuser /app/checkpoints

# Copy built frontend from frontend builder
COPY --from=frontend-builder /app/frontend/build /usr/share/nginx/html

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R appuser:appuser /app/data /app/logs /app/cache

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx-fullstack.conf /etc/nginx/nginx.conf

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_IO_ENABLE_OPENEXR=1 \
    QT_QPA_PLATFORM=offscreen \
    DEBIAN_FRONTEND=noninteractive

EXPOSE 80 8000

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
