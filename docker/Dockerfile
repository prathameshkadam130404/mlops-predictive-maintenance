# ==============================================================================
# Multi-stage Production Dockerfile
# Predictive Maintenance API
#
# Design decisions:
#   - Multi-stage build: keeps final image small (~300MB vs ~1.5GB)
#   - Non-root user: follows principle of least privilege
#   - python:3.11-slim: matches NCM conda env Python version
#   - Health check: built-in container health monitoring
#   - Gunicorn + Uvicorn: production process management
# ==============================================================================

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Stage 2: Production runtime ---
FROM python:3.11-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY src/ ./src/
COPY api/ ./api/
COPY configs/ ./configs/
COPY models/ ./models/
COPY metrics/ ./metrics/
COPY data/features/ ./data/features/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Ensure Python packages are on PATH
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ENVIRONMENT=production

# Expose the port
EXPOSE 8000

# Health check using Python (no curl needed in slim image)
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Production server: Gunicorn with Uvicorn workers
# 2 workers is appropriate for free-tier hosting (512MB RAM)
CMD ["gunicorn", \
     "--workers", "2", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "api.main:app"]
