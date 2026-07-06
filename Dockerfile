# ── Project Skypad – Production Dockerfile ──────────────────────────────────
# Builds a lean Python 3.12 image for Cloud Run.
# Port is controlled by the $PORT env var that Cloud Run injects at runtime.

FROM python:3.12-slim

# Prevent .pyc files and enable unbuffered logs (important for Cloud Run)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the source files needed at runtime
COPY engine.py   .
COPY routing.py  .
COPY app.py      .
COPY index.html  .

# Cloud Run injects PORT (default 8080). app.py already reads os.getenv("PORT","8000").
EXPOSE 8080

# Launch the unified backend
CMD ["python", "app.py"]
