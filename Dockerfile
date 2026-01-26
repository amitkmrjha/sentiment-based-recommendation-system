# Stage 1: Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only requirements first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Stage 2: Final production image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project files (excluding .venv thanks to .dockerignore)
COPY . .

# Expose port
EXPOSE 8000

# Use Gunicorn for Flask production
CMD ["gunicorn", "-b", "0.0.0.0:8000", "app:app", "--workers=3", "--timeout=120"]
