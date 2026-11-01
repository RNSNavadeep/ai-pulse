FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configuration
COPY config.py app.py sources.json ./
COPY src/ ./src/

# Create data directory for persistent SQLite volume mounting
RUN mkdir -p /app/data

# Default volume for SQLite persistence
VOLUME ["/app/data"]

# Default command: continuous polling with 30-minute interval (1800s)
CMD ["python", "app.py", "--poll", "--interval", "1800"]
