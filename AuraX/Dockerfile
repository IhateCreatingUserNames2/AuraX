FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Helper to ensure logs flush immediately
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Command is handled by docker-compose, but setting a default here
CMD ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8000"]