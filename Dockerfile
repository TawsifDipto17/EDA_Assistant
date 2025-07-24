# Use lightweight Python base image
FROM python:3.11-slim

# Create non-root user for security
RUN useradd -m appuser

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app into the container
COPY . .

# Create the .files directory and give ownership to appuser
RUN mkdir -p /app/.files && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Chainlit's default port
EXPOSE 8000

# Ensure stdout/stderr are unbuffered
ENV PYTHONUNBUFFERED=1


CMD python3 -m chainlit run app.py --host 0.0.0.0 --port 7860
