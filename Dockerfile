FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching (Docker-specific, no CUDA)
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy the rest of the application
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data/monitoring data/processed data/embeddings data/model_input data/final

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]