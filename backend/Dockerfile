FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy and install Python dependencies first (caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and documents
COPY my_rag_api.py .
COPY config.py .
COPY docs/ docs/

# Document the port
EXPOSE 8000

# Start the application
CMD ["uvicorn", "my_rag_api:app", "--host", "0.0.0.0", "--port", "8000"]