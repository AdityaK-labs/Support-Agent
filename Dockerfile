FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir openenv-core

# Copy the rest of the application
COPY . .

# Ensure standard port for Hugging Face Spaces (Gradio/FastAPI)
EXPOSE 7860

# Run the backend
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
