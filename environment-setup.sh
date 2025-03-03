#!/bin/bash
# Setup script for Nilfisk Service Manual RAG System

# Create project structure
mkdir -p data/pdfs data/processed logs

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install \
    fastapi[all] \
    uvicorn \
    python-multipart \
    python-dotenv \
    httpx \
    supabase \
    numpy \
    pymupdf \
    tqdm \
    pytest \
    pytest-asyncio

# Create .env file template
cat > .env.template << 'EOL'
# Azure OpenAI API Configuration
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=

# Supabase Configuration
SUPABASE_URL=
SUPABASE_SERVICE_KEY=

# Application Configuration
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=100
EOL

# Create actual .env file (to be filled in by the user)
cp .env.template .env

echo "Please edit the .env file with your actual API keys and configuration."

# Create Dockerfile
cat > Dockerfile << 'EOL'
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# Create requirements.txt
cat > requirements.txt << 'EOL'
fastapi==0.104.1
uvicorn==0.23.2
python-multipart==0.0.6
python-dotenv==1.0.0
httpx==0.25.1
supabase==1.0.4
numpy==1.26.1
pymupdf==1.23.5
tqdm==4.66.1
pytest==7.4.3
pytest-asyncio==0.21.1
EOL

# Create docker-compose file
cat > docker-compose.yml << 'EOL'
version: '3'

services:
  rag-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped
EOL

echo "Environment setup complete!"
