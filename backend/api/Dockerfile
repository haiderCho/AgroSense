# Base Image: Lightweight Python
FROM python:3.10-slim

# Set Working Directory
WORKDIR /app

# Install System Dependencies (if essential for some ML libraries)
# build-essential for compiling some C extensions, although slim usually is enough for wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy Requirements
COPY requirements.txt .

# Install Python Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy Project Code
# We copy the entire project but .dockerignore should exclude .git, .env, frontend, etc.
COPY . .

# Expose Port
EXPOSE 8000

# Run Application
# Using explicit python path to ensure module loading works
ENV PYTHONPATH=/app
CMD ["python", "backend/api/main.py"]
