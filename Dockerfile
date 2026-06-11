# Use official slim Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for building some python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python requirements
COPY flask_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy all project files into the container
COPY . .

# Set default port and bind host
ENV PORT=7860
EXPOSE 7860

# Run with Gunicorn for concurrency and reliability
CMD gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 flask_app.app:app
