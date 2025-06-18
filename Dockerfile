FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    libsndfile1 \
    libmagic1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install Python dependencies (includes sentencepiece)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
