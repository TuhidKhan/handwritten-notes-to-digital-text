# Lightweight official Python 3.11 Debian runtime
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install required operating system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR neural network weights during image build
# This avoids downloading weights during user requests or runtime restarts
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy application source code and dataset
COPY . .

# Set environment variables for Hugging Face Spaces
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# Start Uvicorn ASGI server on port 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
