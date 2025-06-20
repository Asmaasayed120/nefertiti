# ===============================
# Dockerfile for Nefertiti AI API
# ===============================

# 1. Base image
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Install OS dependencies (for moviepy, opencv, gtts, etc.)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy project files
COPY . .

# 5. Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 6. Expose API port
EXPOSE 8000

# 7. Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
