# 1. Isticmaal slim base image (yar & degdeg ah)
FROM python:3.11-slim

# 2. Set environment variables si ay u sahlanaato error detection
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Update system packages & install required system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    cmake \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set working directory
WORKDIR /app

# 5. Copy all source code to container
COPY . /app

# 6. Upgrade pip & install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Expose port to outside (default for FastAPI)
EXPOSE 8080

# 8. Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
