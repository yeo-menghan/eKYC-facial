FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# FIX: Point to the api subfolder for requirements
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary code (Corrected paths)
COPY api/ ./api
COPY quality/ ./quality
COPY model/ ./model
COPY CONSTANTS.py .

ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]