FROM python:3.12-slim
# Stage 1: Build stage - with build tools
FROM python:3.12-slim as builder

WORKDIR /app

# Install system deps if PyTorch / transformers need them (kept minimal)
# Only needed in the build stage
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Use a virtual environment to keep dependencies isolated
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

# Copy model directory and code
COPY model ./model
# Stage 2: Final stage - slim runtime
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code and model assets
COPY model/ ./model/
COPY main.py .

ENV PYTHONUNBUFFERED=1
# Activate the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
