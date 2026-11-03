# AUDI edge / field API (also the maintenance portal backend in cloud mode).
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-cloud.txt ./
# CPU wheels by default; pass --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 for CUDA
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --index-url ${TORCH_INDEX} torch torchvision && pip install -r requirements-cloud.txt
COPY . .
RUN useradd -m -u 10001 audi && mkdir -p /app/data /app/models /app/reports && chown -R audi /app/data /app/models /app/reports
USER audi
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://127.0.0.1:8000/api/health || exit 1
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
