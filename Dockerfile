FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY . .

# Shell form (no brackets) — $PORT expands correctly
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
