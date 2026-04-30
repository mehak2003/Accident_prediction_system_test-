FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY backend/ ./backend/

WORKDIR /app/backend

RUN mkdir -p data output/metrics output/plots output/tables models \
    && chmod -R 777 data output models

ENV FLASK_PORT=7860 \
    FLASK_DEBUG=0 \
    PYTHONPATH=/app/backend

EXPOSE 7860

CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:${FLASK_PORT:-7860} --timeout 300 'app:create_app()'"]
