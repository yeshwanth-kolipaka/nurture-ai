FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN addgroup --system --gid 1001 app && \
    adduser --system --uid 1001 --gid 1001 app

WORKDIR /app

COPY --chown=app:app fusion/ fusion/
COPY --chown=app:app models/ models/
COPY --chown=app:app static/ static/
COPY --chown=app:app templates/ templates/
COPY --chown=app:app app.py db.py ./
COPY --chown=app:app requirements.txt ./

RUN mkdir -p uploads && chown app:app uploads

USER app

EXPOSE 7860

ENV FLASK_ENV=production \
    PYTHONUNBUFFERED=1

CMD gunicorn app:app \
    --bind 0.0.0.0:$(PORT) \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile -
