FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

COPY --chown=user fusion/ fusion/
COPY --chown=user models/ models/
COPY --chown=user static/ static/
COPY --chown=user templates/ templates/
COPY --chown=user app.py db.py ./
COPY --chown=user requirements.txt ./

RUN mkdir -p uploads

EXPOSE 7860

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-"]
