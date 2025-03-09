FROM python:3.10-slim

WORKDIR /demo

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    wget \
    wget \
    postgresql-client

COPY requirements.txt .
COPY src/ /demo

WORKDIR /demo
ENV PYTHONPATH=/demo/src

RUN python -m pip install --no-cache-dir -r requirements.txt
RUN chmod +x /demo/download_static.sh
RUN chmod +x /demo/seed_db.sh

EXPOSE 8000

CMD ["bash", "-c", "/demo/seed_db.sh && gunicorn -w 2 -b 0.0.0.0:8000 --preload wsgi:app"]