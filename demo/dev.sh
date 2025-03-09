#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "Starting development database..."
docker-compose -f docker-compose.dev.yml up -d

if [ ! -d "demo-env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "demo-env"
    source "demo-env/bin/activate"
    python3 -m pip install -r requirements.txt
else
    source "demo-env/bin/activate"
fi

# Load env variables
export $(cat .env | xargs)

cd ./src
chmod +x download_static.sh seed_db.sh
./download_static.sh && ./seed_db.sh

if [ ! -z $1 ] && [ $1 = "nocors" ]; then
    echo "Running in debug mode without CORS..."
    FLASK_DEBUG=1 python3 app.py
else
    echo "Running with gunicorn..."
    exec gunicorn -w 2 -b 0.0.0.0:8000 --preload wsgi:app
fi