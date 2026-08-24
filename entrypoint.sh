#!/bin/bash
set -e

# Wait for PostgreSQL
echo "Waiting for postgres on ${POSTGRES_HOST:-db}:${POSTGRES_PORT:-5432}..."
while ! pg_isready -h "${POSTGRES_HOST:-db}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-postgres}"; do
  sleep 1
done
echo "PostgreSQL is up."

# Run database migrations/initialization
echo "Initializing database..."
python3 -c "from storage.database import init_db; init_db()"

# Seed lookup datasets
echo "Seeding lookup datasets..."
python3 -m storage.seed_lookups

# Start FastAPI and Streamlit
echo "Starting Streamlit..."
python3 -m streamlit run ui/app.py --server.headless true --server.port 8501 --server.address 0.0.0.0 &

echo "Starting FastAPI app..."
exec python3 -m uvicorn server.main:app --host 0.0.0.0 --port 8000