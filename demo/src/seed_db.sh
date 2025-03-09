#!/bin/bash

DB_EXISTS=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USERNAME -lqt | cut -d \| -f 1 | grep -w "$POSTGRES_DATABASE")

if [ -n "$DB_EXISTS" ]; then
  echo "Database  $POSTGRES_DATABASE exists. Skipping setup."

  ITEM_COUNT=$(PGPASSWORD=$POSTGRES_PASSWORD psql -h $POSTGRES_HOST -U $POSTGRES_USERNAME -d $POSTGRES_DATABASE -tAc "SELECT COUNT(*) FROM public.items;" 2>/dev/null | xargs)
  if [ "$ITEM_COUNT" -eq 0 ]; then
    echo "Table is empty. Seeding data."
    python -m postgres_db.postgres_seeddata --model_name "$MODEL_NAME" --kb_path "$SEED_PATH"
  else
    echo "Table already contains data. Skipping seeding."
    python -m utils.model_manager --model_name "$MODEL_NAME"
  fi
else
  echo "Database does not exist. Setting up the database."
  python -m postgres_db.postgres_engine
  
  echo "Seeding data..."
  python -m postgres_db.postgres_seeddata  --model_name "$MODEL_NAME" --kb_path "$SEED_PATH"
fi
