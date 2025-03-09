import os
import argparse
import logging
from typing import List, Tuple, Optional
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.orm import sessionmaker, Session
from dotenv import load_dotenv

from postgres_db.postgres_models import Base


logger = logging.getLogger(__name__)

def create_postgres_engine(host, username, database, password) -> Engine:
    DATABASE_URI = f"postgresql+psycopg://{username}:{password}@{host}/{database}"
    engine = create_engine(DATABASE_URI, echo=True)
    return engine

def create_engine_from_env() -> Engine:
    load_dotenv(override=True)

    return create_postgres_engine(
        username=os.environ["POSTGRES_USERNAME"],
        database=os.environ["POSTGRES_DATABASE"],
        host=os.environ["POSTGRES_HOST"],
        password=os.environ["POSTGRES_PASSWORD"]
    )

def create_engine_from_args(args) -> Engine:
    return create_postgres_engine(
        host=args.host,
        username=args.username,
        database=args.database,
        password=args.password
    )

def create_db_schema(engine):
    with engine.connect() as conn:
        logger.info("Enabling the pgvector extension for Postgres...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("COMMIT;"))
        logger.info("Creating database tables and indexes...")
        Base.metadata.create_all(engine)

    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Create database schema")
    parser.add_argument("--host", type=str, help="Postgres host")
    parser.add_argument("--username", type=str, help="Postgres username")
    parser.add_argument("--password", type=str, help="Postgres password")
    parser.add_argument("--database", type=str, help="Postgres database")

    args = parser.parse_args()
    if args.host is None:
        engine = create_engine_from_env()
    else:
        engine = create_engine_from_args(args)
        
    create_db_schema(engine)
    
    engine.dispose()
    logger.info("Database extension and tables closed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    main()