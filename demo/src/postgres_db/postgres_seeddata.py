import os
import json
import logging
import numpy as np
import sqlalchemy.exc
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

from postgres_db.postgres_engine import create_engine_from_env, create_engine_from_args
from postgres_db.postgres_models import Item
from postgres_db.update_embeddings import update_embeddings

from tqdm import tqdm

logger = logging.getLogger(__name__)

def seed_data(engine, model_name, kb_path):
    with engine.begin() as conn:
        table_name = Item.__tablename__
        result = conn.execute(
            text(
                f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = '{table_name}')"
            )
        )
        if not result.scalar():
            logger.error(f"{table_name} table does not exist. Please run the database setup script first.")
            return
    
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    with SessionLocal() as session:
        current_dir = os.path.dirname(os.path.realpath(__file__))

        with open(os.path.join(current_dir, kb_path)) as f:
            kb_data = [json.loads(line) for line in f]
        
        if "embedding" not in kb_data[0]:
            logger.error(f"Embeddings do not exist in the file. Creating embeddings.")
            kb_data = update_embeddings(model_name, kb_path, in_seed_data=True)
        
        for line in tqdm(kb_data, "Inserting values into db"):
            attrs = {key: value for key, value in line.items()}
            line["embedding"] = np.array(line["embedding"])
            column_names = ", ".join(attrs.keys())
            values = ", ".join([f":{key}" for key in attrs.keys()])
            session.execute(text(f"INSERT INTO {Item.__tablename__} ({column_names}) VALUES ({values})"), attrs)
            
        try:
            session.commit()
            logger.info(f"{Item.__tablename__} table is seeded.") 
        except sqlalchemy.exc.IntegrityError:
            session.rollback()
            logger.info(f"Integrity Error") 
            return
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Postgres host")
    parser.add_argument("--username", type=str, help="Postgres username")
    parser.add_argument("--password", type=str, help="Postgres password")
    parser.add_argument("--database", type=str, help="Postgres database")
    parser.add_argument("--model_name", type=str, help="Embedding model name")
    parser.add_argument("--kb_path", type=str, help="Path to seed file")
    
    
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    
    if args.host is None:
        engine = create_engine_from_env()
    else:
        engine = create_engine_from_args(args)
    
    seed_data(engine, args.model_name, args.kb_path)
    
    engine.dispose()
    
if __name__ == "__main__":
    main()
