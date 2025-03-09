import os
import json
import logging
import argparse
import torch
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from postgres_db.postgres_models import Item
from postgres_db.postgres_engine import create_engine_from_env
from utils.embeddings import compute_emb_batched

logger = logging.getLogger(__name__)

def update_embeddings(model_name: str,  kb_path: str, in_seed_data=True):
    engine = create_engine_from_env()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    embedding_column = "embedding"
    
    if in_seed_data:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        seed_file = os.path.join(current_dir, kb_path)
        
        with open(os.path.join(seed_file)) as f:
            seed_data = [json.loads(line) for line in f]

        concepts = [item["concept"] for item in seed_data]
        embs = compute_emb_batched(concepts, model_name)

        for emb, row in zip(embs, seed_data):
            row[embedding_column] = emb
            
        # Write embeddings to file
        with open(seed_file, "w") as f:
            for item in seed_data:
                json_line = json.dumps(item, separators=(',', ':'), ensure_ascii=False)
                f.write(json_line + "\n")
        
        logger.info("Updated seed data embeddings.")
        return seed_data
    
    # Database update
    with SessionLocal() as session:
        rows_to_update = session.scalars(select(Item)).all()
        if not rows_to_update:
            logger.info("No rows to update.")
            return
            
        concepts = [row.concept for row in rows_to_update]
        embs = compute_emb_batched(concepts, model_name)

        for row, emb in zip(rows_to_update, embs):
            setattr(row, embedding_column, emb)
            
        session.commit()
        logger.info(f"Updated embeddings for {len(rows_to_update)} items.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Embedding model name")
    parser.add_argument("--kb_path", type=str, help="Path to seed file")
    parser.add_argument("--in_seed_data", default=True, action="store_true")
    args = parser.parse_args() 

    update_embeddings(args.model_name, args.kb_path, args.in_seed_data)
