from typing import List, Tuple, Optional, Dict
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from sqlalchemy import Index
from sqlalchemy.orm import Session
from sqlalchemy import text

from utils.embeddings import compute_embedding

class Base(DeclarativeBase):
    pass

class Item(Base):
    __tablename__ = "items"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(nullable=False)
    concept: Mapped[str] = mapped_column(nullable=False)
    embedding: Mapped[Vector] = mapped_column(Vector(768), nullable=True)

table_name = Item.__tablename__

index_hnsw = Index(
    f"hnsw_index_for_cosine_{table_name}",
    Item.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)

class PostgresManager:
    def __init__(self, session: Session, model_name: str, embed_column: Optional[str], embed_dim: Optional[int]):
        self.session = session
        self.model_name = model_name
        self.embed_column = embed_column
        self.embed_dim = embed_dim

    def search(self, query_embedding: list[float], top_k: int = 5, threshold: float = None):
        table_name = Item.__tablename__
        query = text(f"""
            SELECT code, concept, 
                1 - ({self.embed_column} <=> CAST(:embedding AS vector)) AS similarity
            FROM {table_name}
            ORDER BY similarity DESC
            LIMIT {top_k}
        """)

        res = self.session.execute(
            query,
            {"embedding": query_embedding, "top_k": top_k},
        ).fetchall()
        
        top_k_matches = [
            {"code": row.code, "concept": row.concept, "similarity": row.similarity}
            for row in res if (threshold is None or row.similarity >= threshold)
        ]

        return top_k_matches

    def search_embed(self, query_text: Optional[str] = None, top_k: int = 5, threshold: float = None) -> list[Item]:
        embedding = compute_embedding(
                query_text,
                self.model_name,
                )
        return self.search(embedding[:self.embed_dim], top_k, threshold)