from typing import List, Union
import torch

from utils.model_manager import ModelManager

def compute_embedding(query: str, model_name: str) -> List[float]:
    model, tokenizer, device = ModelManager.get_model() or ModelManager.load_model(model_name)
    
    with torch.no_grad():
        encoded = tokenizer(
            query,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=40,
            return_attention_mask=True
        )
        
        outputs = model(
            input_ids=encoded['input_ids'].to(device),
            attention_mask=encoded['attention_mask'].to(device),
        )
        
        # Extract CLS embedding
        cls_embedding: List[float] = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy().tolist()
        
    return cls_embedding


def compute_emb_batched(
    queries: List[str],
    model_name: str,
    batch_size: int = 256
) -> List[List[float]]:
    
    model, tokenizer, device = ModelManager.get_model() or ModelManager.load_model(model_name)
    embs = []
    
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]
        
        with torch.no_grad():
            encoded = tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=40,
                return_attention_mask=True
            )
            
            outputs = model(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device),
            )
            
            # Extract CLS embeddings
            cls_embeddings: List[List[float]] = outputs.last_hidden_state[:, 0, :].cpu().numpy().tolist()
            embs.extend(cls_embeddings)
            
    return embs