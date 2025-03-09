import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
    model = AutoModel.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

def compute_embedding(query, model_path):
    model, tokenizer, device = load_model(model_path)

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

        # Extract cls emebedding
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        embedding = cls_embeddings.squeeze().cpu().numpy().tolist()

    return embedding

def compute_emb_batched(queries, model_path, batch_size=256):
    model, tokenizer, device = load_model(model_path)
    embs = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Processing Entities"):
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
            #Extract cls embedding
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            batch_embeddings = cls_embeddings.squeeze().cpu().numpy().tolist()

            embs.extend(batch_embeddings)

    return embs