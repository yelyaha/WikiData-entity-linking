import os
import json
import pickle
from collections import defaultdict
from typing import List, Tuple, Dict
from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from pytorch_metric_learning import miners, losses
from sklearn.metrics import precision_recall_fscore_support

from dataset import load_data, load_cfg
from train import PairedCollator

def load_model(path_model: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AutoModel.from_pretrained(path_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path_model, do_lower_case=True)
    return model, tokenizer, device

def get_embeddings(model, dataloader, device):
    embeddings, ids = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            batch_embeddings = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device)
            )
            cls_embeddings = batch_embeddings.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embeddings.cpu().detach().numpy())
            ids.extend(batch['labels'].cpu().numpy())
    return embeddings, ids

def create_mappings(
    data_kb: List[Dict],
    output_path: str = None,
) -> Dict[str, Dict]:
    # Create mappings for codes and concepts
    code2concept = {item['code']: item['concept'] for item in data_kb}
    code2idx = {code: idx for idx, code in enumerate(code2concept.keys())}
    idx2code = {idx: code for code, idx in code2idx.items()}

    nil_idx = -100  # Custom index for NIL entities
    code2idx["NIL"] = nil_idx
    idx2code[nil_idx] = "NIL"

    mappings = {
        "code2concept": code2concept,
        "code2idx": code2idx,
        "idx2code": idx2code
    }

    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(mappings, f)

    return mappings

def map_items(
    samples: List[Dict],
    mappings: Dict,
    label: str,
    include_nil: bool = False,
) -> Tuple[List[Tuple], List[Tuple], Dict]:
    # Extract target query entities with relevant label
    query_entities = [
        (code, sample["text"][start:end])
        for sample in samples
        for start, end, lbl, code in sample["label"]
        if lbl == label
    ]
    
    code2cid = mappings["code2idx"]
    code2concept = mappings["code2concept"]
    
    kb_items = [(concept, code2cid[code]) for code, concept in code2concept.items()]
    # Map query entities to KB entities for evaluation
    query_items = []
    for q_code, q_concept in query_entities:
        if q_code in code2concept:
            query_items.append((q_concept, code2cid[q_code]))
        elif include_nil:
            query_items.append((q_concept, -100))

    print(f"Extracted {len(kb_items)} kb items and {len(query_items)} query items")

    return kb_items, query_items

def compute_similarity_matrix(
    model_path: str,
    kb_items: List[Tuple],
    query_items: List[Tuple],
    batch_size: int = 256
) -> Tuple[np.ndarray, List[int], List[int]]:
    # Initialize model and dataloaders
    model, tokenizer, device = load_model(model_path)
    eval_collator = PairedCollator(tokenizer=tokenizer, mode="single")

    kb_loader = DataLoader(kb_items, batch_size=batch_size, collate_fn=eval_collator)
    query_loader = DataLoader(query_items, batch_size=batch_size, collate_fn=eval_collator)

    # Get embeddings
    kb_embs, kb_cids = get_embeddings(model, kb_loader, device)
    query_embs, query_cids = get_embeddings(model, query_loader, device)

    # Normalize and compute similarity
    kb_embs_norm = kb_embs / np.linalg.norm(kb_embs, axis=1, keepdims=True)
    query_embs_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
    score_matrix = np.matmul(query_embs_norm, kb_embs_norm.T)

    return score_matrix, kb_cids, query_cids

def evaluate_matches(
    score_matrix: np.ndarray,
    kb_cids: List[int],
    query_cids: List[int],
    top_k_values: List[int],
    threshold: float,
    model_name: str,
) -> Dict:
    accuracy_scores = {k: 0 for k in top_k_values}
    pred_cids = []
    nil_label = -100  # Special NIL entity label

    # Extract cosine similarity scores for each query vector
    for query_idx in tqdm(range(len(query_cids))):
        query_scores = score_matrix[query_idx]
        true_lbl = query_cids[query_idx]

        # Make prediction based on the threshold
        predicted_lbl = nil_label if np.max(query_scores) < threshold else kb_cids[np.argmax(query_scores)]
        pred_cids.append(predicted_lbl)

        # Calculate top-k accuracy
        for k in top_k_values:
            topk_indices = np.argpartition(query_scores, -k)[-k:]
            topk_cids = [kb_cids[idx] for idx in topk_indices]

            if true_lbl in topk_cids or (true_lbl == nil_label and predicted_lbl == nil_label):
                accuracy_scores[k] += 1

    # Compute final metrics
    accuracy_scores = {f'acc@{k}': round(score/len(query_cids), 4) for k, score in accuracy_scores.items()}
    precision, recall, f1, _ = precision_recall_fscore_support(
        query_cids,
        pred_cids,
        average='micro',
    )

    results = {
        'Model': model_name[:10],
        'Pr': round(precision, 4),
        'Re': round(recall, 4),
        'F1': round(f1, 4),
        **accuracy_scores
    }

    results_df = pd.DataFrame([results])
    print(results_df.to_markdown(index=False))

    return results

def retrieve_concepts(
    score_matrix: np.ndarray,
    kb_cids: List[int],
    query_items: List[Tuple],
    mappings: Dict,
    output_dir: str,
    top_k: int = 5,
    threshold: float = 0,
) -> Dict[str, List[Tuple[str, float]]]:
    query_to_concepts = defaultdict(list)
    cid2code = mappings["idx2code"]
    code2concept = mappings["code2concept"]

    output_file = os.path.join(output_dir, "retrieved.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for query_text, query_cid in query_items:
            query_scores = score_matrix[query_items.index((query_text, query_cid))]
            query_code = cid2code[query_cid]

            # Get top-k indices based on the threshold
            top_k_indices = np.argpartition(query_scores, -top_k)[-top_k:]
            top_k_indices = top_k_indices[query_scores[top_k_indices] >= threshold]

            # Sort by score in descending order
            top_k_indices = top_k_indices[np.argsort(-query_scores[top_k_indices])]

            retrieved_concepts = [
                (cid2code[kb_cids[idx]], code2concept[cid2code[kb_cids[idx]]], query_scores[idx])
                for idx in top_k_indices
            ]

            query_to_concepts[query_text] = retrieved_concepts

            # Write retrieved concepts to file
            f.write(f"Query: {query_text} [{query_code}] [{code2concept.get(query_code, '')}]\n")
            for code, concept, score in retrieved_concepts:
                match = 'match' if code == query_code else ''
                f.write(f"  - {code} | {match} | {concept} | (score: {score:.4f})\n")
            f.write("\n")

    return query_to_concepts

def main(): 
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--doc', type=str, required=True, help='Path to doc JSON file')
    parser.add_argument('--kb', type=str, required=True, help='Path to knowledge base JSON file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')

    parser.add_argument('--label', type=str, default='MEDICATION', help='Label type to extract from doc')
    parser.add_argument('--threshold', type=float, default=0.6, help='Threshold for NIL prediction')
    parser.add_argument('--top_k', type=int, nargs='+', default=[1, 10, 25, 50], help='List of k values for top-k accuracy')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument("--include_nil", action="store_true", default=False, help="Include NIL entities?")

    args = parser.parse_args()
    
    # Load samples doc and kb files
    with open(args.doc, 'r', encoding='utf-8') as f:
        data_doc = [json.loads(l) for l in f.readlines() if l]

    with open(args.kb, 'r', encoding='utf-8') as f:
        data_kb = [json.loads(line.strip()) for line in f if line.strip()]

    # Create (entity, index) pairs for queries and kb entities
    mappings = create_mappings(data_kb)
    kb_items, query_items = map_items(data_doc, mappings, args.label, args.include_nil)

    # Calculate similarity matrix
    score_matrix, kb_cids, query_cids = compute_similarity_matrix(args.model_path, kb_items, query_items)

    # Evaluate retrieved concepts
    results = evaluate_matches(
        score_matrix,
        kb_cids,
        query_cids,
        args.top_k,
        args.threshold,
        args.model_path
    )

if __name__ == "__main__":
    main()
    