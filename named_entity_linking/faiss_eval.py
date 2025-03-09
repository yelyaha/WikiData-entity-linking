import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from eval import create_mappings, map_items
from embeddings import compute_emb_batched

class FaissSearcher:
    def __init__(
        self,
        top_k: int = 64,
        threshold: float = 0.6,
        ef_construction: int = 128,
        ef_search: int = 64
    ):
        self.index = None
        self.mappings = None
        self.top_k = top_k
        self.threshold = threshold
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def build_index(
        self,
        embeddings: np.ndarray,
        index_path: Optional[str] = None
    ) -> faiss.Index:
        dimension = np.array(embeddings).shape[1]
        print("DIMENSION", dimension)
        M = 16  # Number of connections per layer in HNSW graph

        self.index = faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.hnsw.efSearch = self.ef_search
        
        embeddings = np.vstack(embeddings).astype(np.float32) 
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        if index_path:
            faiss.write_index(self.index, index_path)
            print(f"Index saved to {index_path}")
        
        return self.index

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        mappings: Dict,
        index_path: str,
    ) -> List[List[Tuple[str, str, float]]]:
        # Load index if not already loaded
        if self.index is None:
            self.index = faiss.read_index(index_path)
            
        # Normalize query embeddings and search
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)
        scores, indices = self.index.search(query_embeddings, self.top_k)
        
        # Process results
        retrieval_results = []
        for i in range(len(query_embeddings)):
            retrieved = []
            for idx, score in zip(indices[i], scores[i]):
                if idx == -1: continue #NIL entity scenario
                
                elif score >= self.threshold:
                    code = mappings["idx2code"][idx]
                    concept = mappings["code2concept"][code]
                    retrieved.append((code, concept, 1 - float(score)))
                    
            retrieval_results.append(retrieved)
            
        return retrieval_results

def evaluate_matches_faiss(
    retrieval_results: List[List[Tuple[str, str, float]]],
    query_items: List[Tuple[str, str]],
    top_k_values: List[int],
    model_name: str,
    mappings: Dict
) -> Dict:
    # Initialize accuracy scores
    accuracy_scores = {k: 0 for k in top_k_values}
    true_codes, pred_codes = [], []
    true_idxs = [item[1] for item in query_items]
    nil_label = "NIL"
    
    # Process each query result
    for query_idx, results in enumerate(tqdm(retrieval_results, desc="Evaluating")):
        true_idx = true_idxs[query_idx]
        if true_idx in mappings["idx2code"]:
            true_code = mappings["idx2code"][true_idx]
        else:
            true_code = nil_label
        true_codes.append(true_code)
        
        # Empty results are NIL predictions
        if not results:
            predicted_code = nil_label
            pred_codes.append(predicted_code)
            if true_code == nil_label:
                accuracy_scores = {k: accuracy_scores[k] + 1 for k in top_k_values}
            continue
            
        # Get predicted first code
        predicted_code = results[0][0] if results else nil_label
        pred_codes.append(predicted_code)
        
        # Calculate top-k accuracy
        for k in top_k_values:
            topk_codes = [result[0] for result in results[:k]]
            if true_code in topk_codes or (true_code == nil_label and predicted_code == nil_label):
                accuracy_scores[k] += 1
    
    # Compute final metrics
    total_queries = len(query_items)
    accuracy_scores = {
        f'acc@{k}': round(score/total_queries, 4) 
        for k, score in accuracy_scores.items()
    }

    # Calculate precision, recall, and F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_codes,
        pred_codes,
        average='micro',
        # Calculate for all labels, including NIL
    )
    
    # Compile results
    results = {
        'Model': model_name.split('/')[-1][:10],
        'Pr': round(precision, 4),
        'Re': round(recall, 4),
        'F1': round(f1, 4),
        **accuracy_scores
    }
    
    # Create and display results table
    results_df = pd.DataFrame([results])
    print(results_df.to_markdown(index=False))
    
    return results

def main():
    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the embedding model")
    parser.add_argument("--doc", type=str, required=True, help="Path to document data")
    parser.add_argument("--kb", type=str, required=True, help="Path to knowledge base data")
    
    parser.add_argument("--mappings_path", type=str, default=None, help="Path to load/save mappings")
    parser.add_argument("--index_path", type=str, default=None, help="Path to load/save FAISS index")
    parser.add_argument("--top_k", type=int, default=64, help="Maximum number of results to return")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold")
    parser.add_argument("--label", type=str, default="MEDICATION", help="Label type to evaluate")
    parser.add_argument("--include_nil", action="store_true", default=False, help="Whether to include NIL labels")
    
    args = parser.parse_args()
    
    searcher = FaissSearcher(
        top_k=args.top_k, 
        threshold=args.threshold
    )
    
    # Set default path
    if args.index_path is None:
        args.index_path = "faiss_index.bin"
        
    if args.mappings_path is None:
        args.mappings_path = "mappings.pkl"
    
    # Load samples doc and kb files
    with open(args.doc, 'r', encoding='utf-8') as f:
        data_doc = [json.loads(l) for l in f.readlines() if l]
        
    # Load query and kb entities
    build_index = not Path(args.index_path).exists()
    if build_index:
        print("Building new index")
        
        # Load KB document to store in index
        with open(args.kb, 'r', encoding='utf-8') as f:
            data_kb = [json.loads(line.strip()) for line in f if line.strip()]

        mappings = create_mappings(data_kb, args.mappings_path)
        kb_items, query_items = map_items(data_doc, mappings, args.label, args.include_nil)

        # Compute KB embeddings and build index
        kb_concepts = [concept for concept, _ in kb_items]
        kb_embs = compute_emb_batched( kb_concepts, args.model_path)
        searcher.build_index(np.array(kb_embs), args.index_path)
    else:
        with open(args.mappings_path, 'rb') as f:
            mappings = pickle.load(f)
        _, query_items = map_items(data_doc, mappings, args.label, args.include_nil)
        
    # Compute query emebeddings
    query_concepts = [concept for concept, _ in query_items]
    query_embs = compute_emb_batched(query_concepts, args.model_path)
    
    results = searcher.search_batch(
        np.array(query_embs), 
        mappings,
        args.index_path
    )
    
    # Evaluate results
    results = evaluate_matches_faiss(
        retrieval_results=results,
        query_items=query_items,
        top_k_values=[1, 5, 10, 25, 50, 64],
        model_name=args.model_path,
        mappings=mappings,
    )
    
if __name__ == "__main__":
    main()
