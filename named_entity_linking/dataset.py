import os
import json
import itertools

from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from collections import defaultdict

    
def load_cfg(config_path_or_json: str) -> Dict:
    if os.path.exists(config_path_or_json) and config_path_or_json.lower().endswith(".json"):
        with open(config_path_or_json, "r") as f:
            return json.load(f)
    return json.loads(config_path_or_json)

def get_splits(
    path_queries: str,
    path_kb: str,
    dev_test_split: float = 0.1
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, str]]:
    
    # load data
    with open(path_queries, 'r', encoding='utf-8') as f:
        data_queries = [json.loads(line.strip()) for line in f if line.strip()]
    
    with open(path_kb, 'r', encoding='utf-8') as f:
        data_kb = [json.loads(line.strip()) for line in f if line.strip()]
    
    # create KB reference
    kb = {entry['code']: entry['concept'] for entry in data_kb}
    
    # get matched queries for test split
    matched_queries = defaultdict(lambda: {'wiki_items': []})
    unmatched_queries = defaultdict(lambda: {'wiki_items': []})
    
    for sample in data_queries:
        for code, items in sample.items():
            query_items = items.get('wikidata_items', []) + items.get('wikipedia_items', [])
            target_dict = matched_queries if code in kb else unmatched_queries
            target_dict[code]['wiki_items'].extend(query_items)
    
    matched_tuples = [(code, item) for code, value in matched_queries.items() 
                     for item in value['wiki_items']]
    unmatched_tuples = [(code, item) for code, value in unmatched_queries.items() 
                       for item in value['wiki_items']]
    
    dev_matched, test_tuples = train_test_split(matched_tuples, test_size=dev_test_split)
    
    # add unmatched queries to dev set
    dev_tuples = dev_matched + unmatched_tuples
    
    return dev_tuples, test_tuples, kb

def load_data(
    path_queries: str,
    path_kb: str,
    dev_test_split: float = 0.1,
    train_val_split: float = 0.2,
) -> Tuple[Dict[str, List], List[Tuple[str, str]]]:

    dev_tuples, test_tuples, _ = get_splits(path_queries, path_kb, dev_test_split)
    
    # group dev tuples by code into dict
    dev_data = defaultdict(list)
    for code, mention in dev_tuples:
        dev_data[code].append(mention)

    # generate pair combinations for dev set
    dev_pairs = []
    query_id2idx = {}
    current_idx = 0
    for query_id, query_items in dev_data.items():
        if query_id not in query_id2idx:
            query_id2idx[query_id] = current_idx
            current_idx += 1
            
        if len(query_items) > 1:
            pairs = itertools.combinations(query_items, 2)
            dev_pairs.extend((p[0], p[1], query_id2idx[query_id]) for p in pairs)
    
    train_data, val_data = train_test_split(dev_pairs, test_size=train_val_split)
    
    print(f"Train samples: {len(train_data)}, validation samples: {len(val_data)}, test samples: {len(test_tuples)}")

    return {"train": train_data, "eval": val_data}, test_tuples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process and split data for training and evaluation')
    parser.add_argument('path_queries', type=str, help='Path to the queries file (jsonl)')
    parser.add_argument('path_kb', type=str, help='Path to the knowledge base file (jsonl)')
    parser.add_argument('--dev_test_split', type=float, default=0.1, help='Proportion of matched queries for testing')
    parser.add_argument('--train_val_split', type=float, default=0.2, help='Proportion of dev set for validation')
    
    args = parser.parse_args()

    dataset_splits, test_data = load_data(
        args.path_queries, 
        args.path_kb, 
        args.dev_test_split, 
        args.train_val_split
    )