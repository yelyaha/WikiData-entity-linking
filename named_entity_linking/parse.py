import os
import json
import sys
import urllib.parse
from typing import Dict

from SPARQLWrapper import SPARQLWrapper, JSON

def parse_dataset(
    corpus_path: str,
    query: str,
    output_dir: str = "",
    save_to_file: bool = True
) -> Dict[str, Dict]:
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        corpus = [json.loads(line) for line in f if line.strip()]
        
    file_path = os.path.join(output_dir, "corpus.jsonl")
    
    endpoint_url = "https://query.wikidata.org/sparql"
    user_agent = f"WDQS-example Python/{sys.version_info[0]}.{sys.version_info[1]}"
    
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    
    try:
        response = sparql.query().convert()
        results = response["results"]["bindings"]
    except Exception as e:
        print(f"SPARQL query failed: {type(e).__name__}: {e}")
        return

    # transform response into dictionary by QID
    query_items = {}
    for item in results:
        qid = urllib.parse.unquote(item.get('item', {}).get('value', '').split('/')[-1].replace('_', ' '))
        label = item.get('itemLabel', {}).get('value', '')
        code = item.get('code', {}).get('value', '')
        aliases = item.get('itemAltLabel', {}).get('value', '').split('|')
        
        # collect all terms (label + aliases)
        terms = {label.strip()} | {term.strip() for term in aliases if term}
        
        if label != qid:
            if qid not in query_items:
                query_items[qid] = {'codes': {code}, 'wikidata_items': terms}
            else:
                query_items[qid]['wikidata_items'].update(terms)
                query_items[qid]['codes'].add(code)
    
    for qid in query_items:
        query_items[qid]['codes'] = sorted(list(query_items[qid]['codes']))

    # add surface terms from corpus
    for sample in corpus:
        for lbl_start, lbl_end, lbl_class, lbl_qids in sample["label"]:
            surface_term = sample["text"][lbl_start:lbl_end]
            for qid in lbl_qids:
                if qid in query_items:
                    items = list(query_items[qid].get('wikidata_items', [])) + list(query_items[qid].get('wikipedia_items', []))
                    known_items = set(item.lower() for item in items)
                    if surface_term.lower() not in known_items:
                        if 'wikipedia_items' not in query_items[qid]:
                            query_items[qid]['wikipedia_items'] = []
                        query_items[qid]['wikipedia_items'].append(surface_term)

    # merge items by unique code
    dataset = {}
    for qid, info in query_items.items():
        # use first code if multiple codes exist for the same item
        code = info['codes'][0] if info['codes'] else None
        if not code:
            continue
            
        if code not in dataset:
            dataset[code] = {
                'wikidata_items': [],
                'wikipedia_items': [],
                'qids': []
            }
            
        dataset[code]['wikidata_items'].extend(info.get('wikidata_items', []))
        dataset[code]['wikipedia_items'].extend(info.get('wikipedia_items', []))
        dataset[code]['qids'].append(qid)

    final_dataset = {
        code: {
            'wikidata_items': list(set(terms['wikidata_items'])),
            'wikipedia_items': list(set(terms['wikipedia_items'])),
            'qids': terms['qids']
        }
        for code, terms in dataset.items()
    }
    
    if save_to_file:
        with open(file_path, 'w', encoding='utf-8') as f:
            for code, values in final_dataset.items():
                f.write(json.dumps({code: values}) + '\n')
            print(f"Dataset saved to {file_path}")

    return final_dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path", type=str, help="Path to the annotated corpus file.")
    parser.add_argument("--query", type=str, help="The SPARQL query to be executed.")
    parser.add_argument("--output_dir", type=str, help="Path to save the output dataset.")
    parser.add_argument("--save_to_file", action="store_true", default=True, help="Save file?")

    args = parser.parse_args()
    parse_dataset(args.corpus_path, args.query, args.output_dir, args.save_to_file)