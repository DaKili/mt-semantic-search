import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse
from datetime import datetime
import hashlib

def get_cache_path(name):
    cache_dir = 'cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return os.path.join(cache_dir, name)

def get_data_hash(modules):
    return hashlib.md5(json.dumps(modules, sort_keys=True).encode()).hexdigest()

def load_modules():
    modules = []
    json_files = [f for f in os.listdir("input") if f.lower().endswith('.json')]
    for file in json_files:
        with open(os.path.join("input", file), 'r', encoding='utf-8') as f:
            modules += json.load(f)
    print(f"Loaded {len(modules)} total modules")
    return modules

def create_module_text(module):
    '''primitive semantic search without weights'''
    return f"{module['title']} {module['content']} {module['learning_outcomes']}"

def load_or_compute_embeddings(modules, model_name, force_recompute=False):
    cache_file = get_cache_path('embeddings.npz')
    metadata_file = get_cache_path('metadata.json')
    
    if not force_recompute and os.path.exists(cache_file) and os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        if (metadata['data_hash'] == get_data_hash(modules) and 
            metadata['model_name'] == model_name):
            data = np.load(cache_file)
            return data['embeddings']
    
    print("Computing new embeddings...")
    model = SentenceTransformer(model_name)
    
    texts = [create_module_text(module) for module in modules]
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts)
        embeddings.extend(batch_embeddings)
    
    embeddings = np.array(embeddings)
    
    print("Saving embeddings to cache...")
    np.savez_compressed(cache_file, embeddings=embeddings)
    metadata = {
        'data_hash': get_data_hash(modules),
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'num_modules': len(modules)
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return embeddings

def find_similar_modules(module_idx, embeddings, modules, threshold=0.6):
    similarities = cosine_similarity([embeddings[module_idx]], embeddings)[0]
    mask = (similarities >= threshold) & (np.arange(len(similarities)) != module_idx)
    similar_indices = np.where(mask)[0]
    
    results = [
        {
            'module_id': modules[idx]['module_id'],
            'title': modules[idx]['title'],
            'similarity': similarities[idx]
        }
        for idx in similar_indices
    ]
    
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    print(f"\nFound {len(results)} modules above similarity threshold of {threshold}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force-recompute', action='store_true', 
                        help='Force recomputation of embeddings')
    args = parser.parse_args()
    
    modules = load_modules()
    print(f"\nTotal modules loaded: {len(modules)}")
    
    # MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
    MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
    embeddings = load_or_compute_embeddings(modules, MODEL_NAME, args.force_recompute)
    
    number = next((i for i, module in enumerate(modules) 
                    if module['title'].startswith("Parallel Programming")), 42)
    
    similar = find_similar_modules(number, embeddings, modules)
    
    # Print results
    print("\n=== Results ===")
    print(f"\nTop 5 modules similar to: {modules[number]['title']}\n")
    for result in similar[:5]:
        print(f"Module: ({result['module_id']}) {result['title']}")
        print(f"Similarity: {result['similarity']:.2f}\n")
    

if __name__ == "__main__":
    main()