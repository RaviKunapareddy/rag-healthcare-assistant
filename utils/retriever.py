import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_PATH = "../MedQuAD/data/faiss.index"
METADATA_PATH = "../MedQuAD/data/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load FAISS index and metadata
def load_index_and_metadata():
    print("Loading FAISS index and metadata...")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Retrieve top-k passages for a query
def retrieve(query, k=5):
    index, metadata = load_index_and_metadata()
    model = SentenceTransformer(EMBED_MODEL)
    query_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_emb, dtype=np.float32), k)
    results = []
    for idx, score in zip(I[0], D[0]):
        item = metadata[idx].copy()
        item['score'] = float(score)
        results.append(item)
    return results

from sentence_transformers import CrossEncoder

def rerank_cross_encoder(query, candidates, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    cross_encoder = CrossEncoder(model_name)
    pairs = [(query, item['answer']) for item in candidates]
    scores = cross_encoder.predict(pairs)
    for item, score in zip(candidates, scores):
        item['rerank_score'] = float(score)
    candidates = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
    return candidates


def print_retrieved_oneliners(results, max_items=3, maxlen=80):
    """
    Print retrieved Q&A pairs as one-liners, truncating question and answer for readability.
    """
    print(f"\nRetrieved Documents (top {min(max_items, len(results))}):")
    for i, item in enumerate(results[:max_items], 1):
        q = item['question'].replace('\n', ' ').strip()
        a = item['answer'].replace('\n', ' ').strip()
        q = (q[:maxlen] + '...') if len(q) > maxlen else q
        a = (a[:maxlen] + '...') if len(a) > maxlen else a
        print(f"{i}. Q: {q}")
        print(f"   A: {a}")

    print()  # for newline
