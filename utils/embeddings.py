import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# CONFIGURATION
CSV_PATH = "../MedQuAD/data/medquad_cleaned.csv"
INDEX_PATH = "../MedQuAD/data/faiss.index"
METADATA_PATH = "../MedQuAD/data/metadata.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine if normalized)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

def save_index_and_metadata(index, index_path, metadata_path, df):
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    metadata = df.to_dict(orient="records")
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

if __name__ == "__main__":

    # 1. Load data
    df = pd.read_csv(CSV_PATH)

    # 2. Prepare texts to embed (question + answer)
    texts = (df["question"].astype(str) + " [SEP] " + df["answer"].astype(str)).tolist()

    # 3. Generate embeddings
    print("Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"Encoding {len(texts)} passages...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    # 4. Build FAISS index
    index = build_faiss_index(embeddings)

    # 5. Save index and metadata
    save_index_and_metadata(index, INDEX_PATH, METADATA_PATH, df)
    print(f"Metadata saved to {METADATA_PATH}")

    print("Embedding and indexing complete!")
