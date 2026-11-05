
import json
import faiss
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "chunks.json"
VECTOR_DIR = PROJECT_DIR / "vector_store"

VECTOR_DIR.mkdir(parents=True, exist_ok=True)

with open(INPUT_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not chunks:
    raise ValueError("No text chunks found. Run chunk_text.py first.")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

print("Generating embeddings...")

embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

embeddings = np.asarray(embeddings, dtype="float32")

faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(
    index,
    str(VECTOR_DIR / "faiss_index.bin")
)

with open(
    VECTOR_DIR / "chunk_metadata.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(chunks, file, ensure_ascii=False, indent=4)

print("\nEmbedding generation completed!")
print("Total embeddings:", len(embeddings))
print("Embedding dimensions:", dimension)
print("FAISS index size:", index.ntotal)
print("Saved in:", VECTOR_DIR)
