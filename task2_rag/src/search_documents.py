
import json
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent.parent
VECTOR_DIR = PROJECT_DIR / "vector_store"

INDEX_FILE = VECTOR_DIR / "faiss_index.bin"
METADATA_FILE = VECTOR_DIR / "chunk_metadata.json"

UCI_DOCUMENT = "UCI_AI4I_Documentation.pdf"
RESEARCH_DOCUMENT = "Predictive_Maintenance_IIoT_Research_2025.pdf"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index(str(INDEX_FILE))

with open(METADATA_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

vectors = index.reconstruct_n(0, index.ntotal)


def search_documents(question, top_k=3, per_document=False):

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(question_embedding)

    if per_document:

        results = []

        document_queries = {
            UCI_DOCUMENT: (
                "UCI AI4I 2020 dataset purpose, "
                "data characteristics, number of instances, "
                "features, synthetic data and intended use"
            ),
            RESEARCH_DOCUMENT: (
                "2025 Industrial IoT predictive maintenance "
                "research paper methodology, proposed framework, "
                "models, experimental setup, datasets and evaluation"
            )
        }

        for document_name, targeted_query in document_queries.items():

            document_indices = [
                i for i, chunk in enumerate(chunks)
                if chunk["document"] == document_name
            ]

            if not document_indices:
                continue

            targeted_embedding = model.encode(
                [targeted_query],
                convert_to_numpy=True
            ).astype("float32")

            faiss.normalize_L2(targeted_embedding)

            document_vectors = faiss.IndexFlatIP(index.d)

            document_vectors.add(
                vectors[document_indices].copy()
            )

            scores, indices = document_vectors.search(
                targeted_embedding,
                top_k
            )

            for score, local_idx in zip(
                scores[0], indices[0]
            ):

                if local_idx == -1:
                    continue

                original_idx = document_indices[local_idx]

                results.append({
                    "score": float(score),
                    "document": chunks[original_idx]["document"],
                    "page": chunks[original_idx]["page"],
                    "text": chunks[original_idx]["text"]
                })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "score": float(score),
            "document": chunks[idx]["document"],
            "page": chunks[idx]["page"],
            "text": chunks[idx]["text"]
        })

    return results


if __name__ == "__main__":

    question = input("Enter your question: ")

    results = search_documents(question)

    print("\nMost relevant document passages:\n")

    for i, result in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("Document:", result["document"])
        print("Page:", result["page"])
        print("Similarity score:", round(result["score"], 4))
        print("Text:", result["text"][:500])
        print("-" * 60)
