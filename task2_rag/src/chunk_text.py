import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_DIR / "processed_pages.json"
OUTPUT_FILE = PROJECT_DIR / "chunks.json"

CHUNK_SIZE = 1000
OVERLAP = 200


def split_into_chunks(text):

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        if len(paragraph) > CHUNK_SIZE:

            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            start = 0

            while start < len(paragraph):

                piece = paragraph[start:start + CHUNK_SIZE]

                if piece.strip():
                    chunks.append(piece.strip())

                start += CHUNK_SIZE - OVERLAP

            continue

        if len(current_chunk) + len(paragraph) + 2 <= CHUNK_SIZE:

            current_chunk = (
                current_chunk + "\n\n" + paragraph
            ).strip()

        else:

            if current_chunk:
                chunks.append(current_chunk)

            overlap_text = current_chunk[-OVERLAP:]

            current_chunk = (
                overlap_text + "\n\n" + paragraph
            ).strip()

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


with open(INPUT_FILE, "r", encoding="utf-8") as file:
    pages = json.load(file)

all_chunks = []

for page in pages:

    page_chunks = split_into_chunks(page["text"])

    for chunk_number, chunk_text in enumerate(
        page_chunks,
        start=1
    ):

        all_chunks.append({
            "chunk_id": len(all_chunks) + 1,
            "document": page["document"],
            "page": page["page"],
            "chunk_number": chunk_number,
            "text": chunk_text
        })


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    json.dump(
        all_chunks,
        file,
        ensure_ascii=False,
        indent=4
    )


print("Text chunking completed!")
print("Total pages:", len(pages))
print("Total chunks:", len(all_chunks))
print("Chunk size:", CHUNK_SIZE)
print("Overlap:", OVERLAP)
print("Saved to:", OUTPUT_FILE)