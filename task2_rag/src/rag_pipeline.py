
import requests
from search_documents import search_documents

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

SIMILARITY_THRESHOLD = 0.35


def generate_answer(question):

    comparison_words = [
        "compare",
        "comparison",
        "difference",
        "differences",
        "both documents",
        "two documents",
        "contrast"
    ]

    is_comparison = any(
        word in question.lower()
        for word in comparison_words
    )

    if is_comparison:

        results = search_documents(
            question,
            top_k=3,
            per_document=True
        )

    else:

        results = search_documents(
            question,
            top_k=4
        )

    results = [
        result for result in results
        if result["score"] >= SIMILARITY_THRESHOLD
    ]

    if not results:

        return (
            "I could not find sufficient information "
            "in the provided documents."
        ), []

    context = ""

    for i, result in enumerate(results, start=1):

        context += f"""
Source {i}
Document: {result['document']}
Page: {result['page']}
Content:
{result['text']}
"""

    prompt = f"""
You are a careful technical document question-answering assistant.

Answer the user's question using ONLY the supplied document passages.

STRICT RULES:

1. Never use outside knowledge.
2. Never invent facts, numbers, methods, datasets, or experimental results.
3. Only state information that is explicitly supported by the context.
4. If information is missing, clearly say that it is not available.
5. Distinguish between facts stated by the document and uncertain information.
6. If the documents contain conflicting statements, explain the conflict.
7. Do not resolve contradictions by guessing.
8. Do not assume that a dataset is real or synthetic unless the context says so.
9. For comparison questions, explain each document separately before comparing them.
10. If evidence for one document is insufficient, mention this limitation.
11. Use simple, clear language.
12. Do not generate document citations yourself.
13. Do not add unsupported conclusions.
14. If the context cannot answer the question, respond:
"I could not find sufficient information in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1
            }
        },
        timeout=300
    )

    response.raise_for_status()

    answer = response.json()["response"].strip()

    citations = []

    for result in results:

        citation = (
            f"[{result['document']}, Page {result['page']}]"
        )

        if citation not in citations:
            citations.append(citation)

    if citations:

        answer += "\n\nSources: " + "; ".join(citations)

    return answer, results


if __name__ == "__main__":

    print("====================================")
    print(" PREDICTIVE MAINTENANCE RAG SYSTEM")
    print("====================================")

    question = input("\nEnter your question: ")

    answer, sources = generate_answer(question)

    print("\nANSWER:")
    print(answer)

    print("\nRETRIEVED SOURCES:")

    for source in sources:

        print(
            f"- {source['document']} | Page {source['page']} "
            f"| Score: {source['score']:.4f}"
        )
