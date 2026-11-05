import sys
from pathlib import Path

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import generate_answer


st.set_page_config(
    page_title="Predictive Maintenance RAG",
    page_icon="🏭",
    layout="wide"
)


st.title("🏭 Predictive Maintenance RAG Assistant")

st.write(
    "Ask questions about the UCI AI4I dataset "
    "and the 2025 Industrial IoT research paper."
)

st.success(
    "Local deployment: Qwen 2.5 3B through Ollama. "
    "No paid external AI API is used."
)

st.divider()

question = st.text_area(
    "Enter your question",
    placeholder=(
        "Example: What methodology does the "
        "2025 research paper use for predictive maintenance?"
    ),
    height=100
)

if st.button("Ask Question", type="primary"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                answer, sources = generate_answer(
                    question.strip()
                )

            st.subheader("Answer")

            st.write(answer)

            st.divider()

            st.subheader("Retrieved Sources")

            if sources:

                for i, source in enumerate(
                    sources,
                    start=1
                ):

                    with st.expander(
                        f"Source {i}: {source['document']} "
                        f"| Page {source['page']}"
                    ):

                        st.write(
                            "**Similarity Score:**",
                            round(source["score"], 4)
                        )

                        st.write(source["text"])

            else:

                st.info("No relevant sources were retrieved.")

        except Exception as error:

            st.error(
                "Unable to generate an answer. "
                "Please check that Ollama is running "
                "and the Qwen model is available."
            )

            st.code(str(error))


st.divider()

st.caption(
    "Powered by Streamlit, FAISS, "
    "Sentence Transformers and Ollama."
)

st.caption(
    "All question answering and language-model "
    "inference run locally on your computer."
)