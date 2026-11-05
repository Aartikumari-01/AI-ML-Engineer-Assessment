# Task 2 – RAG System Evaluation Report

## 1. Project Overview

The Predictive Maintenance RAG System is a local document question-answering application. It retrieves relevant information from technical PDF documents and uses a locally hosted language model to generate answers with document and page references.

The system uses:

- PyMuPDF and Tesseract OCR for document extraction.
- Paragraph-aware text chunking.
- Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings.
- FAISS for vector similarity search.
- Ollama with Qwen 2.5 3B for answer generation.
- Streamlit for the web interface.

## 2. Evaluation Setup

Two technical documents were used:

1. `UCI_AI4I_Documentation.pdf`
2. `Predictive_Maintenance_IIoT_Research_2025.pdf`

The extracted pages retain their document names and page numbers. The text is divided into chunks of approximately 1,000 characters with 200-character overlap.

The embedding model converts each chunk into a 384-dimensional vector. FAISS uses normalized inner-product similarity for retrieval.

The RAG pipeline retrieves up to four passages for ordinary questions and retrieves passages separately from both documents for comparison questions. A similarity threshold of 0.35 is applied before answer generation.

## 3. Evaluation Questions and Results

### Test 1: Technical methodology

**Question:** How does the DRL agent integrate feedback from Random Forest and Gradient Boosting?

**Result:** Successful.

The system explained that Random Forest classifies faults and contributes a penalty to the DRL reward function, while Gradient Boosting predicts rare failure probabilities that influence the next state transition.

**Retrieved evidence:** Research paper, pages 5–6.

### Test 2: Unavailable information

**Question:** What is the population of India?

**Result:** Successful abstention.

The system responded:

"I could not find sufficient information in the provided documents."

No answer was generated from outside the retrieved documents.

### Test 3: Dataset information

**Question:** How many instances and features are present in the UCI AI4I 2020 Predictive Maintenance dataset?

**Result:** Successful.

The system returned 10,000 instances and 14 columns.

**Retrieved evidence:** UCI documentation, pages 1–3.

### Test 4: Comparison of documents

**Question:** Compare the purpose and methodology of the UCI AI4I 2020 dataset and the 2025 Industrial IoT research paper.

**Result:** Partially successful.

The system retrieved passages from both documents and described their purposes. However, it did not consistently retrieve the most relevant technical methodology passages and generated some claims about the research paper's experimental setup that were not sufficiently supported.

### Test 5: Roles of DRL, RF and GBM

**Question:** What are the specific roles of Deep Reinforcement Learning, Random Forest, and Gradient Boosting in the proposed methodology?

**Result:** Partially successful.

The system identified the general roles of all three methods. However, it added an unsupported statement that Random Forest addresses class imbalance.

**Retrieved evidence:** Research paper, pages 5–6.

## 4. Evaluation Summary

| Evaluation area         | Observation                                                |
| ----------------------- | ---------------------------------------------------------- |
| PDF extraction          | Successfully extracted text and retained page metadata     |
| Chunking                | Paragraph-aware chunking with overlap                      |
| Embeddings              | 384-dimensional sentence embeddings                        |
| Vector retrieval        | FAISS similarity search                                    |
| Document metadata       | Document names and page numbers retained                   |
| Local answer generation | Qwen 2.5 3B through Ollama                                 |
| Source references       | Retrieved document and page references displayed           |
| Unavailable questions   | Successfully abstained in the tested case                  |
| Comparison questions    | Both documents retrieved; answer quality needs improvement |
| Hallucination control   | Some unsupported claims remain                             |

## 5. Limitations

1. The small local language model may generate unsupported statements even when the prompt instructs it not to.
2. Similarity scores do not guarantee that retrieved passages fully answer a question.
3. Comparison questions require relevant evidence from multiple documents and can retrieve less relevant passages.
4. The current citation implementation lists retrieved source pages but does not verify every generated claim against an individual passage.
5. The UCI documentation PDF contains an incomplete variables table because the original webpage printout did not capture all rows.
6. The research paper contains some unclear or inconsistent descriptions of its experimental setup. The system should report these uncertainties instead of resolving them through assumptions.

## 6. Local Deployment

The application runs locally using:

- Ollama at `http://localhost:11434`
- Qwen 2.5 3B as the local language model
- Streamlit at `http://localhost:8501`

No paid external language-model API is used. The embedding model and language model run locally after their initial downloads.

## 7. Conclusion

The RAG system successfully demonstrates PDF ingestion, OCR, paragraph-aware chunking, semantic retrieval, local language-model generation, source-page references, and a Streamlit interface.

The tests show that it can answer factual questions from the supplied documents and abstain when information is unavailable. However, comparison retrieval, citation verification, and hallucination control remain areas for improvement.

The project demonstrates a working local RAG pipeline while documenting its current limitations honestly.
