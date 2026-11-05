\# Task 2 – Predictive Maintenance RAG System



\## 1. Project Overview



The Predictive Maintenance RAG System is a local document question-answering application. It uses Retrieval-Augmented Generation (RAG) to answer technical questions from PDF documents.



Instead of relying only on the language model's existing knowledge, the system retrieves relevant information from supplied documents and uses that information to generate answers with source references.



\### Objectives



\- Extract information from technical PDF documents.

\- Convert extracted text into smaller chunks.

\- Generate semantic embeddings.

\- Store and retrieve document vectors using FAISS.

\- Generate answers using a locally running language model.

\- Display document names and page references.

\- Avoid answering questions when sufficient supporting information is unavailable.



\## 2. Technologies Used



| Technology | Purpose |

|---|---|

| Python | Main programming language |

| PyMuPDF | PDF text extraction |

| Tesseract OCR | Extract text from scanned PDF pages |

| Pillow | Image processing for OCR |

| Sentence Transformers | Generate text embeddings |

| all-MiniLM-L6-v2 | Embedding model |

| FAISS | Vector similarity search |

| Ollama | Run the language model locally |

| Qwen 2.5 3B | Local answer generation |

| Requests | Communicate with Ollama API |

| Streamlit | Web application interface |



\## 3. Documents Used



The system uses two technical documents:



1\. `UCI\_AI4I\_Documentation.pdf`

2\. `Predictive\_Maintenance\_IIoT\_Research\_2025.pdf`



The assessment instruction PDF is excluded from the retrieval process.



\## 4. System Methodology



The RAG pipeline follows these steps:



\### Step 1: Document Extraction



PyMuPDF extracts text from PDF documents. Tesseract OCR is used when a page does not contain extractable text.



The extracted information retains:

\- Document name

\- Page number

\- Extracted text



\### Step 2: Text Preprocessing and Chunking



The extracted text is divided into smaller chunks.



Parameters used:



\- Chunk size: 1,000 characters

\- Chunk overlap: 200 characters

\- Chunking strategy: Paragraph-aware chunking



Overlap helps preserve context between consecutive chunks. Smaller chunks also make it easier to retrieve relevant information without sending entire documents to the language model.



\### Step 3: Embedding Generation



The Sentence Transformers model `all-MiniLM-L6-v2` converts each text chunk into a numerical vector.



The embedding dimension is 384.



These vectors represent the semantic meaning of the text and help identify passages relevant to a user's question.



\### Step 4: Vector Storage



FAISS stores the generated vectors and performs similarity searches.



The implementation uses normalized vectors and inner-product similarity.



\### Step 5: Semantic Retrieval



When a user enters a question:



1\. The question is converted into an embedding.

2\. FAISS searches for relevant document chunks.

3\. The most relevant passages are retrieved.

4\. A similarity threshold of 0.35 filters the retrieved results.



For comparison questions, the current implementation retrieves passages separately from both documents.



\### Step 6: Answer Generation



The retrieved passages are provided as context to the Qwen 2.5 3B language model.



Ollama runs the model locally and generates an answer based on the supplied context.



The prompt instructs the model to:

\- Use only the supplied document passages.

\- Avoid inventing facts or results.

\- Mention missing information.

\- Explain conflicting information.

\- Abstain when sufficient evidence is unavailable.



\### Step 7: Source References



The application displays the document name and page number associated with retrieved passages.



These references help users identify where the retrieved information came from.



\## 5. Answers to the Assessment Questions



\### Q1. What is the objective of the RAG system?



The objective is to build a document-based question-answering system that retrieves relevant information from technical documents and generates grounded answers using a locally hosted language model.



\### Q2. What methodology is used?



The system follows a Retrieval-Augmented Generation approach:



PDF extraction → Text preprocessing → Chunking → Embedding generation → FAISS retrieval → Local LLM generation → Source references.



\### Q3. What parameters are used?



| Parameter | Value |

|---|---|

| Embedding model | all-MiniLM-L6-v2 |

| Embedding dimension | 384 |

| Chunk size | 1,000 characters |

| Chunk overlap | 200 characters |

| Similarity threshold | 0.35 |

| Language model | Qwen 2.5 3B |

| Vector database | FAISS |

| Temperature | 0.1 |



\### Q4. What are the conclusions?



The system successfully demonstrates document extraction, semantic retrieval, local answer generation, and source-page references.



It can answer tested questions about predictive maintenance documents and abstain when the requested information is unavailable.



However, comparison retrieval and hallucination control need further improvement.



\### Q5. Compare the methodology of the two documents.



\*\*UCI AI4I 2020 documentation:\*\*



The UCI document describes a synthetic predictive maintenance dataset containing 10,000 instances. It provides information about the dataset, its variables, and its intended use for machine learning tasks.



\*\*2025 Industrial IoT research paper:\*\*



The research paper proposes an Industrial IoT predictive maintenance framework combining Deep Reinforcement Learning (DRL), Random Forest (RF), and Gradient Boosting Machines (GBM).



\- DRL learns maintenance and network optimization decisions.

\- RF performs fault classification.

\- GBM predicts rare events and failure probabilities.



\*\*Comparison:\*\*



The UCI document primarily describes a dataset, whereas the research paper proposes a predictive maintenance methodology using machine learning and reinforcement learning techniques.



\### Q6. What evidence supports the conclusions?



The conclusions are based on the retrieved passages from the supplied documents.



Examples:



\- The UCI dataset description and characteristics are available in the UCI documentation, pages 1–3.

\- The roles of DRL, RF, and GBM are described in the research paper, particularly pages 5–6.

\- The research paper's reported experimental results are presented in its results section, including the comparison table on page 14.



\### Q7. What happens when information is unavailable?



The system was tested with the question:



"What is the population of India?"



Since the provided documents do not contain this information, the system returned:



"I could not find sufficient information in the provided documents."



This demonstrates the system's abstention mechanism.



\## 6. Evaluation and Limitations



The system was tested with technical methodology, dataset information, document comparison, and unavailable-information questions.



The tests demonstrated that:

\- Technical information can be retrieved from the research paper.

\- Document names and page numbers are retained.

\- The system can abstain when information is unavailable.

\- Comparison questions may retrieve less relevant passages.

\- The small language model may occasionally generate unsupported statements.



The current citation implementation displays retrieved source pages but does not verify every individual generated claim against its supporting passage.



Further improvements could include better comparison retrieval, stronger evidence verification, and additional evaluation metrics.



\## 7. Local Deployment



The application uses local services:



\- Ollama API: `http://localhost:11434`

\- Streamlit application: `http://localhost:8501`

\- Language model: `qwen2.5:3b`



No paid external language-model API is used.



The embedding model and language model are downloaded initially and then run locally.



\### Installation



Install the Python dependencies from the project root:



```bash

pip install -r requirements.txt

```



Install Ollama separately and download the model:



```bash

ollama pull qwen2.5:3b

```



Tesseract OCR must also be installed separately.



\### Run the RAG Command-Line Application



From the project root:



```bash

python .\\task2\_rag\\src\\rag\_pipeline.py

```



\### Run the Streamlit Application



```bash

streamlit run .\\task2\_rag\\app.py

```



Open the local address displayed by Streamlit, normally:



`http://localhost:8501`



\## 8. Project Folder Structure



```text

task2\_rag/

│

├── documents/

│   ├── Assessment\_Instructions.pdf

│   ├── UCI\_AI4I\_Documentation.pdf

│   └── Predictive\_Maintenance\_IIoT\_Research\_2025.pdf

│

├── src/

│   ├── extract\_text.py

│   ├── chunk\_text.py

│   ├── create\_embeddings.py

│   ├── search\_documents.py

│   └── rag\_pipeline.py

│

├── reports/

│   └── evaluation\_report.md

│

├── vector\_store/

│   ├── faiss\_index.bin

│   └── chunk\_metadata.json

│

├── app.py

├── processed\_pages.json

├── chunks.json

└── README.md

```



\## 9. Final Conclusion



The Predictive Maintenance RAG System demonstrates an end-to-end local Retrieval-Augmented Generation pipeline.



It combines document processing, OCR, semantic embeddings, FAISS retrieval, and local language-model inference through Ollama.



The Streamlit interface provides an accessible way to ask questions and inspect retrieved document passages.



The project also identifies limitations in retrieval quality, source verification, and hallucination control, providing clear directions for future improvement.

