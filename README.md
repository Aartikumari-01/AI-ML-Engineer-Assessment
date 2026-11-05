\# AI/ML Engineer Assessment



\## Project Overview



This repository contains two projects developed as part of the AI/ML Engineer Assessment.



1\. \*\*Task 1 – Predictive Maintenance using Machine Learning\*\*

2\. \*\*Task 2 – Predictive Maintenance Document Question Answering using RAG\*\*



Both projects use open-source Python libraries and demonstrate practical machine learning and AI engineering workflows.



\---



\# Task 1 – Predictive Maintenance using Machine Learning



\## 1. Objective



The objective is to predict machine failures using the UCI AI4I 2020 Predictive Maintenance dataset.



The project explores sensor data, trains multiple classification models, compares their performance, and identifies important features associated with machine failure.



\## 2. Dataset



Dataset: UCI AI4I 2020 Predictive Maintenance Dataset.



\- Total records: 10,000

\- Total columns: 14

\- Target variable: `Machine failure`

\- Features include machine type, air temperature, process temperature, rotational speed, torque, and tool wear.



\## 3. Methodology



1\. Exploratory Data Analysis (EDA).

2\. Missing-value and duplicate checks.

3\. Outlier analysis.

4\. Class imbalance analysis.

5\. Feature selection and preprocessing.

6\. Stratified train-test split.

7\. Training multiple classification models.

8\. Model evaluation and comparison.

9\. Random Forest hyperparameter tuning.

10\. Feature importance analysis.

11\. False-positive and false-negative analysis.

12\. Saving the trained model.

13\. Prediction on new machine inputs.



\## 4. Models Used



\- Logistic Regression

\- Random Forest Classifier

\- Support Vector Machine (SVM)

\- Tuned Random Forest Classifier



\## 5. Best Model Results



The tuned Random Forest model achieved the following test results:



| Metric | Score |

|---|---:|

| Accuracy | 98.35% |

| Precision | 90.70% |

| Recall | 57.35% |

| F1-score | 70.27% |

| ROC-AUC | 96.15% |

| PR-AUC | 80.99% |



The model achieved 39 true positives, 29 false negatives, 4 false positives, and 1,928 true negatives on the test set.



\## 6. Feature Importance



The most influential features according to the tuned Random Forest model were:



1\. Torque

2\. Rotational speed

3\. Tool wear

4\. Air temperature

5\. Process temperature



These importance values indicate model contribution, not causation.



\## 7. Run Task 1



From the project root:



```bash

python .\\task1\_ml\\src\\train\_models.py

```



For predictions on new machine inputs:



```bash

python .\\task1\_ml\\src\\predict\_new.py

```



The exploratory analysis notebook is available at:



`task1\_ml/notebooks/eda\_experimentation.ipynb`



\---



\# Task 2 – Predictive Maintenance RAG System



\## 1. Objective



The objective is to develop a Retrieval-Augmented Generation (RAG) application that answers technical questions using supplied PDF documents.



The application retrieves relevant document passages and uses a locally hosted language model to generate answers with document and page references.



\## 2. Technologies Used



\- Python

\- PyMuPDF

\- Tesseract OCR

\- Sentence Transformers

\- all-MiniLM-L6-v2

\- FAISS

\- Ollama

\- Qwen 2.5 3B

\- Streamlit

\- Requests



\## 3. Documents Used



\- UCI AI4I 2020 documentation

\- 2025 Industrial IoT predictive maintenance research paper



\## 4. Methodology



The RAG pipeline follows these steps:



1\. Extract text from PDF documents.

2\. Apply OCR to pages requiring image-based extraction.

3\. Split text into smaller chunks.

4\. Generate 384-dimensional text embeddings.

5\. Store embeddings in a FAISS vector index.

6\. Retrieve relevant passages using semantic similarity.

7\. Pass retrieved passages to the local Qwen language model.

8\. Generate answers using document context.

9\. Display retrieved document names and page numbers.



\## 5. Important Parameters



| Parameter | Value |

|---|---|

| Chunk size | 1,000 characters |

| Chunk overlap | 200 characters |

| Embedding model | all-MiniLM-L6-v2 |

| Embedding dimension | 384 |

| Vector database | FAISS |

| Similarity threshold | 0.35 |

| Language model | Qwen 2.5 3B |

| Temperature | 0.1 |



\## 6. Evaluation



The RAG application was tested using:



\- Technical methodology questions.

\- Dataset information questions.

\- Comparison questions involving both documents.

\- Questions about unavailable information.



The system successfully retrieved technical passages and demonstrated abstention for an unavailable question.



Some limitations remain in comparison retrieval, citation verification, and hallucination control.



\## 7. Run Task 2



First, ensure Ollama is installed and the model is available:



```bash

ollama pull qwen2.5:3b

```



Run the command-line application:



```bash

python .\\task2\_rag\\src\\rag\_pipeline.py

```



Run the Streamlit application:



```bash

streamlit run .\\task2\_rag\\app.py

```



Open the local URL displayed by Streamlit, normally:



`http://localhost:8501`



Ollama's local API normally runs at:



`http://localhost:11434`



No paid external language-model API is used.



\---



\# Installation and Requirements



\## 1. Python Environment



Create a virtual environment:



```bash

python -m venv .venv

```



Activate it in Windows PowerShell:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



\## 2. Install Python Dependencies



```bash

pip install -r requirements.txt

```



\## 3. Additional Software



Install these separately:



\- Ollama for local Qwen model inference.

\- Tesseract OCR for scanned PDF extraction.



Tesseract must be available at the path configured in `task2\_rag/src/extract\_text.py`.



\---



\# Repository Structure



```text

AI\_ML\_Engineer\_Assessment/

│

├── task1\_ml/

│   ├── data/

│   ├── models/

│   ├── notebooks/

│   ├── reports/

│   ├── src/

│   ├── README.md

│   ├── eda.py

│   └── eda\_visualization.py

│

├── task2\_rag/

│   ├── documents/

│   ├── src/

│   ├── reports/

│   ├── vector\_store/

│   ├── app.py

│   ├── processed\_pages.json

│   ├── chunks.json

│   └── README.md

│

├── .venv/

├── requirements.txt

└── README.md

```



\---



\# Conclusion



This repository demonstrates practical skills in machine learning, exploratory data analysis, classification, model evaluation, hyperparameter tuning, feature importance, document processing, semantic search, vector databases, and Retrieval-Augmented Generation.



Task 1 demonstrates predictive maintenance using supervised machine learning.



Task 2 demonstrates a local document question-answering system using open-source embeddings, FAISS, and a locally hosted language model.



Both projects document their methodologies, results, and limitations.

