python -m venv {venv_name}

{venv_name}\Scripts\activate

pip install -r requirements.txt
# 🌍 InterQuest: UN SDG Study Assistant & RAG Pipeline

A robust, LangChain-powered Retrieval-Augmented Generation (RAG) and Question-Generation pipeline designed to process official PDF documentation (the UN Sustainable Development Goals), generate comprehensive study questions through iterative LLM refinement, and provide precise, grounded answers.

---

## 🚀 Key Features

* **Automated Document Ingestion:** Loads and parses PDF resources using `PyPDFLoader`.
* **Iterative Question Generation (`Refine` Chain):** Uses a multi-step `refine` chain (`load_summarize_chain`) with Google Gemini to analyze text chunks sequentially, producing structured exam-style questions without losing critical information.
* **Vector Store & Embeddings:** Indexes document chunks using local embeddings (`OllamaEmbeddings` / `nomic-embed-text`) and a FAISS vector database for lightning-fast similarity searches.
* **Grounded Q&A Loop (`RetrievalQA`):** Employs a retrieval-augmented "stuff" chain to answer student study questions strictly referencing the source material.

---

## 🛠️ Tech Stack & Tools

* **Language:** Python (3.12+)
* **Orchestration:** LangChain (`langchain-classic`, `langchain-community`, `langchain-core`)
* **LLM Providers:** Google Gemini API (`langchain-google-genai`), Ollama (`langchain-ollama`)
* **Vector Database:** FAISS (`faiss-cpu`)[cite: 2, 3]
* **Environment Management:** `python-dotenv`[cite: 1, 2, 3]

---

## 📂 Project Structure

```text
InterQuest/
│
├── data/
│   └── SDG.pdf                 # Source UN Sustainable Development Goals PDF[cite: 1, 2, 3]
│
├── venv/                       # Python Virtual Environment
├── experiment.ipynb            # Jupyter Notebook pipeline implementation
├── answer.txt                  # Exported questions and generated answers[cite: 2, 3]
├── .env                        # API keys configuration (Google Gemini, etc.)
└── README.md
