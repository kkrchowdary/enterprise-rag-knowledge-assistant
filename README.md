# Enterprise RAG Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application built using **Python, FastAPI, Streamlit, Ollama, ChromaDB, and Sentence Transformers**.

The application enables users to upload PDF documents, extract and index their content, and ask natural-language questions using a locally configured Large Language Model (LLM).

It combines semantic search with LLM-based answer generation to retrieve relevant document information and provide supporting source references.

## 1. Project Overview

Traditional Large Language Models rely on their training knowledge and may generate inaccurate responses when answering questions about private or domain-specific documents.

Retrieval-Augmented Generation (RAG) addresses this limitation by retrieving relevant information from an external knowledge base and supplying it as context to an LLM.

This project demonstrates the implementation of a document-based RAG application using open-source technologies.

### Objectives

- Build an end-to-end document question-answering application.
- Extract text from uploaded PDF documents.
- Split extracted text into manageable chunks.
- Generate semantic embeddings using Sentence Transformers.
- Store and retrieve document embeddings using ChromaDB.
- Generate context-grounded responses using Ollama.
- Provide source references and supporting passages.
- Expose backend functionality through REST APIs.
- Develop an interactive user interface using Streamlit.

## 2. Technology Stack

| Category             | Technology                |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Backend Framework    | FastAPI                   |
| Frontend             | Streamlit                 |
| LLM Integration      | Ollama                    |
| Language Model       | Configurable Ollama model |
| Embedding Model      | all-MiniLM-L6-v2          |
| Vector Database      | ChromaDB                  |
| PDF Processing       | PyMuPDF                   |
| API Communication    | Requests                  |
| Configuration        | python-dotenv             |

## 3. System Architecture

```text
                  USER
                   |
                   v
            Streamlit Frontend
                   |
          +--------+--------+
          |                 |
          v                 v
     PDF Upload        User Question
          |                 |
          v                 v
     FastAPI API        FastAPI API
          |                 |
          v                 v
     PDF Extraction     RAG Service
          |                 |
          v                 v
     Text Chunking      Query Embedding
          |                 |
          v                 v
     Embedding Model    ChromaDB Search
          |                 |
          v                 v
     ChromaDB Storage   Relevant Chunks
                            |
                            v
                       Context Builder
                            |
                            v
                        Ollama LLM
                            |
                            v
                     Generated Answer
                            |
                            v
                      Source Citations
                            |
                            v
                     Streamlit Display
```

## 4. Features

### PDF Document Ingestion

Users can upload PDF documents through the Streamlit interface.

The backend:

1. Validates the uploaded file.
2. Saves it using a unique filename.
3. Extracts readable text using PyMuPDF.
4. Splits the extracted text into chunks.
5. Generates embeddings.
6. Stores embeddings and document metadata in ChromaDB.

### Semantic Search

User questions are converted into vector embeddings.

ChromaDB performs similarity search to retrieve relevant document chunks.

### RAG Answer Generation

Retrieved document passages are supplied to an Ollama language model as context.

The model generates an answer based on the retrieved information.

### Source Citations

The application is designed to provide source references alongside generated answers.

Source metadata includes:

- Document filename
- Page number
- Retrieved passage
- Retrieval distance
- Citation identifier

Citation accuracy requires evaluation because generated citation markers do not independently prove factual support.

### Relevance Filtering

The RAG service supports configurable distance-based filtering.

A relevance threshold can be configured to exclude retrieved passages that exceed the selected distance.

Threshold values must be calibrated to the ChromaDB distance metric and embedding model.

### Interactive Chat

The Streamlit interface provides:

- PDF upload
- Document processing status
- Natural-language question answering
- Chat history within the current session
- Source citation display
- Adjustable retrieval settings
- Clear-chat functionality

## 5. Project Structure

```text
enterprise-rag-assistant/
|
|-- app/
|   |-- main.py
|   |
|   |-- ingestion/
|   |   |-- pdf_loader.py
|   |   |-- chunker.py
|   |   |-- ingest.py
|   |
|   |-- retrieval/
|   |   |-- embeddings.py
|   |   |-- vector_store.py
|   |   |-- search.py
|   |
|   |-- generation/
|   |   |-- llm.py
|   |   |-- prompts.py
|   |
|   |-- services/
|       |-- rag_service.py
|
|-- frontend/
|   |-- streamlit_app.py
|
|-- data/
|   |-- documents/
|   |-- chroma_db/
|
|-- .env.example
|-- .gitignore
|-- requirements.txt
|-- README.md
```

## 6. Installation and Setup

### Prerequisites

Install the following:

- Python 3.12 or a compatible Python version
- Git
- Ollama
- A supported Ollama language model

### Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/enterprise-rag-knowledge-assistant.git

cd enterprise-rag-knowledge-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### Create a Virtual Environment

Windows:

```powershell
python -m venv .venv

.venv\Scripts\Activate.ps1
```

### Install Dependencies

```powershell
pip install -r requirements.txt
```

### Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
RELEVANCE_THRESHOLD=0.5
```

The Ollama model must already be available locally.

For example:

```powershell
ollama pull llama3.2
```

Verify the installed models:

```powershell
ollama list
```

## 7. Running the Application

### Start the FastAPI Backend

From the project root:

```powershell
python -m uvicorn app.main:app --reload
```

The backend runs at:

`http://127.0.0.1:8000`

### Start the Streamlit Frontend

Open another terminal:

```powershell
python -m streamlit run frontend/streamlit_app.py
```

The frontend runs at:

`http://localhost:8501`

## 8. API Endpoints

| Method | Endpoint            | Description                                    |
| ------ | ------------------- | ---------------------------------------------- |
| GET    | `/health`           | Application health information                 |
| POST   | `/chat`             | General LLM conversation                       |
| POST   | `/documents/upload` | Upload and ingest a PDF                        |
| POST   | `/rag/ask`          | Ask questions using retrieved document context |

### Example RAG Request

```json
{
  "question": "What is supervised machine learning?",
  "top_k": 3
}
```

### Example Response

Illustrative response format:

```json
{
  "question": "What is supervised machine learning?",
  "answer": "Supervised learning uses labeled training data [1].",
  "sources": [
    {
      "citation_id": "1",
      "source": "sample1.pdf",
      "page": 12,
      "distance": 0.2145,
      "content": "Supervised learning uses labeled examples...",
      "cited": true
    }
  ],
  "total_sources": 1
}
```

Actual results depend on the indexed documents and configured model.

## 9. RAG Processing Workflow

### Document Ingestion

```text
PDF Upload
    |
    v
Text Extraction
    |
    v
Text Chunking
    |
    v
Embedding Generation
    |
    v
ChromaDB Storage
```

### Question Answering

```text
User Question
    |
    v
Question Embedding
    |
    v
Semantic Similarity Search
    |
    v
Relevance Filtering
    |
    v
Context Construction
    |
    v
Ollama LLM
    |
    v
Answer + Source References
```

## 10. Current Limitations

- Scanned or image-only PDFs are not supported without OCR.
- Relevance thresholds require evaluation and calibration.
- Citation identifiers do not guarantee that the corresponding passages support every generated claim.
- Document deduplication and deletion are not fully implemented.
- Chat history is maintained within the Streamlit session.
- Authentication and document-level access controls are not implemented.
- The application is intended for local development and experimentation.

## 11. Planned Enhancements

Future improvements include:

- OCR support for scanned PDFs
- Hybrid retrieval using BM25 and vector search
- Cross-encoder reranking
- Conversation memory
- Document management and deduplication
- Automated RAG evaluation
- API authentication
- Docker containerization
- Automated testing with Pytest
- CI/CD using GitHub Actions
- Cloud deployment

## 12. Learning Outcomes

This project demonstrates practical experience with:

- Retrieval-Augmented Generation
- LLM application development
- Semantic embeddings
- Vector databases
- Document ingestion pipelines
- Prompt engineering
- REST API development
- Local LLM integration
- Source citation handling
- Full-stack AI application development

## 13. Project Status

**Development in progress**

PDF upload, text extraction, chunking, embedding generation, and ChromaDB storage have been tested successfully.

RAG answer generation, citation accuracy, and retrieval quality require further end-to-end testing and evaluation.

## 14. License

No license has been selected yet.

A license file can be added before public distribution.
