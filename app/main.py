import os
import uuid
import logging
from pathlib import Path

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File
)

from pydantic import BaseModel, Field
from ollama import Client, ResponseError

from app.services.rag_service import RAGService
from app.ingestion.ingest import ingest_pdf


# =====================================================
# 1. LOAD ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "nemotron-3.5-super:latest"
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434"
)

UPLOAD_DIRECTORY = Path("data/documents")

UPLOAD_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)

MAX_UPLOAD_SIZE = 20 * 1024 * 1024  # 20 MB


# =====================================================
# 2. LOGGING CONFIGURATION
# =====================================================

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


# =====================================================
# 3. INITIALIZE FASTAPI
# =====================================================

app = FastAPI(
    title="Enterprise RAG Knowledge Assistant",
    version="0.4.0",
    description=(
        "Enterprise RAG application using FastAPI, "
        "Ollama, ChromaDB and Sentence Transformers"
    )
)


# =====================================================
# 4. INITIALIZE OLLAMA CLIENT
# =====================================================

ollama_client = Client(
    host=OLLAMA_BASE_URL
)


# =====================================================
# 5. REQUEST MODELS
# =====================================================

class ChatRequest(BaseModel):

    question: str = Field(
        min_length=1,
        description="Question for the LLM"
    )


class RAGRequest(BaseModel):

    question: str = Field(
        min_length=1,
        description="Question about uploaded documents"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of document chunks to retrieve"
    )


# =====================================================
# 6. HEALTH CHECK ENDPOINT
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "application": "Enterprise RAG Knowledge Assistant",
        "version": "0.4.0",
        "ollama_model": OLLAMA_MODEL
    }


# =====================================================
# 7. GENERAL LLM CHAT ENDPOINT
# =====================================================

@app.post("/chat")
def chat(request: ChatRequest):

    try:

        logger.info("Processing general chat request")

        response = ollama_client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": request.question
                }
            ]
        )

        return {
            "question": request.question,
            "answer": response.message.content,
            "model": OLLAMA_MODEL
        }

    except ResponseError as error:

        logger.exception("Ollama model error")

        raise HTTPException(
            status_code=502,
            detail=f"Ollama error: {error.error}"
        )

    except Exception:

        logger.exception("LLM service unavailable")

        raise HTTPException(
            status_code=503,
            detail="LLM service unavailable"
        )


# =====================================================
# 8. RAG QUESTION ANSWERING ENDPOINT
# =====================================================

@app.post("/rag/ask")
def ask_rag(request: RAGRequest):

    try:

        rag_service = RAGService()

        response = rag_service.ask(
            question=request.question,
            top_k=request.top_k
        )

        return response

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception("RAG processing failed")

        raise HTTPException(
            status_code=500,
            detail="RAG processing failed"
        )

# =====================================================
# 9. DOCUMENT UPLOAD ENDPOINT
# =====================================================

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):

    # Validate filename
    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename is required"
        )

    # Validate file extension
    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Generate unique storage filename
    stored_filename = f"{uuid.uuid4().hex}.pdf"

    file_path = UPLOAD_DIRECTORY / stored_filename

    try:

        logger.info(
            "Uploading document: %s",
            file.filename
        )

        # Read and save file in bounded chunks
        total_size = 0

        with file_path.open("wb") as buffer:

            while True:

                chunk = file.file.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > MAX_UPLOAD_SIZE:

                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 20 MB upload limit"
                    )

                buffer.write(chunk)

        # Reject empty files
        if total_size == 0:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # Basic PDF signature validation
        with file_path.open("rb") as saved_file:

            if saved_file.read(5) != b"%PDF-":

                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file"
                )

        logger.info("Starting document ingestion")

        # Extract text, chunk, embed and store in ChromaDB
        result = ingest_pdf(str(file_path),original_filename= Path(file.filename).name)

        if not isinstance(result, dict):

            raise RuntimeError(
                "ingest_pdf() must return a dictionary"
            )

        # Display original filename in API response
        result["filename"] = file.filename

        logger.info(
            "Document ingestion completed: %s",
            file.filename
        )

        return result

    except HTTPException:

        file_path.unlink(missing_ok=True)

        raise

    except ValueError as error:

        logger.warning(
            "Document validation failed: %s",
            error
        )

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception:

        logger.exception("Document processing failed")

        file_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Document processing failed"
        )

    finally:

        file.file.close()