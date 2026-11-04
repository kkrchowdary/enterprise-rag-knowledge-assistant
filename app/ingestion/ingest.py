from pathlib import Path
import hashlib

from app.ingestion.pdf_loader import extract_pdf_pages
from app.ingestion.chunker import chunk_text
from app.retrieval.vector_store import VectorStore


def ingest_pdf(
    file_path: str,
    original_filename: str | None = None
) -> dict:

    print("\nStarting PDF ingestion...")

    # STEP 1: Extract PDF text
    pages = extract_pdf_pages(file_path)

    texts = []
    metadatas = []
    ids = []

    # Preserve original document filename
    document_name = (
        Path(original_filename).name
        if original_filename
        else Path(file_path).name
    )

    # STEP 2: Chunk extracted pages
    for page in pages:

        chunks = chunk_text(
            page["text"],
            chunk_size=800,
            overlap=150
        )

        for index, chunk in enumerate(chunks):

            # Generate deterministic chunk ID
            chunk_id = hashlib.sha256(
                (
                    f"{document_name}:"
                    f"{page['page']}:"
                    f"{index}:{chunk}"
                ).encode("utf-8")
            ).hexdigest()

            texts.append(chunk)

            metadatas.append({
                "source": document_name,
                "page": page["page"],
                "chunk_index": index
            })

            ids.append(chunk_id)

    # STEP 3: Validate extracted chunks
    if not texts:

        raise ValueError(
            "No usable text chunks generated from PDF."
        )

    # STEP 4: Initialize vector database
    vector_store = VectorStore()

    # STEP 5: Store embeddings and metadata
    vector_store.add_documents(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

    # STEP 6: Print ingestion statistics
    print("\nPDF ingestion completed successfully!")

    print("Total pages:", len(pages))
    print("Total chunks:", len(texts))

    # STEP 7: Return API response
    return {
        "status": "success",
        "filename": document_name,
        "pages": len(pages),
        "chunks": len(texts)
    }


if __name__ == "__main__":

    result = ingest_pdf(
        "data/documents/sample1.pdf"
    )

    print(result)