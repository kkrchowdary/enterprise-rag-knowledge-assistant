def chunk_text( 
    text: str,
    chunk_size : int = 800,
    overlap: int = 150    
) -> list[str]:

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap <0 or overlap >= chunk_size:
        raise ValueError("Invalid overlap")

    chunks = []
    start= 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

if __name__ == "__main__":
    sample = "RAG Combines retrieval and generation." * 50

    chunks = chunk_text(sample)

    print(f"total chunks: {len(chunks)}")

    for index, chunk in enumerate(chunks,1):
        print(f"Chunk {index}: {len(chunk)} characters ")
       