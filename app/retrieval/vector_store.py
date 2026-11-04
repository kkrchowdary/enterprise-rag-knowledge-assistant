from pathlib import Path

import chromadb

from app.retrieval.embeddings import EmbeddingModel

class VectorStore:

    def __init__(self):

        print ("initializing the ChromaDB...")

        db_path = Path("data/chroma_db")
        db_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path= str(db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name = "enterprise_documents",
            metadata={ "hnsw:space": "cosine"}
            
        )

        self.embedding_model = EmbeddingModel()

        print("ChromaDb initiazed successfully!..")

    def add_documents(
            self,
            texts: list[str],
            metadatas:list[dict],
            ids: list[str]
    ):
        if not texts:
            raise ValueError("No document chunks provided")

        if not (len(texts)== len(metadatas)== len(ids)):
            raise ValueError("Texts,metadata and IDs must match")

        embeddings = self.embedding_model.generate_embeddings(
            texts
        )

        self.collection.upsert(
            documents = texts,
            embeddings= embeddings,
            metadatas= metadatas,
            ids=ids
        )    


        print(f"stored {len(texts)} chunks in chromaDB")

    def search(self, query:str , top_k: int = 3):

        query_embedding = self.embedding_model.generate_embeddings(
            [query]
        )[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results= top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        return results 