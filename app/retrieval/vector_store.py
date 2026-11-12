import logging
from pathlib import Path
from time import perf_counter

import chromadb

from app.retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


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
            configuration={"hnsw": {"space": "cosine"}}
        )

        configuration = self.collection.configuration
        metric = (configuration.get("hnsw") or {}).get("space")
        if metric != "cosine":
            raise RuntimeError(
                f"Collection enterprise_documents uses distance metric {metric!r}; "
                "cosine scoring requires 'cosine'. Create a new cosine collection "
                "and re-ingest documents. Existing data has not been deleted."
            )
        logger.info("Verified collection distance metric: %s", metric)
        logger.debug("Collection configuration: %r", configuration)

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

        started = perf_counter()
        logger.info("Vector search started: collection=%s top_k=%d", self.collection.name, top_k)
        logger.debug("Vector search query: %r collection_metadata=%r", query, self.collection.metadata)
        query_embedding = self.embedding_model.generate_embeddings(
            [query]
        )[0]
        logger.info("Query embedding generated: dimensions=%d", len(query_embedding))
        logger.debug("Full query embedding: %r", query_embedding)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results= top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        logger.info("Vector search completed in %.3fs", perf_counter() - started)
        logger.debug("Chroma full query response: %r", results)
        return results