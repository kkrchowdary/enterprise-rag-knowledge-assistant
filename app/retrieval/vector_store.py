import logging
from pathlib import Path
from time import perf_counter

import chromadb
from chromadb.api.types import Metadata

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
            metadatas: list[Metadata],
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

        if type(top_k) is not int or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        started = perf_counter()
        logger.info("Vector search started: collection=%s top_k=%d", self.collection.name, top_k)
        logger.debug("Vector search query: %r collection_metadata=%r", query, self.collection.metadata)
        document_count = self.collection.count()
        if document_count == 0:
            logger.info("Vector search skipped: collection is empty")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_embedding = self.embedding_model.generate_embeddings(
            [query]
        )[0]
        logger.info("Query embedding generated: dimensions=%d", len(query_embedding))
        logger.debug("Full query embedding: %r", query_embedding)

        candidate_count = min(top_k, document_count)
        while True:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=candidate_count,
                include=["documents", "metadatas", "distances"]
            )
            logger.debug("Chroma full candidate response: %r", results)
            documents = (results.get("documents") or [[]])[0]
            selected = []
            seen_passages = set()
            for index, document in enumerate(documents):
                if not isinstance(document, str):
                    continue
                passage_key = " ".join(document.split())
                if not passage_key or passage_key in seen_passages:
                    continue
                seen_passages.add(passage_key)
                selected.append(index)
                if len(selected) == top_k:
                    break

            if len(selected) >= top_k or candidate_count >= document_count:
                break
            # Duplicates must not consume the final evidence budget.
            candidate_count = min(candidate_count * 2, document_count)
            logger.info("Expanding vector candidates: n_results=%d", candidate_count)

        metadatas = results.get("metadatas")
        distances = results.get("distances")
        results = {
            **results,
            "ids": [[results["ids"][0][index] for index in selected]],
            "documents": [[documents[index] for index in selected]],
            "metadatas": [[metadatas[0][index] for index in selected]] if metadatas is not None else None,
            "distances": [[distances[0][index] for index in selected]] if distances is not None else None,
        }

        logger.info("Vector search completed in %.3fs", perf_counter() - started)
        logger.debug("Chroma full query response: %r", results)
        return results