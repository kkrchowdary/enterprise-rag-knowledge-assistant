import logging
import math
import os
import re

from dotenv import load_dotenv

from app.generation.llm import OllamaLLM

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)


class RAGService:
    """
    Enterprise RAG Service.

    Responsibilities:
    1. Validate user questions.
    2. Retrieve relevant document chunks.
    3. Filter irrelevant results.
    4. Build context for the LLM.
    5. Generate grounded answers.
    6. Return source citations.
    """

    # NO_ANSWER_MESSAGE = (
    #     "I couldn't find this information "
    #     "in the uploaded documents."
    # )

    def __init__(self):

        self.vector_store = None
        self.llm = OllamaLLM()

        # ChromaDB cosine distance is in the range [0, 2].
        # For normalized embeddings, cosine similarity is:
        #   similarity = 1.0 - distance
        # The threshold is expressed as a 0.0..1.0 cosine score,
        # not a probability that the passage answers the question.
        self.relevance_threshold = float(
            os.getenv("RELEVANCE_THRESHOLD", "0.5")
        )
        if not math.isfinite(self.relevance_threshold) or not 0 <= self.relevance_threshold <= 1:
            raise ValueError("RELEVANCE_THRESHOLD must be a finite number between 0 and 1.")

    @staticmethod
    def cosine_similarity_from_distance(distance: float) -> float:
        """Return cosine similarity in [-1, 1], not an answer-confidence probability."""
        distance_value = float(distance)
        if not math.isfinite(distance_value) or not -1e-6 <= distance_value <= 2 + 1e-6:
            raise ValueError("Invalid cosine distance.")
        return 1.0 - max(0.0, min(2.0, distance_value))

    def ask(
        self,
        question: str,
        top_k: int = 3
    ) -> dict:

        logger.info("RAG request started: top_k=%s", top_k)
        logger.debug("RAG user query: %r", question)
        # ----------------------------------------
        # STEP 1: Validate question
        # ----------------------------------------

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if type(top_k) is not int or not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10.")

        question = question.strip()
        if self.vector_store is None:
            from app.retrieval.vector_store import VectorStore

            self.vector_store = VectorStore()
        # ----------------------------------------
        # STEP 2: Check indexed documents
        # ----------------------------------------

        document_count = self.vector_store.collection.count()

        logger.info("=" * 60)
        logger.info("RAG DEBUGGING STARTED")
        logger.info("Question: %s", question)
        logger.info(
            "ChromaDB collection: %s",
            self.vector_store.collection.name
        )
        logger.info("Indexed chunks: %d", document_count)
        logger.info(
            "Collection metadata: %s",
            self.vector_store.collection.metadata
        )
        logger.info(
            "Relevance threshold: %.4f (%.2f%%)",
            self.relevance_threshold,
            self.relevance_threshold * 100
        )

        if document_count == 0:
            logger.info("RAG collection is empty; generating with no document context")

        # ----------------------------------------
        # STEP 3: Semantic retrieval
        # ----------------------------------------

        logger.info("RAG retrieval started: top_k=%d", min(top_k, document_count))
        try:
            results = self.vector_store.search(
                query=question,
                top_k=min(top_k, document_count)
            ) if document_count else {}
        except Exception:
            logger.exception("RAG retrieval failed")
            raise

        logger.info("ChromaDB search completed")
        logger.debug("RAG raw retrieval response: %r", results)
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        logger.info("Retrieved chunks: %d", len(documents))
        for index, (document, metadata, distance) in enumerate(
            zip(documents, metadatas, distances),
            start=1
        ):
            logger.info("-" * 60)
            logger.info("Result #%d", index)
            logger.info("Distance: %s", distance)
            logger.info("Metadata: %s", metadata)
            logger.debug("Text: %s", document)

        # ----------------------------------------
        # STEP 4: Relevance filtering
        # ----------------------------------------

        filtered_results = []

        for index, document in enumerate(documents):

            if not isinstance(document, str) or not document.strip() or index >= len(distances):
                logger.debug("RAG skipped candidate %d: empty text or missing distance", index)
                continue

            metadata = metadatas[index] if index < len(metadatas) else {}
            try:
                distance_value = float(distances[index])
                similarity_score = self.cosine_similarity_from_distance(distance_value)
            except (TypeError, ValueError, OverflowError):
                logger.debug("RAG skipped candidate %d: invalid distance %r", index, distances[index])
                continue
            similarity_percent = similarity_score * 100.0

            item = {
                "document": document,
                "metadata": metadata if isinstance(metadata, dict) else {},
                "distance": distance_value,
                "similarity": similarity_score,
                "similarity_percent": similarity_percent
            }

            if similarity_score >= self.relevance_threshold:
                logger.info(
                    "ACCEPTED | Distance: %.4f | Similarity: %.2f%% | Threshold: %.2f%%",
                    distance_value,
                    similarity_percent,
                    self.relevance_threshold * 100
                )
                filtered_results.append(item)
            else:
                logger.warning(
                    "REJECTED | Distance: %.4f | Similarity: %.2f%% | Threshold: %.2f%%",
                    distance_value,
                    similarity_percent,
                    self.relevance_threshold * 100
                )

        final_results = []
        seen_passages = set()
        for item in sorted(filtered_results, key=lambda item: item["distance"]):
            passage_key = " ".join(item["document"].split())
            if passage_key in seen_passages:
                logger.debug("RAG duplicate passage removed: metadata=%r", item["metadata"])
                continue
            seen_passages.add(passage_key)
            final_results.append(item)
            if len(final_results) == top_k:
                break
        logger.debug("RAG ranked and filtered candidates: %r", final_results)

        logger.info(
            "Chunks remaining after filtering: %d",
            len(final_results)
        )

        # ----------------------------------------
        # STEP 5: Handle irrelevant questions
        # ----------------------------------------

        if not final_results:
            logger.info(
                "No relevant chunks passed the threshold. "
                "Generating with no document context."
            )

        # ----------------------------------------
        # STEP 6: Build context and citations
        # ----------------------------------------

        context_parts = []
        sources = []

        for index, item in enumerate(
            final_results,
            start=1
        ):

            document = item["document"]
            metadata = item["metadata"]
            distance = item["distance"]
            similarity_percent = item.get("similarity_percent", 0.0)

            citation_id = str(index)

            source_name = metadata.get(
                "source",
                "Unknown"
            )

            page_number = metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"[{citation_id}]\n"
                f"Document: {source_name}\n"
                f"Page: {page_number}\n"
                f"Content:\n{document}"
            )

            sources.append({
                "citation_id": citation_id,
                "source": source_name,
                "page": page_number,
                "distance": distance,
                "similarity": item["similarity"],
                "similarity_percent": round(similarity_percent, 2),
                "score_type": "cosine_similarity",
                "retrieval_rank": index,
                "content": document,
                "cited": False
            })

        context = "\n\n---\n\n".join(context_parts)

        # ----------------------------------------
        # STEP 7: Generate answer using Ollama
        # ----------------------------------------

        logger.info("=" * 60)
        logger.info("OLLAMA GENERATION STARTED")
        logger.info("Question: %s", question)
        logger.info("Context length: %d characters", len(context))
        logger.debug("RAG full generation context: %s", context)
        try:
            answer = self.llm.generate_answer(
                question=question,
                context=context
            )
            logger.info("LLM response type: %s", type(answer).__name__)
            logger.debug("LLM response: %r", answer)
        except Exception:
            logger.exception("OLLAMA GENERATION FAILED")
            raise
        logger.info("OLLAMA GENERATION COMPLETED")

        if not isinstance(answer, str) or not answer.strip():
            answer = ""
        else:
            answer = answer.strip()

        # ----------------------------------------
        # STEP 8: Identify citations in answer
        # ----------------------------------------

        cited_ids = set(
            re.findall(r"\[(\d+)\]", answer)
        )
        valid_ids = {source["citation_id"] for source in sources}
        logger.debug("RAG citation validation: cited_ids=%r valid_ids=%r", cited_ids, valid_ids)
        if not answer or not cited_ids.issubset(valid_ids):
            logger.warning("RAG answer rejected: empty answer or invalid citation IDs")
            response = {
                "question": question,
                "answer": "I could not find this information in the uploaded documents.",
                "sources": [],
                "total_sources": 0,
            }
            logger.debug("RAG final response: %r", response)
            logger.info("RAG request completed: unsupported answer, sources=0")
            return response

        for source in sources:

            source["cited"] = (
                source["citation_id"] in cited_ids
            )
        sources = [source for source in sources if source["cited"]]

        # ----------------------------------------
        # STEP 9: Return final response
        # ----------------------------------------

        response = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "total_sources": len(sources)
        }
        logger.debug("RAG final response: %r", response)
        logger.info("RAG request completed: sources=%d", len(sources))
        return response


# --------------------------------------------------
# LOCAL TEST
# --------------------------------------------------

if __name__ == "__main__":

    rag = RAGService()

    print("\nEnterprise RAG Knowledge Assistant")
    print("=" * 60)

    question = input("\nEnter your question: ")

    try:

        response = rag.ask(
            question=question,
            top_k=3
        )
        logger.info("response: %s", response)
        print("\nAI ANSWER")
        print("=" * 60)

        print(response["answer"])

        print("\nRETRIEVED SOURCES")
        print("=" * 60)

        if not response["sources"]:

            print("No relevant sources found.")

        else:

            for source in response["sources"]:

                print(
                    f"\n[{source['citation_id']}] "
                    f"{source['source']} | "
                    f"Page {source['page']}"
                )

                print(
                    f"Distance: {source['distance']}"
                )

                print(
                    f"Cited in answer: {source['cited']}"
                )

                print(
                    f"Retrieved passage:\n"
                    f"{source['content'][:300]}"
                )

                print("-" * 60)

    except ValueError as error:

        print(f"\nValidation error: {error}")

    except Exception as error:

        print(f"\nRAG processing failed: {error}")