import os
import re
import logging
from dotenv import load_dotenv


from app.retrieval.vector_store import VectorStore
from app.generation.llm import OllamaLLM

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

load_dotenv()


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

        self.vector_store = VectorStore()
        self.llm = OllamaLLM()

        # ChromaDB cosine distance is in the range [0, 2].
        # For normalized embeddings, cosine similarity is:
        #   similarity = 1.0 - distance
        # A production threshold is expressed as a 0.0..1.0 score,
        # where 0.70 means a 70% match.
        self.relevance_threshold = float(
            os.getenv("RELEVANCE_THRESHOLD", "0.70")
        )

    @staticmethod
    def cosine_similarity_from_distance(distance: float) -> float:
        """Convert Chroma cosine distance into a 0..1 similarity score.

        Chroma returns cosine distance for normalized embeddings, so the best
        approximation is 1.0 - distance. This yields a score that maps directly
        to a human-readable match percentage after multiplying by 100.
        """
        distance_value = max(0.0, min(2.0, float(distance)))
        return max(0.0, min(1.0, 1.0 - distance_value))

    def ask(
        self,
        question: str,
        top_k: int = 3
    ) -> dict:

        # ----------------------------------------
        # STEP 1: Validate question
        # ----------------------------------------

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10.")

        question = question.strip()
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
            raise ValueError(
                "No documents indexed. Upload a PDF first."
            )

        # ----------------------------------------
        # STEP 3: Semantic retrieval
        # ----------------------------------------

        results = self.vector_store.search(
            query=question,
            top_k=min(top_k, document_count)
        )

        logger.info("ChromaDB search completed")
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
            logger.info("Text: %s", (document or "")[:500])

        # ----------------------------------------
        # STEP 4: Relevance filtering
        # ----------------------------------------

        filtered_results = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            if not document or distance is None:
                continue

            distance_value = float(distance)
            similarity_score = self.cosine_similarity_from_distance(
                distance_value
            )
            similarity_percent = similarity_score * 100.0

            item = {
                "document": document,
                "metadata": metadata or {},
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

        final_results = filtered_results

        logger.info(
            "Chunks remaining after filtering: %d",
            len(final_results)
        )

        # ----------------------------------------
        # STEP 5: Handle irrelevant questions
        # ----------------------------------------

        if not final_results:
            logger.warning(
                "No relevant chunks passed the threshold. "
                "Ollama generation skipped."
            )
            return {
                "question": question,
                "answer": (
                    "I couldn't find sufficiently relevant information "
                    "in the uploaded documents."
                ),
                "sources": [],
                "total_sources": 0
            }

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
                "distance": round(distance, 4),
                "similarity_percent": round(similarity_percent, 2),
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
        logger.info("Context preview: %s", context[:1000])
        try:
            answer = self.llm.generate_answer(
                question=question,
                context=context
            )
            logger.info("LLM response type: %s", type(answer).__name__)
            logger.info("LLM response: %r", answer)
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

        for source in sources:

            source["cited"] = (
                source["citation_id"] in cited_ids
            )

        # ----------------------------------------
        # STEP 9: Return final response
        # ----------------------------------------

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "total_sources": len(sources)
        }


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
        logger.info("response:",response)
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