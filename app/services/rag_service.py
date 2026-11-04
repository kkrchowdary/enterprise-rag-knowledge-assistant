import os
import re
from dotenv import load_dotenv

from app.retrieval.vector_store import VectorStore
from app.generation.llm import OllamaLLM


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

    NO_ANSWER_MESSAGE = (
        "I couldn't find this information "
        "in the uploaded documents."
    )

    def __init__(self):

        self.vector_store = VectorStore()
        self.llm = OllamaLLM()

        # Initial experimental threshold.
        # Assumes ChromaDB uses cosine distance.
        self.relevance_threshold = float(
            os.getenv("RELEVANCE_THRESHOLD", "0.5")
        )

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

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

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

            if float(distance) <= self.relevance_threshold:

                filtered_results.append({
                    "document": document,
                    "metadata": metadata or {},
                    "distance": float(distance)
                })

        # ----------------------------------------
        # STEP 5: Handle irrelevant questions
        # ----------------------------------------

        if not filtered_results:

            return {
                "question": question,
                "answer": self.NO_ANSWER_MESSAGE,
                "sources": [],
                "total_sources": 0
            }

        # ----------------------------------------
        # STEP 6: Build context and citations
        # ----------------------------------------

        context_parts = []
        sources = []

        for index, item in enumerate(
            filtered_results,
            start=1
        ):

            document = item["document"]
            metadata = item["metadata"]
            distance = item["distance"]

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
                "content": document,
                "cited": False
            })

        context = "\n\n---\n\n".join(context_parts)

        # ----------------------------------------
        # STEP 7: Generate answer using Ollama
        # ----------------------------------------

        answer = self.llm.generate_answer(
            question=question,
            context=context
        )

        if not isinstance(answer, str) or not answer.strip():
            answer = self.NO_ANSWER_MESSAGE
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