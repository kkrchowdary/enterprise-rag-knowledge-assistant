import os
import unittest
from typing import Any
from unittest.mock import Mock, patch

from app.generation.llm import OllamaLLM
from app.generation.prompts import build_rag_prompt
from app.services.rag_service import RAGService


class DummyCollection:
    name = "enterprise_documents"

    def __init__(self):
        self.metadata = {"hnsw:space": "cosine"}

    def count(self):
        return 2


class DummyStore:
    def __init__(self, distances):
        self.collection = DummyCollection()
        self._distances = distances

    def search(self, query, top_k):
        return {
            "documents": [["Alpha text", "Beta text"]],
            "metadatas": [[{"source": "doc1.pdf", "page": 1}, {"source": "doc2.pdf", "page": 2}]],
            "distances": [self._distances],
        }


class DummyLLM:
    def generate_answer(self, question, context):
        return "This is the answer [1]."


class RAGServiceLogicTests(unittest.TestCase):
    def test_threshold_configuration_rejects_invalid_values(self):
        for threshold in ("nan", "inf", "-0.1", "1.1", "invalid"):
            with (
                self.subTest(threshold=threshold),
                patch.dict(os.environ, {"RELEVANCE_THRESHOLD": threshold}),
                patch("app.services.rag_service.OllamaLLM"),
                self.assertRaises(ValueError),
            ):
                RAGService()

    def test_ollama_honors_existing_environment_settings(self):
        with (
            patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://example.test:11434", "OLLAMA_MODEL": "llama3.2:3b"}),
            patch("app.generation.llm.Client") as client,
        ):
            llm = OllamaLLM()
        client.assert_called_once_with(host="http://example.test:11434")
        self.assertEqual(llm.model, "llama3.2:3b")

    def test_prompt_construction_is_pure(self):
        with patch("ollama.chat") as chat:
            prompt = build_rag_prompt("What is alpha?", "[1] Alpha is a letter.")
        chat.assert_not_called()
        self.assertIn("What is alpha?", prompt)
        self.assertIn("[1] Alpha is a letter.", prompt)

    def test_generation_calls_ollama_once_with_original_context(self):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm.client = Mock()
        llm.model = "llama3.2"
        llm.client.chat.return_value.message.content = "Alpha is a letter [1]."
        self.assertEqual(
            llm.generate_answer("What is alpha?", "[1] Alpha is a letter."),
            "Alpha is a letter [1].",
        )
        llm.client.chat.assert_called_once()
        self.assertFalse(llm.client.chat.call_args.kwargs["stream"])
        prompt = llm.client.chat.call_args.kwargs["messages"][1]["content"]
        self.assertIn("[1] Alpha is a letter.", prompt)

    def make_service(self, distances) -> Any:
        service = RAGService.__new__(RAGService)
        service.vector_store = Mock(wraps=DummyStore(distances))
        service.llm = Mock(wraps=DummyLLM())
        service.relevance_threshold = 0.5
        return service

    def test_accepts_relevant_result(self):
        response = self.make_service([0.2, 0.8]).ask("What is tokenization?", top_k=5)

        self.assertTrue(response["answer"]) 
        self.assertEqual(response["total_sources"], 1)
        self.assertEqual(response["sources"][0]["source"], "doc1.pdf")

    def test_rejects_everything_with_clear_message(self):
        service = self.make_service([0.8, 0.9])
        refusal = "I could not find this information in the uploaded documents."
        service.llm.generate_answer.return_value = refusal
        response = service.ask("What is tokenization?", top_k=5)

        self.assertEqual(response["answer"], refusal)
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["total_sources"], 0)
        service.llm.generate_answer.assert_called_once_with(question="What is tokenization?", context="")

    def test_conversation_uses_same_generation_path_without_classification(self):
        for question in ("hi", " Hello! ", "HEY there", "Good morning.", "Howdy, friend", "Bonjour"):
            with self.subTest(question=question):
                service = self.make_service([0.1, 0.2])
                service.llm.generate_answer.return_value = "Model-generated greeting."
                response = service.ask(question)
                self.assertEqual(response["answer"], "Model-generated greeting.")
                self.assertEqual(response["sources"], [])
                self.assertEqual(response["total_sources"], 0)
                service.llm.generate_answer.assert_called_once()
                self.assertEqual(service.llm.generate_answer.call_args.kwargs["question"], question.strip())
        self.assertFalse(hasattr(OllamaLLM, "generate_greeting"))

    def test_question_starting_with_greeting_still_retrieves(self):
        service = self.make_service([0.1, 0.2])
        response = service.ask("Hello, what is tokenization?")
        service.vector_store.search.assert_called_once()
        self.assertEqual(response["total_sources"], 1)

    def test_rejects_invalid_top_k(self):
        for top_k in (True, False, 0, 11, 1.5, "3"):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                self.make_service([0.1, 0.2]).ask("alpha", top_k)

    def test_rejects_empty_question(self):
        for question in ("", "  ", None, 42):
            with self.subTest(question=question), self.assertRaises(ValueError):
                self.make_service([0.1, 0.2]).ask(question)

    def test_cosine_similarity_preserves_negative_values(self):
        for distance, expected in ((0, 1), (0.25, 0.75), (1, 0), (1.5, -0.5), (2, -1)):
            with self.subTest(distance=distance):
                self.assertEqual(RAGService.cosine_similarity_from_distance(distance), expected)

    def test_invalid_distances_never_become_relevant(self):
        for distance in (float("nan"), float("inf"), -1, 3, None, "invalid"):
            with self.subTest(distance=distance):
                service = self.make_service([distance, 0.9])
                self.assertEqual(service.ask("alpha")["sources"], [])
                service.llm.generate_answer.assert_called_once_with(question="alpha", context="")

    def test_default_top_k_is_three(self):
        service = self.make_service([0.1, 0.2])
        service.vector_store.collection.count.return_value = 20
        service.ask("alpha")
        service.vector_store.search.assert_called_once_with(query="alpha", top_k=3)

    def test_sorts_before_top_k_and_keeps_metadata_aligned(self):
        response = self.make_service([0.4, 0.1]).ask("alpha", top_k=1)
        self.assertEqual(response["total_sources"], 1)
        self.assertEqual(response["sources"][0]["source"], "doc2.pdf")
        self.assertEqual(response["sources"][0]["similarity"], 0.9)
        self.assertTrue(response["sources"][0]["cited"])

    def test_threshold_boundary_is_inclusive(self):
        response = self.make_service([0.5, 0.50001]).ask("alpha")
        self.assertEqual(response["total_sources"], 1)
        self.assertEqual(response["sources"][0]["similarity"], 0.5)

    def test_source_keeps_raw_score_precision(self):
        distance = 0.123456789
        response = self.make_service([distance, 0.9]).ask("alpha")
        source = response["sources"][0]
        self.assertEqual(source["distance"], distance)
        self.assertEqual(source["similarity"], 1 - distance)
        self.assertEqual(source["score_type"], "cosine_similarity")
        self.assertEqual(source["retrieval_rank"], 1)
        self.assertEqual(source["similarity_percent"], round((1 - distance) * 100, 2))

    def test_duplicate_passages_are_not_repeated(self):
        service = self.make_service([0.2, 0.1])
        service.vector_store.search = Mock(return_value={
            "documents": [["Alpha text", "Alpha  text"]],
            "metadatas": [[{"source": "a.pdf"}, {"source": "b.pdf"}]],
            "distances": [[0.2, 0.1]],
        })
        response = service.ask("alpha")
        self.assertEqual(response["total_sources"], 1)
        self.assertEqual(response["sources"][0]["source"], "b.pdf")

    def test_missing_metadata_does_not_drop_evidence(self):
        service = self.make_service([0.1, 0.2])
        service.vector_store.search = Mock(return_value={
            "documents": [["Alpha text"]], "distances": [[0.1]],
        })
        response = service.ask("alpha")
        self.assertEqual(response["sources"][0]["source"], "Unknown")

    def test_invalid_answers_return_no_sources(self):
        for answer in ("", "Wrong source [99]", "Mixed [1] [99]", None,
                       "I could not find this information in the uploaded documents."):
            with self.subTest(answer=answer):
                service = self.make_service([0.1, 0.2])
                service.llm.generate_answer.return_value = answer
                response = service.ask("alpha")
                self.assertEqual(response["sources"], [])
                self.assertIn("could not find", response["answer"])

    def test_empty_collection_still_uses_model_for_conversation_and_refusal(self):
        for question, answer in (
            ("hello", "Hello from Ollama!"),
            ("What is the policy?", "I could not find this information in the uploaded documents."),
        ):
            with self.subTest(question=question):
                service = self.make_service([0.1, 0.2])
                service.vector_store.collection.count.return_value = 0
                service.llm.generate_answer.return_value = answer
                response = service.ask(question)
                self.assertEqual(response["answer"], answer)
                self.assertEqual(response["sources"], [])
                service.vector_store.search.assert_not_called()
                service.llm.generate_answer.assert_called_once_with(question=question, context="")

    def test_debug_logs_include_full_retrieval_context_metadata_and_response(self):
        service = self.make_service([0.1, 0.2])
        document = "Evidence " * 200 + "PASSAGE_END"
        service.vector_store.search.return_value = {
            "ids": [["chunk-123"]],
            "documents": [[document]],
            "metadatas": [[{"source": "debug.pdf", "page": 7, "custom": "metadata-value"}]],
            "distances": [[0.1]],
        }
        with self.assertLogs("app.services.rag_service", level="DEBUG") as captured:
            response = service.ask("debug query")
        output = "\n".join(captured.output)
        for expected in ("debug query", "chunk-123", "metadata-value", "PASSAGE_END",
                         "RAG full generation context", "RAG ranked and filtered candidates",
                         "RAG citation validation", repr(response), "RAG request completed"):
            self.assertIn(expected, output)

    def test_debug_logs_include_greeting_and_refusal_responses(self):
        for answer in ("I could not find this information in the uploaded documents.", "Model greeting"):
            with self.subTest(answer=answer):
                service = self.make_service([0.9, 0.9])
                service.llm.generate_answer.return_value = answer
                with self.assertLogs("app.services.rag_service", level="DEBUG") as captured:
                    response = service.ask("query")
                self.assertIn(repr(response), "\n".join(captured.output))

    def test_generation_failure_logs_traceback(self):
        service = self.make_service([0.1, 0.2])
        service.llm.generate_answer.side_effect = ConnectionError("offline")
        with (
            self.assertLogs("app.services.rag_service", level="ERROR") as captured,
            self.assertRaises(ConnectionError),
        ):
            service.ask("query")
        self.assertIn("Traceback", "\n".join(captured.output))

    def test_ollama_debug_logs_include_request_and_response_metadata(self):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm.model = "llama3.2"
        llm.client = Mock()
        llm.client.chat.return_value.message.content = "Supported answer [1]."
        llm.client.chat.return_value.model_dump.return_value = {
            "model": "llama3.2", "eval_count": 42, "total_duration": 123456,
            "message": {"content": "Supported answer [1]."},
        }
        context = "Full context " * 200 + "CONTEXT_END"
        with self.assertLogs("app.generation.llm", level="DEBUG") as captured:
            llm.generate_answer("original question", context)
        output = "\n".join(captured.output)
        for expected in ("original question", "CONTEXT_END", "eval_count", "42",
                         "total_duration", "Supported answer [1].", "completed in"):
            self.assertIn(expected, output)

    def test_ollama_failure_logs_traceback(self):
        llm = OllamaLLM.__new__(OllamaLLM)
        llm.model = "llama3.2"
        llm.client = Mock()
        llm.client.chat.side_effect = ConnectionError("offline")
        with (
            self.assertLogs("app.generation.llm", level="ERROR") as captured,
            self.assertRaises(ConnectionError),
        ):
            llm.generate_answer("hello", "")
        self.assertIn("Traceback", "\n".join(captured.output))


if __name__ == "__main__":
    unittest.main()
