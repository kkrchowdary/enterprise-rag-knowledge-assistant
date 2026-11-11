import unittest

from app.services.rag_service import RAGService


class DummyCollection:
    name = "enterprise_documents"
    metadata = {"hnsw:space": "cosine"}

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
        return "This is the answer."


class RAGServiceLogicTests(unittest.TestCase):
    def make_service(self, distances):
        service = RAGService.__new__(RAGService)
        service.vector_store = DummyStore(distances)
        service.llm = DummyLLM()
        service.relevance_threshold = 0.5
        return service

    def test_accepts_relevant_result(self):
        response = self.make_service([0.2, 0.8]).ask("What is tokenization?", top_k=5)

        self.assertTrue(response["answer"]) 
        self.assertEqual(response["total_sources"], 1)
        self.assertEqual(response["sources"][0]["source"], "doc1.pdf")

    def test_rejects_everything_with_clear_message(self):
        response = self.make_service([0.8, 0.9]).ask("What is tokenization?", top_k=5)

        self.assertIn("sufficiently relevant information", response["answer"])
        self.assertEqual(response["sources"], [])
        self.assertEqual(response["total_sources"], 0)


if __name__ == "__main__":
    unittest.main()
