import unittest
from unittest.mock import Mock, patch

from app.retrieval.vector_store import VectorStore


class VectorStoreMetricTests(unittest.TestCase):
    def test_creates_cosine_collection_and_verifies_actual_configuration(self):
        collection = Mock()
        collection.configuration = {"hnsw": {"space": "cosine"}}
        with (
            patch("app.retrieval.vector_store.chromadb.PersistentClient") as client,
            patch("app.retrieval.vector_store.EmbeddingModel") as embeddings,
            patch("app.retrieval.vector_store.Path.mkdir"),
        ):
            client.return_value.get_or_create_collection.return_value = collection
            store = VectorStore()
            client.return_value.get_or_create_collection.assert_called_once_with(
                name="enterprise_documents",
                configuration={"hnsw": {"space": "cosine"}},
            )
            embeddings.assert_called_once()
        self.assertIs(store.collection, collection)

    def test_rejects_existing_non_cosine_metric_without_deleting_data(self):
        for metric in ("l2", "ip", None):
            with self.subTest(metric=metric):
                collection = Mock()
                collection.configuration = {"hnsw": {"space": metric}}
                with (
                    patch("app.retrieval.vector_store.chromadb.PersistentClient") as client,
                    patch("app.retrieval.vector_store.EmbeddingModel") as embeddings,
                    patch("app.retrieval.vector_store.Path.mkdir"),
                ):
                    client.return_value.get_or_create_collection.return_value = collection
                    with self.assertRaisesRegex(RuntimeError, "cosine scoring requires"):
                        VectorStore()
                    embeddings.assert_not_called()
                    client.return_value.delete_collection.assert_not_called()
                    collection.modify.assert_not_called()


if __name__ == "__main__":
    unittest.main()