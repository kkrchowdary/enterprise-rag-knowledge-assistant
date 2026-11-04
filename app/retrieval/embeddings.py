from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self):
        print("Loading Embedding model...")

        self.model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding model loaded successfullyy...")

    def generate_embeddings(self,texts: list[str]):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy= True,
            normalize_embeddings= True
        )

        return embeddings.tolist()

if __name__ == "__main__":

    embedding_model = EmbeddingModel()

    texts = [
        "what is artificial intelligence?",
        "Machine Learning is a subset of AI.",
        "Python is used for data engineering."
    ]

    embeddings = embedding_model .generate_embeddings(texts)

    print("Total Embeddings:", len(embeddings))

    print("Vector dimensions:", len(embeddings[0]))

    print("First 5 valuse:", embeddings[0][:5])           