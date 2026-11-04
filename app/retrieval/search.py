from app.retrieval.vector_store import VectorStore

def search_documents(question: str):

    vector_store = VectorStore()

    results = vector_store.search(
        query = question,
        top_k= 3
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not documents:
        print("No retrieval documents found.")

        return
    print("\n earch results ")
    print("=" *60)

    for index, (document, metadata,distance) in enumerate(
        zip(documents, metadatas,distances),
        start=1
    ):

        print(f"\n Result {index}")

        print("source:", metadata["source"])

        print("page:", metadata["page"])

        print("cosine distance:", round(distance, 4))

        print("\n Retrieved text:")

        print(document[:600])

        print("-" * 60)

if __name__ == "__main__":

    question = input("Enter your question:").strip()

    if question:

        search_documents(question)  

    else:
        print("please enter a question.")      