import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")


# -----------------------------
# Connect to ChromaDB
# -----------------------------
client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = client.get_collection(
    name="os_tutor"
)

print("Chunks in database:", collection.count())


# -----------------------------
# Student's question
# -----------------------------
questions = [
    "What is a process?",
    "What are the different states of a process?",
    "What is a thread and why is it useful?",
    "What is a race condition?",
    "What is mutual exclusion?",
    "What is a semaphore and why is it used?",
    "What is a condition variable?",
    "What is a lock and why is it needed?",
    "What is inter-process communication?",
    "What is the critical section problem?"
]


# Convert question into embedding
for question in questions:

    print()
    print("================================")
    print("QUESTION:", question)
    print("================================")

    # Convert question into embedding
    question_embedding = model.encode(question).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )

    # Display results
    for i in range(len(results["documents"][0])):

        print()
        print(f"Result {i + 1}")
        print("--------------------------------")

        print(
            "Source:",
            results["metadatas"][0][i]["source"]
        )

        print(
            "Chunk ID:",
            results["metadatas"][0][i]["chunk_id"]
        )

        print(
            "Distance:",
            results["distances"][0][i]
        )

        print()
        print(results["documents"][0][i])