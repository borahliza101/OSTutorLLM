from pathlib import Path
from datetime import datetime
import chromadb
import requests
from sentence_transformers import SentenceTransformer


# ==============================
# CONFIGURATION
# ==============================

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "os_tutor"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

TOP_K = 3


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded!")


# ==============================
# CONNECT TO CHROMADB
# ==============================

print("Connecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Chunks in database:", collection.count())


# ==============================
# ASK QUESTION
# ==============================

question = input("\nAsk an Operating Systems question: ")


# ==============================
# RETRIEVE RELEVANT CHUNKS
# ==============================

print("\nRetrieving relevant OS material...")

question_embedding = embedding_model.encode(
    question
).tolist()

results = collection.query(
    query_embeddings=[question_embedding],
    n_results=TOP_K
)

retrieved_distances = results["distances"][0]
retrieved_metadatas = results["metadatas"][0]


retrieved_documents = results["documents"][0]


# ==============================
# DISPLAY RETRIEVED CONTEXT
# ==============================

print("\n================================")
print("RETRIEVED CONTEXT")
print("================================")

for i, document in enumerate(retrieved_documents):

    print(f"\n--- Context {i + 1} ---")

    print("Source:", retrieved_metadatas[i]["source"])
    print("Chunk ID:", retrieved_metadatas[i]["chunk_id"])
    print("Distance:", retrieved_distances[i])

    print()
    print(document)


# ==============================
# BUILD CONTEXT
# ==============================

context = "\n\n".join(
    retrieved_documents
)


# ==============================
# CREATE RAG PROMPT
# ==============================

system_prompt = """
You are an Operating Systems tutor.

Answer the student's question using the provided
Operating Systems study material.

Use the retrieved material as your primary source.

If the material does not contain enough information
to answer the question, say so instead of inventing
facts.

Explain concepts clearly and at an appropriate level
for a university student.
"""


user_prompt = f"""
Here is the retrieved Operating Systems study material:

--------------------
{context}
--------------------

Student question:

{question}

Answer the question clearly using the study material.
"""


# ==============================
# SEND TO QWEN
# ==============================

print("\nGenerating answer...")

response = requests.post(
    LLM_URL,
    json={
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "max_tokens": 300,
        "temperature": 0.2
    }
)


# ==============================
# DISPLAY ANSWER
# ==============================

response.raise_for_status()

answer = response.json()["choices"][0]["message"]["content"]

# ==============================
# SAVE EVALUATION RESULT
# ==============================

EVAL_DIR = Path("data/evaluation")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

eval_file = EVAL_DIR / f"rag_test_{timestamp}.txt"

with eval_file.open("w", encoding="utf-8") as f:

    f.write("OSTutorLLM RAG Evaluation\n")
    f.write("=========================\n\n")

    f.write(f"Question:\n{question}\n\n")

    f.write("Retrieved Contexts\n")
    f.write("------------------\n\n")

    for i, document in enumerate(retrieved_documents):

        f.write(f"Context {i + 1}\n")
        f.write(f"Source: {retrieved_metadatas[i]['source']}\n")
        f.write(f"Chunk ID: {retrieved_metadatas[i]['chunk_id']}\n")
        f.write(f"Distance: {retrieved_distances[i]}\n\n")
        f.write(document)
        f.write("\n\n")

    f.write("Generated Answer\n")
    f.write("----------------\n\n")
    f.write(answer)

print()
print("Evaluation saved to:")
print(eval_file)

print("\n================================")
print("OSTutorLLM ANSWER")
print("================================")

print(answer)