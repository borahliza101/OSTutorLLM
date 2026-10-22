from pathlib import Path
from datetime import datetime
import json

import chromadb
import requests
from sentence_transformers import SentenceTransformer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "os_tutor"

LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"

TOP_K = 3

QUESTIONS_FILE = Path(
    "data/evaluation/evaluation_questions.json"
)

OUTPUT_DIR = Path(
    "data/evaluation/results"
)

TEST_LIMIT = None


print("Loading evaluation questions...")

with QUESTIONS_FILE.open(
    "r",
    encoding="utf-8"
) as f:
    questions = json.load(f)

if TEST_LIMIT is not None:
    questions = questions[:TEST_LIMIT]

print(f"Questions to evaluate: {len(questions)}")


# ==============================
# LOAD EMBEDDING MODEL
# ==============================

print("\nLoading embedding model...")

embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5"
)

print("Embedding model loaded!")


# ==============================
# CONNECT TO CHROMADB
# ==============================

print("\nConnecting to ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

print("Chunks in database:", collection.count())


# ==============================
# CHECK LLM SERVER
# ==============================

print("\nChecking LLM server...")

health_response = requests.get(
    "http://127.0.0.1:8080/health",
    timeout=10
)

health_response.raise_for_status()

print("LLM server is ready!")


# ==============================
# EVALUATE QUESTIONS
# ==============================

results = []

for index, item in enumerate(questions, start=1):

    question_id = item["id"]
    topic = item["topic"]
    difficulty = item["difficulty"]
    question = item["question"]
    expected_key_points = item["expected_key_points"]

    print()
    print("================================")
    print(f"QUESTION {index}/{len(questions)}")
    print("================================")

    print("ID:", question_id)
    print("Topic:", topic)
    print("Difficulty:", difficulty)
    print("Question:", question)

    # --------------------------
    # RETRIEVAL
    # --------------------------

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    retrieval_results = collection.query(
        query_embeddings=[question_embedding],
        n_results=TOP_K
    )

    documents = retrieval_results["documents"][0]
    metadatas = retrieval_results["metadatas"][0]
    distances = retrieval_results["distances"][0]

    print("\nRetrieved sources:")

    for i in range(len(documents)):

        print(
            f"  {i + 1}. "
            f"{metadatas[i]['source']} "
            f"(distance={distances[i]:.4f})"
        )

    # --------------------------
    # BUILD CONTEXT
    # --------------------------

    context = "\n\n".join(documents)

    # --------------------------
    # RAG PROMPT
    # --------------------------

    system_prompt = """
You are an Operating Systems tutor for university students.

Answer the student's question using the retrieved Operating
Systems study material as your primary source.

Follow these rules:

1. Answer exactly what the question asks.
2. First determine what type of answer the question requires:
   definition, comparison, list of problems, causes, steps,
   advantages, disadvantages, or solutions.
3. If the question asks for PROBLEMS, list only problems.
   Do not list prevention methods, synchronization mechanisms,
   or solutions as problems.
4. If the question asks for SOLUTIONS or HOW TO PREVENT something,
   list the appropriate solutions or mechanisms.
5. Do not mix problems and solutions unless the question explicitly
   asks for both.
6. Use the retrieved material as evidence.
7. Do not introduce unrelated Operating Systems concepts.
8. For comparisons, clearly distinguish the two concepts.
9. For processes or mechanisms, describe the steps in the correct order.
10. Do not contradict yourself.
11. If the retrieved material is insufficient, say so instead of
    inventing information.
12. Keep the answer concise and appropriate for a university student.
"""

    user_prompt = f"""
Retrieved Operating Systems study material:

--------------------
{context}
--------------------

Student question:

{question}

Answer the question directly using the retrieved material.

Important:
- Answer only the type of information requested by the question.
- If the question asks for problems, give problems only.
- If the question asks for solutions, give solutions only.
- Do not classify a solution or synchronization mechanism as a problem.
- Do not add unrelated concepts.

Give a clear, concise university-level answer.
"""

    # --------------------------
    # GENERATE ANSWER
    # --------------------------

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
        },
        timeout=120
    )

    response.raise_for_status()

    answer = response.json()["choices"][0]["message"]["content"]

    print("\nGenerated answer:")
    print(answer)

    # --------------------------
    # SAVE RESULT
    # --------------------------

    results.append({
        "id": question_id,
        "topic": topic,
        "difficulty": difficulty,
        "question": question,
        "expected_key_points": expected_key_points,
        "retrieved_contexts": [
            {
                "source": metadatas[i]["source"],
                "chunk_id": metadatas[i]["chunk_id"],
                "distance": distances[i],
                "text": documents[i]
            }
            for i in range(len(documents))
        ],
        "generated_answer": answer
    })


# ==============================
# SAVE ALL RESULTS
# ==============================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = OUTPUT_DIR / (
    f"evaluation_{timestamp}.json"
)

with output_file.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=2,
        ensure_ascii=False
    )


print()
print("================================")
print("EVALUATION COMPLETE")
print("================================")

print("Questions evaluated:", len(results))
print("Results saved to:")
print(output_file)