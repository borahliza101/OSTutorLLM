from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------
# Paths
# -----------------------------
CHUNKS_DIR = Path("data/processed/chunks")
CHROMA_DIR = "chroma_db"


# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Embedding model loaded!")


# -----------------------------
# Read chunks
# -----------------------------
chunks = []
metadatas = []
ids = []

chunk_files = sorted(CHUNKS_DIR.glob("*__chunks.txt"))

print(f"Found {len(chunk_files)} chunk files.")


for chunk_file in chunk_files:

    text = chunk_file.read_text(encoding="utf-8")

    # Split the file into individual chunks
    parts = text.split("--- CHUNK ")

    for part in parts[1:]:

        lines = part.split("\n", 1)

        chunk_id = int(lines[0].strip().replace("---", ""))

        chunk_text = lines[1].strip()

        if not chunk_text:
            continue

        chunks.append(chunk_text)

        metadatas.append({
            "source": chunk_file.name.replace("__chunks.txt", ".txt"),
            "chunk_id": chunk_id
        })

        ids.append(
            f"{chunk_file.stem}_{chunk_id}"
        )


print(f"Total chunks found: {len(chunks)}")


# -----------------------------
# Create embeddings
# -----------------------------
print("Creating embeddings...")

embeddings = model.encode(
    chunks,
    batch_size=16,
    show_progress_bar=True
)

print("Embeddings created!")


# -----------------------------
# Create ChromaDB
# -----------------------------
print("Creating ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_or_create_collection(
    name="os_tutor"
)


# -----------------------------
# Store everything
# -----------------------------
print("Storing chunks in ChromaDB...")

collection.upsert(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist(),
    metadatas=metadatas
)


# -----------------------------
# Verify
# -----------------------------
print()
print("================================")
print("Embedding pipeline completed!")
print("================================")
print("Chunks stored:", collection.count())
print("ChromaDB location:", CHROMA_DIR)