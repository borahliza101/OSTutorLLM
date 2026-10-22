from pathlib import Path
from transformers import AutoTokenizer


CHUNKS_DIR = Path("data/processed/chunks")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

TOKEN_THRESHOLD = 50


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Tokenizer loaded!")

short_chunks = []

chunk_files = sorted(CHUNKS_DIR.glob("*__chunks.txt"))

for chunk_file in chunk_files:

    text = chunk_file.read_text(encoding="utf-8")

    parts = text.split("--- CHUNK ")

    for part in parts[1:]:

        lines = part.split("\n", 1)

        chunk_id = int(
            lines[0].strip().replace("---", "")
        )

        chunk_text = lines[1].strip()

        if not chunk_text:
            continue

        token_ids = tokenizer.encode(
            chunk_text,
            add_special_tokens=True
        )

        token_count = len(token_ids)

        if token_count < TOKEN_THRESHOLD:

            short_chunks.append({
                "source": chunk_file.name,
                "chunk_id": chunk_id,
                "tokens": token_count,
                "text": chunk_text
            })


print()
print("================================")
print("SHORT CHUNK INSPECTION")
print("================================")

print("Threshold:", TOKEN_THRESHOLD, "tokens")
print("Short chunks found:", len(short_chunks))

for item in short_chunks:

    print()
    print("--------------------------------")
    print("Source:", item["source"])
    print("Chunk ID:", item["chunk_id"])
    print("Tokens:", item["tokens"])
    print("--------------------------------")
    print(item["text"])