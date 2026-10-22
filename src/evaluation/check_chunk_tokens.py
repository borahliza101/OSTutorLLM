from pathlib import Path
from transformers import AutoTokenizer
import statistics


# ==============================
# CONFIGURATION
# ==============================

CHUNKS_DIR = Path("data/processed/chunks")
MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ==============================
# LOAD TOKENIZER
# ==============================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded!")


# ==============================
# READ CHUNKS
# ==============================

chunk_files = sorted(CHUNKS_DIR.glob("*__chunks.txt"))

token_counts = []
chunk_info = []

print(f"Found {len(chunk_files)} chunk files.")


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

        token_counts.append(token_count)

        chunk_info.append({
            "source": chunk_file.name,
            "chunk_id": chunk_id,
            "tokens": token_count
        })


# ==============================
# CALCULATE STATISTICS
# ==============================

token_counts_sorted = sorted(token_counts)

minimum = min(token_counts)
maximum = max(token_counts)
average = statistics.mean(token_counts)
median = statistics.median(token_counts)

p90 = token_counts_sorted[
    int(0.90 * len(token_counts_sorted))
]

p95 = token_counts_sorted[
    int(0.95 * len(token_counts_sorted))
]


# ==============================
# DISPLAY RESULTS
# ==============================

print()
print("================================")
print("CHUNK TOKEN ANALYSIS")
print("================================")

print("Total chunks:", len(token_counts))

print()
print("Minimum tokens:", minimum)
print("Maximum tokens:", maximum)
print("Average tokens:", round(average, 2))
print("Median tokens:", median)
print("90th percentile:", p90)
print("95th percentile:", p95)


# ==============================
# SHOW EXTREME CHUNKS
# ==============================

largest = sorted(
    chunk_info,
    key=lambda x: x["tokens"],
    reverse=True
)[:5]

smallest = sorted(
    chunk_info,
    key=lambda x: x["tokens"]
)[:5]


print()
print("================================")
print("5 LARGEST CHUNKS")
print("================================")

for item in largest:
    print(
        f"{item['source']} | "
        f"Chunk {item['chunk_id']} | "
        f"{item['tokens']} tokens"
    )


print()
print("================================")
print("5 SMALLEST CHUNKS")
print("================================")

for item in smallest:
    print(
        f"{item['source']} | "
        f"Chunk {item['chunk_id']} | "
        f"{item['tokens']} tokens"
    )