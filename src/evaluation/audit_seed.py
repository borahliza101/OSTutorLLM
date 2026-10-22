import json
from collections import Counter

path = "data/fine_tuning/synchronization_seed_fixed.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print("=" * 60)
print("DATASET AUDIT")
print("=" * 60)

# 1. Basic checks
print("\nTotal records:", len(rows))

ids = [r["id"] for r in rows]
print("Unique IDs:", len(set(ids)))
print("Duplicate IDs:", len(ids) - len(set(ids)))

# 2. Required fields
required = {"id", "topic", "instruction", "input", "output"}

bad_fields = []

for i, r in enumerate(rows, 1):
    missing = required - set(r.keys())
    if missing:
        bad_fields.append((i, missing))

print("Records with missing fields:", len(bad_fields))

# 3. Topic distribution
print("\nTopic distribution:")
for topic, count in Counter(r["topic"] for r in rows).most_common():
    print(f"  {topic:25} {count}")

# 4. Question length
lengths = [len(r["instruction"].split()) for r in rows]

print("\nQuestion length:")
print("  Minimum:", min(lengths), "words")
print("  Maximum:", max(lengths), "words")
print("  Average:", round(sum(lengths) / len(lengths), 2), "words")

# 5. Answer length
answer_lengths = [len(r["output"].split()) for r in rows]

print("\nAnswer length:")
print("  Minimum:", min(answer_lengths), "words")
print("  Maximum:", max(answer_lengths), "words")
print("  Average:", round(sum(answer_lengths) / len(answer_lengths), 2), "words")

# 6. Empty inputs
print("\nEmpty input fields:", sum(not r["input"].strip() for r in rows))

# 7. Exact duplicate questions
questions = [r["instruction"].strip().lower() for r in rows]
duplicates = Counter(questions)

print("\nExact duplicate questions:")
for q, count in duplicates.items():
    if count > 1:
        print(f"  {count}x: {q}")

print("\nAudit complete.")