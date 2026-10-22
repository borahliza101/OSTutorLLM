import json
from difflib import SequenceMatcher

path = "data/fine_tuning/synchronization_seed_fixed.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

questions = [
    (r["id"], r["topic"], r["instruction"])
    for r in rows
]

pairs = []

for i in range(len(questions)):
    for j in range(i + 1, len(questions)):
        id1, topic1, q1 = questions[i]
        id2, topic2, q2 = questions[j]

        similarity = SequenceMatcher(
            None,
            q1.lower(),
            q2.lower()
        ).ratio()

        if similarity >= 0.65:
            pairs.append((similarity, id1, id2, topic1, topic2, q1, q2))

pairs.sort(reverse=True)

print("=" * 80)
print("NEAR-DUPLICATE QUESTION AUDIT")
print("=" * 80)

print(f"\nPairs with similarity >= 0.65: {len(pairs)}\n")

for similarity, id1, id2, topic1, topic2, q1, q2 in pairs:
    print(f"{similarity:.2f} | {id1} | {id2}")
    print(f"  [{topic1}] {q1}")
    print(f"  [{topic2}] {q2}")
    print()