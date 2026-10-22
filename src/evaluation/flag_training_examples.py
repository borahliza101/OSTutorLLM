import json
import re

path = "data/fine_tuning/synchronization_seed_fixed.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]


# Terms that often deserve manual review because they can hide
# overly strong or context-dependent claims.
warning_patterns = {
    "always": r"\balways\b",
    "never": r"\bnever\b",
    "guaranteed": r"\bguarantee(?:d|s)?\b",
    "cannot": r"\bcannot\b",
    "must": r"\bmust\b",
    "only": r"\bonly\b",
    "immediately": r"\bimmediately\b",
    "automatically": r"\bautomatically\b",
    "ensures": r"\bensures?\b",
    "prevents": r"\bprevents?\b",
}

print("=" * 80)
print("TRAINING DATA TECHNICAL AUDIT — FLAGGED EXAMPLES")
print("=" * 80)

flagged = []

for r in rows:
    text = r["output"].lower()

    matched = []

    for label, pattern in warning_patterns.items():
        if re.search(pattern, text):
            matched.append(label)

    if matched:
        flagged.append((r, matched))


print(f"\nFlagged examples: {len(flagged)} / {len(rows)}\n")

for r, matched in flagged:
    print("-" * 80)
    print(f"ID: {r['id']}")
    print(f"Topic: {r['topic']}")
    print(f"Warnings: {', '.join(matched)}")
    print(f"Question: {r['instruction']}")
    print(f"Answer: {r['output']}")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)