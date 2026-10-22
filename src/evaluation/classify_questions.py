import json
import re
from collections import Counter

path = "data/fine_tuning/synchronization_seed_fixed.jsonl"

with open(path, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]


def classify(question):
    q = question.lower()

    # Scenario / execution
    if any(x in q for x in [
        "thread a", "thread b", "a thread", "two threads",
        "what happens when", "what can happen when",
        "is initialized to", "performs wait", "performs post",
        "acquires", "holds", "interrupted"
    ]):
        return "scenario_execution"

    # Debugging / identifying an error
    if any(x in q for x in [
        "what can go wrong",
        "what problem can occur",
        "what is wrong",
        "why can this",
        "why does this fail",
        "why is this dangerous",
        "bug",
        "deadlock"
    ]) and any(x in q for x in [
        "why", "what", "problem", "wrong", "fail", "bug"
    ]):
        return "debugging_problem"

    # Comparisons
    if any(x in q for x in [
        "difference between",
        "compare",
        "which",
        "more suitable"
    ]):
        return "comparison_choice"

    # Misconception / correctness
    if any(x in q for x in [
        "student says",
        "is this statement correct",
        "is this always",
        "is it correct"
    ]):
        return "misconception"

    # Why / explanation
    if q.startswith("why ") or " why " in q:
        return "why_explanation"

    # How / mechanism
    if q.startswith("how ") or " how " in q:
        return "how_mechanism"

    # Definitions
    if q.startswith("what is") or q.startswith("what are"):
        return "definition"

    return "other"


counts = Counter()

print("=" * 80)
print("QUESTION-TYPE AUDIT")
print("=" * 80)

for r in rows:
    qtype = classify(r["instruction"])
    counts[qtype] += 1

print("\nDistribution:\n")

for qtype, count in counts.most_common():
    percentage = count / len(rows) * 100
    print(f"{qtype:25} {count:3} ({percentage:5.1f}%)")


print("\n" + "=" * 80)
print("EXAMPLES BY TYPE")
print("=" * 80)

for qtype in counts:
    print(f"\n--- {qtype.upper()} ---")

    shown = 0

    for r in rows:
        if classify(r["instruction"]) == qtype:
            print(f'{r["id"]}: {r["instruction"]}')
            shown += 1

            if shown >= 5:
                break