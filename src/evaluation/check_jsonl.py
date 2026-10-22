import json

path = "data/fine_tuning/synchronization_seed.jsonl"

with open(path, encoding="utf-8") as f:
    for line_number, line in enumerate(f, 1):
        if not line.strip():
            continue

        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f"BAD LINE: {line_number}")
            print(f"ERROR: {e}")
            print(f"CONTENT: {line[:300]!r}")
            print("-" * 60)