import json

input_path = "data/fine_tuning/synchronization_seed.jsonl"
output_path = "data/fine_tuning/synchronization_seed_fixed.jsonl"

with open(input_path, encoding="utf-8") as f:
    content = f.read()

decoder = json.JSONDecoder()
pos = 0
records = []

while pos < len(content):
    # Skip whitespace
    while pos < len(content) and content[pos].isspace():
        pos += 1

    if pos >= len(content):
        break

    obj, end = decoder.raw_decode(content, pos)
    records.append(obj)
    pos = end

with open(output_path, "w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("Records found:", len(records))
print("Created:", output_path)