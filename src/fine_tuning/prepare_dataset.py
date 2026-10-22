import json
from pathlib import Path

DATA_DIR = Path("data/fine_tuning")
OUTPUT_DIR = Path("data/fine_tuning/processed")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_file(input_file, output_file):
    converted = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            converted_record = {
                "messages": [
                    {
                        "role": "user",
                        "content": record["instruction"]
                    },
                    {
                        "role": "assistant",
                        "content": record["output"]
                    }
                ]
            }

            converted.append(converted_record)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in converted:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Converted {len(converted)} examples")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":

    train_file = DATA_DIR / "synchronization_train.jsonl"
    validation_file = DATA_DIR / "synchronization_validation.jsonl"

    train_output = OUTPUT_DIR / "train.jsonl"
    validation_output = OUTPUT_DIR / "validation.jsonl"

    convert_file(train_file, train_output)
    convert_file(validation_file, validation_output)

    print("\nDataset preparation complete.")