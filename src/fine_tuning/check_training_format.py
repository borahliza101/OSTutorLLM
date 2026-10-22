import json
from pathlib import Path


DATA_FILE = Path("data/fine_tuning/processed/train.jsonl")


def main():

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        first_line = f.readline()

    record = json.loads(first_line)

    print("Loaded first training example successfully.\n")

    print("Messages:")
    for message in record["messages"]:
        print(f"Role: {message['role']}")
        print(f"Content: {message['content']}")
        print()

    # Basic validation
    assert "messages" in record
    assert len(record["messages"]) == 2
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][1]["role"] == "assistant"

    print("Format validation: PASSED")
    print("\nThis example is ready for conversion to a Qwen chat template.")


if __name__ == "__main__":
    main()