import json
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-14B"

TRAIN_FILE = Path(
    "data/fine_tuning/processed/train.jsonl"
)

VALIDATION_FILE = Path(
    "data/fine_tuning/processed/validation.jsonl"
)


def load_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_token_lengths(records, tokenizer):

    lengths = []

    for record in records:

        formatted = tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=False
        )

        token_ids = tokenizer(
            formatted,
            add_special_tokens=False
        )["input_ids"]

        lengths.append(len(token_ids))

    return lengths


def percentile(values, p):

    values = sorted(values)

    index = (len(values) - 1) * p

    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    fraction = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * fraction
    )


def print_statistics(name, lengths):

    print(f"\n{name}")
    print("-" * 40)

    print("Examples:", len(lengths))
    print("Minimum:", min(lengths))
    print("Maximum:", max(lengths))
    print("Mean:", round(mean(lengths), 2))
    print("Median:", median(lengths))
    print("90th percentile:", round(percentile(lengths, 0.90), 2))
    print("95th percentile:", round(percentile(lengths, 0.95), 2))
    print("99th percentile:", round(percentile(lengths, 0.99), 2))


def main():

    print("Loading Qwen3 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("Tokenizer loaded.")

    train_records = load_records(TRAIN_FILE)
    validation_records = load_records(VALIDATION_FILE)

    train_lengths = get_token_lengths(
        train_records,
        tokenizer
    )

    validation_lengths = get_token_lengths(
        validation_records,
        tokenizer
    )

    print_statistics(
        "TRAINING DATA",
        train_lengths
    )

    print_statistics(
        "VALIDATION DATA",
        validation_lengths
    )

    combined = train_lengths + validation_lengths

    print_statistics(
        "TRAIN + VALIDATION",
        combined
    )


if __name__ == "__main__":
    main()