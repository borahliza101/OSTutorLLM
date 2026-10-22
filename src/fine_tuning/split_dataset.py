import json
import random
from pathlib import Path
from collections import defaultdict, Counter


DATA_DIR = Path("data/fine_tuning")

INPUT_FILE = DATA_DIR / "synchronization_seed_fixed.jsonl"

TRAIN_FILE = DATA_DIR / "synchronization_train.jsonl"
VALIDATION_FILE = DATA_DIR / "synchronization_validation.jsonl"
TEST_FILE = DATA_DIR / "synchronization_test.jsonl"

SEED = 42


def load_dataset():
    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def save_dataset(records, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

    records = load_dataset()

    print(f"Loaded {len(records)} examples.")

    if len(records) != 140:
        raise ValueError(
            f"Expected 140 examples, but found {len(records)}."
        )

    # Group examples by topic
    by_topic = defaultdict(list)

    for record in records:
        by_topic[record["topic"]].append(record)

    train = []
    validation = []
    test = []

    # Use a fixed seed for reproducibility
    rng = random.Random(SEED)

    # Stratified split
    for topic, examples in sorted(by_topic.items()):

        examples = examples.copy()
        rng.shuffle(examples)

        n = len(examples)

        # Approximately 10% validation and 10% test
        n_validation = max(1, round(n * 0.10))
        n_test = max(1, round(n * 0.10))

        # Keep at least one example for training
        if n_validation + n_test >= n:
            n_validation = 1
            n_test = 1

        validation.extend(
            examples[:n_validation]
        )

        test.extend(
            examples[
                n_validation:n_validation + n_test
            ]
        )

        train.extend(
            examples[
                n_validation + n_test:
            ]
        )

    # Shuffle each final split
    random.Random(SEED + 1).shuffle(train)
    random.Random(SEED + 2).shuffle(validation)
    random.Random(SEED + 3).shuffle(test)

    # Save
    save_dataset(train, TRAIN_FILE)
    save_dataset(validation, VALIDATION_FILE)
    save_dataset(test, TEST_FILE)

    # Print summary
    print("\nSplit complete!")
    print(f"Train:       {len(train)}")
    print(f"Validation:  {len(validation)}")
    print(f"Test:        {len(test)}")
    print(f"Total:       {len(train) + len(validation) + len(test)}")

    print("\nTopic distribution:")

    topics = sorted(by_topic.keys())

    for topic in topics:

        train_count = sum(
            r["topic"] == topic for r in train
        )

        validation_count = sum(
            r["topic"] == topic for r in validation
        )

        test_count = sum(
            r["topic"] == topic for r in test
        )

        print(
            f"{topic:22s} "
            f"train={train_count:2d} "
            f"val={validation_count:2d} "
            f"test={test_count:2d}"
        )

    print("\nFiles created:")
    print(TRAIN_FILE)
    print(VALIDATION_FILE)
    print(TEST_FILE)


if __name__ == "__main__":
    main()