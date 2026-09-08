from pathlib import Path


INPUT_DIR = Path("data/processed/cleaned")
OUTPUT_DIR = Path("data/processed/chunks")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    print("OSTutorLLM - Text Chunking")
    print("---------------------------")

    if not INPUT_DIR.exists():
        print("Cleaned text directory does not exist.")
        exit()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_DIR.glob("*.txt"))

    print(f"Found {len(files)} cleaned text files.\n")

    total_chunks = 0

    for file in files:

        print(f"Chunking: {file.name}")

        text = file.read_text(encoding="utf-8")

        chunks = create_chunks(text)

        output_file = OUTPUT_DIR / f"{file.stem}__chunks.txt"

        with output_file.open("w", encoding="utf-8") as f:

            for i, chunk in enumerate(chunks):

                f.write(f"--- CHUNK {i} ---\n")
                f.write(chunk)
                f.write("\n\n")

        print(f"Characters: {len(text)}")
        print(f"Chunks created: {len(chunks)}")
        print(f"Saved to: {output_file}\n")

        total_chunks += len(chunks)

    print("---------------------------")
    print(f"Total chunks created: {total_chunks}")