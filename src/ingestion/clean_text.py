from pathlib import Path
import re


INPUT_DIR = Path("data/processed/extracted")
OUTPUT_DIR = Path("data/processed/cleaned")


def clean_text(text):
    # Replace Windows-style line endings
    text = text.replace("\r\n", "\n")

    # Remove excessive spaces at the beginning/end of lines
    lines = [line.strip() for line in text.split("\n")]

    # Remove empty lines
    lines = [line for line in lines if line]

    # Join lines back together
    text = "\n".join(lines)

    # Replace multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Replace more than two consecutive newlines with two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


if __name__ == "__main__":

    print("OSTutorLLM - Text Cleaning")
    print("---------------------------")

    if not INPUT_DIR.exists():
        print("Input directory does not exist.")
        exit()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(INPUT_DIR.glob("*.txt"))

    print(f"Found {len(files)} text files.\n")

    successful = 0

    for file in files:

        print(f"Cleaning: {file.name}")

        text = file.read_text(encoding="utf-8")

        cleaned_text = clean_text(text)

        output_path = OUTPUT_DIR / file.name

        output_path.write_text(cleaned_text, encoding="utf-8")

        print(f"Original characters: {len(text)}")
        print(f"Cleaned characters:  {len(cleaned_text)}")
        print(f"Saved to: {output_path}\n")

        successful += 1

    print("---------------------------")
    print(f"Successfully cleaned: {successful}/{len(files)} files")