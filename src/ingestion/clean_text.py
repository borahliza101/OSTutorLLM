from pathlib import Path


INPUT_DIR = Path("data/processed/extracted")
OUTPUT_DIR = Path("data/processed/cleaned")


def clean_text(text):
    kept = []

    for line in text.splitlines():
        line = line.rstrip()            # remove spaces at the END of the line only
        stripped = line.strip()         # a trimmed copy, used only for checking
        compact = stripped.upper().replace(" ", "")   # uppercase, no spaces

        # 1. Drop OSTEP header/footer lines
        if "WWW.OSTEP.ORG" in compact:
            continue
        if "[VERSION" in compact:
            continue
        if stripped == "OPERATING SYSTEMS":
            continue

        # 2. Drop lines that are only a page number (1 to 3 digits)
        if stripped.isdigit() and len(stripped) <= 3:
            continue

        # 3. Keep at most one blank line in a row
        if stripped == "" and kept and kept[-1] == "":
            continue

        kept.append(line)

    return "\n".join(kept).strip()


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