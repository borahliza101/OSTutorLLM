from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

text = "What is a process in an operating system?"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.encode(text)

print("\n==============================")
print("TOKENIZATION CHECK")
print("==============================")

print("\nOriginal text:")
print(text)

print("\nTokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nNumber of tokens:")
print(len(token_ids))