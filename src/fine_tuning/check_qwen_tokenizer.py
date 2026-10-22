from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-14B"


def main():

    print("Loading Qwen3 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )

    print("Tokenizer loaded successfully.\n")

    messages = [
        {
            "role": "user",
            "content": "What are the four conditions that must hold for deadlock to occur?"
        },
        {
            "role": "assistant",
            "content": (
                "The four conditions are mutual exclusion, hold-and-wait, "
                "no preemption, and circular wait."
            )
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    print("Formatted Qwen chat input:")
    print("----------------------------------------")
    print(formatted)
    print("----------------------------------------")

    tokens = tokenizer(
        formatted,
        return_tensors=None
    )

    print("\nNumber of tokens:", len(tokens["input_ids"]))

    print("\nTokenizer check: PASSED")


if __name__ == "__main__":
    main()