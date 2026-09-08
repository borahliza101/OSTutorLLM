from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("Model loaded!")

messages = [
    {
        "role": "system",
        "content": "You are a helpful Operating Systems tutor."
    },
    {
        "role": "user",
        "content": "What is a process in an operating system?"
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer(
    [text],
    return_tensors="pt"
).to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=200
)

generated_ids = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(
        model_inputs.input_ids,
        generated_ids
    )
]

response = tokenizer.batch_decode(
    generated_ids,
    skip_special_tokens=True
)[0]

print()
print("================================")
print("LLM RESPONSE")
print("================================")
print(response)