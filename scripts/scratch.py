from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.3-70B-Instruct")

# Check model device
print(f"Model is on device: {model.device}")

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(model.device.index)}")
else:
    print("CUDA is not available.")

messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Check inputs device
print(f"Inputs are on device: {inputs['input_ids'].device}")

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))