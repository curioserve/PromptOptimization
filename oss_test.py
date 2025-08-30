# oss_example_min.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os

model_dir = "./gpt-oss-20b"  # your local folder

tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# (You already saw the device map split across 0/1)
max_memory = {0: "36GiB", 1: "36GiB", "cpu": "120GiB"}
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,   # unquantized BF16
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print("Device map:", model.hf_device_map)

# Optional speedups on A100s
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

messages = [
    {"role": "system", "content": "You are a helpful, concise assistant."},
    {"role": "user",   "content": "Explain quantum mechanics clearly and concisely."},
]

prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    out = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.2,      # deterministic-ish
        do_sample=False,      # or True + top_p/top_k if you want sampling
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
    )

text = tok.decode(out[0], skip_special_tokens=True)

# Optional: extract only the last assistant message if your template encloses roles
def extract_last_assistant(s: str) -> str:
    # naive split; adjust if your chat_template differs
    sep = "<|assistant|>"
    if sep in s:
        return s.split(sep)[-1].strip()
    return s

print("\n=== Assistant ===\n")
print(extract_last_assistant(text))
