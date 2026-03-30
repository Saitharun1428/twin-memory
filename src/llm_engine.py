"""
src/llm_engine.py
Loads Qwen2.5-1.5B-Instruct and exposes ask_qwen().
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = None
model = None

def load_model():
    global tokenizer, model

    if model is not None:
        return 

    print(f"[llm_engine] Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[llm_engine] Loading model to GPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,  
        device_map="cuda:0"    
    )
    print(f"[llm_engine] Model loaded on: {model.device}")

def ask_qwen(user_prompt: str, max_new_tokens: int = 512) -> str:
    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a logical AI reasoning agent. "
                "When requested, strictly output valid JSON and nothing else."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False, 
        pad_token_id=tokenizer.eos_token_id
    )

    generated_ids = out[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n[DEBUG] RAW QWEN OUTPUT:\n{response.strip()}\n" + "="*40)
    
    return response.strip()
