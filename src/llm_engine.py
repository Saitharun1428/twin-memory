"""
src/llm_engine.py
Loads Qwen2.5-1.5B-Instruct and exposes ask_qwen().
Translated directly from notebook Cells 1 & 2.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# These are module-level globals — loaded once when this file is imported
tokenizer = None
model = None


def load_model():
    """Download and load the model into GPU memory. Call this once at startup."""
    global tokenizer, model

    if model is not None:
        return  # Already loaded — don't reload

    print(f"[llm_engine] Loading tokenizer from {MODEL_ID} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"[llm_engine] Loading model to GPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",   # automatically uses GPU if available
    )
    print(f"[llm_engine] Model loaded on: {model.device}")


def ask_qwen(user_prompt: str, max_new_tokens: int = 512) -> str:
    """
    Send a prompt to Qwen and return the response string.
    Identical to the ask_qwen() function in notebook Cell 2.

    Args:
        user_prompt:    The text prompt to send to the model.
        max_new_tokens: Maximum tokens to generate (default 512).

    Returns:
        The model's response as a plain string.
    """
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

    # Format using Qwen's chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,   # Low temperature = more deterministic JSON output
    )

    # Strip the input tokens from the output
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response
