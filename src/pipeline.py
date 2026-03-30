"""
src/pipeline.py
The core agent pipeline — factual reasoning + counterfactual variant generation.
Translated directly from notebook Cells 5 & 7.
"""

import json

from src.llm_engine import ask_qwen
from src.memory_db import (
    store_factual,
    retrieve_factual,
    store_counterfactual,
)


def agent_pipeline(user_query: str, chroma_path: str) -> str:
    """
    Full Twin Memory pipeline for a single query.
    Translated directly from agent_pipeline() in notebook Cell 5.

    Steps:
      1. Check M_f for a cached causal graph.
      2. Generate an answer (with or without memory context).
      3. If cache miss: extract a causal graph from the answer.
      4. If cache miss: store the new graph in M_f.

    Returns:
        The model's answer string.
    """
    print(f"\n{'='*55}")
    print(f"[pipeline] Query: '{user_query[:80]}'")
    print(f"{'='*55}")

    # ── Step 1: Check M_f ────────────────────────────────────────
    print("\n[Step 1] Checking Factual Memory (M_f)...")
    memory_graph = retrieve_factual(user_query, chroma_path)

    # ── Step 2: Generate answer ───────────────────────────────────
    print("\n[Step 2] Generating answer...")
    if memory_graph:
        prompt = (
            f"Use the following known causal graph fact to help answer "
            f"the user's question.\n\n"
            f"Causal Fact Memory:\n{memory_graph}\n\n"
            f"User Question: {user_query}"
        )
    else:
        prompt = (
            f"Answer the following question logically and concisely.\n"
            f"Question: {user_query}"
        )

    answer = ask_qwen(prompt)
    print(f"\n[Agent Answer]\n{answer}")

    # ── Steps 3 & 4: Extract + store (only on cache miss) ─────────
    if not memory_graph:
        print("\n[Step 3] Extracting causal graph from answer...")
        graph_prompt = (
            f"Convert the following scientific reasoning into a causal graph.\n"
            f"Output strictly as a JSON object with 'nodes' and 'edges'. "
            f"Do not include any other text.\n\n"
            f"Text: {answer}\n\n"
            f"JSON Graph:"
        )
        extracted_graph = ask_qwen(graph_prompt)

        print("\n[Step 4] Storing new graph in M_f...")
        store_factual(user_query, extracted_graph, chroma_path)
    else:
        print("\n[Steps 3 & 4] Skipped — memory was used.")

    return answer


def generate_and_store_counterfactual_variants(
    factual_query: str,
    factual_graph: str,
    chroma_path: str,
    k: int = 2,
) -> list[dict]:
    """
    Perturb a factual graph to produce k counterfactual variants and store
    them in M_cf. Translated from generate_and_store_k_variants() in Cell 7.

    Args:
        factual_query:  The original query whose graph we are perturbing.
        factual_graph:  JSON string of the factual causal graph from M_f.
        chroma_path:    Path to the ChromaDB persistent storage directory.
        k:              Number of counterfactual variants to generate.

    Returns:
        List of parsed variant dicts (may be empty if model output is invalid).
    """
    print(f"\n{'='*55}")
    print(f"[pipeline] Generating {k} counterfactual variant(s)...")
    print(f"{'='*55}")

    variant_prompt = (
        f"You are a logical AI. I will provide a factual causal graph.\n"
        f"Your task is to generate {k} counterfactual variants of this graph "
        f"by changing one core node (e.g., changing an action or property).\n\n"
        f"Factual Graph:\n{factual_graph}\n\n"
        f"Output STRICTLY as a JSON array containing {k} graph objects. "
        f"Each object must have 'divergence_node', 'nodes', and 'edges'. "
        f"Do not output any other text.\n\n"
        f"JSON Array:"
    )

    variants_output = ask_qwen(variant_prompt)

    try:
        variants = json.loads(variants_output)
    except json.JSONDecodeError:
        print("[pipeline] ERROR: Model did not return valid JSON for variants.")
        return []

    stored = []
    for i, variant in enumerate(variants):
        variant_json_str = json.dumps(variant, indent=2)
        print(f"\n--- Variant {i + 1} ---\n{variant_json_str}")
        store_counterfactual(factual_query, variant_json_str, chroma_path)
        stored.append(variant)

    return stored
