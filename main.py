"""
main.py
Quick smoke-test of the full Twin Memory pipeline.
Run this first to verify everything works before running evals.

Usage:
    python main.py --chroma_path ./chroma_db
"""

import argparse
from src.llm_engine import load_model
from src.pipeline import agent_pipeline, generate_and_store_counterfactual_variants
from src.memory_db import retrieve_factual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chroma_path", default="./chroma_db")
    args = parser.parse_args()

    print("=" * 60)
    print("  TWIN MEMORY ARCHITECTURE — SMOKE TEST")
    print("=" * 60)

    # Step 1: Load the model
    print("\n[1] Loading Qwen model ...")
    load_model()

    # Step 2: Run a factual query (same as notebook test)
    print("\n[2] Testing factual pipeline (M_f) ...")
    query = "Explain why water boils at 100 degrees Celsius at sea level."
    agent_pipeline(query, args.chroma_path)

    # Step 3: Run a semantically similar query — should HIT memory
    print("\n[3] Testing memory retrieval ...")
    similar_query = "What causes water to boil when heated?"
    agent_pipeline(similar_query, args.chroma_path)

    # Step 4: Generate counterfactual variants (M_cf)
    print("\n[4] Testing counterfactual variant generation (M_cf) ...")
    factual_graph = retrieve_factual(query, args.chroma_path)
    if factual_graph:
        generate_and_store_counterfactual_variants(
            query, factual_graph, args.chroma_path, k=2
        )
    else:
        print("  Skipped — no factual graph found in M_f yet.")

    print("\n" + "=" * 60)
    print("  Smoke test complete. Ready to run evaluations.")
    print("  Next steps:")
    print("    python src/eval_clomo.py --data_path data/clomo_zero_test.json")
    print("    python src/eval_advbench.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
