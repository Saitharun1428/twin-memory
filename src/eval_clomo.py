"""
src/eval_clomo.py
3-Condition ablation study on the CLOMO dataset.

Conditions:
  A — No memory   (raw Qwen baseline)
  B — M_f only    (factual memory)
  C — M_f + M_cf  (both memories)

Metric:
  Qwen-as-Judge: binary YES/NO — does the generated argument satisfy
  the required logical relation with the new premise?

Usage:
  python src/eval_clomo.py \
      --data_path  data/clomo_zero_test.json \
      --chroma_path ./chroma_db \
      --n_samples  100 \
      --conditions A B C
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from llm_engine import ask_qwen
from memory_db import retrieve_factual, retrieve_counterfactual

# ── CLOMO relation type mapping ───────────────────────────────────────────────
RELATION_MAP = {
    0: "Necessary Assumption",
    1: "Sufficient Assumption",
    2: "Strengthen",
    3: "Weaken",
}


# ── 1. Data loading ───────────────────────────────────────────────────────────

def load_clomo(path: str, n: int, seed: int = 42) -> list[dict]:
    """
    Load CLOMO zero-shot JSON and return a stratified subsample of n items.
    Stratified by qtype so all 4 relation types are represented.
    """
    with open(path) as f:
        data = json.load(f)

    by_type: dict[int, list] = {}
    for item in data:
        by_type.setdefault(item["qtype"], []).append(item)

    rng = random.Random(seed)
    sampled = []
    per_type = n // len(by_type)
    remainder = n % len(by_type)

    for i, (qtype, items) in enumerate(sorted(by_type.items())):
        k = per_type + (1 if i < remainder else 0)
        sampled.extend(rng.sample(items, min(k, len(items))))

    rng.shuffle(sampled)
    return sampled[:n]


# ── 2. Prompt builders (one per condition) ────────────────────────────────────

SYSTEM_TASK = (
    "You are a logical reasoning expert.\n"
    "Your task: given an original argument and a NEW premise, rewrite the "
    "argument so that it logically satisfies the specified relation with the "
    "NEW premise.\n"
    "Output ONLY the rewritten argument. No explanation, no preamble."
)


def _base_block(item: dict) -> str:
    relation = RELATION_MAP[item["qtype"]]
    return (
        f"ORIGINAL ARGUMENT:\n{item['input_info']['P']}\n\n"
        f"NEW PREMISE:\n{item['input_info']['O']}\n\n"
        f"LOGICAL RELATION REQUIRED: {relation}\n\n"
        f"Rewrite the argument so it {relation.lower()}s the NEW PREMISE:\n"
    )


def build_prompt_A(item: dict) -> str:
    """Condition A — no memory context."""
    return SYSTEM_TASK + "\n\n" + _base_block(item)


def build_prompt_B(item: dict, mf_graphs: list[str]) -> str:
    """Condition B — M_f context injected."""
    if not mf_graphs:
        return build_prompt_A(item)  # graceful degradation

    graph_ctx = ""
    for g_str in mf_graphs[:2]:
        try:
            g = json.loads(g_str)
            nodes = ", ".join(
                n.get("label", str(n)) for n in g.get("nodes", [])
            )
            edges = "; ".join(
                f"{e['source']}→{e['target']}" for e in g.get("edges", [])
            )
            graph_ctx += f"\n  Nodes: [{nodes}]\n  Edges: [{edges}]\n"
        except (json.JSONDecodeError, TypeError):
            graph_ctx += f"\n  {g_str[:120]}\n"

    return (
        SYSTEM_TASK
        + "\n\n"
        + f"[FACTUAL MEMORY CONTEXT]\n"
        + f"Verified causal knowledge relevant to this argument:{graph_ctx}\n"
        + "Use this factual context to ground your rewrite.\n\n"
        + _base_block(item)
    )


def build_prompt_C(item: dict, mf_graphs: list[str],
                   mcf_variants: list[str]) -> str:
    """Condition C — both M_f and M_cf context injected."""
    if not mf_graphs and not mcf_variants:
        return build_prompt_A(item)

    graph_ctx = ""
    for g_str in mf_graphs[:2]:
        try:
            g = json.loads(g_str)
            nodes = ", ".join(
                n.get("label", str(n)) for n in g.get("nodes", [])
            )
            graph_ctx += f"\n  Causal Nodes: [{nodes}]\n"
        except (json.JSONDecodeError, TypeError):
            pass

    cf_ctx = ""
    for v_str in mcf_variants[:2]:
        try:
            v = json.loads(v_str)
            perturbed = v.get("divergence_node", "")
            alt_nodes = ", ".join(
                n.get("label", str(n)) for n in v.get("nodes", [])
            )
            cf_ctx += f"\n  If [{perturbed}] changes → affects [{alt_nodes}]\n"
        except (json.JSONDecodeError, TypeError):
            pass

    return (
        SYSTEM_TASK
        + "\n\n"
        + f"[FACTUAL MEMORY CONTEXT]{graph_ctx}\n"
        + f"[COUNTERFACTUAL MEMORY CONTEXT]\n"
        + f"Hypothetical alternative causal paths:{cf_ctx}\n"
        + "Use BOTH factual grounding and counterfactual alternatives "
        + "to produce the best rewrite.\n\n"
        + _base_block(item)
    )


# ── 3. Qwen-as-Judge (proxy for SES) ─────────────────────────────────────────

JUDGE_TMPL = (
    "You are a strict logical reasoning evaluator.\n\n"
    "ORIGINAL ARGUMENT: {original}\n"
    "NEW PREMISE: {premise}\n"
    "REQUIRED LOGICAL RELATION: {relation}\n"
    "REWRITTEN ARGUMENT: {rewritten}\n\n"
    "Does the REWRITTEN ARGUMENT logically {relation_lower} the NEW PREMISE?\n"
    "Answer with exactly one word — YES or NO."
)


def judge(original: str, premise: str, relation: str, rewritten: str) -> int:
    prompt = JUDGE_TMPL.format(
        original=original,
        premise=premise,
        relation=relation,
        relation_lower=relation.lower(),
        rewritten=rewritten,
    )
    verdict = ask_qwen(prompt, max_new_tokens=8).strip().upper()
    return 1 if verdict.startswith("YES") else 0


# ── 4. Single-sample evaluator ────────────────────────────────────────────────

@dataclass
class SampleResult:
    id_string: str
    qtype: int
    relation: str
    condition: str
    prediction: str
    judge_score: int
    latency_s: float
    mf_hits: int = 0
    mcf_hits: int = 0
    error: str = ""


def evaluate_sample(item: dict, condition: str,
                    chroma_path: str) -> SampleResult:
    relation = RELATION_MAP[item["qtype"]]
    original_p = item["input_info"]["P"]
    new_premise = item["input_info"]["O"]
    mf_hits, mcf_hits = 0, 0

    t0 = time.time()
    try:
        # ── build prompt ──────────────────────────────────────────
        if condition == "A":
            prompt = build_prompt_A(item)

        elif condition == "B":
            query = f"{relation}: {original_p[:200]}"
            mf_graphs_raw = retrieve_factual(
                query, chroma_path, threshold=0.5
            )
            # retrieve_factual returns a single string or None
            mf_graphs = [mf_graphs_raw] if mf_graphs_raw else []
            mf_hits = len(mf_graphs)
            prompt = build_prompt_B(item, mf_graphs)

        elif condition == "C":
            query = f"{relation}: {original_p[:200]}"
            mf_graphs_raw = retrieve_factual(
                query, chroma_path, threshold=0.5
            )
            mf_graphs = [mf_graphs_raw] if mf_graphs_raw else []
            mf_hits = len(mf_graphs)

            mcf_variants = retrieve_counterfactual(
                query, chroma_path, n_results=2
            )
            mcf_hits = len(mcf_variants)
            prompt = build_prompt_C(item, mf_graphs, mcf_variants)

        else:
            raise ValueError(f"Unknown condition: {condition}")

        # ── generate rewrite ──────────────────────────────────────
        prediction = ask_qwen(prompt, max_new_tokens=256).strip()

        # ── judge ─────────────────────────────────────────────────
        score = judge(original_p, new_premise, relation, prediction)

    except Exception as exc:
        return SampleResult(
            id_string=item["id_string"],
            qtype=item["qtype"],
            relation=relation,
            condition=condition,
            prediction="",
            judge_score=0,
            latency_s=round(time.time() - t0, 3),
            mf_hits=mf_hits,
            mcf_hits=mcf_hits,
            error=str(exc),
        )

    return SampleResult(
        id_string=item["id_string"],
        qtype=item["qtype"],
        relation=relation,
        condition=condition,
        prediction=prediction,
        judge_score=score,
        latency_s=round(time.time() - t0, 3),
        mf_hits=mf_hits,
        mcf_hits=mcf_hits,
    )


# ── 5. Condition runner ───────────────────────────────────────────────────────

def run_condition(condition: str, samples: list[dict],
                  chroma_path: str, output_dir: Path) -> dict:
    print(f"\n{'='*60}")
    print(f"  CONDITION {condition}  ({len(samples)} samples)")
    print(f"{'='*60}")

    results: list[SampleResult] = []

    for i, item in enumerate(samples):
        print(f"  [{i+1:3d}/{len(samples)}] {item['id_string']} "
              f"({RELATION_MAP[item['qtype']]})  ", end="", flush=True)

        result = evaluate_sample(item, condition, chroma_path)
        results.append(result)

        status = "✓" if result.judge_score == 1 else ("E" if result.error else "✗")
        print(f"{status}  ({result.latency_s:.1f}s)")

    valid = [r for r in results if not r.error]
    accuracy = sum(r.judge_score for r in valid) / len(valid) if valid else 0.0

    per_rel: dict[str, dict] = {}
    for r in valid:
        per_rel.setdefault(r.relation, {"correct": 0, "total": 0})
        per_rel[r.relation]["total"] += 1
        per_rel[r.relation]["correct"] += r.judge_score

    summary = {
        "condition": condition,
        "n_total": len(results),
        "n_valid": len(valid),
        "n_errors": len(results) - len(valid),
        "accuracy": round(accuracy, 4),
        "per_relation_accuracy": {
            k: round(v["correct"] / v["total"], 4)
            for k, v in per_rel.items()
        },
        "avg_latency_s": round(
            sum(r.latency_s for r in results) / len(results), 3
        ),
        "avg_mf_hits": round(
            sum(r.mf_hits for r in results) / len(results), 2
        ),
        "avg_mcf_hits": round(
            sum(r.mcf_hits for r in results) / len(results), 2
        ),
    }

    raw_path = output_dir / f"clomo_condition_{condition}_raw.json"
    with open(raw_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print(f"\n  → Accuracy: {accuracy:.1%}  |  "
          f"Errors: {summary['n_errors']}  |  Saved: {raw_path}")
    return summary


# ── 6. Report writer ──────────────────────────────────────────────────────────

def write_report(summaries: list[dict], output_dir: Path) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline = next(
        (s["accuracy"] for s in summaries if s["condition"] == "A"), 0.0
    )

    lines = [
        "=" * 70,
        "  TWIN MEMORY — CLOMO ABLATION REPORT",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "  A = No memory (baseline)   B = M_f only   C = M_f + M_cf",
        "  Metric: Qwen-as-Judge accuracy (logical relation satisfied)",
        "",
        "-" * 70,
        f"  {'Cond':<8} {'Accuracy':>10} {'Δ vs A':>9} "
        f"{'Lat(s)':>8} {'MF hits':>9} {'MCF hits':>10}",
        "-" * 70,
    ]

    for s in summaries:
        delta = s["accuracy"] - baseline
        d_str = ("—" if s["condition"] == "A"
                 else f"+{delta:.1%}" if delta >= 0 else f"{delta:.1%}")
        lines.append(
            f"  {s['condition']:<8} {s['accuracy']:>9.1%} {d_str:>9} "
            f"{s['avg_latency_s']:>8.2f} {s['avg_mf_hits']:>9.1f} "
            f"{s['avg_mcf_hits']:>10.1f}"
        )

    lines += ["", "-" * 70, "", "  PER-RELATION BREAKDOWN:", ""]
    all_rels = sorted({r for s in summaries
                       for r in s["per_relation_accuracy"]})
    header = f"  {'Relation':<26}" + "".join(
        f"  {'Cond ' + s['condition']:>10}" for s in summaries
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for rel in all_rels:
        row = f"  {rel:<26}"
        for s in summaries:
            acc = s["per_relation_accuracy"].get(rel, None)
            row += f"  {acc:>10.1%}" if acc is not None else f"  {'N/A':>10}"
        lines.append(row)

    lines += [
        "", "=" * 70,
        "  GUIDE:",
        "    C > B > A  →  both memories improve reasoning",
        "    C > B on Weaken/Strengthen  →  M_cf adds value on perturbative tasks",
        "=" * 70,
    ]

    report = "\n".join(lines)
    txt_path = output_dir / f"clomo_report_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(report)

    json_path = output_dir / f"clomo_summary_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(summaries, f, indent=2)

    print("\n" + report)
    print(f"\n  Saved: {txt_path}")
    print(f"  Saved: {json_path}")


# ── 7. Entry point ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path",   required=True,
                   help="Path to data/clomo_zero_test.json")
    p.add_argument("--chroma_path", default="./chroma_db")
    p.add_argument("--output_dir",  default="./results")
    p.add_argument("--n_samples",   type=int, default=100)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--conditions",  nargs="+", default=["A", "B", "C"],
                   choices=["A", "B", "C"])
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from llm_engine import load_model
    print("[1/4] Loading model ...")
    load_model()

    print(f"\n[2/4] Loading CLOMO data from {args.data_path} ...")
    samples = load_clomo(args.data_path, args.n_samples, args.seed)
    dist = {}
    for s in samples:
        dist[RELATION_MAP[s["qtype"]]] = dist.get(RELATION_MAP[s["qtype"]], 0) + 1
    print(f"      {len(samples)} samples loaded: {dist}")

    print(f"\n[3/4] Running conditions: {args.conditions}")
    summaries = []
    for cond in args.conditions:
        summary = run_condition(cond, samples, args.chroma_path, output_dir)
        summaries.append(summary)

    print("\n[4/4] Writing report ...")
    write_report(summaries, output_dir)


if __name__ == "__main__":
    main()
