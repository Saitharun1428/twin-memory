# Twin Memory Architecture for LLM Reasoning and Safety

A Retrieval-Augmented Generation system that improves LLM logical reasoning and adversarial safety by grounding responses in structured causal memory.

## Key Results

| Test | Baseline | With Twin Memory | Improvement |
|---|---|---|---|
| CLOMO Logical Reasoning (Accuracy) | 88.8% | 94.4% | +5.6% |
| AdvBench Safety (Attack Success Rate) | 38.0% | 14.0% | -24.0% |

---

## How It Works

The system uses two separate vector memory banks stored in ChromaDB:

- **Factual Memory (Mf)** — Causal graphs of real-world facts. Example: heat causes kinetic energy to rise, which causes water to boil.
- **Counterfactual Memory (Mcf)** — Causal graphs of alternative what-if scenarios. Example: if the boiling point changed, what causal chain would follow?

When a user query arrives, it is converted into a vector. Both memory banks are searched for the closest matching graphs. The LLM uses both to reason before generating an answer. This forces the model to consider real consequences before responding to harmful or logically tricky prompts.

---

## Repository Structure

```
twin-memory/
├── main.py                    # Smoke test - run this first to verify setup
├── show_memory.py             # Inspect what is stored inside ChromaDB
├── requirements.txt           # Python dependencies
├── data/
│   ├── clomo_zero_test.json   # CLOMO logical reasoning dataset
│   └── advbench.parquet       # AdvBench safety dataset
├── final_thesis_data/         # Final evaluation reports
├── logs/                      # Execution logs
├── results/                   # Raw evaluation output per condition
├── chroma_db/                 # ChromaDB vector storage (auto-created on first run)
└── src/
    ├── llm_engine.py          # Loads Qwen 2.5 model, provides ask_qwen()
    ├── memory_db.py           # ChromaDB store and retrieve functions
    ├── pipeline.py            # Core agent pipeline + counterfactual generation
    ├── eval_clomo.py          # CLOMO reasoning evaluation
    └── eval_advbench.py       # AdvBench safety evaluation
```

---

## Requirements

- Python 3.11+
- NVIDIA GPU with at least 8GB VRAM
- CUDA drivers installed

---

## Setup

### Step 1 - Clone the repository

```bash
git clone https://github.com/Saitharun1428/twin-memory.git
cd twin-memory
```

### Step 2 - Install dependencies

**On a local machine:**
```bash
pip install -r requirements.txt
```

**On the craft-1 server** (does not support venv):
```bash
pip3 install --user --break-system-packages -r requirements.txt
```

### Step 3 - Add pip to PATH (server only)

```bash
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
. ~/.bashrc
```

### Step 4 - Download the CLOMO dataset

```bash
mkdir -p data logs results
wget -O data/clomo_zero_test.json "https://raw.githubusercontent.com/Eleanor-H/CLOMO/main/data/zero_test.json"
```

### Step 5 - Run the smoke test

This downloads the Qwen model (~3GB) on first run and verifies the full pipeline works end to end.

**On a local machine:**
```bash
python main.py --chroma_path ./chroma_db
```

**On the craft-1 server:**
```bash
PYTHONPATH=. python3 main.py --chroma_path ./chroma_db
```

> Note: PYTHONPATH=. is only needed on the server. It tells Python to look in the current directory for imports from the src/ folder. On a local machine with a normal Python environment this is not required.

You should see the model load on cuda:0, a factual query run, memory store and retrieve, and two counterfactual variants generated. If this completes without errors the system is ready.

---

## Running Evaluations

The evaluations test the model under three conditions:

- **Condition A** — No memory (raw Qwen baseline)
- **Condition B** — Factual memory only (Mf)
- **Condition C** — Both memories (Mf + Mcf)

### CLOMO Logical Reasoning Evaluation

**Local:**
```bash
python src/eval_clomo.py \
    --data_path data/clomo_zero_test.json \
    --chroma_path ./chroma_db \
    --n_samples 100 \
    --conditions A B C
```

**Server (run in background):**
```bash
PYTHONPATH=. nohup python3 -u src/eval_clomo.py \
    --data_path data/clomo_zero_test.json \
    --chroma_path ./chroma_db \
    --n_samples 100 \
    --conditions A B C \
    > logs/clomo.log 2>&1 &
```

Monitor progress:
```bash
tail -f logs/clomo.log
```

### AdvBench Safety Evaluation

**Server (run in background):**
```bash
PYTHONPATH=. nohup python3 -u src/eval_advbench.py \
    --chroma_path ./chroma_db \
    --n_samples 100 \
    --conditions A B C \
    > logs/advbench.log 2>&1 &
```

Monitor progress:
```bash
tail -f logs/advbench.log
```

---

## Inspecting What is Stored in Memory

To see exactly what is stored inside the ChromaDB vector database run:

**Local:**
```bash
python show_memory.py
```

**Server:**
```bash
PYTHONPATH=. python3 show_memory.py
```

This prints every record in both Mf and Mcf collections showing:

- **THE ID** — unique hash that identifies each stored entry
- **THE DOCUMENT** — the original query text that was embedded
- **THE EMBEDDING** — the 384-dimensional vector ChromaDB uses for similarity search
- **THE METADATA** — the full causal graph JSON stored alongside the vector

---

## Reading Results

After each evaluation completes, results are saved in the `results/` folder:

```bash
cat results/clomo_report_*.txt
cat results/advbench_report_*.txt
```

Two types of output files are generated:

- `*_report.txt` and `*_summary.json` — Overall accuracy or ASR, latency, and memory hit rates per condition
- `*_raw.json` — Per-sample logs with the exact prompt, retrieved graphs, model response, and judge score for every sample

---

## Metrics Explained

**CLOMO - Accuracy**
Measures whether the model correctly rewrites a logical argument to satisfy a given relation (Weaken, Strengthen, Necessary Assumption, Sufficient Assumption). Judged by Qwen itself as a binary YES/NO per sample. Higher is better.

**AdvBench - Attack Success Rate (ASR)**
Measures the fraction of harmful prompts where the model failed to refuse. Detected using keyword matching on refusal phrases. Lower is better. ASR of 0% means the model refused every harmful prompt.

---

## Tech Stack

| Component | Tool |
|---|---|
| LLM | Qwen2.5-1.5B-Instruct |
| Vector Database | ChromaDB (persistent) |
| Embeddings | all-MiniLM-L6-v2 (SentenceTransformers) |
| Similarity Metric | Cosine Similarity |
| Server | craft-1.cs.binghamton.edu (6x NVIDIA RTX 6000 Ada, 49GB each) |

---

## Citation

CLOMO dataset from:
Huang et al., CLOMO: Counterfactual Logical Modification with Large Language Models, ACL 2024.

AdvBench dataset from:
walledai/AdvBench on HuggingFace.
