"""
src/memory_db.py
ChromaDB + SentenceTransformer memory layer.
Translated directly from notebook Cell 3.

Two collections are used (always kept separate):
  - "factual_memory"         → M_f
  - "counterfactual_memory"  → M_cf
"""

import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.60   # same value used in Cell 5

# Module-level globals — loaded once
_embedder: SentenceTransformer = None
_chroma_client = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print("[memory_db] Loading embedding model ...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("[memory_db] Embedding model ready.")
    return _embedder


def _get_client(chroma_path: str):
    """
    Return a persistent ChromaDB client rooted at chroma_path.
    Using PersistentClient so data survives across runs (unlike the
    ephemeral chromadb.Client() in the notebook).
    """
    global _chroma_client
    if _chroma_client is None:
        print(f"[memory_db] Connecting to ChromaDB at: {chroma_path}")
        _chroma_client = chromadb.PersistentClient(path=chroma_path)
    return _chroma_client


def get_collection(collection_name: str, chroma_path: str):
    """
    Return (or create) a ChromaDB collection using cosine similarity.
    This replaces the chroma_client.get_or_create_collection() calls
    scattered through the notebook.
    """
    client = _get_client(chroma_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Factual Memory (M_f) ──────────────────────────────────────────────────────

def store_factual(query: str, graph_json_str: str, chroma_path: str) -> None:
    """
    Embed query and store the JSON causal graph in M_f.
    Translated from store_in_memory() in notebook Cell 3.
    """
    embedder = _get_embedder()
    collection = get_collection("factual_memory", chroma_path)

    embedding = embedder.encode(query).tolist()
    memory_id = str(hash(query))

    collection.add(
        embeddings=[embedding],
        documents=[query],
        metadatas=[{"graph": graph_json_str}],
        ids=[memory_id],
    )
    print(f"[M_f] Stored: '{query[:80]}'")


def retrieve_factual(query: str, chroma_path: str,
                     threshold: float = SIMILARITY_THRESHOLD) -> str | None:
    """
    Search M_f for the closest matching past query.
    Returns the stored graph JSON string, or None if no match above threshold.
    Translated from retrieve_memory() in notebook Cell 3.
    """
    embedder = _get_embedder()
    collection = get_collection("factual_memory", chroma_path)

    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["metadatas", "documents", "distances"],
    )

    if not results["distances"][0]:
        return None

    distance = results["distances"][0][0]
    similarity = 1.0 - distance

    if similarity >= threshold:
        print(f"[M_f] HIT  (similarity={similarity:.2f})  "
              f"matched: '{results['documents'][0][0][:60]}'")
        return results["metadatas"][0][0]["graph"]
    else:
        print(f"[M_f] MISS (best similarity={similarity:.2f})")
        return None


# ── Counterfactual Memory (M_cf) ──────────────────────────────────────────────

def store_counterfactual(query: str, variant_json_str: str,
                         chroma_path: str) -> None:
    """
    Embed the original factual query but store the counterfactual variant graph.
    Translated from the loop inside generate_and_store_k_variants() in Cell 7.
    """
    embedder = _get_embedder()
    collection = get_collection("counterfactual_memory", chroma_path)

    embedding = embedder.encode(query).tolist()
    memory_id = str(hash(variant_json_str))

    collection.add(
        embeddings=[embedding],
        documents=[query],
        metadatas=[{"graph": variant_json_str}],
        ids=[memory_id],
    )
    print(f"[M_cf] Stored variant for: '{query[:60]}'")


def retrieve_counterfactual(query: str, chroma_path: str,
                             n_results: int = 2) -> list[str]:
    """
    Return up to n_results counterfactual variant JSON strings for a query.
    """
    embedder = _get_embedder()
    collection = get_collection("counterfactual_memory", chroma_path)

    embedding = embedder.encode(query).tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["metadatas", "documents", "distances"],
    )

    graphs = []
    for meta in results["metadatas"][0]:
        graphs.append(meta["graph"])
    print(f"[M_cf] Retrieved {len(graphs)} variant(s) for: '{query[:60]}'")
    return graphs
