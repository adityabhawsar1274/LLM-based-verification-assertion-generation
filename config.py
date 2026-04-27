# config.py
# ─────────────────────────────────────────────────────────
# Central configuration for the LLM-based SVA generation pipeline.
# Edit this file to change API keys, models, and pipeline parameters.
# ─────────────────────────────────────────────────────────

# ── API ──────────────────────────────────────────────────
API_KEY = "<YOUR_GROQ_API_KEY>"

# ── LLM models ───────────────────────────────────────────
GROQ_MODEL_LARGE = "llama-3.3-70b-versatile"   # Used for assertion generation
GROQ_MODEL_SMALL = "llama-3.1-8b-instant"       # Used for tree summarization

# ── Embedding model ───────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Spec tree building ────────────────────────────────────
CHUNK_SIZE    = 35   # words per leaf chunk
CHUNK_OVERLAP = 7    # overlapping words between adjacent chunks

# ── Retrieval ─────────────────────────────────────────────
RETRIEVAL_K         = 7     # top-k leaves to retrieve
RETRIEVAL_THRESHOLD = 0.25  # cosine similarity pruning threshold

# ── Formal verification ───────────────────────────────────
BMC_DEPTH      = 10  # SymbiYosys bounded model checking depth
MAX_ITERATIONS = 5   # maximum LLM→verify refinement iterations


PARAM_TO_CONCRETE = {"BDW": 32, "OWN": 1, "BAW": 1}