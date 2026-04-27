# spec_tree.py
# ---------------------------------------------------------
# Builds a hierarchical binary similarity tree from a hardware
# specification document.  No Streamlit dependency.
# ---------------------------------------------------------

import uuid
import time
import random
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import GROQ_MODEL_SMALL, EMBEDDING_MODEL

# -- Singleton embedding model (loaded once, shared by tree_retriever) --------
_embedding_model_instance = None

def get_embedding_model():
    global _embedding_model_instance
    if _embedding_model_instance is None:
        print(f"[embedding] Loading '{EMBEDDING_MODEL}' (first use)...")
        _embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model_instance


# ---------------------------------------------------------
# Add json to your imports at the top if not already present
import json 
import numpy as np

class TreeNode:
    """A node in the Hierarchical Binary Similarity Tree."""

    def __init__(self, node_id, is_leaf=False, page_text="",
                 chunk_summary="", embedding=None, signals=None):
        self.node_id       = node_id
        self.is_leaf       = is_leaf
        self.page_text     = page_text       
        self.chunk_summary = chunk_summary   
        self.embedding     = embedding       
        self.signals       = signals or []   

        self.parent_id   = None
        self.left_child  = None
        self.right_child = None

    # --- ADD THESE METHODS ---
    def to_dict(self):
        """Serializes the TreeNode and its children to a dictionary."""
        return {
            "node_id": self.node_id,
            "is_leaf": self.is_leaf,
            "page_text": self.page_text,
            "chunk_summary": self.chunk_summary,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "signals": self.signals,
            "parent_id": self.parent_id,
            "left_child": self.left_child.to_dict() if self.left_child else None,
            "right_child": self.right_child.to_dict() if self.right_child else None,
        }

    @classmethod
    def from_dict(cls, data):
        """Deserializes a dictionary back into a TreeNode hierarchy."""
        if data is None:
            return None
            
        node = cls(
            node_id=data["node_id"],
            is_leaf=data["is_leaf"],
            page_text=data["page_text"],
            chunk_summary=data["chunk_summary"],
            embedding=np.array(data["embedding"]) if data.get("embedding") is not None else None,
            signals=data.get("signals", [])
        )
        node.parent_id = data.get("parent_id")

        left_data = data.get("left_child")
        if left_data:
            node.left_child = cls.from_dict(left_data)
            
        right_data = data.get("right_child")
        if right_data:
            node.right_child = cls.from_dict(right_data)

        return node


# ---------------------------------------------------------
class SpecTreeBuilder:
    """
    Segments a spec document into chunks (leaves), summarises and
    extracts signals from each via the Groq LLM, then merges nodes
    bottom-up by embedding similarity to form a binary tree.
    """

    def __init__(self, api_key, chunk_size=35, chunk_overlap=7):
        self.client        = Groq(api_key=api_key)
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model         = GROQ_MODEL_SMALL
        self.emb_model     = get_embedding_model()

    # -- Helpers ------------------------------------------

    def _retry(self, func, *args, max_retries=5, **kwargs):
        """Exponential-backoff retry for Groq rate-limit errors."""
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                code = getattr(e, "status_code", None)
                if code in (429, 503) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                else:
                    raise

    def _embed(self, text):
        return np.array(self.emb_model.encode(text))

    def _summarise(self, text, is_combination=False):
        """
        Ask the LLM to summarise a chunk and extract hardware signal names.
        Returns (summary: str, signals: list[str]).
        """
        if is_combination:
            prompt = (
                "Combine these two hardware spec summaries into a single "
                "cohesive 2-line summary:\n" + text +
                "\n\nReturn EXACTLY:\nSUMMARY: <combined summary>"
            )
        else:
            prompt = (
                "Analyse the following hardware specification text.\n"
                "1. Provide a concise 1-3 line functional summary.\n"
                "2. List all explicit hardware signal/register/port names "
                "   mentioned (e.g., clk, stall, exc_flag).\n\n"
                "Text:\n" + text +
                "\n\nReturn EXACTLY:\n"
                "SUMMARY: <summary>\n"
                "SIGNALS: <comma-separated list, or NONE>"
            )

        def _call():
            r = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
            )
            return r.choices[0].message.content.strip()

        raw = self._retry(_call)
        summary, signals = "", []

        if "SIGNALS:" in raw:
            parts   = raw.split("SIGNALS:", 1)
            summary = parts[0].replace("SUMMARY:", "").strip()
            sig_txt = parts[1].strip()
            if sig_txt.upper() != "NONE":
                signals = [s.strip() for s in sig_txt.split(",") if s.strip()]
        else:
            summary = raw.replace("SUMMARY:", "").strip()

        return summary, signals

    def _segment(self, text):
        """Split text into overlapping word-level chunks."""
        words  = text.split()
        step   = max(1, self.chunk_size - self.chunk_overlap)
        chunks = [
            " ".join(words[i: i + self.chunk_size])
            for i in range(0, len(words), step)
        ]
        return [c for c in chunks if c.strip()]

    # -- Main build ---------------------------------------

    def build_tree(self, spec_text):
        """
        Build the full hierarchical tree.
        Returns (root_node, all_nodes_dict).
        """
        print("[spec_tree] Segmenting specification...")
        chunks = self._segment(spec_text)
        active = []

        # 1. Create leaf nodes
        for i, chunk in enumerate(chunks):
            print(f"  Leaf {i+1}/{len(chunks)}...")
            summary, signals = self._summarise(chunk)
            node = TreeNode(
                node_id       = f"L0_p{i}",
                is_leaf       = True,
                page_text     = chunk,
                chunk_summary = summary,
                embedding     = self._embed(summary),
                signals       = signals,
            )
            active.append(node)
            time.sleep(0.5)  # respect rate limits

        all_nodes = {n.node_id: n for n in active}
        level = 1

        # 2. Bottom-up agglomerative merging
        while len(active) > 1:
            print(f"  [Tree] Level {level}: merging {len(active)} nodes...")
            embs = np.array([n.embedding for n in active])
            sim  = cosine_similarity(embs)
            np.fill_diagonal(sim, -1)
            i1, i2 = np.unravel_index(np.argmax(sim), sim.shape)

            n1, n2    = active[i1], active[i2]
            parent_id = f"L{level}_{uuid.uuid4().hex[:6]}"

            merged_summary, _ = self._summarise(
                f"1. {n1.chunk_summary}\n2. {n2.chunk_summary}",
                is_combination=True,
            )
            parent = TreeNode(
                node_id       = parent_id,
                is_leaf       = False,
                chunk_summary = merged_summary,
                embedding     = (n1.embedding + n2.embedding) / 2.0,
                signals       = list(set(n1.signals + n2.signals)),
            )
            parent.left_child = n1
            parent.right_child = n2
            n1.parent_id = n2.parent_id = parent_id

            for idx in sorted([i1, i2], reverse=True):
                active.pop(idx)
            active.append(parent)
            all_nodes[parent_id] = parent
            level += 1

        root = active[0]
        print(f"[spec_tree] Tree built. Root: {root.node_id}")
        return root, all_nodes
