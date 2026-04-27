# tree_retriever.py
# ─────────────────────────────────────────────────────────
# Query-time top-down traversal of the spec tree.
# Returns the most relevant leaf chunks for a given query.
# ─────────────────────────────────────────────────────────

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from spec_tree import get_embedding_model   # reuse singleton, no double-load


class HierarchicalTreeRetriever:
    """
    Traverses the spec tree top-down.
    At each node the cosine similarity to the query embedding is computed.
    Branches below the threshold are pruned.
    The top-k passing leaf nodes are returned.
    """

    def __init__(self):
        self.emb_model = get_embedding_model()

    def _embed(self, text):
        return np.array(self.emb_model.encode(text))

    # ── Public API ───────────────────────────────────────

    def retrieve_leaves(self, root_node, query, k=5, threshold=0.25):
        """
        Returns list of dicts:
          { node_id, similarity, summary, signals, text }
        sorted by similarity descending, capped at k.
        """
        query_emb = self._embed(query)
        survivors = []

        print("\n" + "=" * 60)
        print("TREE TRAVERSAL")
        print(f"  Query     : {query[:80]}")
        print(f"  Threshold : {threshold}   Top-k : {k}")
        print("=" * 60)

        self._traverse(root_node, query_emb, threshold, survivors, depth=0)

        survivors.sort(key=lambda x: x["similarity"], reverse=True)
        result = survivors[:k]

        print(f"\n  Leaves passing threshold : {len(survivors)}")
        print(f"  Returning top-{len(result)}")
        print("-" * 60)
        return result

    # ── Internal ─────────────────────────────────────────

    def _traverse(self, node, query_emb, threshold, survivors, depth):
        if node is None:
            return
        
        

        sim = cosine_similarity(
            query_emb.reshape(1, -1),
            node.embedding.reshape(1, -1)
        )[0][0]
        
        if depth == 0:
            threshold = sim - 0.05

        tag    = "LEAF  " if node.is_leaf else "PARENT"
        indent = "  " * depth
        status = "✅" if sim >= threshold else "❌ PRUNED"
        print(f"{indent}├─ [{tag}] {node.node_id} | sim={sim:.4f} {status}")

        if sim < threshold:
            return

        if node.is_leaf:
            survivors.append({
                "node_id"   : node.node_id,
                "similarity": float(sim),
                "summary"   : node.chunk_summary,
                "signals"   : node.signals,
                "text"      : node.page_text,
            })
        else:
            self._traverse(node.left_child,  query_emb, threshold, survivors, depth + 1)
            self._traverse(node.right_child, query_emb, threshold, survivors, depth + 1)
