#!/usr/bin/env python3
# batch_eval.py
# ---------------------------------------------------------
# Batch runner + metrics collector for the LLM-based SVA
# generation pipeline (main2.py).
#
# Usage:
#   python batch_eval.py test_rtl.v \
#       --queries      queries.txt       \
#       --tree-file    spec_tree.json    \
#       --depth        15                \
#       --max-iterations 3              \
#       --retrieval-k  3                \
#       --save-report  report.json
#
# queries.txt format:
#   One query per "block"; blocks are separated by a blank line.
#   Each query can span multiple lines.
#
# Metrics reported:
#   1. Pass / Fail counts  (fail sub-types: counterexample, syntax, missing signals)
#   2. Leaf usage %        (leaves used / total leaves in tree)
#   3. Iterations to pass  (average, min, max, 1st-try pass rate)
#   4. Wall-clock time per query
#   5. Per-query table for a quick overview
# ---------------------------------------------------------

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ═══════════════════════════════════════════════════════════
#  Tree helpers
# ═══════════════════════════════════════════════════════════

def count_leaves(node: dict) -> int:
    """
    Recursively count leaf nodes in the serialized tree JSON.

    TreeNode.to_dict() stores children as 'left_child' / 'right_child'
    (both may be None for an actual leaf).  We also accept the generic
    'children' / 'subtrees' list keys just in case.
    """
    # Prefer the explicit is_leaf flag when present
    if node.get("is_leaf") is True:
        return 1

    # Binary-tree layout used by TreeNode.to_dict()
    left  = node.get("left_child")
    right = node.get("right_child")
    if left is None and right is None:
        return 1  # no children → leaf

    total = 0
    if left:
        total += count_leaves(left)
    if right:
        total += count_leaves(right)

    # Fall back to generic list-based children if present
    for child in node.get("children") or node.get("subtrees") or []:
        total += count_leaves(child)

    return total


# ═══════════════════════════════════════════════════════════
#  Query file parser
# ═══════════════════════════════════════════════════════════

def parse_queries(path: str) -> list:
    """
    Read queries.txt.
    Queries are separated by one or more blank lines; each query
    may itself span multiple lines.
    Returns a list of stripped query strings.
    """
    raw = Path(path).read_text(encoding="utf-8")
    # Split on one-or-more blank lines
    blocks = re.split(r"\n[ \t]*\n", raw.strip())
    queries = []
    for block in blocks:
        q = block.strip()
        if q:
            queries.append(q)
    return queries


# ═══════════════════════════════════════════════════════════
#  stdout parser
# ═══════════════════════════════════════════════════════════

def parse_stdout(stdout: str) -> dict:
    """
    Extract structured metrics from main2.py's printed output.

    Keys returned
    ─────────────
    status              : final pipeline status string
    iterations_used     : last iteration index reached (int)
    initial_leaves      : leaves retrieved in Phase 2 (int, if printed)
    extra_leaves_added  : extra leaves added during FAIL_MISSING_SIGNALS loops
    missing_signals     : list of invented signal names (may be empty)
    has_syntax_warning  : bool — LLM produced a syntax warning
    converged           : bool — pipeline said "did not converge"
    """
    m = {
        "status"             : "UNKNOWN",
        "iterations_used"    : 0,
        "initial_leaves"     : 0,
        "extra_leaves_added" : 0,
        "missing_signals"    : [],
        "has_syntax_warning" : False,
        "converged"          : True,
    }

    # ── Status ────────────────────────────────────────────
    # Last "Status : XYZ" line wins (there may be one per iteration)
    for hit in re.finditer(r"Status\s*:\s*(\S+)", stdout):
        m["status"] = hit.group(1)

    if "ASSERTION VERIFIED" in stdout:
        m["status"] = "PASS"

    if "did not converge" in stdout:
        m["converged"] = False
        if m["status"] == "UNKNOWN":
            m["status"] = "FAIL_NO_CONVERGENCE"

    # ── Iterations ────────────────────────────────────────
    iter_hits = re.findall(r"ITERATION\s+(\d+)\s*/", stdout)
    if iter_hits:
        m["iterations_used"] = int(iter_hits[-1])

    # ── Leaf counts ───────────────────────────────────────
    # main2.py / tree_retriever prints lines like:
    #   "Leaves passing threshold : 8"   ← total candidates
    #   "Returning top-7"                ← actually used (capped by K)
    # Prefer "Returning top-N"; fall back to "Retrieved N leaves".
    returning_hit = re.search(r"Returning\s+top[-\s](\d+)", stdout, re.I)
    threshold_hit = re.search(r"Leaves\s+passing\s+threshold\s*:\s*(\d+)", stdout, re.I)
    generic_hit   = re.search(r"[Rr]etrieved\s+(\d+)\s+lea(?:f|ves)", stdout)

    if returning_hit:
        m["initial_leaves"] = int(returning_hit.group(1))
    elif threshold_hit:
        m["initial_leaves"] = int(threshold_hit.group(1))
    elif generic_hit:
        m["initial_leaves"] = int(generic_hit.group(1))

    extra_hits = re.findall(r"Added\s+(\d+)\s+extra\s+leaf", stdout)
    m["extra_leaves_added"] = sum(int(x) for x in extra_hits)

    # ── Missing / invented signals ────────────────────────
    missing_hits = re.findall(
        r"LLM invented signals not in RTL:\s*\{([^}]+)\}", stdout
    )
    signals = []
    for hit in missing_hits:
        for s in hit.split(","):
            s = s.strip().strip("'\"")
            if s:
                signals.append(s)
    m["missing_signals"] = list(dict.fromkeys(signals))  # deduplicate, keep order

    # ── Syntax warnings ───────────────────────────────────
    if re.search(r"Syntax warnings|syntax_warn", stdout, re.I):
        m["has_syntax_warning"] = True

    return m


# ═══════════════════════════════════════════════════════════
#  Fail classifier
# ═══════════════════════════════════════════════════════════

def classify_fail(status: str, stdout: str, result_json: dict) -> str | None:
    """
    Return one of: None | 'counterexample' | 'syntax_error' |
                   'missing_signals' | 'no_convergence' | 'pipeline_error' | 'unknown'
    """
    if status == "PASS":
        return None

    if "MISSING_SIGNALS" in status:
        return "missing_signals"

    if "NO_CONVERGENCE" in status or "did not converge" in stdout:
        return "no_convergence"

    # Pipeline failed before any verification ran (file not found, import error…)
    # Detected by: no ITERATION line was ever printed.
    if status == "UNKNOWN":
        return "pipeline_error"

    # Look at result JSON for more detail
    suggestions = ""
    if result_json:
        suggestions = result_json.get("suggestions_for_next_iteration", "")
        rjstatus    = result_json.get("status", "")
        if "MISSING_SIGNALS" in rjstatus:
            return "missing_signals"

    if re.search(r"syntax|SyntaxError|parse\s*error", stdout + suggestions, re.I):
        return "syntax_error"

    # Only call it a counterexample if sby explicitly said so — not just any "FAIL" word
    if re.search(r"\bcounterexample\b", stdout, re.I):
        return "counterexample"

    if status.startswith("FAIL"):
        return "counterexample"

    return "unknown"


# ═══════════════════════════════════════════════════════════
#  Single-query runner
# ═══════════════════════════════════════════════════════════

def run_one_query(
    query: str,
    rtl_file: str,
    tree_file: str,
    depth: int,
    max_iterations: int,
    python_bin: str,
) -> tuple:
    """
    Spawn main2.py for one query.
    Returns (result_json | None, stdout_str, stderr_str, elapsed_sec).
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        out_path = tf.name

    cmd = [
        python_bin, "main2.py",
        rtl_file,
        "--query", query,
        "--tree-file", tree_file,
        "--depth", str(depth),
        "--max-iterations", str(max_iterations),
        "--output", out_path,
    ]

    t0   = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    stdout = proc.stdout
    stderr = proc.stderr

    result_json = None
    try:
        text = Path(out_path).read_text(encoding="utf-8").strip()
        if text:
            result_json = json.loads(text)
    except Exception:
        pass

    # Clean up temp file
    try:
        Path(out_path).unlink(missing_ok=True)
    except Exception:
        pass

    return result_json, stdout, stderr, elapsed


# ═══════════════════════════════════════════════════════════
#  Report printer
# ═══════════════════════════════════════════════════════════

def _avg(lst):
    return sum(lst) / len(lst) if lst else 0.0


def print_report(results: list, total_leaves: int, retrieval_k: int):
    n      = len(results)
    passed = [r for r in results if r["status"] == "PASS"]
    failed = [r for r in results if r["status"] != "PASS"]

    # Fail sub-type counts
    fail_counts = {
        "counterexample"  : 0,
        "syntax_error"    : 0,
        "missing_signals" : 0,
        "no_convergence"  : 0,
        "pipeline_error"  : 0,
        "unknown"         : 0,
    }
    for r in failed:
        key = r["fail_reason"] or "unknown"
        fail_counts[key] = fail_counts.get(key, 0) + 1

    # Iteration stats
    iters_pass = [r["iterations_used"] for r in passed if r["iterations_used"] > 0]
    iters_all  = [r["iterations_used"] for r in results if r["iterations_used"] > 0]
    first_try  = sum(1 for r in passed if r["iterations_used"] == 1)

    # Leaf usage — prefer the count parsed from stdout (accurate);
    # fall back to the --retrieval-k CLI arg when stdout gave nothing.
    def leaf_used(r):
        parsed = r.get("initial_leaves", 0)
        base   = parsed if parsed > 0 else retrieval_k
        return base + r["extra_leaves_added"]

    # Actual initial-leaf counts seen across queries (for the header line)
    actual_initial = [r["initial_leaves"] for r in results if r["initial_leaves"] > 0]
    display_k = (
        f"{min(actual_initial)}–{max(actual_initial)}"
        if actual_initial and min(actual_initial) != max(actual_initial)
        else str(actual_initial[0]) if actual_initial
        else str(retrieval_k)          # nothing parsed → fall back to CLI arg
    )

    leaf_pcts = []
    for r in results:
        if total_leaves > 0:
            pct = min(100.0, 100.0 * leaf_used(r) / total_leaves)
            leaf_pcts.append(pct)

    # Timing
    times = [r["elapsed_sec"] for r in results]

    W = 65
    SEP = "═" * W

    print(f"\n{SEP}")
    print("  BATCH EVALUATION REPORT")
    print(SEP)

    # ── 1. Pass / Fail ───────────────────────────────────────
    print(f"\n  ┌─ Pass / Fail ({'─'*40})")
    print(f"  │  Total queries          : {n}")
    print(f"  │  Passed                 : {len(passed)}  ({100*len(passed)/n:.1f}%)")
    print(f"  │  Failed                 : {len(failed)}  ({100*len(failed)/n:.1f}%)")
    print(f"  │")
    print(f"  │  Failure breakdown:")
    print(f"  │    Counterexample        : {fail_counts['counterexample']}")
    print(f"  │    Syntax error          : {fail_counts['syntax_error']}")
    print(f"  │    Missing / invented    : {fail_counts['missing_signals']}")
    print(f"  │    No convergence        : {fail_counts['no_convergence']}")
    print(f"  │    Pipeline/setup error  : {fail_counts['pipeline_error']}")
    print(f"  │    Other / unknown       : {fail_counts['unknown']}")

    # ── 2. Leaf usage ────────────────────────────────────────
    print(f"\n  ┌─ Context / Leaf Usage ({'─'*36})")
    print(f"  │  Total leaves in tree   : {total_leaves}")
    print(f"  │  Leaves retrieved (avg) : {display_k}  (parsed from 'Returning top-N' in output)")
    if leaf_pcts:
        print(f"  │  Avg leaf usage         : {_avg(leaf_pcts):.1f}%")
        print(f"  │  Min / Max leaf usage   : {min(leaf_pcts):.1f}% / {max(leaf_pcts):.1f}%")
        print(f"  │  → Low % = minimal context is sufficient (good!)")
    else:
        print(f"  │  (leaf count unavailable)")

    # ── 3. Iterations ────────────────────────────────────────
    print(f"\n  ┌─ Iteration Efficiency ({'─'*37})")
    if iters_all:
        print(f"  │  Avg iterations (all)   : {_avg(iters_all):.2f}")
    if iters_pass:
        print(f"  │  Avg iterations (pass)  : {_avg(iters_pass):.2f}")
        print(f"  │  Min / Max (pass)       : {min(iters_pass)} / {max(iters_pass)}")
    print(f"  │  1st-try passes          : {first_try} / {len(passed)}"
          + (f"  ({100*first_try/len(passed):.1f}%)" if passed else ""))

    # ── 4. Timing ────────────────────────────────────────────
    print(f"\n  ┌─ Wall-clock Time ({'─'*41})")
    print(f"  │  Total                  : {sum(times):.1f}s")
    print(f"  │  Avg per query          : {_avg(times):.1f}s")
    print(f"  │  Min / Max              : {min(times):.1f}s / {max(times):.1f}s")

    # ── 5. Per-query table ───────────────────────────────────
    print(f"\n  Per-query summary:")
    hdr = f"  {'#':<4} {'Status':<24} {'Iters':<7} {'Leaves%':<10} {'Time(s)':<9} Fail reason"
    print(hdr)
    print("  " + "─" * (W - 2))
    for r in results:
        lp   = (f"{min(100.0, 100.0*leaf_used(r)/total_leaves):.1f}%"
                if total_leaves else "N/A")
        fr   = r["fail_reason"] or "—"
        icon = "✅" if r["status"] == "PASS" else "❌"
        print(f"  {r['query_index']:<4} {icon} {r['status']:<22} "
              f"{r['iterations_used']:<7} {lp:<10} {r['elapsed_sec']:<9.1f} {fr}")

    print(f"\n{SEP}\n")


# ═══════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Batch evaluator for main2.py — runs all queries and "
                    "produces pass/fail + leaf-usage + iteration metrics."
    )
    ap.add_argument("rtl_file",
                    help="RTL Verilog file (.v / .sv) — same as main2.py arg")
    ap.add_argument("--queries",    default="queries.txt",
                    help="Path to queries.txt (default: queries.txt)")
    ap.add_argument("--tree-file",  default="spec_tree.json",
                    help="Serialized spec tree JSON (default: spec_tree.json)")
    ap.add_argument("--depth",      type=int, default=10,
                    help="BMC unroll depth passed to main2.py (default: 10)")
    ap.add_argument("--max-iterations", type=int, default=5,
                    help="Max LLM iterations passed to main2.py (default: 5)")
    ap.add_argument("--retrieval-k", type=int, default=3,
                    help="RETRIEVAL_K used inside main2.py (default: 3). "
                         "Needed for the leaf-usage metric.")
    ap.add_argument("--python",     default=sys.executable,
                    help="Python interpreter to invoke main2.py "
                         "(default: same interpreter running this script)")
    ap.add_argument("--save-report", default="",
                    help="If set, save the full JSON report to this path")
    ap.add_argument("--quiet", action="store_true",
                    help="Suppress per-query subprocess output "
                         "(summary table still printed)")

    args = ap.parse_args()

    # ── Load tree and count leaves ────────────────────────
    total_leaves = 0
    try:
        with open(args.tree_file, encoding="utf-8") as f:
            tree_data = json.load(f)
        total_leaves = count_leaves(tree_data)
        print(f"[info] Spec tree loaded — {total_leaves} leaf nodes detected.")
    except Exception as e:
        print(f"[warn] Could not load tree file for leaf count: {e}")

    # ── Parse queries ─────────────────────────────────────
    try:
        queries = parse_queries(args.queries)
    except FileNotFoundError:
        print(f"[!] queries file not found: {args.queries}")
        sys.exit(1)

    print(f"[info] {len(queries)} quer{'y' if len(queries)==1 else 'ies'} "
          f"found in '{args.queries}'.\n")

    all_records = []

    for idx, query in enumerate(queries, start=1):
        banner = f"  QUERY {idx} / {len(queries)}"
        print(f"\n{'═'*65}")
        print(banner)
        short = query[:100].replace("\n", " ")
        print(f"  {short}{'…' if len(query) > 100 else ''}")
        print(f"{'═'*65}")

        result_json, stdout, stderr, elapsed = run_one_query(
            query          = query,
            rtl_file       = args.rtl_file,
            tree_file      = args.tree_file,
            depth          = args.depth,
            max_iterations = args.max_iterations,
            python_bin     = args.python,
        )

        # Print subprocess output unless --quiet
        if not args.quiet:
            print(stdout)
            if stderr.strip():
                print(f"[STDERR]\n{stderr[:800]}")

        # Parse and classify
        sm       = parse_stdout(stdout)
        fr       = classify_fail(sm["status"], stdout, result_json)

        record = {
            "query_index"       : idx,
            "query_short"       : query[:200],
            "status"            : sm["status"],
            "iterations_used"   : sm["iterations_used"],
            "initial_leaves"    : sm["initial_leaves"],
            "extra_leaves_added": sm["extra_leaves_added"],
            "missing_signals"   : sm["missing_signals"],
            "has_syntax_warning": sm["has_syntax_warning"],
            "converged"         : sm["converged"],
            "fail_reason"       : fr,
            "elapsed_sec"       : round(elapsed, 1),
            "final_assertion"   : (result_json or {}).get(
                                      "generated_assertion_sv", ""),
        }
        all_records.append(record)

        icon = "✅ PASS" if record["status"] == "PASS" else f"❌ {record['status']}"
        print(f"\n  → {icon} | iters={record['iterations_used']} | "
              f"extra_leaves={record['extra_leaves_added']} | "
              f"time={elapsed:.1f}s")

    # ── Print summary report ──────────────────────────────
    print_report(all_records, total_leaves, args.retrieval_k)

    # ── Optionally save JSON report ───────────────────────
    if args.save_report:
        report = {
            "meta": {
                "rtl_file"      : args.rtl_file,
                "tree_file"     : args.tree_file,
                "total_leaves"  : total_leaves,
                "retrieval_k"   : args.retrieval_k,
                "depth"         : args.depth,
                "max_iterations": args.max_iterations,
            },
            "summary": {
                "total"  : len(all_records),
                "passed" : sum(1 for r in all_records if r["status"] == "PASS"),
                "failed" : sum(1 for r in all_records if r["status"] != "PASS"),
            },
            "queries": all_records,
        }
        with open(args.save_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[info] Full JSON report saved → {args.save_report}")


if __name__ == "__main__":
    main()