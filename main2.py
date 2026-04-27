#!/usr/bin/env python3
# main2.py  (pipeline entry-point — uses pre-built spec tree)
# ---------------------------------------------------------
# Pipeline stages:
#   1. Load hierarchical similarity tree from JSON
#   2. Retrieve the most relevant spec chunks for the query
#   3. Loop (up to --max-iterations):
#        a. LLM generates a Yosys-compatible SystemVerilog assertion
#        b. Hard pre-flight: check forbidden operators + DUT instantiation
#        c. Signal pre-flight: validate signals against RTL port list
#        d. SymbiYosys formal verification
#        e. If PASS  → done
#        f. If FAIL  → analyse, extend context if needed, retry
# ---------------------------------------------------------

import argparse
import json
import re
import sys

from config import (API_KEY, BMC_DEPTH, MAX_ITERATIONS,
                    RETRIEVAL_K, RETRIEVAL_THRESHOLD, PARAM_TO_CONCRETE)
from spec_tree import TreeNode
from tree_retriever import HierarchicalTreeRetriever
from rtl_utils import (get_rtl_ports, get_rtl_port_widths,
                       build_port_table, find_modules)
from assertion_generator import (build_context_string, generate_assertion,
                                 preflight_signal_check)
from formal_runner import run_formal_check

SAMPLE_QUERY = (
    "Generate an assertion to ensure that when valid_in is asserted "
    "and den equals zero, exc_flag is set exactly 2 clock cycles later "
    "(assuming no stall cycles occur)."
)

def run_pipeline(rtl_file, user_query, tree_file, depth, max_iterations, output_path):

    print("=" * 60)
    print("  LLM-BASED SVA GENERATION PIPELINE")
    print("=" * 60)
    print(f"  RTL file   : {rtl_file}")
    print(f"  Query      : {user_query[:80]}")
    print(f"  Tree file  : {tree_file}")
    print(f"  Depth      : {depth}    Max iterations : {max_iterations}")
    print("=" * 60)

    # -- Phase 1: Load spec tree -------------------------------------------
    print("\n[Phase 1] Loading hierarchical spec tree...")
    try:
        with open(tree_file, "r") as f:
            tree_data = json.load(f)
        root_node = TreeNode.from_dict(tree_data)
        print("  Tree loaded successfully.")
    except Exception as e:
        print(f"[!] Failed to load tree from {tree_file}: {e}")
        sys.exit(1)

    # -- Phase 2: Retrieve relevant leaves ---------------------------------
    print("\n[Phase 2] Retrieving relevant context...")
    retriever = HierarchicalTreeRetriever()
    top_leaves = retriever.retrieve_leaves(
        root_node, query=user_query,
        k=RETRIEVAL_K, threshold=RETRIEVAL_THRESHOLD,
    )
    context_str = build_context_string(top_leaves)

    # -- Phase 3: RTL static analysis --------------------------------------
    print("\n[Phase 3] Analysing RTL ports...")
    rtl_inputs, rtl_outputs = get_rtl_ports(rtl_file)
    width_map = get_rtl_port_widths(rtl_file, param_overrides=PARAM_TO_CONCRETE)
    port_table_str          = build_port_table(rtl_inputs, rtl_outputs, width_map)

    rtl_mods = find_modules(rtl_file)
    if not rtl_mods:
        print("[!] Cannot detect RTL top module name. Exiting.")
        sys.exit(1)
    rtl_top = rtl_mods[0]

    all_rtl_sigs = rtl_inputs | rtl_outputs
    clock_resets = {
        s for s in rtl_inputs
        if re.search(r"^(clk|clock|rst|reset|rst_n|reset_n|arst|arst_n)$", s, re.I)
    }
    data_inputs = rtl_inputs - clock_resets

    print(f"  RTL top    : {rtl_top}")
    print(f"  Inputs     : {sorted(rtl_inputs)}")
    print(f"  Outputs    : {sorted(rtl_outputs)}")
    print(f"  Data inputs: {sorted(data_inputs)}")

    # -- Phase 4: Iteration loop -------------------------------------------
    prev_result  = None
    final_result = None

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'-'*60}")
        print(f"  ITERATION {iteration} / {max_iterations}")
        print(f"{'-'*60}")

        # -- Step 4a: LLM assertion generation ----------------------------
        print("\n[Step 4a] Generating assertion with LLM...")
        try:
            llm_result = generate_assertion(
                api_key        = API_KEY,
                query          = user_query,
                context_str    = context_str,
                port_table_str = port_table_str,
                rtl_top        = rtl_top,
                prev_result    = prev_result,
            )
            print(port_table_str)
        except Exception as e:
            print(f"[!] LLM call failed: {e}")
            break

        assertion_sv   = llm_result["assertion_sv"]
        signals_used   = llm_result.get("signals_used", [])
        pipeline_depth = llm_result.get("pipeline_depth", "?")
        reasoning      = llm_result.get("reasoning", "")
        hard_errors    = llm_result.get("hard_errors", [])
        soft_warns     = llm_result.get("soft_warnings", [])

        print(f"  Assertion generated : {assertion_sv}")
        print(f"  Pipeline depth : {pipeline_depth}")
        print(f"  Signals used   : {signals_used}")
        print(f"  Reasoning      : {reasoning}")

        # -- Step 4b: Hard pre-flight (forbidden ops + missing DUT) -------
        if hard_errors:
            print(f"\n[Step 4b] ⛔ Hard pre-flight FAILED — blocking sby run:")
            for e in hard_errors:
                print(f"  • {e}")

            # Package errors as Yosys-style error dicts so the feedback
            # hints section in build_generation_prompt can display them
            # with source-line context where possible.
            yosys_style_errors = []
            for err_msg in hard_errors:
                # Try to find the offending line in the assertion for better feedback
                offending_line = ""
                if "Forbidden" in err_msg:
                    # Extract the operator from the message and find it in code
                    op_match = re.search(r"'([^']+)'", err_msg)
                    if op_match:
                        op = op_match.group(1)
                        for lineno, line in enumerate(assertion_sv.splitlines(), 1):
                            if op in line:
                                offending_line = line.strip()
                                yosys_style_errors.append({
                                    "line":        lineno,
                                    "message":     err_msg,
                                    "source_line": offending_line,
                                })
                                break
                if not offending_line:
                    yosys_style_errors.append({"line": 0, "message": err_msg})

            prev_result = {
                "status"                    : "ERROR",
                "iteration"                 : iteration,
                "generated_assertion_sv"    : assertion_sv,
                "sby_log"                   : "",
                "error"                     : "\n".join(hard_errors),
                "yosys_errors"              : yosys_style_errors,
                "soft_warnings"             : soft_warns,
                "suggestions_for_next_iteration": (
                    "Fix ALL hard pre-flight errors listed above. "
                    "Common causes: using '->' implication operator (replace with !A || B), "
                    "or missing DUT instantiation (you must include "
                    f"'{rtl_top} DUT (.clk(clk), .rst(rst), ...all ports...);')."
                ),
            }
            final_result = prev_result
            continue

        # Soft warnings: print but do not block
        if soft_warns:
            print(f"\n[Step 4b] ⚠  Soft pre-flight warnings (non-blocking):")
            for w in soft_warns:
                print(f"  • {w[:120]}")

        # -- Step 4c: Signal pre-flight ------------------------------------
        print("\n[Step 4c] Pre-flight signal validation...")
        bad = preflight_signal_check(signals_used, all_rtl_sigs)
        if bad:
            print(f"  [!] LLM invented signals not in RTL: {bad}")
            prev_result = {
                "status"    : "FAIL_MISSING_SIGNALS",
                "iteration" : iteration,
                "generated_assertion_sv": assertion_sv,
                "sby_log"   : "",
                "soft_warnings": soft_warns,
                "missing_signals": [
                    {
                        "name"      : s,
                        "role"      : "signal not found in RTL port list",
                        "in_coi_of" : [],
                        "fix_hint"  : (
                            f"'{s}' does not exist. Valid data inputs: "
                            f"{sorted(data_inputs)}. Valid outputs: "
                            f"{sorted(rtl_outputs)}."
                        ),
                    }
                    for s in sorted(bad)
                ],
                "suggestions_for_next_iteration": (
                    f"Invented signals {sorted(bad)} — use only ports from the "
                    f"RTL Port-Signal Table."
                ),
            }
            final_result = prev_result
            continue

        print("  All signals validated ✓")

        # -- Step 4d: Formal verification ----------------------------------
        print("\n[Step 4d] Running SymbiYosys formal check...")
        result = run_formal_check(
            rtl_file     = rtl_file,
            assertion_sv = assertion_sv,
            rtl_top      = rtl_top,
            depth        = depth,
            all_rtl_sigs = all_rtl_sigs,
            rtl_inputs   = rtl_inputs,
            rtl_outputs  = rtl_outputs,
            data_inputs  = data_inputs,
            clock_resets = clock_resets,
            iteration    = iteration,
        )

        status = result.get("status", "ERROR")
        print(f"\n  Status : {status}")
        final_result = result

        # -- PASS → done --------------------------------------------------
        if status == "PASS":
            print("\n" + "=" * 60)
            print("  ✅  ASSERTION VERIFIED — PIPELINE COMPLETE")
            print("=" * 60)
            print("\nFinal Assertion:\n")
            print(result["generated_assertion_sv"])
            break

        # -- FAIL_MISSING_SIGNALS → re-retrieve for missing signals -------
        if status == "FAIL_MISSING_SIGNALS":
            missing_names = [s["name"] for s in result.get("missing_signals", [])]
            if missing_names:
                print(f"\n  [Context] Re-retrieving spec for missing signals: {missing_names}")
                extra_query  = (
                    f"Signals {', '.join(missing_names)}: their role, "
                    f"behavior and effect on the pipeline timing"
                )
                extra_leaves = retriever.retrieve_leaves(
                    root_node, query=extra_query, k=5, threshold=0.25
                )
                if extra_leaves:
                    extra_ctx    = build_context_string(extra_leaves)
                    context_str  = (
                        context_str
                        + "\n\n--- ADDITIONAL CONTEXT (missing signals) ---\n"
                        + extra_ctx
                    )
                    print(f"  [Context] Added {len(extra_leaves)} extra leaf(ves)")

        prev_result = result

    else:
        print(f"\n[!] Pipeline did not converge after {max_iterations} iterations.")

    # -- Write output JSON -------------------------------------------------
    if final_result and output_path:
        save = {k: v for k, v in final_result.items() if k != "sby_log"}
        with open(output_path, "w") as f:
            json.dump(save, f, indent=2)
        print(f"\n  Result written to: {output_path}")
    elif final_result:
        save = {k: v for k, v in final_result.items() if k != "sby_log"}
        print("\n--- Final Result (JSON) ---")
        print(json.dumps(save, indent=2))

    return final_result


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="LLM-based SVA generation using cached spec tree.")
    ap.add_argument("rtl_file", help="Path to the RTL Verilog file (.v / .sv)")
    ap.add_argument("--query", default=SAMPLE_QUERY, help="Natural-language verification requirement")
    ap.add_argument("--tree-file", default="spec_tree.json", help="Path to the serialized JSON tree file")
    ap.add_argument("--depth", type=int, default=BMC_DEPTH, help=f"SymbiYosys BMC unroll depth (default: {BMC_DEPTH})")
    ap.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS, help=f"Maximum LLM refinement iterations (default: {MAX_ITERATIONS})")
    ap.add_argument("--output", default="", help="Write final result JSON to this file")

    args = ap.parse_args()

    run_pipeline(
        rtl_file       = args.rtl_file,
        user_query     = args.query,
        tree_file      = args.tree_file,
        depth          = args.depth,
        max_iterations = args.max_iterations,
        output_path    = args.output,
    )


if __name__ == "__main__":
    main()
