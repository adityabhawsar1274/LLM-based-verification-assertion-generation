#!/usr/bin/env python3
# main1.py
# ---------------------------------------------------------
# Builds the hierarchical similarity tree from the spec text
# and serializes it to a JSON file.
# ---------------------------------------------------------

import argparse
import json
import sys

from config import API_KEY, CHUNK_SIZE, CHUNK_OVERLAP
from spec_tree import SpecTreeBuilder

SAMPLE_SPEC = """\
PAGE 1 - Module Overview

The divider_top module is a pipelined integer divider that accepts a numerator
and denominator as inputs and computes the quotient. The module operates on the
rising edge of clk and is reset by an active-high synchronous reset signal rst.
The module exposes a valid_in signal which must be asserted for one cycle to
latch a new division request. The result appears on quotient after a fixed
pipeline latency.

PAGE 2 - Input/Output Interface

Inputs:
  - clk        : 1-bit clock signal. All logic is synchronous to posedge clk.
  - rst        : 1-bit synchronous active-high reset.
  - valid_in   : 1-bit. Assert for one cycle to submit a new division request.
  - num        : 16-bit unsigned numerator.
  - den        : 16-bit unsigned denominator.

Outputs:
  - quotient   : 16-bit unsigned result of num/den.
  - exc_flag   : 1-bit exception flag. Asserted when a divide-by-zero is detected.
  - valid_out  : 1-bit. Asserted when quotient and exc_flag carry valid results.

PAGE 3 - Pipeline Architecture

The divider is a 2-stage pipeline. Stage 1 latches the inputs and checks for
exceptional conditions (divide-by-zero). Stage 2 computes the quotient and
propagates the exception flag to the output. Due to this 2-stage structure,
results appear exactly 2 clock cycles after valid_in is asserted.
The stall signal may be asserted by downstream logic to freeze the pipeline for
one or more cycles. When stall is high, all pipeline registers hold their values
and no new inputs are accepted.

PAGE 4 - Divide-by-Zero Exception Handling

When den equals zero at the time valid_in is asserted, the module detects a
divide-by-zero condition. The exc_flag output is set to 1 exactly two clock
cycles after the cycle in which den==0 and valid_in==1. This two-cycle latency
is due to the pipeline: Stage 1 detects the condition, Stage 2 propagates it.
If a stall occurs between Stage 1 and Stage 2, the latency increases by the
number of stall cycles. In all cases, when exc_flag is asserted, the quotient
output is set to 0xFFFF (all ones) to indicate an invalid result.

PAGE 5 - Reset Behavior

On assertion of rst (synchronous, active-high), all pipeline registers are
cleared to zero within one clock cycle. exc_flag and valid_out are deasserted.
quotient is set to 0x0000. The module is ready to accept a new request on the
cycle immediately after rst is deasserted.

PAGE 6 - Stall Behavior

The stall input is driven by downstream logic to apply backpressure.
When stall==1 on any posedge clk, all pipeline stage registers freeze.
No new inputs are accepted while stall is asserted. valid_out is also
suppressed (held low) during stall cycles. The stall condition does not
affect the correctness of the exception flag — exc_flag will still be
asserted once the stall is released and the pipeline drains normally.

PAGE 7 - Formal Verification Requirements

The following properties must hold for all reachable states:
1. Safety: exc_flag must never be asserted unless den==0 was presented.
2. Timing: When den==0 and valid_in==1 and no stall occurs, exc_flag must
   be asserted exactly 2 cycles later.
3. Stall-tolerant timing: When a stall of duration S cycles occurs between
   Stage 1 and Stage 2, exc_flag must assert exactly 2+S cycles after
   the original valid_in.
4. Mutual exclusion: valid_out and exc_flag should not both be high unless
   the exception itself is the valid output for that transaction.
5. Reset guarantee: Within one cycle of rst, exc_flag must deassert.
"""

def main():
    ap = argparse.ArgumentParser(description="Generate and save Spec Tree to JSON.")
    ap.add_argument("--spec-file", default="", help="Path to the specification text file (default: built-in sample spec)")
    ap.add_argument("--output-tree", default="spec_tree.json", help="Output path for the serialized tree JSON")
    args = ap.parse_args()

    if args.spec_file:
        try:
            with open(args.spec_file, "r") as f:
                spec_text = f.read()
        except OSError as e:
            print(f"[!] Cannot read spec file: {e}")
            sys.exit(1)
    else:
        print("[info] No --spec-file given, using built-in sample spec.")
        spec_text = SAMPLE_SPEC

    print("\n[Phase 1] Building hierarchical spec tree...")
    builder = SpecTreeBuilder(
        api_key=API_KEY, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    
    root_node, _ = builder.build_tree(spec_text)

    print(f"\n[Phase 1] Saving tree to {args.output_tree}...")
    with open(args.output_tree, "w") as f:
        json.dump(root_node.to_dict(), f, indent=2)
    print("  Done.")

if __name__ == "__main__":
    main()