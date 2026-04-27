# result_builder.py
# ─────────────────────────────────────────────────────────
# Builds the structured result JSON for each pipeline iteration
# and generates human-readable improvement suggestions.
# ─────────────────────────────────────────────────────────

import re


# ═════════════════════════════════════════════════════════
#  Signal role + fix hints (RTL file analysis, no LLM)
# ═════════════════════════════════════════════════════════

def describe_signal_role(sig, rtl_file):
    """
    Infer a human-readable role for a signal by scanning the RTL source.
    Returns a short description string.
    """
    try:
        lines = open(rtl_file).readlines()
    except OSError:
        return "unknown role (RTL file not readable)"

    hits = [
        (i + 1, ln.rstrip())
        for i, ln in enumerate(lines)
        if re.search(rf"\b{re.escape(sig)}\b", ln)
    ]
    if not hits:
        return "drives output(s) through RTL logic"

    for _, ln in hits:
        if re.search(r"\binput\b", ln):
            if re.search(r"stall", ln, re.I):
                return (
                    "pipeline stall control — when high, pipeline registers do not "
                    "advance and output timing is extended by one cycle per stall"
                )
            if re.search(r"valid", ln, re.I):
                return "data-valid qualifier — marks whether current-cycle inputs are meaningful"
            if re.search(r"\ben\b|enable", ln, re.I):
                return "enable — gates pipeline progression"
            return "input port — directly controls datapath or pipeline behaviour"
        if re.search(r"\boutput\b", ln):
            return "output port of the RTL"

    return (
        "internal/input signal driving the output via sequential/combinational logic "
        f"(RTL lines: {', '.join(str(l) for l, _ in hits[:5])})"
    )


def make_fix_hint(sig, sv_text, data_inputs, rtl_outputs):
    """
    Generate a concrete fix hint for a signal missing from the assertion.
    """
    port_match = re.search(r"module\s+\w+\s*\(([^)]+)\)", sv_text or "", re.DOTALL)
    port_list  = port_match.group(1) if port_match else ""
    already    = bool(re.search(rf"\b{re.escape(sig)}\b", port_list))

    parts = []
    if not already:
        parts.append(
            f"Add 'input wire {sig}' to the dynamic_checker port list "
            f"and connect it in the DUT instantiation as '.{sig}({sig})'"
        )
    parts.append(
        f"Constrain '{sig}' in the checker: either 'assume(!{sig})' to simplify "
        f"timing or guard the assert with 'if (!{sig})' to skip stalled cycles"
    )
    return "; ".join(parts)


# ═════════════════════════════════════════════════════════
#  Improvement suggestions (pure logic, no LLM)
# ═════════════════════════════════════════════════════════

def build_suggestions(missing, coi_data, assert_inputs, counterexample,
                      assertion_sv, data_inputs, rtl_outputs):
    """
    Build a free-text suggestions string for the next LLM iteration.

    Parameters
    ----------
    missing        : set of COI signal names absent from the assertion
    coi_data       : full COI signal set (data inputs only)
    assert_inputs  : signals currently covered by the assertion
    counterexample : dict of {signal: value} from VCD
    assertion_sv   : the generated SV text
    data_inputs    : set of RTL data inputs (no clock/reset)
    rtl_outputs    : set of RTL output names
    """
    parts = []

    if missing:
        names = sorted(missing)
        parts.append(
            f"MISSING_SIGNALS {names}: these signals are in the cone-of-influence "
            f"of the asserted output(s) but are absent from the assertion. "
            f"The spec retrieval should fetch context for {names}. "
            f"Each must be added as an input port and constrained."
        )
        for sig in names:
            hint = make_fix_hint(sig, assertion_sv, data_inputs, rtl_outputs)
            parts.append(f"  [{sig}] fix: {hint}")

    elif counterexample:
        active   = [k for k, v in counterexample.items() if v == "1"]
        inactive = [k for k, v in counterexample.items() if v == "0"]
        parts.append(
            "LOGIC_TIMING_ERROR: all COI signals are covered yet the assertion fails. "
            f"Counterexample: active={active}, inactive={inactive}. "
            "Check: (1) pipeline depth N in $past(x,N) — count stages from spec; "
            "(2) per-stage stall guards — each stage needs its own !$past(stall,k); "
            "(3) the cycle guard (cyc >= N) must be large enough for $past history."
        )

    return "\n".join(parts)


# ═════════════════════════════════════════════════════════
#  Result dict builder
# ═════════════════════════════════════════════════════════

def build_result(
    status, iteration, assertion_sv, sby_log="",
    asserted_outputs=None, covered_signals=None, coi_signals=None,
    missing_signals_raw=None, per_output_coi=None,
    counterexample=None, suggestions=None,
    error_msg=None, rtl_file=None,
    data_inputs=None, rtl_outputs=None,
    yosys_errors=None,
):
    """
    Construct the standardised result dict for one pipeline iteration.

    Status values
    -------------
    PASS                  assertion holds for all reachable states
    FAIL_MISSING_SIGNALS  COI signals absent from the assertion
    FAIL_LOGIC_ERROR      all COI signals present but assertion fails (bad timing/logic)
    ERROR                 sby/syntax/infrastructure error
    """
    missing_list = []
    if missing_signals_raw and rtl_file and per_output_coi and asserted_outputs:
        for sig in sorted(missing_signals_raw):
            in_coi = sorted(
                o for o in asserted_outputs
                if sig in per_output_coi.get(o, set())
            )
            missing_list.append({
                "name"     : sig,
                "role"     : describe_signal_role(sig, rtl_file),
                "in_coi_of": in_coi,
                "fix_hint" : make_fix_hint(
                    sig,
                    assertion_sv or "",
                    data_inputs  or set(),
                    rtl_outputs  or set(),
                ),
            })

    result = {
        "status"                : status,
        "iteration"             : iteration,
        "generated_assertion_sv": assertion_sv or "",
        "sby_log"               : sby_log,
    }

    if asserted_outputs is not None:
        result["asserted_outputs"] = sorted(asserted_outputs)
    if covered_signals is not None:
        result["covered_signals"]  = sorted(covered_signals)
    if coi_signals is not None:
        result["coi_signals"]      = sorted(coi_signals)

    result["missing_signals"] = missing_list

    if counterexample is not None:
        result["counterexample"] = counterexample
    if yosys_errors is not None:
        result["yosys_errors"] = yosys_errors
    if suggestions:
        result["suggestions_for_next_iteration"] = suggestions
    if error_msg:
        result["error"] = error_msg

    return result
