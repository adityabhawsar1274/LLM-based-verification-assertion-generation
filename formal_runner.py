# formal_runner.py
# ─────────────────────────────────────────────────────────
# Orchestrates the SymbiYosys formal verification step:
#   1. Write the generated .sv to a temp dir
#   2. Generate the .sby configuration file
#   3. Run sby
#   4. Parse PASS / FAIL / ERROR from output
#   5. On FAIL: read VCD counterexample, run Yosys COI, gap analysis
#   6. Return a standardised result dict (built by result_builder)
#
# KEY FIX: prep -top uses rtl_top (e.g. divider_top), NOT the checker
# module.  Using the checker as top makes the formal model standalone
# with free inputs, producing spurious PASSes.
# ─────────────────────────────────────────────────────────

import os
import re
import subprocess
import tempfile

from rtl_utils      import get_assertion_signals, run_yosys_coi
from result_builder import build_result, build_suggestions


# ═════════════════════════════════════════════════════════
#  sby output filtering
# ═════════════════════════════════════════════════════════

# Lines matching any of these patterns are suppressed from terminal output.
# The full log is still written to yosys_output.txt and stored in the result.
_SBY_SUPPRESS = re.compile(
    r"Removing directory"
    r"|Copy '[^']*' to '"       # file copy notifications
    r"|starting process"        # subprocess start lines
    r"|finished \(returncode=0\)"  # successful sub-process exits (keep rc=1)
    r"|Warning: Wire .* is used but has no driver"  # harmless undriven-wire warnings
    r"|Writing trace to"        # VCD/YW/SMT file writing notifications
    r"|Writing trace to Verilog"
    r"|Writing trace to Yosys"
    r"|Writing trace to constraints"
)


def _print_sby_summary(combined: str) -> None:
    """
    Print a filtered view of sby output:
      - Remove verbose infrastructure lines (file copies, sub-process starts, etc.)
      - Remove harmless Yosys warnings about undriven wires
      - Keep all BMC progress steps, PASS/FAIL status, error messages
    """
    filtered = [
        line for line in combined.splitlines()
        if not _SBY_SUPPRESS.search(line)
    ]
    print("\n".join(filtered))


# ═════════════════════════════════════════════════════════
#  sby file generation
# ═════════════════════════════════════════════════════════

def generate_sby(rtl_file, sv_path, rtl_top, depth, work_dir):
    """
    Write a SymbiYosys .sby configuration file.

    The prep -top directive is set to dynamic_checker.
    The bind statement inside the checker SV causes dynamic_checker
    to be instantiated inside the RTL, so Yosys sees the assertions
    in the context of the real RTL logic.
    """
    rtl_abs = os.path.abspath(rtl_file)
    sv_abs  = os.path.abspath(sv_path)
    content = (
        "[options]\n"
        "mode bmc\n"
        f"depth {depth}\n\n"
        "[engines]\n"
        "smtbmc z3\n\n"
        "[script]\n"
        f"read -formal {os.path.basename(rtl_abs)}\n"
        f"read -formal {os.path.basename(sv_abs)}\n"
        f"prep -top dynamic_checker\n\n"
        "[files]\n"
        f"{rtl_abs}\n"
        f"{sv_abs}\n"
    )
    path = os.path.join(work_dir, "check.sby")
    with open(path, "w") as f:
        f.write(content)
    with open("sby_output.txt", "w") as file:
        file.write(content)
    return path


# ═════════════════════════════════════════════════════════
#  sby execution
# ═════════════════════════════════════════════════════════

def run_sby(sby_path, work_dir):
    return subprocess.run(
        ["sby", "-f", sby_path],
        cwd=work_dir,
        capture_output=True,
        text=True,
    )


# ═════════════════════════════════════════════════════════
#  VCD counterexample reader
# ═════════════════════════════════════════════════════════

def _vcd_basename(name):
    return name.rsplit(".", 1)[-1]


def read_counterexample(vcd_path, known_signals=None):
    """
    Parse a Yosys VCD trace and return {signal_name: last_value}.
    Drops solver-internal names that start with $ or _.
    Returns {} if the file doesn't exist.
    """
    try:
        content = open(vcd_path).read()
    except FileNotFoundError:
        return {}

    aliases = {}
    for m in re.finditer(r"\$var\s+\w+\s+\d+\s+(\S+)\s+([\w.]+)", content):
        sym  = m.group(1)
        bare = _vcd_basename(m.group(2))
        if known_signals is None or bare in known_signals:
            aliases[sym] = bare

    state = {}
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if len(line) >= 2 and line[0] in "01xz" and " " not in line:
            name = aliases.get(line[1:])
            if name:
                state[name] = line[0]
        elif line.startswith("b") and " " in line:
            parts = line.split(None, 1)
            if len(parts) == 2:
                name = aliases.get(parts[1])
                if name:
                    state[name] = parts[0][1:]
        i += 1
    return state


# ═════════════════════════════════════════════════════════
#  Signal coverage logging
# ═════════════════════════════════════════════════════════

def _log_coverage(coi_data, asserted_outputs, per_output, missing):
    col = max((len(s) for s in coi_data), default=6)
    print(f"\n  {'Signal':<{col}}  {'Drives':^24}  {'Covered':^8}")
    print("  " + "─" * (col + 40))
    for sig in sorted(coi_data):
        drives = ", ".join(
            o for o in sorted(asserted_outputs)
            if sig in per_output.get(o, set())
        ) or "—"
        tick   = "✓" if sig not in missing else "✗ MISSING"
        print(f"  {sig:<{col}}  {drives:^24}  {tick}")


# ═════════════════════════════════════════════════════════
#  Yosys error extractor
# ═════════════════════════════════════════════════════════

def extract_yosys_errors(combined_output, sv_content=None):
    """
    Parse Yosys/sby stderr for structured error entries.
    Returns list of dicts with line number, message, source context.
    """
    errors = []
    pattern = re.compile(
        r"(\w+\.sv):(\d+):\s*(?:ERROR|error|WARNING|warning):\s*(.+)"
    )
    sv_lines = sv_content.splitlines() if sv_content else []

    for m in pattern.finditer(combined_output):
        lineno  = int(m.group(2))
        message = m.group(3).strip()
        entry   = {"file": m.group(1), "line": lineno, "message": message}

        if sv_lines:
            idx   = lineno - 1
            start = max(0, idx - 2)
            end   = min(len(sv_lines), idx + 3)
            entry["source_line"] = sv_lines[idx].strip() if 0 <= idx < len(sv_lines) else ""
            entry["context"]     = [
                f"  {'>>>' if i == idx else '   '} L{i+1}: {sv_lines[i]}"
                for i in range(start, end)
            ]
        errors.append(entry)

    return errors


# ═════════════════════════════════════════════════════════
#  Main formal check entry-point
# ═════════════════════════════════════════════════════════

def run_formal_check(
    rtl_file, assertion_sv, rtl_top, depth,
    all_rtl_sigs, rtl_inputs, rtl_outputs,
    data_inputs, clock_resets, iteration,
):
    """
    Write the assertion to a temp dir, run sby, and return a result dict.

    Returns one of:
      {"status": "PASS",                  ...}
      {"status": "FAIL_MISSING_SIGNALS",  ...}
      {"status": "FAIL_LOGIC_ERROR",      ...}
      {"status": "ERROR",                 ...}
    """
    work_dir = "/Users/hemanggautam/Desktop/fmsv_new/testfiles"
    os.makedirs(work_dir, exist_ok=True)

    sv_path = os.path.join(work_dir, "generated_checker.sv")
    with open(sv_path, "w") as f:
        f.write(assertion_sv)
    with open("assertion_output", "w") as f:
        f.write(assertion_sv)

    sby_path = generate_sby(rtl_file, sv_path, rtl_top, depth, work_dir)

    print(f"  [sby] Running SymbiYosys (depth={depth}, top={rtl_top})...")
    proc     = run_sby(sby_path, work_dir)
    combined = proc.stdout + proc.stderr

    # Always write the full log to disk for debugging
    with open("yosys_output.txt", "w") as file:
        file.write(combined)

    # Print filtered summary to terminal
    _print_sby_summary(combined)
    print("\n")

    # ── PASS ─────────────────────────────────────────────
    if "DONE (PASS" in combined:
        print("  ✅  PASS — all assertions hold")
        return build_result(
            status       = "PASS",
            iteration    = iteration,
            assertion_sv = assertion_sv,
            sby_log      = combined,
        )

    # ── Syntax / infrastructure error ────────────────────
    if "DONE (FAIL" not in combined:
        tail = "\n".join(combined.strip().splitlines()[-40:])
        print("  ⚠   SymbiYosys did not complete cleanly")

        yosys_errors = extract_yosys_errors(combined, sv_content=assertion_sv)

        if yosys_errors:
            err_summary_lines = ["Yosys parse errors in generated checker:"]
            for e in yosys_errors:
                err_summary_lines.append(f"  Line {e['line']}: {e['message']}")
                if "source_line" in e:
                    err_summary_lines.append(f"    Offending code: {e['source_line']}")
                if "context" in e:
                    err_summary_lines.append("    Context:")
                    err_summary_lines.extend(e["context"])
            err_summary = "\n".join(err_summary_lines)
        else:
            err_summary = f"SymbiYosys did not complete cleanly:\n{tail}"

        print(err_summary)
        return build_result(
            status       = "ERROR",
            iteration    = iteration,
            assertion_sv = assertion_sv,
            sby_log      = combined,
            error_msg    = err_summary,
            yosys_errors = yosys_errors,
            suggestions  = (
                "Fix the exact lines shown in 'yosys_errors'. "
                "Common causes: forbidden operators (-> |-> |=> ## $rose $fell), "
                "signal not declared as input port, or wrong bit-width. "
                "Replace any implication operator '->' with its boolean equivalent: "
                "(!antecedent || consequent)."
            ),
        )

    # ── Counterexample found ──────────────────────────────
    print("  ✗   FAIL — counterexample found")

    vcd_path = os.path.join(work_dir, "check", "engine_0", "trace.vcd")
    ce = read_counterexample(vcd_path, known_signals=all_rtl_sigs)
    if ce:
        print("  Counterexample signals:")
        for k, v in sorted(ce.items()):
            print(f"    {k:<22} = {v}")
    else:
        print("  (no signal values found in VCD)")

    # Signal coverage analysis
    strict_sigs, all_sigs = get_assertion_signals(assertion_sv)
    asserted_outputs = rtl_outputs & strict_sigs
    assert_inputs    = (all_sigs & data_inputs) - clock_resets

    if not asserted_outputs:
        return build_result(
            status       = "ERROR",
            iteration    = iteration,
            assertion_sv = assertion_sv,
            sby_log      = combined,
            counterexample = ce,
            error_msg    = "Generated assertion does not reference any RTL output port.",
            suggestions  = (
                f"The assert() expression must reference at least one of: "
                f"{sorted(rtl_outputs)}"
            ),
        )

    # COI analysis
    coi_union, per_output = run_yosys_coi(
        rtl_file, rtl_top, asserted_outputs, work_dir
    )
    coi_data = coi_union & data_inputs
    missing  = coi_data - assert_inputs

    print(f"\n  Asserted outputs : {sorted(asserted_outputs)}")
    print(f"  Covered inputs   : {sorted(assert_inputs)}")
    print(f"  COI union        : {sorted(coi_data)}")
    _log_coverage(coi_data, asserted_outputs, per_output, missing)

    status_str  = "FAIL_MISSING_SIGNALS" if missing else "FAIL_LOGIC_ERROR"
    suggestions = build_suggestions(
        missing, coi_data, assert_inputs, ce,
        assertion_sv, data_inputs, rtl_outputs
    )

    return build_result(
        status              = status_str,
        iteration           = iteration,
        assertion_sv        = assertion_sv,
        sby_log             = combined,
        asserted_outputs    = asserted_outputs,
        covered_signals     = assert_inputs,
        coi_signals         = coi_data,
        missing_signals_raw = missing,
        per_output_coi      = per_output,
        counterexample      = ce,
        suggestions         = suggestions,
        rtl_file            = rtl_file,
        data_inputs         = data_inputs,
        rtl_outputs         = rtl_outputs,
    )
