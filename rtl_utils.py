# rtl_utils.py
# ─────────────────────────────────────────────────────────
# All static analysis utilities that operate on RTL source files
# and generated SystemVerilog.  No LLM calls here.
# ─────────────────────────────────────────────────────────

import os
import re
import subprocess


# ═════════════════════════════════════════════════════════
#  RTL file parsing
# ═════════════════════════════════════════════════════════

def find_modules(source, is_path=True):
    """Return all module names declared in a file or raw string."""
    text = open(source).read() if is_path else source
    text = re.sub(r"//[^\n]*", "", text)
    return re.findall(r"\bmodule\s+(\w+)\s*[#(;]", text)


def get_rtl_ports(rtl_file):
    """Return (inputs: set, outputs: set) of port signal names."""
    with open(rtl_file) as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    inputs, outputs = set(), set()
    for m in re.finditer(
        r"\b(input|output)\b\s*"
        r"(?:wire|reg|logic|tri)?\s*"
        r"(?:signed|unsigned)?\s*"
        r"(?:\[[^\]]+\]\s*)?(\w+)",
        text,
    ):
        (inputs if m.group(1) == "input" else outputs).add(m.group(2))
    return inputs, outputs


# ═════════════════════════════════════════════════════════
#  Parameter extraction & width resolution  (NEW)
# ═════════════════════════════════════════════════════════

def extract_parameters(rtl_file, extra_overrides=None):
    """
    Parse all `parameter` / `localparam` integer declarations from an RTL
    file and return them as {name: int}.

    extra_overrides: optional {name: int} that takes precedence over
    anything found in the file.  Pass PARAM_TO_CONCRETE from config.py
    here when parameters are set at the instantiation level and therefore
    not visible inside the module body itself.
    """
    with open(rtl_file) as f:
        text = f.read()
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    params = {}
    pat = re.compile(
        r"\b(?:parameter|localparam)\b"
        r"(?:\s+(?:integer|int|logic|wire|reg|bit))?"   # optional type keyword
        r"(?:\s*\[[^\]]+\])?"                            # optional packed range
        r"\s+(\w+)\s*=\s*([^,;)]+)",
    )

    # First pass: collect simple integer literals
    for m in pat.finditer(text):
        name    = m.group(1).strip()
        val_str = m.group(2).strip()
        try:
            params[name] = int(val_str, 0)
        except ValueError:
            pass

    # Second pass: resolve params whose value references another already-known param
    changed = True
    while changed:
        changed = False
        for m in pat.finditer(text):
            name    = m.group(1).strip()
            val_str = m.group(2).strip()
            if name in params:
                continue
            substituted = val_str
            for p, v in params.items():
                substituted = re.sub(r"\b" + re.escape(p) + r"\b",
                                     str(v), substituted)
            if re.fullmatch(r"[\d\s\+\-\*\/\(\)\%\&\|\^~<>]+", substituted):
                try:
                    params[name] = int(eval(substituted))
                    changed = True
                except Exception:
                    pass

    if extra_overrides:
        params.update(extra_overrides)

    return params


def _eval_expr(expr, param_map):
    """
    Substitute known parameter names into an arithmetic expression and
    evaluate it to a concrete integer string.
    Returns the original expression unchanged if resolution fails.
    Longest names are substituted first to avoid partial-name collisions.
    """
    substituted = expr
    for name in sorted(param_map, key=len, reverse=True):
        substituted = re.sub(r"\b" + re.escape(name) + r"\b",
                             str(param_map[name]), substituted)
    # Guard: only call eval when the string is pure arithmetic
    if re.fullmatch(r"[\d\s\+\-\*\/\(\)\%\&\|\^~<>]+", substituted):
        try:
            return str(int(eval(substituted)))
        except Exception:
            pass
    return expr  # unresolvable — return as-is


def resolve_width_str(width_str, param_map):
    """
    Resolve a parameterised width string to a concrete one using param_map.

    Examples:
      '[BDW-1:0]'  + {'BDW': 32}  ->  '[31:0]'
      '[OWN-1:0]'  + {'OWN':  1}  ->  '[0:0]'
      '[BAW-1:0]'  + {'BAW':  1}  ->  '[0:0]'
      '[0:0]'      + {}           ->  '[0:0]'   (already concrete, unchanged)
    """
    if not param_map:
        return width_str
    m = re.match(r"\[\s*([^\]:]+?)\s*:\s*([^\]]+?)\s*\]", width_str.strip())
    if not m:
        return width_str
    hi = _eval_expr(m.group(1), param_map)
    lo = _eval_expr(m.group(2), param_map)
    return f"[{hi}:{lo}]"


# ═════════════════════════════════════════════════════════
#  Port-width map
# ═════════════════════════════════════════════════════════

def get_rtl_port_widths(rtl_file, param_overrides=None):
    """
    Return {signal_name: (direction, width_str)} for all ports.

    Width strings are resolved to concrete bit-ranges, e.g.:
      '[BAW-1:0]' -> '[0:0]'
      '[BDW-1:0]' -> '[31:0]'

    param_overrides: optional {str: int} — pass PARAM_TO_CONCRETE from
    config.py for parameters defined outside the module (at the
    instantiation site) that are not visible in the RTL file body.
    """
    with open(rtl_file) as f:
        text = f.read()
    text_clean = re.sub(r"//[^\n]*", "", text)
    text_clean = re.sub(r"/\*.*?\*/", "", text_clean, flags=re.DOTALL)

    param_map = extract_parameters(rtl_file, extra_overrides=param_overrides)

    widths = {}
    for m in re.finditer(
        r"\b(input|output)\b\s*"
        r"(?:wire|reg|logic|tri)?\s*"
        r"(?:signed|unsigned)?\s*"
        r"(\[[^\]]+\])?\s*(\w+)",
        text_clean,
    ):
        direction = m.group(1)
        raw_width = (m.group(2) or "").strip() or "[0:0]"
        name      = m.group(3)
        widths[name] = (direction, resolve_width_str(raw_width, param_map))

    return widths


def build_port_table(rtl_inputs, rtl_outputs, width_map):
    """
    Build a compact plain-text port-signal table.
    This is the ONLY RTL-derived information sent to the LLM.
    Raw RTL logic is NEVER included.
    Widths are always concrete (e.g. [31:0]) so the LLM copies them
    correctly into the generated dynamic_checker module ports.
    """
    lines = [
        "RTL Port-Signal Table (auto-extracted — no RTL logic shown):",
        f"  {'Signal':<22} {'Dir':<8} Width",
        "  " + "─" * 44,
    ]
    for sig in sorted(rtl_inputs):
        _, w = width_map.get(sig, ("input", "[0:0]"))
        lines.append(f"  {sig:<22} input    {w}")
    for sig in sorted(rtl_outputs):
        _, w = width_map.get(sig, ("output", "[0:0]"))
        lines.append(f"  {sig:<22} output   {w}")
    lines += [
        "",
        "CONSTRAINT: Every signal in assert() / assume() MUST appear in this table.",
        "Do NOT invent signal names that are not listed above.",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════
#  SV assertion text analysis
# ═════════════════════════════════════════════════════════

_SV_KW = {
    "posedge", "negedge", "always", "always_ff", "always_comb",
    "if", "else", "begin", "end", "case", "endcase", "default",
    "module", "endmodule", "input", "output", "inout", "wire",
    "reg", "logic", "bit", "int", "integer", "byte", "assign",
    "initial", "parameter", "localparam", "genvar", "assert",
    "assume", "cover", "property", "sequence", "endproperty",
    "endsequence", "disable", "iff", "throughout", "within",
    "first_match", "and", "or", "not", "true", "false", "clk",
    "clock", "rst", "reset", "rst_n", "reset_n", "bind",
    "generate", "endgenerate", "for",
}


def get_assertion_signals(sv_text):
    """
    Analyse generated SV text and return:
      strict_sigs : signals referenced inside assert() / property bodies
      all_sigs    : all identifiers in the module body (excluding declaration)
    Both are sets of bare identifiers with SV keywords removed.
    """
    text = re.sub(r"//[^\n]*", "", sv_text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    strict = set()
    for body in re.findall(r"\bassert\s*\(([^;]+)\)", text, re.DOTALL):
        strict.update(re.findall(r"\b([a-zA-Z_]\w*)\b", body))
    for body in re.findall(
        r"\bproperty\s+\w+\s*;(.+?)\bendproperty\b", text, re.DOTALL
    ):
        strict.update(re.findall(r"\b([a-zA-Z_]\w*)\b", body))

    body_text = re.sub(r"\bmodule\s+\w+[\s\S]*?;", "", text)
    all_sigs  = set(re.findall(r"\b([a-zA-Z_]\w*)\b", body_text))

    strict   = {s for s in strict   - _SV_KW if not s.isdigit()}
    all_sigs = {s for s in all_sigs - _SV_KW if not s.isdigit()}
    return strict, all_sigs


# ═════════════════════════════════════════════════════════
#  Yosys Cone of Influence  (backwards reachability)
# ═════════════════════════════════════════════════════════

def run_yosys_coi(rtl_file, rtl_top, asserted_outputs, work_dir):
    """
    Use Yosys to compute the cone-of-influence (backward from each output).
    Returns (coi_union: set, per_output: dict[output → set]).
    Silently returns empty sets if Yosys is unavailable.
    """
    if not asserted_outputs:
        return set(), {}

    rtl_abs    = os.path.abspath(rtl_file)
    per_output = {}

    for out in sorted(asserted_outputs):
        coi_file = os.path.join(work_dir, f"coi_{out}.txt")
        ys_script = (
            f"read -formal {rtl_abs}\n"
            f"hierarchy -top {rtl_top}\n"
            f"proc; opt; flatten\n"
            f"select -set coi_{out} o:{out} %ci*\n"
            f"select -write {coi_file} @coi_{out}\n"
            f"exit\n"
        )
        ys_path = os.path.join(work_dir, f"coi_{out}.ys")
        with open(ys_path, "w") as f:
            f.write(ys_script)

        subprocess.run(
            ["yosys", "-s", ys_path],
            capture_output=True, text=True
        )

        sigs = set()
        if os.path.isfile(coi_file):
            for raw in open(coi_file):
                m = re.match(r"^\S+/(\w+)$", raw.strip())
                if m and not m.group(1).startswith(("$", "_")):
                    sigs.add(m.group(1))
        per_output[out] = sigs

    union = set().union(*per_output.values()) if per_output else set()
    return union, per_output