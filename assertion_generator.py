# assertion_generator.py
# ─────────────────────────────────────────────────────────
# Responsible for:
#   1. Detecting whether the query expresses a positive or negative
#      requirement (polarity) so the LLM cannot confuse them.
#   2. Assembling the full system + user prompt (with port table,
#      polarity directive, previous sby log, counterexample hints).
#   3. Calling the LLM and parsing the structured JSON reply.
#   4. Pre-flight checks (forbidden operators, unmatched parens,
#      invented signal names, missing DUT instantiation).
# ─────────────────────────────────────────────────────────

import re
import json
import time
import os
import itertools

from llm_client import call_groq
from config    import GROQ_MODEL_LARGE, API_KEY


# ═════════════════════════════════════════════════════════
#  API Key Manager
# ═════════════════════════════════════════════════════════

def _load_api_keys():
    """Load API keys from API_list.txt or fallback to config API_KEY."""
    if os.path.exists("API_list.txt"):
        with open("API_list.txt", "r") as f:
            keys = [line.strip() for line in f if line.strip()]
            if keys:
                return keys
    return [API_KEY]

API_KEYS = _load_api_keys()
KEY_CYCLE = itertools.cycle(API_KEYS)


# ═════════════════════════════════════════════════════════
#  Parametric width resolution
# ═════════════════════════════════════════════════════════

PARAM_TO_CONCRETE: dict[str, int] = {
    "BDW": 32,   # [BDW-1:0] -> [31:0]  bus data width
    "OWN":  1,   # [OWN-1:0] -> [0:0]   1-wire port count
    "BAW":  1,   # [BAW-1:0] -> [0:0]   bus address width (=1 when BDW=32)
}

_PARAM_RE = re.compile(r"\[([A-Za-z_]\w*)-1:0\]")


def _resolve_width_expr(expr: str) -> str:
    def _replace(m):
        param = m.group(1)
        bits  = PARAM_TO_CONCRETE.get(param, 32)
        hi    = bits - 1
        return f"[{hi}:0]"
    return _PARAM_RE.sub(_replace, expr)


def concretize_port_table(port_table_str: str) -> str:
    lines = []
    for line in port_table_str.splitlines():
        lines.append(_resolve_width_expr(line))
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════
#  Query polarity detection
# ═════════════════════════════════════════════════════════

_NEGATIVE_PATTERNS = [
    r"does\s+not", r"do\s+not", r"must\s+not", r"should\s+not",
    r"shall\s+not", r"must\s+never", r"should\s+never",
    r"\bnever\b", r"doesn't", r"don't", r"cannot", r"can't",
    r"prohibit", r"prevent", r"ensure\s+.*\s+not", r"verify\s+.*\s+not",
    r"no\s+\w+\s+should", r"must\s+remain\s+(low|zero|deasserted|0)",
]

def detect_query_polarity(query):
    q = query.lower()
    for pat in _NEGATIVE_PATTERNS:
        m = re.search(pat, q)
        if m:
            return "NEGATIVE", m.group(0)
    return "POSITIVE", None


# ═════════════════════════════════════════════════════════
#  Context string builder
# ═════════════════════════════════════════════════════════

def build_context_string(retrieved_leaves):
    blocks = []
    for leaf in retrieved_leaves:
        b  = f"--- CHUNK ID: {leaf['node_id']} ---\n"
        b += f"SIGNALS EXTRACTED: {', '.join(leaf['signals'])}\n"
        b += f"TEXT:\n{leaf['text']}\n"
        blocks.append(b)
    return "\n".join(blocks)


# ═════════════════════════════════════════════════════════
#  System prompt template
# ═════════════════════════════════════════════════════════

_SYSTEM_PROMPT_TMPL = """\
You are an expert in formal hardware verification using SymbiYosys (sby) + Yosys.

{port_table}

━━━ QUERY POLARITY — READ CAREFULLY ━━━
{polarity_block}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ABSOLUTE PROHIBITION: THE → IMPLICATION OPERATOR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEVER use  ->  anywhere — in assert(), assume(), if(), or any expression.
It is NOT supported by Yosys and ALWAYS produces a parse error.

  WRONG  (Yosys parse error):  assert(rst == 1 -> owr_p == 0);
  CORRECT (boolean NAND form):  assert(!(rst == 1) || (owr_p == 0));

  WRONG:   assert(A -> B);
  CORRECT: assert(!A || B);

  General rule:   assert(ANTECEDENT -> CONSEQUENT)
  Becomes:        assert(!ANTECEDENT || CONSEQUENT)

Use an if/else guard instead whenever the antecedent is complex:
  PREFERRED:
    if (ANTECEDENT) begin
        assert(CONSEQUENT);
    end
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━ YOSYS-COMPATIBLE SYNTAX RULES ━━━
ALLOWED inside always @(posedge clk):
  assert( <boolean_expr> );
  assume( <boolean_expr> );
  $past(signal, N)   — look back N clock cycles
  $initstate         — true only at time-step 0; use ONLY inside assume() for reset bootstrapping
  if/else guards and reg counters

FORBIDDEN — these cause Yosys parse errors, do NOT use any of them:
  assert property(...)   |->   |=>   ##N   [*N]   [->N]
  ->  (Verilog implication operator — use if/else or (!a || b) instead)
  property...endproperty    sequence...endsequence
  $rose()   $fell()   $stable()   disable iff(...)
  @(posedge clk) inside an assert or assume expression
  SystemVerilog `bind` keyword (FORBIDDEN)
  Wildcard auto-connect `.*` (FORBIDDEN)

━━━ PORT WIDTH RULES — CRITICAL ━━━
Every signal declaration MUST include its exact bit-width from the port table above.
Never omit the width.  Never use a parameter name; the port table already shows
resolved concrete ranges.

  CORRECT:  input wire [7:0]  bus_adr,
            input wire [31:0] bus_wdt,
            input wire [0:0]  bus_wen,
            wire       [31:0] bus_rdt;

  WRONG:    input wire bus_adr,          // missing width → Yosys infers 1-bit
            input wire [BAW-1:0] bus_adr // parameter name not resolved
            
━━━ $past() INDEXING — HARD SYNTAX RULE ━━━
ALWAYS move bit-selects INSIDE the $past() argument. Indexing the return value
of a system function is a Yosys parse error.
 
  WRONG  (Yosys parse error): assert(out[7] == $past(in, 1)[7]);
  CORRECT:                     assert(out[7] == $past(in[7], 1));
 
  WRONG  (Yosys parse error): assert(out[3:0] == $past(in, 2)[3:0]);
  CORRECT:                     assert(out[3:0] == $past(in[3:0], 2));
 
This applies to ALL bit-selects and part-selects after $past().
 
━━━ MODULE STRUCTURE RULES (STRICT) ━━━
When generating the verification wrapper, follow these steps in order.
Deviating from this structure causes Yosys failures.
 
1. Write all input signals in the module port list as `input wire`. Do not write `output wire` here.
2. Write all output signals as `wire` declarations inside the module body.
3. Instantiate the Design Under Test as `arbiter DUT (...)`, connecting
   ALL ports from the port table — inputs and outputs. This line is MANDATORY.
4. Write the formal verification body (`always @(posedge clk)` block with
   assumes/asserts) following the syntax rules above.
 
Example Structure:
module dynamic_checker (
    // 1. all input signals
    input wire clk,
    input wire rst,
    ... all other input signals
);
    // 2. output signals as wires
    wire out_a;
    wire [31:0] out_b;
 
    // 3. DUT instantiation — REQUIRED, must appear before the always block
    arbiter DUT (
        .clk(clk), .rst(rst), ... all ports from the port table
    );
 
    // 4. verification body
    reg initialized = 0;
    always @(posedge clk) begin
        initialized <= 1;
        assume(rst == !initialized);
 
        if (initialized) begin
            ... assertions here
        end
    end
 
endmodule



━━━ OUTPUT FORMAT ━━━
ALL string values must be wrapped in double quotes. The "assertion_sv" value MUST be a single, flat string with all newlines properly escaped as \\n. Do NOT output raw multi-line strings.
{{
  "assertion_sv":   "module dynamic_checker (\\n ...",
  "signals_used":   ["clk", "rst", "bus_wen", "bus_adr", "bus_irq"],
  "pipeline_depth": <int>,
  "reasoning":      "<one paragraph explaining the assertion logic and polarity>"
}}
"""

_POLARITY_POSITIVE = """\
The query expresses a POSITIVE requirement — something MUST HAPPEN.
→ Generate an assertion that FAILS when the described event does NOT occur.
Example: "ensure exc_flag is set" → assert(exc_flag == 1)\
"""

_POLARITY_NEGATIVE = """\
⚠  The query expresses a NEGATIVE requirement — something MUST NOT HAPPEN.
    Detected negation phrase: "{phrase}"
→ Generate an assertion that FAILS when the PROHIBITED event DOES occur.
Example: "ensure exc_flag is NOT set" → assert(exc_flag == 0)

CRITICAL WARNING:
  The specification context below describes the normal (positive) design behavior.
  You MUST IGNORE what the spec says should happen and instead STRICTLY follow
  the query's negation.  The query overrides the spec for polarity.\
"""


# ═════════════════════════════════════════════════════════
#  Prompt assembly
# ═════════════════════════════════════════════════════════

def _make_implication_fix(source_line: str) -> str:
    """
    If source_line contains a simple assert(A -> B) pattern, return the
    exact corrected version assert(!A || B).  Returns "" if not parseable.
    """
    stripped = source_line.strip()
    # Match: assert( <antecedent> -> <consequent> );
    m = re.match(
        r'assert\s*\(\s*(.+?)\s*->\s*(.+?)\s*\)\s*;?\s*$',
        stripped, re.DOTALL
    )
    if m:
        ante = m.group(1).strip()
        cons = m.group(2).strip()
        return f"assert(!({ante}) || ({cons}));"
    return ""


def build_generation_prompt(query, context_str, port_table_str,
                            rtl_top, prev_result=None):
    polarity, phrase = detect_query_polarity(query)
 
    if polarity == "NEGATIVE":
        polarity_block = _POLARITY_NEGATIVE.format(phrase=phrase)
    else:
        polarity_block = _POLARITY_POSITIVE
 
    system = _SYSTEM_PROMPT_TMPL.format(
        port_table     = port_table_str,
        polarity_block = polarity_block,
        rtl_top        = rtl_top,
    )
 
    # ── Detect DUT missing from previous result ──────────
    # Hard pre-flight stores the message in the "error" field; check both
    # that and the "hard_errors" list if present.
    dut_was_missing = False
    if prev_result:
        error_str       = prev_result.get("error", "")
        hard_errors_prev = prev_result.get("hard_errors", [])
        dut_was_missing  = (
            "Missing DUT" in error_str
            or any("Missing DUT" in (e or "") for e in hard_errors_prev)
        )
 
    # ── Pin DUT reminder to top of user message ──────────
    # Appears FIRST, before the query, so the LLM cannot miss it.
    dut_pin = ""
    if dut_was_missing:
        dut_pin = (
            f"⛔ MANDATORY FIX — DO NOT SKIP:\n"
            f"Your previous checker was MISSING the arbiter DUT instantiation.\n"
            f"The very first statement inside the module body MUST be:\n"
            f"arbiter DUT (.clk(clk), .rst(rst), /* all ports from port table */);\n"
            f"A checker without the arbiter DUT proves nothing — outputs are free variables.\n\n"
        )
 
    # ── Build hints from previous iteration ─────────────
    hints = ""
    if prev_result:
        status  = prev_result.get("status", "")
        missing = prev_result.get("missing_signals", [])
        suggest = prev_result.get("suggestions_for_next_iteration", "")
        sby_log = prev_result.get("sby_log", "")
 
        yosys_errors = prev_result.get("yosys_errors", [])
        if yosys_errors:
            hints += "\n### Yosys / Pre-flight Errors in Your Previous Assertion\n"
            hints += "You MUST fix EVERY item below before regenerating:\n\n"
            for e in yosys_errors:
                hints += f"  Line {e['line']}: {e['message']}\n"
                if "source_line" in e:
                    hints += f"  Offending code: `{e['source_line']}`\n"
                    if "->" in e["source_line"]:
                        fix = _make_implication_fix(e["source_line"])
                        if fix:
                            hints += f"  ✅ EXACT FIX: `{fix}`\n"
                        else:
                            hints += (
                                "  ✅ RULE: replace  assert(A -> B)  with  assert(!A || B)\n"
                                "           OR use:  if (A) begin assert(B); end\n"
                            )
                if "context" in e:
                    hints += "  Context:\n"
                    hints += "\n".join(f"    {c}" for c in e["context"]) + "\n"
                hints += "\n"
            hints += (
                "CRITICAL FIX RULES:\n"
                "  1. NEVER use '->' in any context. It is always a Yosys parse error.\n"
                "     Replace  assert(A -> B)  with  assert(!A || B)\n"
                "     Or use   if (A) begin assert(B); end\n"
                "  2. Do NOT use |-> |=> ## $rose $fell $stable assert property.\n"
                "  3. NEVER index $past() return value: write $past(sig[N], cycles) "
                "not $past(sig, cycles)[N].\n"
                "  4. Only use: assert(<bool>), assume(<bool>), $past(sig,N), "
                "if/else, reg counters, inside always @(posedge clk).\n"
            )
 
        # DUT missing hint (after the pinned header above, also insert here
        # so it appears in the hints section alongside other issues)
        if dut_was_missing:
            hints += f"\n### Missing DUT Instantiation\n"
            hints += (
                f"⛔ '{rtl_top} DUT (...)' was absent from your previous checker.\n"
                f"Without it, all output wires are free unconstrained variables and\n"
                f"the checker trivially proves (or disproves) nothing meaningful.\n"
                f"You MUST add it as the first statement inside the module body,\n"
                f"connecting EVERY port listed in the port table.\n"
            )
 
        elif sby_log and sby_log.strip() and not yosys_errors:
            log_lines   = sby_log.strip().splitlines()
            error_lines = [l for l in log_lines if any(
                kw in l for kw in ("ERROR", "error", "WARNING", "FAILED", "syntax")
            )]
            tail_lines  = log_lines[-30:]
            combined    = list(dict.fromkeys(error_lines + tail_lines))
            hints += "\n### SymbiYosys / Yosys Error Log (previous run)\n"
            hints += "\n".join(combined[:50]) + "\n"
            hints += "Action: read the above errors and fix the syntax or logic issues.\n"
 
        if missing:
            names = [s["name"] for s in missing]
            hints += "\n### Missing Signals from Previous Attempt\n"
            for s in missing:
                hints += f"  - {s['name']}: {s.get('role', '')}\n"
                hints += f"    fix: {s.get('fix_hint', '')}\n"
            hints += (
                f"\nAction required: add ALL of {names} as input wire ports "
                f"in dynamic_checker and connect them in the DUT instantiation.\n"
            )
 
        if status == "FAIL_LOGIC_ERROR":
            ce = prev_result.get("counterexample", {})
            if ce:
                hints += "\n### Counterexample Values (previous run)\n"
                for k, v in sorted(ce.items()):
                    hints += f"  {k} = {v}\n"
                hints += (
                    "\nAction: trace why these values violate the assertion. "
                    "Common causes:\n"
                    "  1. Antecedent is too broad — you must include BOTH the "
                    "strobe/enable signal AND the correct address or select "
                    "qualifier in your $past() condition. Consult the "
                    "specification context to determine the exact qualifying "
                    "conditions for the register or operation you are targeting.\n"
                    "  2. Wrong $past() depth N — count pipeline stages from the spec.\n"
                    "  3. Missing stall or cycle-in-progress guard.\n"
                )
 
        if suggest:
            hints += f"\n### Improvement Hints\n{suggest}\n"
 
    user_msg = (
        dut_pin
        + f"### Verification Query\n{query}\n\n"
        + f"### Specification Context\n{context_str}\n"
        + (hints if hints else "\n### Previous Run\nFirst iteration — no previous result.\n")
        + "\nGenerate the checker now. Respond ONLY with the JSON object."
    )
 
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_msg},
    ]



# ═════════════════════════════════════════════════════════
#  Response parsing
# ═════════════════════════════════════════════════════════

def parse_llm_response(llm_text):
    clean = re.sub(r"```[a-zA-Z]*\n?", "", llm_text)
    clean = clean.replace("```", "").strip()

    m = re.search(r"\{[\s\S]+\}", clean)
    if not m:
        raise ValueError(f"No JSON object in LLM reply:\n{llm_text[:500]}")

    json_str = m.group(0)
    try:
        return json.loads(json_str, strict=False)
    except json.JSONDecodeError as e:
        print(f"\n[!] JSON Parse Error: {e}")
        print(f"--- RAW LLM OUTPUT ---\n{json_str}\n----------------------\n")
        raise


# ═════════════════════════════════════════════════════════
#  Pre-flight checks
# ═════════════════════════════════════════════════════════

# Forbidden operators that cause Yosys parse errors
_FORBIDDEN_OPS = re.compile(
    r"\bassert\s+property\b"
    r"|\|\->"                       # SVA overlapping implication
    r"|\|\=>"                       # SVA non-overlapping implication
    r"|(?<![|!&=<>])\->"            # standalone -> implication (not preceded by operator chars)
    r"|##\d|\[\*\d|\[\-\>\d"
    r"|\$rose\b|\$fell\b|\$stable\b|\bdisable\s+iff\b",
    re.IGNORECASE,
)

_ALWAYS_VALID = {
    "clk", "clock", "rst", "reset", "rst_n", "reset_n", "arst", "arst_n",
}


def _strip_comments(text: str) -> str:
    """Remove // line comments and /* block comments */ from SV text."""
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def validate_sv_syntax(sv_text: str, rtl_top: str = None):
    """
    Pre-flight syntax and structural check.

    Returns a tuple:
        hard_errors : list[str]  — must block execution (forbidden ops, missing DUT)
        soft_warns  : list[str]  — advisory only
    """
    hard_errors = []
    soft_warns  = []

    clean = _strip_comments(sv_text)

    # ── Hard check 1: forbidden SVA operators ─────────────────────────
    for m in _FORBIDDEN_OPS.finditer(clean):
        op = m.group(0)
        hard_errors.append(f"Forbidden Yosys operator: '{op}' — Yosys will reject this file.")

    # ── Hard check 2: DUT instantiation ───────────────────────────────
    if rtl_top:
        # Look for: <rtl_top> <identifier> (
        dut_pattern = re.compile(
            rf"\b{re.escape(rtl_top)}\s+\w+\s*\(", re.MULTILINE
        )
        if not dut_pattern.search(clean):
            hard_errors.append(
                f"Missing DUT instantiation: '{rtl_top} DUT (...)' not found. "
                "Without the DUT, output signals are free unconstrained variables "
                "and the checker verifies nothing meaningful. "
                f"You MUST include: {rtl_top} DUT (.clk(clk), .rst(rst), ...all ports...);"
            )

    # ── Soft check 1: reset bootstrap ─────────────────────────────────
    has_initialized = bool(re.search(r"\breg\s+initialized\b", clean))
    has_initstate   = bool(re.search(r"\$initstate\b", clean))
    if not has_initialized and not has_initstate:
        soft_warns.append(
            "No reset bootstrap detected. Add either:\n"
            "  reg initialized = 0; always @(posedge clk) { initialized <= 1; "
            "assume(rst == !initialized); }\n"
            "  OR: assume($initstate -> rst == 1'b1);"
        )

    # ── Soft check 2: assert must be inside an initialized / !rst guard ──
    if re.search(r"\bassert\s*\(", clean):
        has_init_guard = bool(re.search(r"if\s*\(\s*initialized\b", clean))
        has_rst_guard  = bool(re.search(r"if\s*\(\s*!", clean))
        if not has_init_guard and not has_rst_guard:
            soft_warns.append(
                "assert() found but no 'if (initialized)' or 'if (!rst)' guard detected. "
                "All assertions should be guarded to avoid spurious failures at reset."
            )

    # ── Soft check 3: literal RTL-top placeholder ─────────────────────
    if re.search(r"bind\s+(<\w+>|MY_RTL_TOP)\b", clean):
        soft_warns.append(
            "bind statement contains a placeholder instead of the real RTL module name."
        )

    # ── Soft check 4: unmatched parentheses ───────────────────────────
    depth = 0
    for ch in clean:
        if ch == "(":   depth += 1
        elif ch == ")": depth -= 1
        if depth < 0:
            soft_warns.append("Unmatched closing parenthesis.")
            break
    if depth > 0:
        soft_warns.append(f"Unmatched opening parenthesis (depth={depth} at end).")

    return hard_errors, soft_warns


def preflight_signal_check(signals_used, all_rtl_sigs):
    return [
        s for s in signals_used
        if s not in all_rtl_sigs and s.lower() not in _ALWAYS_VALID
    ]


def fix_rtl_top_placeholder(sv_text, rtl_top):
    sv_text = re.sub(r"bind\s+<[^>]+>",      f"bind {rtl_top}", sv_text)
    sv_text = re.sub(r"bind\s+MY_RTL_TOP\b",  f"bind {rtl_top}", sv_text)
    return sv_text


# ═════════════════════════════════════════════════════════
#  Main generation entry-point
# ═════════════════════════════════════════════════════════

def generate_assertion(api_key, query, context_str, port_table_str,
                       rtl_top, prev_result=None):
    """
    Full generation pipeline:
      build prompt → call LLM → parse JSON → fix placeholders → pre-flight

    Returns
    -------
    dict with keys:
      assertion_sv      : the complete .sv string ready for Yosys
      signals_used      : list of signal names the LLM declared
      pipeline_depth    : int
      reasoning         : LLM explanation string
      hard_errors       : list of blocking pre-flight errors (forbidden ops, missing DUT)
      soft_warnings     : list of advisory pre-flight warnings
    """
    messages = build_generation_prompt(
        query, context_str, port_table_str, prev_result
    )

    max_attempts = len(API_KEYS) * 2
    sleep_time   = 10
    raw          = None

    for attempt in range(max_attempts):
        current_api_key = next(KEY_CYCLE)
        print(f"  [LLM] Calling {GROQ_MODEL_LARGE} (Attempt {attempt+1}/{max_attempts})...")
        try:
            raw = call_groq(current_api_key, messages, model=GROQ_MODEL_LARGE, max_tokens=1800)
            break
        except Exception as e:
            print(f"  [!] API call failed: {e}")
            if attempt < max_attempts - 1:
                print(f"  [!] Halting for {sleep_time} seconds before trying the next key...")
                time.sleep(sleep_time)
            else:
                raise RuntimeError("All API call attempts failed across available keys.") from e

    parsed = parse_llm_response(raw)

    assertion_sv = parsed.get("assertion_sv", "").strip()
    if not assertion_sv:
        raise ValueError("LLM returned empty assertion_sv field")

    assertion_sv = fix_rtl_top_placeholder(assertion_sv, rtl_top)
    parsed["assertion_sv"] = assertion_sv

    # Pre-flight: hard errors block sby; soft warns are advisory
    hard_errors, soft_warns = validate_sv_syntax(assertion_sv, rtl_top=rtl_top)
    parsed["hard_errors"]   = hard_errors
    parsed["soft_warnings"] = soft_warns

    # Keep legacy key for any downstream code that reads syntax_warnings
    parsed["syntax_warnings"] = hard_errors + soft_warns

    return parsed
