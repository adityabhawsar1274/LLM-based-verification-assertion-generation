"""
extract_spec.py  v2
-------------------
Extracts hardware-relevant specification text from a PDF (designed for
sockit_owm 1-wire master, but adaptable) and produces a clean .txt file
suitable as LLM context for RTL assertion generation.

Removes:
  • Table of Contents / Drawing Index / Abbreviations pages
  • License, References sections
  • Demo hardware / software / tool-flow details
  • File-manifest tables (not needed for assertions)
  • Bare page numbers, URL-only lines, figure captions
  • Garbled / doubled text from diagram rendering
  • Empty or near-empty table rows

Keeps:
  • Module parameters and their ranges
  • Port list with widths and directions
  • All timing constraints (min/max tables)
  • Register bit-field definitions
  • Address space map
  • Supported cycle table with timings
  • Driver access sequences (polling + interrupt pseudo-code)
  • Verilog port-level code snippets
  • RTL state-machine description

Usage:
    pip install pdfplumber
    python extract_spec.py [input.pdf] [output.txt]

Defaults:
    input  : sockit.pdf  (same directory)
    output : spec_clean.txt
"""

import sys
import re
import pdfplumber

# ── I/O ──────────────────────────────────────────────────────────────────────
INPUT_PDF  = sys.argv[1] if len(sys.argv) > 1 else "sockit.pdf"
OUTPUT_TXT = sys.argv[2] if len(sys.argv) > 2 else "spec_clean.txt"

# ── Sections to DROP (matched against heading text, numbers stripped) ────────
# Any numbered section whose title matches is skipped, including all children.
DROP_SECTIONS = [
    r"table of contents",
    r"index of tables",
    r"drawing index",
    r"abbreviations",
    r"terminology",
    r"list of source files",
    r"altera development tools",
    r"sopc builder",
    r"nios ii eds",
    r"demo hardware",
    r"demo software",
    r"software driver",
    r"port of public domain",
    r"adding support for new",
    r"possible improvements",
    r"testing todo",
    r"c driver tests",
    r"license",
    r"references",
]
_DROP_SECTION_RE = [re.compile(p, re.IGNORECASE) for p in DROP_SECTIONS]


# ── Individual lines that are always noise ───────────────────────────────────
DROP_LINES = [
    r"^\s*\d{1,3}\s*$",                   # lone page numbers (1-3 digits)
    r"^(http|ftp)s?://\S+$",              # bare URLs
    r"^copyright",
    r"^cc\s+by",
    r"^jean\s+j\.",
    r"iztok jeras",                        # author line on cover
    r"project home pages",
    r"^sockit_owm\s*$",                    # bare module-name on cover
    r"^1-wire \(onewire\) master\s*$",     # cover subtitle
    r"drawing\s+\d+\s*:",                  # "Drawing N: caption"
    # TOC numbered entries: "1.2 Foo bar....5"
    r"^\s*[\d.]+\s+.{5,}\.{4,}\s*\d+\s*$",
    # TOC unnumbered sub-entries: "Reset and presence....16"
    r"^[A-Z][^|].{5,}\.{4,}\s*\d+\s*$",
    # "Table of Contents" heading itself
    r"^table of contents\s*$",
]
_DROP_LINE_RE = [re.compile(p, re.IGNORECASE) for p in DROP_LINES]

# ── Table rows that are nearly empty (all cells blank or single char) ────────
_ALL_EMPTY_CELLS = re.compile(r"^\|(\s*\|)+\s*$")


def is_empty_table_row(line: str) -> bool:
    if not line.startswith("|"):
        return False
    cells = [c.strip() for c in line.strip("|").split("|")]
    non_empty = [c for c in cells if len(c) > 1]
    return len(non_empty) == 0


# ── Heading detection ────────────────────────────────────────────────────────
_HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.+)$")


def parse_heading(line: str):
    """Return (level, title_text) or None."""
    m = _HEADING_RE.match(line.strip())
    if not m:
        return None
    number = m.group(1)
    title  = m.group(2).strip()
    level  = number.count(".") + 1
    return level, title


def is_drop_heading(title: str) -> bool:
    clean = re.sub(r"[.]{3,}.*$", "", title).strip()   # strip TOC trailing dots
    for pat in _DROP_SECTION_RE:
        if pat.search(clean):
            return True
    return False


def _is_doubled_text(s: str) -> bool:
    """
    Detect lines where characters are rendered twice (diagram OCR artefact).
    E.g. "DDrraawwiinngg 67:: WWrriittee" or "88 77 66 55 44".
    Heuristic: if >50% of adjacent character pairs are identical letters/digits
    and the line has ≥6 characters, flag it.
    """
    alnum = re.sub(r"\s+", "", s)
    if len(alnum) < 6:
        return False
    pairs = sum(1 for i in range(len(alnum) - 1) if alnum[i] == alnum[i+1] and alnum[i].isalnum())
    return pairs / max(len(alnum) - 1, 1) > 0.45


def is_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    for pat in _DROP_LINE_RE:
        if pat.search(s):
            return True
    if is_empty_table_row(s):
        return True
    if _is_doubled_text(s):
        return True
    return False


# ── Table extraction ─────────────────────────────────────────────────────────

def extract_tables(page):
    """Return list of (top_y, pipe-delimited text) for this page."""
    result = []
    try:
        tables   = page.extract_tables()
        t_objs   = page.find_tables()
    except Exception:
        return result
    for obj, rows in zip(t_objs, tables):
        if not rows:
            continue
        lines = []
        for row in rows:
            cells = []
            for c in row:
                val = (c or "").replace("\n", " ").strip()
                cells.append(val)
            pipe_row = "| " + " | ".join(cells) + " |"
            # skip rows where every cell is empty
            if all(c == "" for c in cells):
                continue
            lines.append(pipe_row)
        if lines:
            result.append((obj.bbox[1], "\n".join(lines)))
    return result


def page_text_no_tables(page, table_tops_bots):
    """Extract prose text, excluding vertical bands occupied by tables."""
    try:
        if not table_tops_bots:
            return page.extract_text() or ""
        pw   = page.width
        ph   = page.height
        segs = []
        prev = 0
        for top, bot in sorted(table_tops_bots):
            if top > prev + 2:
                crop = page.within_bbox((0, prev, pw, top))
                segs.append(crop.extract_text() or "")
            prev = bot
        if prev < ph - 2:
            crop = page.within_bbox((0, prev, pw, ph))
            segs.append(crop.extract_text() or "")
        return "\n".join(s for s in segs if s)
    except Exception:
        return page.extract_text() or ""


# ── Section-skip state machine ───────────────────────────────────────────────

class SectionFilter:
    """Track which sections are dropped, respecting nesting."""

    def __init__(self):
        self.drop_level = None   # level of the outermost dropped heading

    def process_heading(self, level: int, title: str) -> bool:
        """
        Call when a heading is encountered.
        Returns True if this heading (and subsequent content) should be dropped.
        """
        if self.drop_level is not None:
            if level <= self.drop_level:
                # exiting the dropped section
                self.drop_level = None
            else:
                return True   # still inside dropped section

        if is_drop_heading(title):
            self.drop_level = level
            return True
        return False

    @property
    def skipping(self):
        return self.drop_level is not None


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(pdf_path: str) -> str:
    sf     = SectionFilter()
    blocks = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:

            # gather tables
            tbl_list = extract_tables(page)              # [(top_y, text), ...]
            tbl_bboxes = []
            for obj in (page.find_tables() if tbl_list else []):
                tbl_bboxes.append((obj.bbox[1], obj.bbox[3]))

            # gather prose
            prose = page_text_no_tables(page, tbl_bboxes)

            # interleave: prose first, then tables (tables keyed at bottom)
            items = []
            if prose:
                for i, ln in enumerate(prose.split("\n")):
                    items.append((i, ln, "prose"))
            for top_y, txt in tbl_list:
                items.append((top_y + 99999, txt, "table"))

            page_out = []
            for _, text, kind in sorted(items, key=lambda x: x[0]):
                for line in text.split("\n"):
                    s = line.strip()
                    if not s:
                        continue

                    # check for section heading
                    h = parse_heading(s)
                    if h:
                        level, title = h
                        dropped = sf.process_heading(level, title)
                        if dropped:
                            continue
                    elif sf.skipping:
                        continue

                    if is_noise_line(s):
                        continue

                    page_out.append(line)

            if page_out:
                blocks.append("\n".join(page_out))

    return "\n\n".join(blocks)


# ── Post-processing ───────────────────────────────────────────────────────────

def post_process(text: str) -> str:
    lines      = text.split("\n")
    out        = []
    blank_run  = 0
    prev_blank = False

    for line in lines:
        s = line.strip()

        if not s:
            blank_run += 1
            if blank_run == 1:
                out.append("")
            continue
        blank_run = 0

        # section dividers for readability
        if re.match(r"^\d+\s+[A-Z]", s):
            if out and out[-1] != "":
                out.append("")
            out.append("=" * 72)
        elif re.match(r"^\d+\.\d+\s+[A-Z]", s):
            if out and out[-1] != "":
                out.append("")
            out.append("-" * 48)

        out.append(line)

    return "\n".join(out).strip()


# ── LLM context header ────────────────────────────────────────────────────────

HEADER = """\
================================================================================
HARDWARE SPECIFICATION: sockit_owm — 1-wire (OneWire) Master
Source: sockit_owm documentation (CC BY-SA 3.0, Iztok Jeras)
Purpose: RTL assertion generation context
Content: module parameters · port list · timing constraints · register map ·
         address space · supported cycle table · driver access sequences
Removed: ToC, licenses, references, tool-flow, demo, file manifests
================================================================================

"""


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"[+] Reading  : {INPUT_PDF}")
    raw   = extract(INPUT_PDF)
    clean = post_process(raw)
    final = HEADER + clean

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final)

    lines  = final.count("\n")
    chars  = len(final)
    tokens = chars // 4
    print(f"[+] Written  : {OUTPUT_TXT}")
    print(f"[+] Lines    : {lines:,}")
    print(f"[+] Chars    : {chars:,}")
    print(f"[+] Est. tokens (~÷4): {tokens:,}")