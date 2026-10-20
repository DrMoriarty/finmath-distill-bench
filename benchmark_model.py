#!/usr/bin/env python3
"""Benchmark a chat-completions LLM endpoint against the finmath-distill datasets.

Usage:
    python3 benchmark_model.py --host <host[:port]> --token <token> --model <model> [options]

Required arguments:
    --host HOST        Endpoint host, optionally with port (e.g. localhost:8000
                       or http://host:8000/v1)
    --token TOKEN      Access token (sent as Bearer token)
    --model MODEL      Model name to request

Options:
    -c, --concurrency N    Number of parallel requests (default: 1)
    -d, --datasets LIST    Comma-separated dataset names: enum,formula,math,yesno
                           or "all" (default: all)
    -n, --limit N          Max questions per dataset; random sample if smaller
                           than dataset size (default: 1000)
        --seed S           Random seed for the sampling (default: none)
        --timeout SECONDS  Per-request timeout in seconds (default: 120)
        --retries N        Attempts per question on transient network/HTTP
                           errors (429, 5xx, connection drops, timeouts);
                           after the last attempt the question is skipped and
                           excluded from the accuracy denominator (default: 10)
        --max-tokens N     Max tokens for model responses (default: 2048)
        --temperature T    Sampling temperature sent to the endpoint
                           (default: 0)
    -v, --verbose          Print every mismatch, not only the first few

Grading rules:
    enum    order-insensitive set comparison of the selected items
            (case/punctuation/extra-whitespace tolerant);
    formula symbolic comparison via sympy (LaTeX -> sympy expression, then
            numeric sampling + simplify), falling back to normalized string
            comparison when parsing is impossible; no char-by-char match;
    math    numeric comparison of the final "Answer: <number>" against the
            gold answer with a small relative tolerance (percent signs and
            multi-value gold answers are handled);
    yesno   case-insensitive yes/no extraction, whitespace-insensitive.

The math dataset stores newlines inside prompts as literal "\\n" sequences;
they are unescaped before being sent to the model (real LaTeX commands such
as \\nu / \\neq are preserved).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import keyword
import random
import re
import sys
import time
from pathlib import Path

import warnings

import aiohttp
import numpy as np
import sympy

# Parsing LaTeX like "\text{Cov}(R_p, R_b)" (unknown function with several
# arguments) makes implicit_multiplication build a Mul with a Tuple argument,
# and "\not"-style tokens inside \sqrt{...} similarly produce a Pow with a Not
# argument. The resulting expressions are still correct; silence the spurious
# sympy deprecation noise instead of mangling the formulas.
warnings.filterwarnings(
    "ignore",
    category=sympy.utilities.exceptions.SymPyDeprecationWarning,
    # warning text starts with a newline, hence the (?s).* prefix
    message=r"(?s).*Using non-Expr arguments in (Mul|Pow) is deprecated",
)
from sympy.parsing.sympy_parser import (
    implicit_multiplication,
    parse_expr,
    standard_transformations,
)

ROOT = Path(__file__).resolve().parent
DATASET_NAMES = ("enum", "formula", "math", "yesno")

# Optional system prompt sent along with each question, per dataset.
SYSTEM_PROMPTS = {
    "enum": (
        "You are taking a multiple-select test. Reply with the correct "
        "options only, listed exactly as they appear in the question and "
        "separated by commas. Do not add explanations, comments or any "
        "text formatting (no markdown, bullets, numbering or bold)."
    ),
}
MAX_TOKENS_DEFAULT = 2048
MATH_REL_TOL = 5e-3  # gold answers are rounded, allow ~0.5% relative error

# --------------------------------------------------------------------------
# Prompt preparation
# --------------------------------------------------------------------------

# LaTeX commands starting with \n that must NOT be turned into newlines when
# unescaping the math dataset.
_LATEX_N_COMMANDS = {
    "nu", "neq", "ne", "nabla", "neg", "not", "notin", "newline",
    "nonumber", "nolimits", "notag", "normal",
}


def unescape_math_text(text: str) -> str:
    """Replace literal backslash-n newline placeholders with real newlines.

    Real LaTeX commands starting with \\n (e.g. \\nu, \\neq) are preserved.
    """
    if "\\n" not in text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "\\" and i + 1 < n and text[i + 1] == "n":
            k = i + 2
            while k < n and (text[k].isalnum() or text[k] == "_"):
                k += 1
            word = text[i + 1:k]  # full command name, starting at the 'n'
            if word and word in _LATEX_N_COMMANDS:
                out.append("\\n")  # genuine LaTeX command, keep it
            else:
                out.append("\n")
            i += 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def build_prompt(item: dict) -> str:
    content = item["messages"][0]["content"]
    if item.get("type") == "math":
        content = unescape_math_text(content)
    return content


# --------------------------------------------------------------------------
# Graders
# --------------------------------------------------------------------------

_NUM_RE = r"[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?"


def _strip_grouping_commas(token: str) -> str:
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", token)


def parse_numbers(text: str) -> list[float]:
    """Extract all plausible numbers from text ('200,000' -> 200000)."""
    out = []
    for tok in re.findall(_NUM_RE, text):
        try:
            out.append(float(_strip_grouping_commas(tok)))
        except ValueError:
            continue
    return out


# ------------------------------- enum -------------------------------------

# Check/cross marks models put in front of (or around) list items.
_MARK_RE = re.compile(r"[\u2705\u2611\u2714\u2713\u274c\u2717\u2718\u2716\uFE0F]")
_SELECT_MARK_RE = re.compile(r"[\u2705\u2611\u2714\u2713]")
_REJECT_MARK_RE = re.compile(r"[\u274c\u2717\u2718\u2716]")

# A line that looks like a list entry: "- item", "* item", "• item", "•item",
# "1. item", "2) item", "✅ item" (a bullet may precede the mark). A bare "*"
# followed by a non-space (i.e. markdown bold "**text**") is NOT a list marker.
_LIST_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[-\u2022\u00b7\u2013\u2014>+]\s*\S"
    r"|\*\s+\S"
    r"|\d{1,3}[.)]\s+\S"
    r"|[\u2705\u2611\u2714\u2713\u274c\u2717\u2718\u2716]"
    r")"
)

# Leading bullets / numbering of a single list item ("1. ", "- ", "-- ", "> ").
# A numbering marker must be followed by whitespace (so "1.5x", "10-K",
# "401(k)", "2%", "-MPC", "10.5%" stay intact).
_BULLET_PREFIX_RE = re.compile(
    r"^(?:\s*(?:[-\u2022\u00b7\u2013\u2014>+]+|\d{1,3}[.)])(?=\s|$))+\s*"
)

# Lead-in before the answer body. "Answers questions" (no is/are/colon) must
# NOT match — gold items can contain the word "answers".
_ANSWER_TAIL_RE = re.compile(
    r"\banswers?\b\s*(?:is|are\b)?\s*[:\-]\s*(.*)"
    r"|\banswers?\s+(?:is|are)\b\s*[:\-]?\s*(.*)"
    r"|^\s*answers?\b\s*[:\-]?\s*(.*)",
    re.IGNORECASE | re.DOTALL,
)

# Inline numbered lists written without commas: "The answer is 1. a 2. b".
# Lookahead keeps the digits out of the match, so a skipped "…16)" marker does
# not swallow the whitespace that precedes the next real marker. Applied per
# comma/newline chunk with paren-depth awareness, so that a trailing
# "…(pre-IFRS 16)" or "…, 0)" is not mistaken for a list marker.
_NUM_MARKER_RE = re.compile(r"\s+(?=\d{1,3}[.)]\s+)")
_BULLETISH_RE = re.compile(r"[\s\-*\u2022\u00b7\u2013\u2014>]*")


def _split_numbered(part: str) -> list[str]:
    """Split "1. a 2. b" style chunks at top-level numbering markers.
    Markers inside unclosed parentheses and the chunk's own leading
    numbering ("- 1. a ...", "1. a ...") are not split points."""
    if not _NUM_MARKER_RE.search(part):
        return [part]
    parts, last, scanned, depth = [], 0, 0, 0
    for m in _NUM_MARKER_RE.finditer(part):
        depth += part[scanned:m.start()].count("(") - part[scanned:m.start()].count(")")
        scanned = m.end()
        seg = part[last:m.start()]
        if depth != 0 or _BULLETISH_RE.fullmatch(seg):
            continue  # inside parentheses, or the marker numbers the chunk itself
        parts.append(part[last:m.start()])
        last = m.end()
    parts.append(part[last:])
    return parts

# Sentences that start these ways are commentary about the answer, not items.
_COMMENTARY_RE = re.compile(
    r"^(?:note\b|nb\b|explanation\b|therefore\b|thus\b|hence\b|in summary\b|"
    r"to summarize\b|this (?:is|means|shows|implies)\b|"
    r"these (?:are|represent|options|selections)\b|that (?:is|means)\b|"
    r"respectively\b|the (?:other|remaining)\b|let me\b|i (?:hope|can)\b|"
    r"feel free\b)\b",
    re.IGNORECASE,
)

_PROMPT_OPTIONS_RE = re.compile(r"\boptions?\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)

# "all of the above"-style answers: resolve them against the option list from
# the prompt. The lookbehind keeps "not all of the above" from matching.
_ALL_OPTIONS_RE = re.compile(
    r"(?<!\bnot\s)\ball of the (?:above|listed|following)\b"
    r"|^\W*(?:all|both|each)\b[^.\n]*\b(?:above|listed|following|options?|choices?)\b"
    r"|^\W*both\b[.\s]*$",
    re.IGNORECASE | re.DOTALL,
)
_NONE_OPTIONS_RE = re.compile(
    r"\b(?:none|neither)\b[^.\n]*\b(?:above|listed|following|options?|choices?)\b",
    re.IGNORECASE,
)


def _clean_enum_item(item: str) -> str:
    s = _MARK_RE.sub("", item)
    s = s.strip().lower()
    # markdown emphasis first, so that "**1. item**" exposes the numbering
    # to the bullet stripper below; a second pass handles "- **1. item**"
    # (the "**" between the dash and the number blocks the first pass)
    s = s.strip(" \t\r\n\"'`*_~")
    s = _BULLET_PREFIX_RE.sub("", s)  # bullets / numbering
    s = s.strip(" \t\r\n\"'`*_~")
    s = _BULLET_PREFIX_RE.sub("", s)
    s = re.sub(r"^only\s+", "", s)  # "only capital raising" == "capital raising"
    s = s.strip(" \t\r\n\"'`*_~")
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".").strip()
    return s


def _answer_body(text: str) -> str:
    """Strip the prose around the answer: take everything after the last
    "answer is/are:" style lead-in, if the reply contains one."""
    t = text.strip()
    body = None
    for m in _ANSWER_TAIL_RE.finditer(t):
        body = m.group(1) or m.group(2) or m.group(3)
    return body if body is not None else t


def _marked_answer_lines(text: str) -> list[str] | None:
    """Lines of a bulleted/numbered/check-marked list, if the reply has one.

    Headers ("**Correct selections:**"), introductory sentences and trailing
    commentary are not part of the list and are dropped. Explicitly rejected
    options ("❌ item") are dropped as well. Returns None when the reply does
    not contain a recognizable list."""
    t = _answer_body(text)
    marked = [ln for ln in t.splitlines() if _LIST_LINE_RE.match(ln)]
    if not marked:
        return None
    kept = [ln for ln in marked
            if not (_REJECT_MARK_RE.search(ln) and not _SELECT_MARK_RE.search(ln))]
    return kept or None


def parse_enum_answer(text: str, split_and: bool = False,
                      expand_single: bool = True) -> set[str]:
    t = _answer_body(text)
    marked = _marked_answer_lines(text)
    if marked:
        # The reply contains a list: only list lines are the answer, anything
        # before/after them is formatting or commentary.
        t = "\n".join(marked)
    t = t.strip(" \t\r\n\"'`*")
    parts = [p for p in re.split(r"[;\n]+|\s*,\s*(?![^()]*\))", t) if p.strip()]
    parts = [q for p in parts for q in _split_numbered(p) if q.strip()]
    if split_and:
        # "a, b and c" -> ["a", "b", "c"]; only comma-free chunks are split so
        # that items legitimately containing "and" are not torn apart.
        parts = [
            piece
            for p in parts
            for piece in (
                re.split(r"\s+and\s+", p, flags=re.IGNORECASE)
                if "," not in p and re.search(r"\sand\s", p, re.IGNORECASE)
                else [p]
            )
        ]
    elif expand_single and len(parts) == 1 and "," not in parts[0]:
        parts = re.split(r"\s+and\s+", parts[0], flags=re.IGNORECASE)
    out = set()
    for p in parts:
        c = _clean_enum_item(p)
        # Gold items never end with a colon, so "...:" parts are headers like
        # "Correct selections:" / "The following factors:", not answers.
        if c and not c.endswith(":"):
            out.add(c)
    return out


def _extract_prompt_options(prompt: str) -> list[str]:
    """The enumerated choices from the prompt ("... Options: a, b, c.")."""
    matches = list(_PROMPT_OPTIONS_RE.finditer(prompt))
    if not matches:
        return []
    tail = matches[-1].group(1).strip().rstrip(".")
    parts = re.split(r"[;\n]+|\s*,\s*(?![^()]*\))", tail)
    return [c for p in parts if (c := _clean_enum_item(p))]


def _resolve_all_of_above(response: str, options: list[str]) -> str | None:
    """Resolve "all of the above" / "none of the above" / "all except X"
    style replies into an explicit list of options. None = nothing to resolve.
    """
    if not options:
        return None
    if _ALL_OPTIONS_RE.search(response):
        m = re.search(r"\bexcept\b|\bapart from\b|\bother than\b", response, re.IGNORECASE)
        if m:
            excl_clean = _clean_enum_item(response[m.end():])
            excluded = set()
            if excl_clean:
                for chunk in re.split(r"\s+and\s+", excl_clean, flags=re.IGNORECASE):
                    excluded.update(o for o in options if o in chunk or chunk in o)
            if excluded:
                return ", ".join(o for o in options if o not in excluded)
            return None  # exclusions do not match any option; don't guess
        return ", ".join(options)
    if _NONE_OPTIONS_RE.search(response):
        return ""
    return None


def _without_commentary(parsed: set[str], gold: set[str]) -> set[str]:
    """Drop parts that are clearly commentary ("Note: the rest are wrong.");
    a part that overlaps some gold item text is always kept."""
    return {
        c for c in parsed
        if not (_COMMENTARY_RE.match(c) and not any(c in g or g in c for g in gold))
    }


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._/\u2011-][a-z0-9]+)*")


def _fuzzy_eligible(gold: set[str]) -> bool:
    """Fuzzy matching is only meaningful for phrase-like gold items: at least
    two word tokens each, and no item's token set nested inside another one's
    ("1 - (Total Dividends / Net Income)" vs "(Net Income - Total Dividends) /
    Net Income" share their words; "e*" is a single token). Otherwise a
    missing or extra item can silently "cover" another one."""
    token_sets = []
    for g in gold:
        ts = set(_TOKEN_RE.findall(g))
        if len(ts) < 2:
            return False
        token_sets.append(ts)
    for a in token_sets:
        for b in token_sets:
            if a is not b and a <= b:
                return False
    return True


def _fuzzy_list_match(response: str, gold: set[str]) -> bool:
    """Fallback for replies whose list items are reformulated or extended
    ("✅ A reverse cash-and-carry arbitrage opportunity, i.e. buy spot..."):
    every gold item must be covered (word-wise) by a list line, and every
    list line must cover at least one gold item (so extra selected options
    still fail). Word-level containment, not substrings: short gold items
    like "e" must not match inside "e_actual" or "1.5"."""
    if not _fuzzy_eligible(gold):
        return False
    lines = _marked_answer_lines(response)
    if not lines:
        return False
    line_tokens = [set(_TOKEN_RE.findall(c)) for ln in lines if (c := _clean_enum_item(ln))]
    gold_tokens = []
    for g in gold:
        gt = set(_TOKEN_RE.findall(g))
        if not gt:
            return False
        gold_tokens.append(gt)
    # every gold item must be covered by some line...
    if not all(any(gt <= lt for lt in line_tokens) for gt in gold_tokens):
        return False
    # ...and every line must cover at least one gold item
    return all(any(gt <= lt for gt in gold_tokens) for lt in line_tokens)


def grade_enum(item: dict, response: str) -> bool:
    # Gold answers are always comma-separated; "and" inside a gold item is
    # part of the item itself and must never be split.
    gold = parse_enum_answer(item["messages"][1]["content"],
                             split_and=False, expand_single=False)
    if not gold:
        return False
    candidates = [response]
    resolved = _resolve_all_of_above(
        response, _extract_prompt_options(item["messages"][0]["content"]))
    if resolved is not None:
        candidates.append(resolved)
    parsed_variants = [
        parse_enum_answer(cand, split_and=sa, expand_single=es)
        for cand in candidates
        for sa, es in ((False, False), (True, False), (True, True))
    ]
    # The model reply may use "and"-separated lists; accept if any reasonable
    # interpretation of the reply matches the gold set.
    if any(parsed == gold for parsed in parsed_variants):
        return True
    # Retry ignoring trailing/leading commentary sentences...
    if any(_without_commentary(parsed, gold) == gold
           for parsed in parsed_variants):
        return True
    # ...and finally a fuzzy match against the reply's list lines.
    return _fuzzy_list_match(response, gold)


# ------------------------------ yesno -------------------------------------

def grade_yesno(item: dict, response: str) -> bool:
    gold = item["messages"][1]["content"].strip().lower()
    t = re.sub(r"\s+", " ", response).strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    return bool(m) and m.group(1) == gold


# ------------------------------- math -------------------------------------

def parse_math_answer(response: str) -> float | None:
    matches = re.findall(
        r"\banswer\s*(?:is|:|=)?\s*\$?\s*(" + _NUM_RE + r")",
        response,
        re.IGNORECASE,
    )
    if matches:
        try:
            return float(_strip_grouping_commas(matches[-1]))
        except ValueError:
            pass
    nums = parse_numbers(response)
    return nums[-1] if nums else None


def grade_math(item: dict, response: str) -> bool:
    gold_text = str(item.get("answer", "")).strip().rstrip(".")
    pred = parse_math_answer(response)
    if pred is None:
        return False
    gold_text = gold_text.replace("%", "")
    # Grouped thousands ("200,000") is one number; otherwise "a,b" is a list.
    if re.fullmatch(r"\s*-?\d{1,3}(,\d{3})+(\.\d+)?\s*", gold_text):
        gold_nums = [float(gold_text.replace(",", ""))]
    else:
        gold_nums = []
        for chunk in gold_text.split(","):
            chunk = chunk.strip()
            try:
                gold_nums.append(float(chunk))
            except ValueError:
                gold_nums.extend(parse_numbers(chunk))
    if not gold_nums:
        return False
    return any(
        abs(pred - g) <= MATH_REL_TOL * max(abs(g), 1e-9) for g in gold_nums
    )


# ------------------------------ formula ------------------------------------

_EXPR_TRANSFORMATIONS = standard_transformations + (implicit_multiplication,)

# Names that sympy's global namespace would otherwise shadow (S, N, I, O, Q, E
# are live sympy objects) or that are Python keywords (\lambda); they must be
# treated as plain symbols.
_PARSE_LOCALS = {"e": sympy.E, "pi": sympy.pi}
for _name in ("S", "N", "I", "O", "Q", "E", "beta", "gamma", "zeta", "Gamma"):
    _PARSE_LOCALS[_name] = sympy.Symbol(_name)

_NAME_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _split_name_tokens(s: str) -> str:
    """Split multi-letter name tokens into products of single-letter symbols
    ("kP" -> "k P", "EMA_prev" -> "E M A_prev"). Tokens that denote known
    functions/constants (followed by '(' or present in _PARSE_LOCALS) and
    Python keywords are left intact. Applied identically to both compared
    formulas, so the transformation is equality-preserving."""
    out = []
    pos = 0
    for m in _NAME_TOKEN_RE.finditer(s):
        name = m.group(0)
        out.append(s[pos:m.start()])
        pos = m.end()
        nxt = pos
        while nxt < len(s) and s[nxt] in " \t":
            nxt += 1
        if (nxt < len(s) and s[nxt] == "(") or name in _PARSE_LOCALS \
                or keyword.iskeyword(name.rstrip("_")):
            out.append(name)
            continue
        if "_" in name:
            head, tail = name.split("_", 1)
            out.append(" ".join(head[:-1]) + (" " if len(head) > 1 else "")
                       + head[-1:] + "_" + tail if head else "_" + tail)
        else:
            out.append(" ".join(name))
    out.append(s[pos:])
    return "".join(out)

_LATEX_FUNCS = {
    "ln": "log", "log": "log", "log10": "log10", "exp": "exp",
    "sin": "sin", "cos": "cos", "tan": "tan", "arcsin": "asin",
    "arccos": "acos", "arctan": "atan", "sinh": "sinh", "cosh": "cosh",
    "tanh": "tanh", "min": "Min", "max": "Max", "erf": "erf",
    "abs": "Abs",
}

_UNSUPPORTED_RE = re.compile(r"\\(sum|prod|int|lim|max_|min_|bigcup|bigcap)\b")
_TEXTEXT_RE = re.compile(r"\\(?:text|mbox|mathrm|operatorname|mathbf|boldsymbol)\s*\{([^{}]*)\}")
_FRAC_RE = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_IDX_RE = re.compile(r"\\sqrt\s*\[([^\[\]]*)\]\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_SUP_BRACE_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_SUP_PLAIN_RE = re.compile(r"\^\s*(-?\d+(?:\.\d+)?|-?[A-Za-z](?![A-Za-z0-9]))")
_SUB_BRACE_RE = re.compile(r"_\s*\{([^{}]*)\}")
_SUB_PLAIN_RE = re.compile(r"_\s*([A-Za-z0-9]+)")
_WORD_RE = re.compile(r"\\([A-Za-z]+)")


def _cleanup_text_arg(content: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", content)


def normalize_formula(s: str) -> str:
    """Whitespace/notation-insensitive normalization used both for the
    fast-path string comparison and as the symbolic fallback."""
    s = s.strip().strip("$").strip().rstrip(".").strip()
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\quad|\\qquad|\\displaystyle", " ", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\ast", "*")
    s = s.replace("\\approx", "=").replace("\\sim", "=")
    s = re.sub(r"[\s]+", "", s)
    return s


def latex_to_expr(latex: str, split_symbols_mode: bool = False):
    """Best-effort LaTeX -> sympy expression conversion.

    Returns a sympy expression, or None when the formula uses constructs that
    cannot be safely converted (sums, integrals, ...). With split_symbols_mode
    multi-letter names are split into products of single-letter symbols
    ("kP" -> k*P); the same transformation is applied to both sides being
    compared, keeping the comparison symmetric."""
    s = latex.strip().strip("$").strip()

    if _UNSUPPORTED_RE.search(s):
        return None

    # separate adjacent commands ("\sigma\sqrt{T}" -> " \sigma \sqrt {T}")
    s = re.sub(r"\\([A-Za-z]+)", r" \\\1 ", s)
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\quad|\\qquad|\\displaystyle|\\ ", " ", s)
    s = re.sub(r"\\(?:d|t)?frac", r"\\frac", s)
    s = _TEXTEXT_RE.sub(lambda m: _cleanup_text_arg(m.group(1)), s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\ast", "*")
    s = s.replace("\\div", "/").replace("\\approx", "=").replace("\\sim", "=")

    for src, dst in _LATEX_FUNCS.items():
        s = re.sub(r"\\" + src + r"\b", dst, s)

    while True:
        m = re.search(r"\\[dt]?frac\s*([0-9]|[A-Za-z])\s*([0-9]|[A-Za-z])", s)
        if not m:
            break
        s = s[: m.start()] + "((%s)/(%s))" % (m.group(1), m.group(2)) + s[m.end():]
    while True:
        m = _SQRT_IDX_RE.search(s)
        if not m:
            break
        s = s[: m.start()] + "((%s)**(1/(%s)))" % (m.group(2), m.group(1)) + s[m.end():]
    while True:
        m = _SQRT_RE.search(s)
        if not m:
            break
        s = s[: m.start()] + "sqrt(%s)" % m.group(1) + s[m.end():]
    while True:
        m = _FRAC_RE.search(s)
        if not m:
            break
        s = s[: m.start()] + "((%s)/(%s))" % (m.group(1), m.group(2)) + s[m.end():]

    s = _SUP_BRACE_RE.sub(r"**(\1)", s)
    s = _SUP_PLAIN_RE.sub(r"**(\1)", s)
    s = _SUB_BRACE_RE.sub(lambda m: "_" + _cleanup_text_arg(m.group(1)), s)
    s = _SUB_PLAIN_RE.sub(r"_\1", s)

    s = s.replace("{", "(").replace("}", ")")
    s = s.replace("[", "(").replace("]", ")")
    # remaining commands -> symbol names (\pi -> pi); a trailing space keeps
    # adjacent tokens from merging ("\sigma\sqrt{T}" -> "sigma sqrt(T)");
    # Python keywords (e.g. \lambda) get a trailing underscore
    s = _WORD_RE.sub(
        lambda m: " " + m.group(1) + ("_" if keyword.iskeyword(m.group(1)) else "") + " ",
        s,
    )
    s = s.replace("\\", "").replace("$", "")

    if not s.strip():
        return None
    if split_symbols_mode:
        s = _split_name_tokens(s)
    try:
        expr = parse_expr(
            s,
            local_dict=dict(_PARSE_LOCALS),
            transformations=_EXPR_TRANSFORMATIONS,
            evaluate=False,
        )
    except Exception:
        return None
    return expr


def _split_equation(expr):
    """Split a normalized string at top-level '='; returns (lhs, rhs) or None."""
    depth = 0
    for i, ch in enumerate(expr):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "=" and depth == 0:
            return expr[:i], expr[i + 1:]
    return None


def _expr_pair(latex: str, split_symbols_mode: bool = False):
    """Parse 'lhs = rhs' (or a bare expression) into a pair of sympy
    expressions, (lhs, rhs) or (expr, None). None on parse failure.

    The split happens on the raw LaTeX (\\approx is pre-unified to '='), so
    that whitespace-sensitive token adjacency inside each side
    (e.g. 'K e^{-rT}') survives for the implicit-multiplication parser."""
    prep = re.sub(r"\\(?:approx|sim)\b", "=", latex.strip().strip("$"))
    sides = _split_equation(prep)
    if sides:
        lhs = latex_to_expr(sides[0], split_symbols_mode)
        rhs = latex_to_expr(sides[1], split_symbols_mode)
        if lhs is None or rhs is None:
            return None
        return lhs, rhs
    expr = latex_to_expr(prep, split_symbols_mode)
    return None if expr is None else (expr, None)


_KNOWN_CONSTANTS = {"pi", "E", "I", "oo", "NaN", "zoo"}


def _numeric_equal(a, b, rng: np.random.Generator, trials: int = 6) -> bool | None:
    """Sample random positive values for all free symbols and compare both
    expressions numerically. None means 'no reliable conclusion'."""
    symbols = sorted(
        (s for e in (a, b) for s in e.free_symbols if s.name not in _KNOWN_CONSTANTS),
        key=lambda s: s.name,
    )
    ok = False
    for _ in range(trials):
        subs = {s: float(rng.uniform(0.4, 3.0)) for s in symbols}
        vals = []
        for e in (a, b):
            try:
                v = complex(sympy.N(e.subs(subs)))
            except Exception:
                return None
            if not np.isfinite(v.real) or not np.isfinite(v.imag):
                return None
            if abs(v.imag) > 1e-8 * max(1.0, abs(v.real)):
                return None  # left the real domain, sampling unreliable
            vals.append(v.real)
        scale = max(1.0, abs(vals[0]), abs(vals[1]))
        if abs(vals[0] - vals[1]) > 1e-7 * scale:
            return False
        ok = True
    return ok


def _exprs_equal(a, b, rng: np.random.Generator) -> bool:
    numeric = _numeric_equal(a, b, rng)
    if numeric is not None:
        return numeric
    try:
        diff = sympy.expand(sympy.simplify(a - b))
        if diff == 0:
            return True
        return bool(diff.equals(0))
    except Exception:
        return False


def _pairs_equal(pa, pb, rng: np.random.Generator) -> bool:
    (la, ra), (lb, rb) = pa, pb
    if (ra is None) != (rb is None):
        return False  # equation vs bare expression
    if ra is None:
        return _exprs_equal(la, lb, rng)
    # equations: sides may be written in either order (lhs<->rhs)
    return (_exprs_equal(la, lb, rng) and _exprs_equal(ra, rb, rng)) or \
           (_exprs_equal(la, rb, rng) and _exprs_equal(ra, lb, rng))


def grade_formula(item: dict, response: str) -> bool:
    gold = item["messages"][1]["content"]
    if normalize_formula(gold) == normalize_formula(response):
        return True

    rng = np.random.default_rng(0)
    # Two parsing interpretations: multi-letter names kept whole ("sigma",
    # "EMA_prev") and names split into single letters ("kP" -> k*P). The same
    # interpretation is applied to both sides; equality in either wins.
    for split_mode in (False, True):
        pair_gold = _expr_pair(gold, split_mode)
        pair_pred = _expr_pair(response, split_mode)
        if pair_gold is None or pair_pred is None:
            continue  # symbolic parsing impossible for this interpretation
        if _pairs_equal(pair_gold, pair_pred, rng):
            return True
    return False


_GRADERS = {
    "enum": grade_enum,
    "formula": grade_formula,
    "math": grade_math,
    "yesno": grade_yesno,
}


def grade(item: dict, response: str) -> bool:
    grader = _GRADERS.get(item.get("type"))
    if grader is None:
        return False
    try:
        return grader(item, response)
    except Exception:
        return False


# --------------------------------------------------------------------------
# Model endpoint
# --------------------------------------------------------------------------

def build_base_url(host: str) -> str:
    if "://" not in host:
        host = "http://" + host
    base = host.rstrip("/")
    if not re.search(r"/v\d+$", base):
        base += "/v1"
    return base


class FatalEndpointError(RuntimeError):
    """Endpoint rejection that will not get better with retries (auth, bad
    request, wrong path) — aborts the whole benchmark run."""


async def ask_model(
    session: aiohttp.ClientSession,
    url: str,
    headers: dict,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout: float,
    temperature: float = 0.0,
    retries: int = 10,
    system: str | None = None,
) -> str:
    """Send one chat-completions request with retries on transient failures.

    Transient (retryable) failures: HTTP 429, HTTP 5xx, connection errors,
    connection resets and timeouts (aiohttp.ClientError, asyncio.TimeoutError).
    Fatal errors (400/401/403/404, malformed response payload) raise
    FatalEndpointError immediately and abort the whole benchmark run.

    Backoff is exponential (capped at 30s) with jitter; for HTTP 429 the
    server-provided Retry-After delay is honored when present.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    last_error = None
    for attempt in range(retries):
        retry_after = None
        try:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status in (400, 401, 403, 404):
                    body = await resp.text()
                    raise FatalEndpointError(
                        f"fatal HTTP {resp.status} from endpoint: {body[:300]}"
                    )
                if resp.status >= 500 or resp.status == 429:
                    last_error = f"HTTP {resp.status}"
                    if resp.status == 429:
                        retry_after = resp.headers.get("Retry-After")
                else:
                    data = await resp.json()
                    try:
                        return data["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, TypeError) as exc:
                        raise FatalEndpointError(
                            f"unexpected response shape: {json.dumps(data)[:300]}"
                        ) from exc
        except FatalEndpointError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        # transient failure -> wait and retry
        if attempt + 1 < retries:
            await asyncio.sleep(_retry_delay(attempt, retry_after))
    raise RuntimeError(f"request failed after {retries} attempts: {last_error}")


def _retry_delay(attempt: int, retry_after: str | None) -> float:
    """Exponential backoff capped at 30s, with ±20% jitter. When the server
    sends a valid Retry-After header, use it (jittered upwards slightly)."""
    base = min(2 ** attempt, 30.0)
    if retry_after is not None:
        try:
            return min(float(retry_after), 60.0)
        except ValueError:
            pass
    return base * random.uniform(0.8, 1.2)


# --------------------------------------------------------------------------
# Benchmark driver
# --------------------------------------------------------------------------

def load_dataset(name: str) -> list[dict]:
    path = ROOT / "datasets" / f"finmath-distill-{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def sample_items(items: list[dict], limit: int | None, rng: random.Random) -> list[dict]:
    if limit is None or limit >= len(items):
        return items
    return [items[i] for i in rng.sample(range(len(items)), limit)]


async def run(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)

    requested = (
        list(DATASET_NAMES)
        if args.datasets.strip().lower() == "all"
        else [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    )
    unknown = [d for d in requested if d not in DATASET_NAMES]
    if unknown:
        print(f"error: unknown dataset(s): {', '.join(unknown)} "
              f"(available: {', '.join(DATASET_NAMES)}, or 'all')", file=sys.stderr)
        return 2

    plan = []
    for name in requested:
        items = load_dataset(name)
        chosen = sample_items(items, args.limit, rng)
        if not chosen:
            print(f"warning: dataset '{name}' produced 0 questions, skipping",
                  file=sys.stderr)
            continue
        plan.append((name, chosen))

    if not plan:
        print("error: nothing to test", file=sys.stderr)
        return 2

    url = build_base_url(args.host) + "/chat/completions"
    headers = {"Authorization": f"Bearer {args.token}"}
    total = sum(len(items) for _, items in plan)
    print(f"endpoint : {url}")
    print(f"model    : {args.model}")
    print(f"questions: {total} "
          f"({', '.join(f'{n}={len(items)}' for n, items in plan)})")
    print(f"workers  : {args.concurrency}")
    print(f"temp     : {args.temperature:g}")
    print()

    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    done_counter = {"done": 0}
    skipped_counter = {"skipped": 0}
    correct_counter = {"correct": 0}

    async def worker(session, name, item, results):
        async with sem:
            try:
                response = await ask_model(
                    session, url, headers, args.model,
                    build_prompt(item), args.max_tokens, args.timeout,
                    temperature=args.temperature,
                    retries=args.retries,
                    system=SYSTEM_PROMPTS.get(name),
                )
            except FatalEndpointError:
                raise
            except Exception as exc:
                # question exhausted all retries: skip it and exclude it from
                # the accuracy denominator
                results[name]["skipped"] += 1
                results[name].setdefault("error_examples", []).append(
                    (item.get("id", "?"), str(exc)[:200])
                )
                response = None
                skipped_counter["skipped"] += 1
            if response is not None:
                correct = await asyncio.to_thread(grade, item, response)
                results[name]["correct"] += int(correct)
                correct_counter["correct"] += int(correct)
                if not correct:
                    results[name]["mismatches"].append(
                        (item.get("id", "?"),
                         item["messages"][1]["content"][:120],
                         response[:160])
                    )
            done_counter["done"] += 1
            # skipped = request errors after all retries; excluded from the
            # graded total, so they never count towards the accuracy
            print(f"\rprogress: {done_counter['done']}/{total} — "
                  f"correct: {correct_counter['correct']}, "
                  f"wrong: {done_counter['done'] - correct_counter['correct'] - skipped_counter['skipped']}, "
                  f"skipped: {skipped_counter['skipped']}",
                  end="", flush=True)

    results = {
        name: {"correct": 0, "skipped": 0, "mismatches": [], "error_examples": [],
               "total": len(items)}
        for name, items in plan
    }

    started = time.monotonic()
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 10))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for name, items in plan:
            for item in items:
                tasks.append(asyncio.create_task(worker(session, name, item, results)))
        try:
            await asyncio.gather(*tasks)
        except FatalEndpointError as exc:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            print(f"\nfatal: {exc}", file=sys.stderr)
            return 1
    elapsed = time.monotonic() - started
    print(f" — finished in {elapsed:.1f}s\n")

    # ------------------------------ report --------------------------------
    # "graded" excludes questions skipped after exhausting all retries, so
    # they never affect the accuracy.
    print("=" * 60)
    print(f"{'dataset':<12}{'correct/graded':>18}{'accuracy':>12}{'skipped':>9}")
    print("-" * 60)
    sum_correct = sum_total = sum_skipped = 0
    for name, res in results.items():
        correct, total_ds, skipped = res["correct"], res["total"], res["skipped"]
        graded = total_ds - skipped
        pct = 100.0 * correct / graded if graded else 0.0
        print(f"{name:<12}{correct:>10}/{graded:<7}{pct:>11.2f}%{skipped:>9}")
        sum_correct += correct
        sum_total += total_ds
        sum_skipped += skipped
    print("-" * 60)
    graded_total = sum_total - sum_skipped
    pct = 100.0 * sum_correct / graded_total if graded_total else 0.0
    print(f"{'TOTAL':<12}{sum_correct:>10}/{graded_total:<7}"
          f"{pct:>11.2f}%{sum_skipped:>9}")
    print("=" * 60)
    if sum_skipped:
        print(f"note: {sum_skipped} question(s) skipped after "
              f"{args.retries} retries and excluded from the score")

    if not args.verbose:
        for name, res in results.items():
            del res["mismatches"][3:]
            del res["error_examples"][3:]
    for name, res in results.items():
        if res["mismatches"]:
            print(f"\nmismatch examples [{name}]:")
            for qid, expected, got in res["mismatches"]:
                print(f"  {qid}\n    expected: {expected!r}\n    got     : {got!r}")
        if res["error_examples"]:
            print(f"\nrequest errors [{name}]:")
            for qid, err in res["error_examples"]:
                print(f"  {qid}: {err}")

    return 0


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a model on the finmath-distill datasets.",
    )
    parser.add_argument("--host", required=True,
                        help="endpoint host, optionally with port "
                             "(e.g. localhost:8000 or http://host:8000/v1)")
    parser.add_argument("--token", required=True,
                        help="access token (sent as Bearer token)")
    parser.add_argument("--model", required=True,
                        help="model name to request")
    parser.add_argument("-c", "--concurrency", type=int, default=1,
                        help="parallel requests (default: 1)")
    parser.add_argument("-d", "--datasets", default="all",
                        help='comma-separated datasets (enum,formula,math,yesno) '
                             'or "all" (default: all)')
    parser.add_argument("-n", "--limit", type=int, default=1000,
                        help="max questions per dataset, random sample "
                             "(default: 1000)")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for sampling")
    parser.add_argument("--timeout", type=float, default=120.0,
                        help="per-request timeout in seconds (default: 120)")
    parser.add_argument("--retries", type=int, default=10,
                        help="attempts per question on transient errors (429, "
                             "5xx, network drops, timeouts); a question that "
                             "fails all attempts is skipped and excluded from "
                             "the score (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_DEFAULT,
                        help="max tokens per response (default: 2048)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="sampling temperature sent to the endpoint "
                             "(default: 0)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print all mismatches instead of a few examples")
    args = parser.parse_args(argv)
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0")
    if args.retries < 1:
        parser.error("--retries must be >= 1")
    return args


def main() -> int:
    try:
        return asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
