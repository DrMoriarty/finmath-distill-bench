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
            numeric sampling + simplify); tolerant to synonymous variable
            names via a name-plausible symbol-bijection fallback, to a
            missing "X =" prefix and to percent-sign notation
            ("\\times 100\\%" == "\\times 100", "15\\%" == "(15/100)");
            no char-by-char match;
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
import concurrent.futures
import faulthandler
import itertools
import json
import keyword
import multiprocessing
import os
import random
import re
import signal
import sys
import time
import traceback
from pathlib import Path

import warnings

import aiohttp
import numpy as np
import sympy

# Parsing LaTeX with implicit multiplication can still build a Mul with a
# Tuple argument in odd corners ("K (a, b)" written with a space), and
# "\not"-style tokens inside \sqrt{...} similarly produce a Pow with a Not
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
    "formula": (
        "You are answering a finance formula test. Reply with a single "
        "formula only: one line in LaTeX, no explanations, derivations, "
        "comments or markdown (no $$ blocks, no code fences). Use exactly "
        "the same variable names and symbols for quantities as the question "
        "does — do not rename, abbreviate or invent new symbols. Write "
        "multi-word quantity names inside \\text{...} (e.g. "
        "\\text{Net Income}_t) and keep any subscripts used in the question."
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


def _undefined_function_names(s: str) -> set[str]:
    r"""Identifiers applied to 2+ comma-separated arguments ("Cov(x, y)").

    With implicit_multiplication such a call cannot be a plain multiplication
    (a comma inside the parentheses makes a Tuple), yet sympy still parses
    "Cov(r_i, r_m)" as Cov*(r_i, r_m) — a Mul with a Tuple argument, which it
    can neither evaluate numerically nor simplify. Registering the name as an
    undefined Function upfront keeps the application intact so that
    structurally equal formulas compare as equal after symbol substitution.
    """
    names = set()
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", s):
        depth, end = 0, None
        for i in range(m.end() - 1, len(s)):
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            continue
        d = 0
        for ch in s[m.end():end]:
            if ch in "([{":
                d += 1
            elif ch in ")]}":
                d -= 1
            elif ch == "," and d == 0:
                names.add(m.group(1))
                break
    return names


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
_SQRT_IDX_RE = re.compile(r"\\sqrt\s*\[([^\[\]]*)\]\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_SUP_BRACE_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_SUP_PLAIN_RE = re.compile(r"\^\s*(-?\d+(?:\.\d+)?|-?[A-Za-z](?![A-Za-z0-9]))")
_SUB_BRACE_RE = re.compile(r"_\s*\{([^{}]*)\}")
_SUB_PLAIN_RE = re.compile(r"_\s*([A-Za-z0-9]+)")
_WORD_RE = re.compile(r"\\([A-Za-z]+)")


def _cleanup_text_arg(content: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", content)


def _find_matching_brace(s: str, start: int) -> int | None:
    """Index of the '}' matching the '{' at `start`, or None."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _replace_single_char_fracs(s: str) -> str:
    while True:
        m = re.search(r"\\[dt]?frac\s*([0-9]|[A-Za-z])\s*([0-9]|[A-Za-z])", s)
        if not m:
            return s
        s = s[: m.start()] + "((%s)/(%s))" % (m.group(1), m.group(2)) + s[m.end():]


def _replace_sqrt_indexed(s: str) -> str:
    while True:
        m = _SQRT_IDX_RE.search(s)
        if not m:
            return s
        s = s[: m.start()] + "((%s)**(1/(%s)))" % (m.group(2), m.group(1)) + s[m.end():]


def _replace_sqrt(s: str) -> str:
    while True:
        m = _SQRT_RE.search(s)
        if not m:
            return s
        s = s[: m.start()] + "sqrt(%s)" % m.group(1) + s[m.end():]


def _replace_braced_fracs(s: str) -> str:
    r"""Replace \frac{num}{den} with balanced-brace arguments (the arguments
    may themselves contain braces, e.g. \frac{W_{nom}}{(1+\pi)^t})."""
    res, pos = [], 0
    pat = re.compile(r"\\[dt]?frac\s*")
    while True:
        m = pat.search(s, pos)
        if not m:
            res.append(s[pos:])
            return "".join(res)
        res.append(s[pos:m.start()])  # text before \frac; the \frac itself is replaced
        pos = m.end()
        if pos >= len(s) or s[pos] != "{":
            res.append("\\frac")  # malformed here; leave for later stages
            continue
        e1 = _find_matching_brace(s, pos)
        if e1 is None:
            res.append(s[pos:])
            return "".join(res)
        num, pos = s[pos + 1:e1], e1 + 1
        j = pos
        while j < len(s) and s[j] in " \t":
            j += 1
        if j < len(s) and s[j] == "{":
            e2 = _find_matching_brace(s, j)
            if e2 is None:
                den, pos = s[j:], len(s)
            else:
                den, pos = s[j + 1:e2], e2 + 1
        else:
            den, pos = (s[j] if j < len(s) else ""), j + 1
        res.append("((%s)/(%s))" % (num, den))


# The three roles "%" plays in the dataset, handled by _normalize_percent:
# - "\times 100\%" is a unit marker ("express the result in percent"),
#   numerically the same as "\times 100" — the "%" is dropped;
# - "15\%" is a percent value -> (15/100);
# - "%" glued to a name ("\text{Bullish}\%", "\%_{\text{actual},i}") is part
#   of the symbol name -> "pct".
_PERCENT_X100_RE = re.compile(r"(\\(?:times|cdot|ast)\s*|\*\s*)100\s*(?:\\[,;!])*\s*\\?%")
_PERCENT_VALUE_RE = re.compile(
    r"(?<![\w\\])(\d+(?:\.\d+)?)\s*(?:\\[,;!])*\s*\\?%(?![A-Za-z0-9_{])"
)
_PERCENT_NAME_RE = re.compile(r"\\%(?=[A-Za-z0-9_{])|(?<=[A-Za-z0-9_}])\s*\\?%")


def _normalize_percent(s: str) -> str:
    r"""Fold percent signs into the expression so it stays parseable.

    Must run AFTER \text{...} arguments are cleaned up: a "%" inside
    \text{...} (e.g. the "(%)" column-header suffix) is pure formatting and
    disappears together with the other non-alphanumeric characters there.
    """
    s = _PERCENT_X100_RE.sub(r"\g<1>100", s)
    s = _PERCENT_VALUE_RE.sub(r"(\1/100)", s)
    s = _PERCENT_NAME_RE.sub("pct", s)
    return s.replace("\\%", "pct").replace("%", " pct ")


def normalize_formula(s: str) -> str:
    """Whitespace/notation-insensitive normalization used both for the
    fast-path string comparison and as the symbolic fallback."""
    s = s.strip().strip("$").strip().rstrip(".").strip()
    s = _normalize_percent(s)
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

    # unwrap \text{...}/\mathrm{...} etc BEFORE separating the commands.
    # A "\text{Cov}" directly followed by "(" must stay glued to it, so that
    # Cov(r_i, r_m) parses as a function application; with the separation
    # running first, the injected space yields the product Cov*(r_i, r_m)
    # — a Mul with a Tuple argument, which sympy can neither evaluate
    # numerically nor simplify, so such formulas could only match verbatim.
    def _text_repl(m):
        rest = m.string[m.end():]
        glued = rest[:1] in ("_", "^") or rest.startswith("(") \
            or re.match(r"\\left\s*\(", rest)
        return " " + _cleanup_text_arg(m.group(1)) + ("" if glued else " ")

    s = _TEXTEXT_RE.sub(_text_repl, s)

    # separate adjacent commands ("\sigma\sqrt{T}" -> " \sigma \sqrt {T}"),
    # but a command directly followed by a subscript/superscript keeps it
    # attached ("\beta_i" must parse as the single symbol "beta_i", not as
    # the product "beta * _i")
    s = re.sub(
        r"\\([A-Za-z]+)",
        lambda m: " \\" + m.group(1)
        + ("" if s[m.end():m.end() + 1] in ("_", "^") else " "),
        s,
    )
    s = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\quad|\\qquad|\\displaystyle|\\ ", " ", s)
    s = re.sub(r"\\(?:d|t)?frac", r"\\frac", s)
    s = _normalize_percent(s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\ast", "*")
    s = s.replace("\\div", "/").replace("\\approx", "=").replace("\\sim", "=")

    for src, dst in _LATEX_FUNCS.items():
        s = re.sub(r"\\" + src + r"\b", dst, s)

    # frac/sqrt replacement with nested arguments needs several passes:
    # inner fracs are expanded first, then outer ones see brace-free args
    while True:
        new = _replace_sqrt(_replace_sqrt_indexed(
            _replace_braced_fracs(_replace_single_char_fracs(s))))
        if new == s:
            break
        s = new

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
        lambda m: " " + m.group(1)
        + ("_" if keyword.iskeyword(m.group(1)) else "")
        + ("" if s[m.end():m.end() + 1] in ("_", "^") else " "),
        s,
    )
    s = s.replace("\\", "").replace("$", "")

    if not s.strip():
        return None
    if split_symbols_mode:
        s = _split_name_tokens(s)
    local = dict(_PARSE_LOCALS)
    fnames = _undefined_function_names(s)
    for fname in fnames:
        # known sympy names (Max, Min, log, ...) must keep their live
        # definitions, otherwise e.g. Max(a, b) stops evaluating numerically
        if fname not in local and not hasattr(sympy, fname):
            local[fname] = sympy.Function(fname)
    # every other identifier must parse as a plain Symbol: a name that
    # clashes with a sympy callable ("Float", "Beta") would otherwise
    # resolve to the sympy object itself and break the expression handling
    for name in set(_NAME_TOKEN_RE.findall(s)):
        if name in local or name in fnames or keyword.iskeyword(name) \
                or name in _LIVE_SYMPY_NAMES:
            continue
        local[name] = sympy.Symbol(name)
    try:
        expr = parse_expr(
            s,
            local_dict=local,
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

# Complexity gate: no gold formula in the datasets contains an Integer with
# abs value > 1e6, and evaluating one (e.g. 10**(10**18), a 3-node tree) is
# C-level big-integer arithmetic that holds the GIL and cannot be interrupted.
# Any such integer in either expression means we refuse to evaluate/simplify.
_MAX_SAFE_INTEGER = 10**6


def _has_huge_integer(*exprs) -> bool:
    """True when any Integer with abs(value) > _MAX_SAFE_INTEGER appears in
    any of the expression trees. Pure tree walk: evaluates nothing."""
    for e in exprs:
        if e is None or not isinstance(e, sympy.Basic):
            continue
        try:
            for node in sympy.preorder_traversal(e):
                if isinstance(node, sympy.Integer) \
                        and abs(int(node)) > _MAX_SAFE_INTEGER:
                    return True
        except Exception:
            return True  # unwalkable tree: refuse to evaluate it
    return False


def _safe_count_ops(expr) -> int:
    """count_ops that never raises; -1 for non-expression sides."""
    if expr is None or not isinstance(expr, sympy.Basic):
        return -1
    try:
        return int(sympy.count_ops(expr))
    except Exception:
        return -1


def _pair_ops(pair) -> int:
    """Total count_ops of an (lhs, rhs) pair parsed in one split mode."""
    lhs, rhs = pair
    return _safe_count_ops(lhs) + max(_safe_count_ops(rhs), 0)


def _numeric_equal(a, b, rng: np.random.Generator, trials: int = 6) -> bool | None:
    """Sample random positive values for all free symbols and compare both
    expressions numerically. None means 'no reliable conclusion'."""
    if _has_huge_integer(a, b):
        return None  # numeric evaluation could run effectively forever
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
    if _has_huge_integer(a, b):
        # simplify/equals on huge-integer trees can hang for hours and hold
        # the GIL; treat as "no reliable conclusion" -> not equal
        return False
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
        # equation vs bare expression: the question asks for "the formula
        # for X", so answering just the right-hand side is still correct
        if ra is None:
            return _exprs_equal(la, rb, rng)
        return _exprs_equal(lb, ra, rng)
    if ra is None:
        return _exprs_equal(la, lb, rng)
    # equations: sides may be written in either order (lhs<->rhs)
    return (_exprs_equal(la, lb, rng) and _exprs_equal(ra, rb, rng)) or \
           (_exprs_equal(la, rb, rng) and _exprs_equal(ra, lb, rng))


# -------------------- rename-tolerant formula fallback ---------------------
#
# Models often answer with synonymous variable names ("R_t" for
# "\text{Revenue}_t", "Net Working Capital_before" for "NWC_before"). Those
# parse to different sympy symbols and fail the direct symbolic comparison
# even though the formula is identical. Fallback: find a bijection between
# the free symbols of both sides that makes the formulas equal AND whose
# name pairs plausibly denote the same quantity (equal / substring /
# acronym / short-subsequence). This keeps wrong formulas out: a renamed
# match still requires the same expression structure, same operand order
# (checked numerically) and related names on every mapped pair.

_MAX_RENAME_SYMBOLS = 6  # 6! = 720 candidate bijections max

# sympy names that must keep their live meaning when they appear as a token
# in a formula (single-argument applications like log(x)/sqrt(T) are not
# captured by _undefined_function_names, which only handles comma-args)
_LIVE_SYMPY_NAMES = set(_LATEX_FUNCS.values()) | {"sqrt", "pi", "E", "oo", "zoo"}


def _is_subsequence(short: str, long: str) -> bool:
    it = iter(long)
    return all(c in it for c in short)


def _acronym(name: str) -> str:
    """Initials of the words of a symbol name: 'NetWorkingCapital_before'
    -> 'nwc', 'PriorYearTaxRate' -> 'pytr'."""
    parts = []
    for chunk in name.split("_"):
        parts += re.findall(r"[A-Z][a-z]*|[a-z]+|[0-9]+", chunk)
    return "".join(p[0] for p in parts if p).lower()


def _name_compatible(gold_name: str, pred_name: str) -> bool:
    """Plausible 'same quantity' relation between two symbol names."""
    a = re.sub(r"[^a-z0-9]", "", gold_name.lower())
    b = re.sub(r"[^a-z0-9]", "", pred_name.lower())
    if not a or not b:
        return True  # nothing meaningful to compare (e.g. numeric-only name)
    if a == b or a in b or b in a:
        return True  # 'refund' ~ 'taxrefund', 'g' ~ 'growthrate'
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) <= 3 and _is_subsequence(short, long):
        return True  # 'rt' ~ 'revenuet', 'wr' ~ 'workingcapitalrelease'
    acr_a, acr_b = _acronym(gold_name), _acronym(pred_name)
    if acr_a and acr_b and (acr_a == acr_b or acr_a in acr_b or acr_b in acr_a):
        return True  # 'nwc' ~ 'NetWorkingCapital'
    return bool(
        (len(acr_a) <= 3 and _is_subsequence(acr_a, acr_b))
        or (len(acr_b) <= 3 and _is_subsequence(acr_b, acr_a))
    )


def _mapped_pair_equal(pair_gold, pair_pred, mapping, rng) -> bool:
    """Compare both sides of an equation pair after substituting a symbol
    mapping into the gold side; sides may be swapped (lhs<->rhs)."""
    (lg, rg), (lp, rp) = pair_gold, pair_pred
    try:
        lg2 = lg.subs(mapping)
        rg2 = rg.subs(mapping) if rg is not None else None
    except Exception:
        return False
    if _exprs_equal(lg2, lp, rng) and \
            (rg2 is None or _exprs_equal(rg2, rp, rng)):
        return True
    if rg is not None:
        return _exprs_equal(lg2, rp, rng) and _exprs_equal(rg2, lp, rng)
    return False


def _renames_equal(pair_gold, pair_pred, rng: np.random.Generator) -> bool:
    """True when a name-plausible bijection of free symbols makes the two
    (equation or bare) pairs equal."""
    (lg, rg), (lp, rp) = pair_gold, pair_pred
    if (rg is None) != (rp is None):
        return False
    # degenerate parses (e.g. a bare python tuple from "f(a, b)"-style
    # LaTeX) cannot be compared symbolically
    if not all(isinstance(e, sympy.Basic)
               for e in (lg, rg, lp, rp) if e is not None):
        return False
    gold_syms = sorted(
        (s for e in (lg, rg) if e is not None
         for s in e.atoms(sympy.Symbol) if s.name not in _KNOWN_CONSTANTS),
        key=lambda s: s.name,
    )
    pred_syms = sorted(
        (s for e in (lp, rp) if e is not None
         for s in e.atoms(sympy.Symbol) if s.name not in _KNOWN_CONSTANTS),
        key=lambda s: s.name,
    )
    if not gold_syms or len(gold_syms) != len(pred_syms):
        return False
    if len(gold_syms) > _MAX_RENAME_SYMBOLS:
        return False
    # symbol->symbol subs() never changes count_ops, so a correct bijection
    # must preserve the per-side node counts; skip the (up to 720) expensive
    # comparisons when the structures already differ in size.
    if _safe_count_ops(lg) != _safe_count_ops(lp):
        return False
    if rg is not None and _safe_count_ops(rg) != _safe_count_ops(rp):
        return False
    for perm in itertools.permutations(pred_syms):
        mapping = dict(zip(gold_syms, perm))
        if not all(_name_compatible(g.name, p.name)
                   for g, p in mapping.items()):
            continue
        if _mapped_pair_equal(pair_gold, pair_pred, mapping, rng):
            return True
    return False


def _response_candidates(response: str) -> list[str]:
    """The full reply plus its last line containing '=' (models sometimes
    prepend prose like 'The formula is: ...' despite the instructions)."""
    cands = [response]
    eq_lines = [ln.strip() for ln in response.splitlines()
                if "=" in ln and ln.strip()]
    if eq_lines and eq_lines[-1] != response.strip():
        cands.append(eq_lines[-1])
    return cands


def grade_formula(item: dict, response: str) -> bool:
    gold = item["messages"][1]["content"]

    rng = np.random.default_rng(0)
    for cand in _response_candidates(response):
        if normalize_formula(gold) == normalize_formula(cand):
            return True
        # Two parsing interpretations: multi-letter names kept whole ("sigma",
        # "EMA_prev") and names split into single letters ("kP" -> k*P). The
        # same interpretation is applied to both sides; equality in either
        # wins.
        for split_mode in (False, True):
            pair_gold = _expr_pair(gold, split_mode)
            pair_pred = _expr_pair(cand, split_mode)
            if pair_gold is None or pair_pred is None:
                continue  # symbolic parsing impossible for this interpretation
            # Complexity gate: a prediction with vastly more nodes than the
            # gold formula (e.g. a 100+-node pathological reply against a
            # ~10-node gold) cannot be an equal formula written differently;
            # skip the expensive symbolic comparison entirely. Threshold is
            # generous so expanded-vs-factored forms of the same formula
            # ((a+b)**2 vs a**2+2*a*b+b**2) still pass. Counts are computed
            # within this same split mode.
            if _pair_ops(pair_pred) > 4 * _pair_ops(pair_gold) + 20:
                continue
            if _pairs_equal(pair_gold, pair_pred, rng):
                return True
            # Rename-tolerant fallback only for whole names: with split
            # names the symbol identities are meaningless, so a "rename"
            # there would accept any structurally equal formula.
            if not split_mode and _renames_equal(pair_gold, pair_pred, rng):
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


# The grader runs sympy/numpy on the model's reply. Some replies (very long
# or symbolically nasty LaTeX) can make a comparison take a very long time
# without raising — and big-integer arithmetic inside sympy holds the GIL and
# cannot be interrupted by signals, so a worker *thread* can hang forever and,
# with low concurrency, freeze the whole benchmark (it also blocks interpreter
# shutdown, making Ctrl-C useless). Cap it with a real process boundary:
# grading runs in a ProcessPoolExecutor, and on timeout the whole pool is
# killed and recreated, so a stuck worker cannot poison later gradings. A
# reply that does not grade within the timeout is treated as "not correct",
# matching the grader's own error handling.
GRADE_TIMEOUT = 60.0

# spawn (not fork): the parent runs an asyncio event loop with helper threads,
# and forking a multi-threaded process can deadlock the child. Spawned workers
# import this module (module-level sympy/numpy imports only) and are reused
# across gradings; the pool is recreated only after a timeout/crash.
_GRADER_MP_CONTEXT = multiprocessing.get_context("spawn")

# Counters/observability shared between the grading machinery, the watchdog
# and the progress line.
_GRADE_STATS = {
    "slow": 0,       # grades that took longer than GRADE_SLOW_SECONDS
    "timeouts": 0,   # grades killed by the process timeout
    "inflight": 0,   # grades currently running in pool workers
}
_GRADE_SLOW_SECONDS = 10.0
_GRADER_POOL = None


def _get_grader_pool() -> concurrent.futures.ProcessPoolExecutor:
    global _GRADER_POOL
    if _GRADER_POOL is None:
        # a broken pool surfaces as BrokenProcessPool on submit(); callers
        # recreate it there — no eager health check needed
        _GRADER_POOL = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            mp_context=_GRADER_MP_CONTEXT,
        )
    return _GRADER_POOL


def _grader_pool_size() -> int:
    """Number of live pool worker processes (0 when no pool exists)."""
    if _GRADER_POOL is None:
        return 0
    return len(getattr(_GRADER_POOL, "_processes", {}) or {})


def _kill_grader_pool() -> None:
    """Kill the grading pool and all its worker processes.

    Called after a timeout (a worker is stuck in uninterruptible C-level
    arithmetic — it must be SIGKILLed, not joined) and on shutdown, so that
    the concurrent.futures atexit join can never block on a stuck worker."""
    global _GRADER_POOL
    pool, _GRADER_POOL = _GRADER_POOL, None
    if pool is None:
        return
    for proc in list(getattr(pool, "_processes", {}).values()):
        try:
            proc.kill()
        except Exception:
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


async def _grade_timed(item: dict, response: str, timeout: float) -> bool:
    """Grade in a pool worker *process*, bounded by a wall-clock timeout.

    On timeout the worker process is terminated with the whole pool and a
    fresh pool is created for subsequent gradings, so one pathological answer
    can never stall the run. Timed-out grades are scored as "not correct"."""
    loop = asyncio.get_running_loop()
    qid = item.get("id", "?")
    ds = item.get("type", "?")
    started = time.monotonic()
    _GRADE_STATS["inflight"] += 1
    try:
        for attempt in range(2):
            try:
                pool = _get_grader_pool()
                fut = loop.run_in_executor(pool, grade, item, response)
                return await asyncio.wait_for(fut, timeout)
            except asyncio.TimeoutError:
                _GRADE_STATS["timeouts"] += 1
                # wait_for only cancels the asyncio wrapper; the worker keeps
                # running sympy forever -> SIGKILL the whole pool
                _kill_grader_pool()
                print(
                    f"\n[grade] TIMEOUT after {timeout:.0f}s, grader worker "
                    f"killed; qid={qid} dataset={ds}",
                    file=sys.stderr, flush=True,
                )
                return False
            except concurrent.futures.process.BrokenProcessPool as exc:
                _kill_grader_pool()
                if attempt == 0:
                    # this grade was likely an innocent bystander of a
                    # sibling timeout-kill: retry once in a fresh pool
                    continue
                print(
                    f"\n[grade] grader worker died ({exc}); pool recreated; "
                    f"qid={qid} dataset={ds}",
                    file=sys.stderr, flush=True,
                )
                return False
        return False
    except Exception as exc:
        # never let a grading bug stall the run, but don't swallow it silently
        print(
            f"\n[grade] grader error ({type(exc).__name__}: {exc}); "
            f"scored as not correct; qid={qid} dataset={ds}",
            file=sys.stderr, flush=True,
        )
        return False
    finally:
        _GRADE_STATS["inflight"] -= 1
        elapsed = time.monotonic() - started
        if elapsed > _GRADE_SLOW_SECONDS:
            _GRADE_STATS["slow"] += 1
            print(
                f"\n[grade] slow grade took {elapsed:.1f}s; "
                f"qid={qid} dataset={ds}",
                file=sys.stderr, flush=True,
            )


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


class WatchdogStall(RuntimeError):
    """The run stopped making progress (done-counter frozen for minutes) —
    the watchdog dumps diagnostics and aborts with a non-zero exit code."""


# Watchdog tuning: poll the done-counter every 30s; a run that has not
# completed a single question for WATCHDOG_STALL_SECONDS is considered stalled
# (process-bounded grading should make real stalls nearly impossible, so a
# generous threshold keeps false positives away while still catching hangs).
WATCHDOG_POLL_SECONDS = 30.0
WATCHDOG_STALL_SECONDS = 420.0


def _dump_stall_diagnostics(inflight: dict, stalled_for: float) -> None:
    """Dump everything needed to debug a stalled run to stderr: thread
    stacks, asyncio task stacks, in-flight question ids and grader state."""
    print(
        f"\nwatchdog: no question finished for {stalled_for:.0f}s "
        f"(> {WATCHDOG_STALL_SECONDS:.0f}s) — dumping diagnostics",
        file=sys.stderr, flush=True,
    )
    print(
        f"watchdog: grader pool workers={_grader_pool_size()} "
        f"busy_grades={_GRADE_STATS['inflight']} "
        f"slow_grades={_GRADE_STATS['slow']} "
        f"grade_timeouts={_GRADE_STATS['timeouts']}",
        file=sys.stderr, flush=True,
    )
    if inflight:
        print("watchdog: in-flight questions:", file=sys.stderr, flush=True)
        for qid, ds in list(inflight.values()):
            print(f"  qid={qid} dataset={ds}", file=sys.stderr, flush=True)
    print("watchdog: thread stacks:", file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr)
    print("watchdog: asyncio task stacks:", file=sys.stderr, flush=True)
    for task in asyncio.all_tasks():
        print(f"  task {task.get_name()!r}:", file=sys.stderr, flush=True)
        for frame in (task.get_stack() or []):
            traceback.print_stack(frame, file=sys.stderr)


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


def interleave_plan(plan: list[tuple[str, list[dict]]]) -> list[tuple[str, dict]]:
    """Round-robin the per-dataset question lists: one question from each
    dataset in turn (1st of A, 1st of B, 1st of C, 2nd of A, ...) instead of
    exhausting the datasets one by one. Shorter datasets simply drop out of
    the rotation once exhausted."""
    grouped = [[(name, item) for item in items] for name, items in plan]
    return [
        pair
        for row in itertools.zip_longest(*grouped)
        for pair in row
        if pair is not None
    ]


def format_duration(seconds: float) -> str:
    """Human-readable duration with at most two significant units
    ('2d 5h', '3h 12m', '7m 45s', '42s'); '0s' for zero/negative input."""
    seconds = max(int(seconds), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    units = [(days, "d"), (hours, "h"), (minutes, "m"), (secs, "s")]
    # take the largest non-zero unit and the next one after it
    for i, (value, suffix) in enumerate(units):
        if value:
            second = units[i + 1] if i + 1 < len(units) else None
            parts = [f"{value}{suffix}"]
            if second is not None:
                parts.append(f"{second[0]}{second[1]}")
            return " ".join(parts)
    return "0s"


async def run(args: argparse.Namespace) -> int:
    # asyncio.Runner installs its own SIGINT handler for the duration of
    # asyncio.run(); override it so the two-phase handler below is active
    # while the benchmark is running (main() keeps one installed for the
    # phases outside the event loop; Runner restores it on exit).
    signal.signal(signal.SIGINT, _sigint_handler)
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
    # task -> (question id, dataset), used by the stall watchdog
    inflight = {}
    started = time.monotonic()

    results = {
        name: {"correct": 0, "skipped": 0, "done": 0, "mismatches": [],
               "error_examples": [], "total": len(items)}
        for name, items in plan
    }
    ds_names = [name for name, _ in plan]

    def progress_line() -> str:
        # per-dataset breakdown: done = graded + skipped for that dataset
        parts = [
            f"{name}: {results[name]['correct']} / "
            f"{results[name]['done'] - results[name]['correct'] - results[name]['skipped']}"
            for name in ds_names
        ]
        # skipped = request errors after all retries; excluded from the
        # graded total, so they never count towards the accuracy
        done = done_counter["done"]
        eta = ""
        if done:
            remaining = (time.monotonic() - started) / done * (total - done)
            eta = f" — ETA: {format_duration(remaining)}"
        # grader observability: live worker/busy counts and slow/timeout
        # counters turn silent degradation into something visible in-flight
        grader = (f"grader: {_GRADE_STATS['inflight']}/{_grader_pool_size()} "
                  f"busy, slow(>{_GRADE_SLOW_SECONDS:.0f}s): "
                  f"{_GRADE_STATS['slow']}, timeouts: {_GRADE_STATS['timeouts']}")
        return (f"\rprogress: {done}/{total}, "
                f"skipped: {skipped_counter['skipped']} — "
                f"{' | '.join(parts)} — [{grader}]{eta}")

    async def worker(session, name, item, results):
        task = asyncio.current_task()
        inflight[task] = (item.get("id", "?"), name)
        try:
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
                    # question exhausted all retries: skip it and exclude it
                    # from the accuracy denominator
                    results[name]["skipped"] += 1
                    results[name].setdefault("error_examples", []).append(
                        (item.get("id", "?"), str(exc)[:200])
                    )
                    response = None
                    skipped_counter["skipped"] += 1
                if response is not None:
                    correct = await _grade_timed(item, response, GRADE_TIMEOUT)
                    results[name]["correct"] += int(correct)
                    if not correct:
                        results[name]["mismatches"].append(
                            (item.get("id", "?"),
                             item["messages"][1]["content"][:120],
                             response[:160])
                        )
                results[name]["done"] += 1
                done_counter["done"] += 1
                print(progress_line(), end="", flush=True)
        finally:
            inflight.pop(task, None)

    async def watchdog():
        """Abort the run when nothing finishes for WATCHDOG_STALL_SECONDS:
        dump all stacks + in-flight questions, then raise WatchdogStall."""
        last_done = done_counter["done"]
        last_change = time.monotonic()
        while True:
            await asyncio.sleep(WATCHDOG_POLL_SECONDS)
            done = done_counter["done"]
            now = time.monotonic()
            if done != last_done:
                last_done, last_change = done, now
                continue
            stalled_for = now - last_change
            if stalled_for >= WATCHDOG_STALL_SECONDS:
                _dump_stall_diagnostics(inflight, stalled_for)
                raise WatchdogStall(
                    f"no progress for {stalled_for:.0f}s "
                    f"(done={done}/{total})")

    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 10))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for name, item in interleave_plan(plan):
            tasks.append(asyncio.create_task(worker(session, name, item, results)))
        workers_task = asyncio.gather(*tasks)
        watchdog_task = asyncio.create_task(watchdog(), name="watchdog")
        try:
            done, _pending = await asyncio.wait(
                {workers_task, watchdog_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            # whichever way we leave this block: stop the watchdog, cancel
            # (and reap) any still-running workers, and make sure no grader
            # worker process survives into interpreter shutdown (a killed/
            # stuck pool would block the concurrent.futures atexit join)
            watchdog_task.cancel()
            workers_task.cancel()
            await asyncio.gather(watchdog_task, workers_task,
                                 return_exceptions=True)
            _kill_grader_pool()

        if watchdog_task in done and not watchdog_task.cancelled() \
                and watchdog_task.exception() is not None:
            exc = watchdog_task.exception()
            if isinstance(exc, WatchdogStall):
                print(f"\nfatal: watchdog aborted the run: {exc}",
                      file=sys.stderr)
                return 3
            raise exc  # unexpected watchdog crash
        try:
            workers_task.result()  # propagates FatalEndpointError and crashes
        except FatalEndpointError as exc:
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


_SIGINT_STATE = {"count": 0}


def _sigint_handler(signum, frame):
    """First Ctrl-C: graceful cancel (raise KeyboardInterrupt, as the default
    handler does — asyncio.run then cancels all tasks; the grader pool is
    killed in run()'s cleanup so shutdown cannot block on a stuck worker).
    Second Ctrl-C: dump the stacks of ALL threads and exit immediately."""
    _SIGINT_STATE["count"] += 1
    if _SIGINT_STATE["count"] == 1:
        raise KeyboardInterrupt
    print("\nsecond SIGINT: dumping thread stacks and exiting",
          file=sys.stderr, flush=True)
    faulthandler.dump_traceback(file=sys.stderr)
    os._exit(130)


def main() -> int:
    signal.signal(signal.SIGINT, _sigint_handler)
    # Crash diagnostics for silent hangs: if the process stops making
    # progress (stuck C-level sympy arithmetic, event-loop stall, ...),
    # dump the stacks of all threads to stderr every 10 minutes.
    faulthandler.dump_traceback_later(600, repeat=True, file=sys.stderr)
    try:
        return asyncio.run(run(parse_args()))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        # guarantee no grader worker process can be joined at interpreter
        # exit (concurrent.futures' atexit hook would otherwise block on a
        # stuck worker and make even the graceful path hang)
        _kill_grader_pool()
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    sys.exit(main())
