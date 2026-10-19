# AGENTS.md

## What this repo is

- Data-only repo: four JSONL benchmark/dataset files, no source code, no build/test/lint config, no CI. There is nothing to build or run; verification means inspecting the JSONL files.
- Git repo has no commits yet (fresh `master`); only `.drcode/` is gitignored, everything else is untracked.

## Dataset format (`finmath-distill-*.jsonl`)

- One JSON object per line. Common schema: `source` (always `"finmath-distill"`), `messages` (a user/assistant pair), `id`, `type`.
- One file per task type: `enum` (~4.9k), `formula` (~3.9k), `math` (~4.1k), `yesno` (~13.2k). Counts live in the filename; the `type` field always matches the file.
- `id` pattern: `section-<n>-<n>-dataset-<type>:<index>` — sections are interleaved across the file, not sorted.
- Only `finmath-distill-math.jsonl` has an extra top-level `answer` field (the numeric final answer; assistant messages end with `Answer: <number>`).
- Quirk: in the `math` file, newlines inside `messages[].content` are stored as literal `\n` two-character sequences (double-escaped), not real newlines. Unescape them before rendering or diffing.
- Files total ~12 MB; don't read them whole — use `head`/`tail` or stream with `python3 -c` / `jq`.

## Repo config

- `drcode.json` holds local DrCode model config and contains an API key — treat it as sensitive; don't copy the key into docs, logs, or committed files.
- `.drcode/` contains DrCode task backlog files and is gitignored.
