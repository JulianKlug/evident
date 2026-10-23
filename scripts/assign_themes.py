"""Assign every current recommendation to one theme; the frozen output is read by `evident publish`.

    out/tables/recommendations.csv (current editions) + corpus/manifest.csv
        │ guideline topic_id in topic_themes.csv?  yes → that theme              (method "topic")
        │ no → LLM: 2–5 word clinical topic ("phrase"), then one taxonomy theme   (method "llm")
        ▼
    corpus/themes/assignments.csv   existing keys are kept: only new recommendations are assigned

The taxonomy was drafted by an LLM from the recommendations' topics and edited by a clinician;
change it there, then delete the affected rows here to reassign them.

Usage:
    python scripts/assign_themes.py [--model qwen3:14b]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evident.corpus import load_manifest  # noqa: E402
from evident.themes import OTHER, rec_key, specialty  # noqa: E402
from extraction.llm_client import OllamaClient, ThinkMode  # noqa: E402

_RECS = "out/tables/recommendations.csv"
_MANIFEST = "corpus/manifest.csv"
_TAXONOMY = "corpus/themes/taxonomy.csv"
_TOPIC_THEMES = "corpus/themes/topic_themes.csv"
_OUT = "corpus/themes/assignments.csv"
_MODEL = "qwen3:14b"

_CURRENT = "current"
_TOPIC, _LLM = "topic", "llm"
_COLUMNS = ("key", "doi", "society", "method", "phrase", "theme")
_BATCH = 25
_TEXT_CHARS = 400     # phrase prompt
_CONTEXT_CHARS = 300  # assignment prompt
_NUM_CTX = 16384
_SEED = 0
_ANY_OBJECT = {"type": "object"}

_PHRASE_PROMPT = (
    "For each clinical recommendation, write the clinical topic it is about in 2-5 words: the condition and "
    "intervention domain (e.g. 'empirical antimicrobial therapy', 'enteral nutrition timing', 'fluid "
    "resuscitation', 'glucose control'). Ignore the population (adults, children) and the strength or "
    'direction. Reply as JSON {{"1": "topic", "2": "topic", ...}}.\n\n{listing}')
_ASSIGN_PROMPT = (
    "Assign each {specialty} recommendation (its topic in brackets, then its text) to exactly one theme from "
    "this list; use the theme name verbatim:\n{menu}\n- " + OTHER + ": fits none of the above\n\n"
    'Recommendations:\n{listing}\n\nReply as JSON {{"1": "theme name", "2": "theme name", ...}}.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recommendations", default=_RECS)
    parser.add_argument("--manifest", default=_MANIFEST)
    parser.add_argument("--taxonomy", default=_TAXONOMY)
    parser.add_argument("--topic-themes", default=_TOPIC_THEMES)
    parser.add_argument("--out", default=_OUT)
    parser.add_argument("--model", default=_MODEL)
    args = parser.parse_args()

    topic_of = {e.doi.lower(): e.topic_id for e in load_manifest(args.manifest)}
    topic_themes = {r["topic_id"]: r["theme"] for r in _read(args.topic_themes)}
    taxonomy: dict[str, list[tuple[str, str]]] = {}
    for r in _read(args.taxonomy):
        taxonomy.setdefault(r["specialty"], []).append((r["theme"], r["scope"]))

    kept = {r["key"]: r for r in _read(args.out)} if os.path.exists(args.out) else {}
    recs = [r for r in _read(args.recommendations) if r["edition"] == _CURRENT]
    new = {}
    for r in recs:
        key = rec_key(r["doi"], r["text"])
        if key not in kept and key not in new:
            new[key] = r

    # Single-condition guidelines: the theme comes from the topic, no LLM
    out = dict(kept)
    to_llm = []
    for key, r in new.items():
        theme = topic_themes.get(topic_of.get(r["doi"].lower(), ""))
        if theme:
            out[key] = _row(key, r, _TOPIC, "", theme)
        else:
            to_llm.append((key, r))

    missing = sorted({specialty(r["society"]) for _, r in to_llm} - set(taxonomy))
    if missing:
        print(f"no taxonomy for specialty: {', '.join(missing)}", file=sys.stderr)
        return 1

    client = OllamaClient(args.model)
    for name in sorted({specialty(r["society"]) for _, r in to_llm}):
        batch = [(k, r) for k, r in to_llm if specialty(r["society"]) == name]
        for start in range(0, len(batch), _BATCH):
            chunk = batch[start:start + _BATCH]
            phrases = _phrases(client, [r["text"] for _, r in chunk])
            themes = _assign(client, name, taxonomy[name], [(p, r["text"]) for p, (_, r) in zip(phrases, chunk)])
            for (key, r), phrase, theme in zip(chunk, phrases, themes):
                out[key] = _row(key, r, _LLM, phrase, theme)
            print(f"{name}: {start + len(chunk)}/{len(batch)}", flush=True)

    _write(args.out, out)
    print(f"{len(new)} new ({len(to_llm)} by {args.model}), {len(kept)} kept → {args.out}")
    return 0


def _phrases(client: OllamaClient, texts: list[str]) -> list[str]:
    listing = "\n".join(f"{i}. {t[:_TEXT_CHARS]}" for i, t in enumerate(texts, start=1))
    answer = _ask(client, _PHRASE_PROMPT.format(listing=listing))
    return [str(answer.get(str(i), "")).strip().lower() for i in range(1, len(texts) + 1)]


def _assign(client: OllamaClient, name: str, themes: list[tuple[str, str]], recs: list[tuple[str, str]]) -> list[str]:
    """Theme per rec; an answer outside the list is OTHER. "Fluids: fluid resuscitation…" counts as "Fluids"."""
    menu = "\n".join(f"- {t}: {s}" for t, s in themes)
    listing = "\n".join(f"{i}. [{p}] {t[:_CONTEXT_CHARS]}" for i, (p, t) in enumerate(recs, start=1))
    answer = _ask(client, _ASSIGN_PROMPT.format(specialty=name.lower(), menu=menu, listing=listing))
    by_lower = {t.lower(): t for t, _ in themes}
    return [by_lower.get(str(answer.get(str(i), "")).split(":")[0].strip().lower(), OTHER)
            for i in range(1, len(recs) + 1)]


def _ask(client: OllamaClient, prompt: str) -> dict:
    response = client.generate_json(prompt, _ANY_OBJECT, num_ctx=_NUM_CTX, seed=_SEED, think=ThinkMode.OFF)
    try:
        answer = json.loads(response.raw_text)
    except json.JSONDecodeError:
        return {}
    return answer if isinstance(answer, dict) else {}


def _row(key: str, r: dict, method: str, phrase: str, theme: str) -> dict:
    return {"key": key, "doi": r["doi"], "society": r["society"], "method": method, "phrase": phrase,
            "theme": theme}


def _read(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path: str, rows: dict[str, dict]) -> None:
    """Sorted by guideline then key: a diff shows which guideline changed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows.values(), key=lambda r: (r["doi"], r["key"])))


if __name__ == "__main__":
    sys.exit(main())
