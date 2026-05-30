"""
One-shot ingestion pipeline for BibleGraphRAG.

Steps:
    A. Parse kjv.txt ("<book chap:verse>\t<text>" per line) into data/verses.parquet
    B. Embed every verse with OpenAI text-embedding-3-small -> data/verse_embeddings.npy
    C. LLM-extract people/places/events/themes per chapter (resumable)
    D. Build the graph -> data/graph.json

Run with:
    python bible_ingest.py            # all steps, skipping cached work
    python bible_ingest.py --step A   # just parse
    python bible_ingest.py --step B   # just embed
    python bible_ingest.py --step C   # just extract
    python bible_ingest.py --step D   # just build graph
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
# Accept either OPENAI_API_KEY (standard) or OPENAI_API (this repo's .env).
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]

from openai import OpenAI  # noqa: E402

ROOT = Path(__file__).parent
KJV_PATH = ROOT / "kjv.txt"
DATA_DIR = ROOT / "data"
EXTRACT_DIR = DATA_DIR / "extractions"
VERSES_PARQUET = DATA_DIR / "verses.parquet"
EMBED_NPY = DATA_DIR / "verse_embeddings.npy"
GRAPH_JSON = DATA_DIR / "graph.json"

EMBED_MODEL = "text-embedding-3-small"
EXTRACT_MODEL = "gpt-4o-mini"

_client_singleton: OpenAI | None = None


def _client() -> OpenAI:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = OpenAI()
    return _client_singleton

# Canonical 66-book Protestant Bible ordering. The KJV file uses "Psalm"
# (singular); we normalize to "Psalms" to match conventional citation style.
BOOK_ORDER: list[str] = [
    "Genesis", "Exodus", "Leviticus", "Numbers", "Deuteronomy",
    "Joshua", "Judges", "Ruth", "1 Samuel", "2 Samuel",
    "1 Kings", "2 Kings", "1 Chronicles", "2 Chronicles", "Ezra",
    "Nehemiah", "Esther", "Job", "Psalms", "Proverbs",
    "Ecclesiastes", "Song of Solomon", "Isaiah", "Jeremiah", "Lamentations",
    "Ezekiel", "Daniel", "Hosea", "Joel", "Amos",
    "Obadiah", "Jonah", "Micah", "Nahum", "Habakkuk",
    "Zephaniah", "Haggai", "Zechariah", "Malachi",
    "Matthew", "Mark", "Luke", "John", "Acts",
    "Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
    "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
    "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
    "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude",
    "Revelation",
]
BOOK_RANK = {name: i for i, name in enumerate(BOOK_ORDER)}


_VERSE_LINE = re.compile(r"^([1-3]?\s?[A-Za-z][A-Za-z ]+?) (\d+):(\d+)\t(.+)$")


def _normalize_book(name: str) -> str:
    return "Psalms" if name == "Psalm" else name


def step_a_parse() -> pd.DataFrame:
    print(f"[A] reading {KJV_PATH}")
    with open(KJV_PATH, "r", encoding="utf-8-sig") as f:
        lines = [ln.rstrip("\r\n") for ln in f]
    print(f"[A] file lines: {len(lines)}")

    rows = []
    skipped = 0
    for i, line in enumerate(lines):
        m = _VERSE_LINE.match(line)
        if not m:
            skipped += 1
            continue
        book = _normalize_book(m.group(1))
        chapter = int(m.group(2))
        verse = int(m.group(3))
        text = m.group(4)
        if book not in BOOK_RANK:
            raise RuntimeError(f"unknown book at line {i}: {book!r}")
        rows.append({
            "idx": len(rows),
            "book": book,
            "chapter": chapter,
            "verse": verse,
            "ref": f"{book} {chapter}:{verse}",
            "text": text,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["book", "chapter", "verse"],
        key=lambda c: c.map(BOOK_RANK) if c.name == "book" else c,
        kind="stable",
    ).reset_index(drop=True)
    df["idx"] = df.index

    n_books = df["book"].nunique()
    n_chapters = df.groupby(["book", "chapter"]).ngroups
    print(f"[A] parsed {len(df)} verses across {n_books} books, "
          f"{n_chapters} chapters (skipped {skipped} non-verse lines)")
    print(f"[A] first: {df.iloc[0].ref} | {df.iloc[0].text[:60]!r}")
    print(f"[A] last : {df.iloc[-1].ref} | {df.iloc[-1].text[:60]!r}")

    if n_books != 66:
        raise RuntimeError(f"expected 66 books, got {n_books}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(VERSES_PARQUET)
    print(f"[A] wrote {VERSES_PARQUET}")
    return df



def step_b_embed(df: pd.DataFrame, batch: int = 128) -> np.ndarray:
    if EMBED_NPY.exists():
        arr = np.load(EMBED_NPY)
        if arr.shape[0] == len(df):
            print(f"[B] cached embeddings present: {arr.shape}")
            return arr
        print(f"[B] cached embeddings shape {arr.shape} != verses {len(df)}; regenerating")

    w = _client()
    n = len(df)
    out: list[list[float]] = [None] * n  # type: ignore[list-item]

    print(f"[B] embedding {n} verses with {EMBED_MODEL} (batch={batch})")
    t0 = time.perf_counter()
    last_print = t0
    for start in range(0, n, batch):
        end = min(start + batch, n)
        texts = df["text"].iloc[start:end].tolist()
        resp = w.embeddings.create(model=EMBED_MODEL, input=texts)
        for j, item in enumerate(resp.data):
            out[start + j] = item.embedding
        if time.perf_counter() - last_print > 5.0 or end == n:
            done = end
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed else 0
            eta = (n - done) / rate if rate else 0
            print(f"[B] {done}/{n}  rate={rate:.1f}/s  eta={eta:.0f}s")
            last_print = time.perf_counter()

    arr = np.asarray(out, dtype=np.float32)
    np.save(EMBED_NPY, arr)
    print(f"[B] wrote {EMBED_NPY} shape={arr.shape}")
    return arr


EXTRACT_SYSTEM = """You extract structured information from a Bible passage.
Return ONLY valid JSON, no prose, no markdown fences. Schema:
{
  "people":   [{"name": str, "canonical_id": str, "verses": [int]}],
  "places":   [{"name": str, "canonical_id": str, "verses": [int]}],
  "events":   [{"title": str, "verses": [int], "people": [str], "place": str|null}],
  "themes":   [{"label": str, "verses": [int]}]
}
Rules:
- "verses" lists verse numbers (1-based within the chapter) where the entity appears.
- "canonical_id" is a stable lowercase slug (e.g. "moses", "jerusalem", "the_lord").
  Use the same canonical_id for the same person/place across mentions.
- "themes" should be 2-5 short noun phrases capturing what the chapter is about
  (e.g. "creation", "covenant", "exodus", "forgiveness").
- If a category has no entries, return an empty list.
- Be conservative: only include named or clearly identified people/places.
"""


def _extract_one(book: str, chapter: int, verses: pd.DataFrame) -> dict:
    out_path = EXTRACT_DIR / f"{book.replace(' ', '_')}_{chapter}.json"
    if out_path.exists():
        try:
            return json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    passage_lines = [f"{r.verse}. {r.text}" for r in verses.itertuples()]
    user = f"Book: {book}\nChapter: {chapter}\nPassage:\n" + "\n".join(passage_lines)

    w = _client()
    resp = w.chat.completions.create(
        model=EXTRACT_MODEL,
        messages=[
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": user},
        ],
        max_tokens=4096,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    text = (resp.choices[0].message.content or "") if resp.choices else ""
    text = text.strip()
    try:
        data = json.loads(text)
    except Exception as exc:
        print(f"[C] parse error {book} {chapter}: {exc}; len={len(text)}; head={text[:120]!r}")
        data = {"people": [], "places": [], "events": [], "themes": [], "_error": str(exc), "_raw": text}
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def step_c_extract(df: pd.DataFrame, workers: int = 8, only_books: list[str] | None = None) -> dict:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    work: list[tuple[str, int, pd.DataFrame]] = []
    for (book, ch), grp in df.groupby(["book", "chapter"], sort=False):
        if only_books and book not in only_books:
            continue
        work.append((book, int(ch), grp.reset_index(drop=True)))
    total = len(work)
    print(f"[C] {total} chapters to process with {EXTRACT_MODEL} (workers={workers})")

    results: dict[tuple[str, int], dict] = {}
    t0 = time.perf_counter()
    done = 0
    last_print = t0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_key = {
            ex.submit(_extract_one, b, c, g): (b, c) for b, c, g in work
        }
        for fut in as_completed(fut_to_key):
            key = fut_to_key[fut]
            try:
                results[key] = fut.result()
            except Exception as exc:
                print(f"[C] FAILED {key}: {exc!r}")
                results[key] = {"people": [], "places": [], "events": [], "themes": [], "_error": repr(exc)}
            done += 1
            if time.perf_counter() - last_print > 10.0 or done == total:
                elapsed = time.perf_counter() - t0
                rate = done / elapsed if elapsed else 0
                eta = (total - done) / rate if rate else 0
                print(f"[C] {done}/{total}  rate={rate:.1f}/s  eta={eta:.0f}s")
                last_print = time.perf_counter()
    return results


def step_d_build_graph(df: pd.DataFrame) -> None:
    # Local import so steps A/B/C don't drag in networkx/pyvis.
    from bible_graph import build_graph, save_graph

    extractions: dict[tuple[str, int], dict] = {}
    if EXTRACT_DIR.exists():
        for p in EXTRACT_DIR.glob("*.json"):
            stem = p.stem  # "Genesis_1" or "1_Samuel_3"
            parts = stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            book = parts[0].replace("_", " ")
            try:
                chapter = int(parts[1])
            except ValueError:
                continue
            try:
                extractions[(book, chapter)] = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[D] skip bad extraction {p.name}: {exc}")
    print(f"[D] loaded {len(extractions)} chapter extractions")

    g = build_graph(df, extractions)
    save_graph(g, GRAPH_JSON)
    print(f"[D] wrote {GRAPH_JSON}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["A", "B", "C", "D", "all"], default="all")
    ap.add_argument("--books", nargs="*", default=None,
                    help="(C only) limit extraction to these book names")
    args = ap.parse_args()

    if args.step in ("A", "all"):
        df = step_a_parse()
    else:
        df = pd.read_parquet(VERSES_PARQUET)

    if args.step in ("B", "all"):
        step_b_embed(df)

    if args.step in ("C", "all"):
        step_c_extract(df, only_books=args.books)

    if args.step in ("D", "all"):
        step_d_build_graph(df)

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
