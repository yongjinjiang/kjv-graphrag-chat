"""
Graph schema, build, query helpers, and pyvis visualization for BibleGraphRAG.

Node types: Book, Chapter, Verse, Person, Place, Event, Theme.
Edge types: CONTAINS, MENTIONS, OCCURS_IN, INVOLVES, EXPRESSES.

Node IDs:
    Book          ->  "B::Genesis"
    Chapter       ->  "C::Genesis::1"
    Verse         ->  "V::Genesis::1::1"
    Person/Place  ->  "P::moses" / "L::jerusalem"   (canonical_id slug)
    Event         ->  "E::Exodus::14::crossing_the_red_sea"
    Theme         ->  "T::covenant"
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd

# ---------- ID helpers ----------

def book_id(book: str) -> str:
    return f"B::{book}"

def chapter_id(book: str, chapter: int) -> str:
    return f"C::{book}::{chapter}"

def verse_id(book: str, chapter: int, verse: int) -> str:
    return f"V::{book}::{chapter}::{verse}"

def person_id(canonical: str) -> str:
    return f"P::{_slug(canonical)}"

def place_id(canonical: str) -> str:
    return f"L::{_slug(canonical)}"

def event_id(book: str, chapter: int, title: str) -> str:
    return f"E::{book}::{chapter}::{_slug(title)}"

def theme_id(label: str) -> str:
    return f"T::{_slug(label)}"


_SLUG_RE = re.compile(r"[^a-z0-9]+")
def _slug(s: str) -> str:
    return _SLUG_RE.sub("_", (s or "").lower()).strip("_") or "unknown"


# ---------- Build ----------

def build_graph(verses_df: pd.DataFrame, extractions: dict[tuple[str, int], dict] | None = None) -> nx.MultiDiGraph:
    """Build the BibleGraphRAG graph from verses + per-chapter LLM extractions.

    extractions: dict keyed by (book, chapter) -> {people, places, events, themes}.
                 If None or empty, only the structural backbone is built.
    """
    g = nx.MultiDiGraph()

    # ---- structural backbone: Book -> Chapter -> Verse ----
    for book, book_df in verses_df.groupby("book", sort=False):
        b_id = book_id(book)
        g.add_node(b_id, kind="Book", label=book, name=book)

        for ch, ch_df in book_df.groupby("chapter", sort=False):
            c_id = chapter_id(book, int(ch))
            g.add_node(c_id, kind="Chapter", label=f"{book} {ch}",
                       book=book, chapter=int(ch))
            g.add_edge(b_id, c_id, kind="CONTAINS")

            for r in ch_df.itertuples():
                v_id = verse_id(book, int(ch), int(r.verse))
                g.add_node(
                    v_id, kind="Verse",
                    label=r.ref, ref=r.ref, idx=int(r.idx),
                    book=book, chapter=int(ch), verse=int(r.verse),
                    text=r.text,
                )
                g.add_edge(c_id, v_id, kind="CONTAINS")

    if not extractions:
        return g

    # ---- entity overlay ----
    # Track display names so we can pick the most common surface form.
    person_names: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    place_names: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    theme_count: dict[str, int] = defaultdict(int)

    for (book, chapter), data in extractions.items():
        if not isinstance(data, dict):
            continue
        b_id_ = book_id(book)
        if b_id_ not in g:
            # Skip extractions for books we don't have; should not happen.
            continue

        for p in data.get("people", []) or []:
            if isinstance(p, str):
                p = {"name": p, "canonical_id": p, "verses": []}
            elif not isinstance(p, dict):
                continue
            cid = p.get("canonical_id") or p.get("name")
            if not cid:
                continue
            slug = _slug(cid)
            pid = person_id(slug)
            person_names[slug][p.get("name") or cid] += 1
            g.add_node(pid, kind="Person", label=p.get("name") or cid,
                       canonical_id=slug, name=p.get("name") or cid)
            for v_no in p.get("verses") or []:
                try:
                    v_id_ = verse_id(book, int(chapter), int(v_no))
                except (TypeError, ValueError):
                    continue
                if v_id_ in g:
                    g.add_edge(v_id_, pid, kind="MENTIONS")

        for pl in data.get("places", []) or []:
            if isinstance(pl, str):
                pl = {"name": pl, "canonical_id": pl, "verses": []}
            elif not isinstance(pl, dict):
                continue
            cid = pl.get("canonical_id") or pl.get("name")
            if not cid:
                continue
            slug = _slug(cid)
            lid = place_id(slug)
            place_names[slug][pl.get("name") or cid] += 1
            g.add_node(lid, kind="Place", label=pl.get("name") or cid,
                       canonical_id=slug, name=pl.get("name") or cid)
            for v_no in pl.get("verses") or []:
                try:
                    v_id_ = verse_id(book, int(chapter), int(v_no))
                except (TypeError, ValueError):
                    continue
                if v_id_ in g:
                    g.add_edge(v_id_, lid, kind="MENTIONS")

        for ev in data.get("events", []) or []:
            if isinstance(ev, str):
                ev = {"title": ev, "verses": [], "people": [], "place": None}
            elif not isinstance(ev, dict):
                continue
            title = ev.get("title")
            if not title:
                continue
            eid = event_id(book, int(chapter), title)
            g.add_node(eid, kind="Event", label=title, name=title,
                       book=book, chapter=int(chapter))
            for v_no in ev.get("verses") or []:
                try:
                    v_id_ = verse_id(book, int(chapter), int(v_no))
                except (TypeError, ValueError):
                    continue
                if v_id_ in g:
                    g.add_edge(eid, v_id_, kind="OCCURS_IN")
            for pname in ev.get("people") or []:
                if not isinstance(pname, str):
                    continue
                slug = _slug(pname)
                pid = person_id(slug)
                if pid in g:
                    g.add_edge(eid, pid, kind="INVOLVES")
            place_name = ev.get("place")
            if isinstance(place_name, str) and place_name:
                lid = place_id(_slug(place_name))
                if lid in g:
                    g.add_edge(eid, lid, kind="INVOLVES")

        for th in data.get("themes", []) or []:
            if isinstance(th, str):
                th = {"label": th, "verses": []}
            elif not isinstance(th, dict):
                continue
            label = th.get("label")
            if not label:
                continue
            slug = _slug(label)
            tid = theme_id(slug)
            theme_count[slug] += 1
            g.add_node(tid, kind="Theme", label=label, name=label, canonical_id=slug)
            for v_no in th.get("verses") or []:
                try:
                    v_id_ = verse_id(book, int(chapter), int(v_no))
                except (TypeError, ValueError):
                    continue
                if v_id_ in g:
                    g.add_edge(v_id_, tid, kind="EXPRESSES")

    # Pick the most common surface form as the display label for entity nodes.
    for slug, counts in person_names.items():
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        nid = person_id(slug)
        if nid in g:
            g.nodes[nid]["label"] = best
            g.nodes[nid]["name"] = best
    for slug, counts in place_names.items():
        best = max(counts.items(), key=lambda kv: kv[1])[0]
        nid = place_id(slug)
        if nid in g:
            g.nodes[nid]["label"] = best
            g.nodes[nid]["name"] = best

    return g


# ---------- Persistence ----------

def save_graph(g: nx.MultiDiGraph, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(g, edges="links")
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def load_graph(path: Path) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, directed=True, multigraph=True, edges="links")


def node_counts(g: nx.MultiDiGraph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for _, d in g.nodes(data=True):
        counts[d.get("kind", "?")] += 1
    return dict(counts)


# ---------- Query helpers ----------

_REF_PAT = re.compile(
    r"^\s*([1-3]?\s?[A-Za-z][A-Za-z ]+?)\s+(\d+)(?::(\d+)(?:\s*-\s*(\d+))?)?\s*$"
)

def parse_ref(ref: str) -> tuple[str, int, int | None, int | None] | None:
    """Parse 'Genesis 1:1', 'John 3:16-17', 'Psalms 23'.
    Returns (book, chapter, verse_start, verse_end) or None.
    """
    m = _REF_PAT.match(ref)
    if not m:
        return None
    book = re.sub(r"\s+", " ", m.group(1)).strip()
    book = "Psalms" if book.lower() == "psalm" else book
    chapter = int(m.group(2))
    v1 = int(m.group(3)) if m.group(3) else None
    v2 = int(m.group(4)) if m.group(4) else v1
    return book, chapter, v1, v2


def verses_by_ref(g: nx.MultiDiGraph, ref: str) -> list[str]:
    """Return verse node IDs matching a textual reference."""
    parsed = parse_ref(ref)
    if not parsed:
        return []
    book, chapter, v1, v2 = parsed
    c_id = chapter_id(book, chapter)
    if c_id not in g:
        return []
    if v1 is None:
        return [v for _, v, d in g.out_edges(c_id, data=True) if d.get("kind") == "CONTAINS"]
    out = []
    end = v2 if v2 is not None else v1
    for v in range(v1, end + 1):
        vid = verse_id(book, chapter, v)
        if vid in g:
            out.append(vid)
    return out


def verses_mentioning(g: nx.MultiDiGraph, query: str) -> list[str]:
    """Return verse node IDs that MENTION an entity matching the query."""
    slug = _slug(query)
    candidates = [person_id(slug), place_id(slug)]
    out: list[str] = []
    for nid in candidates:
        if nid not in g:
            continue
        for u, _, d in g.in_edges(nid, data=True):
            if d.get("kind") == "MENTIONS":
                out.append(u)
    # Fallback: substring match on entity names.
    if not out:
        ql = query.lower()
        for nid, data in g.nodes(data=True):
            if data.get("kind") in ("Person", "Place") and ql in (data.get("name") or "").lower():
                for u, _, d in g.in_edges(nid, data=True):
                    if d.get("kind") == "MENTIONS":
                        out.append(u)
    return list(dict.fromkeys(out))


def verses_by_theme(g: nx.MultiDiGraph, label: str) -> list[str]:
    slug = _slug(label)
    tid = theme_id(slug)
    if tid not in g:
        # Fallback: substring on theme names
        ql = label.lower()
        candidates = [n for n, d in g.nodes(data=True)
                      if d.get("kind") == "Theme" and ql in (d.get("name") or "").lower()]
    else:
        candidates = [tid]
    out: list[str] = []
    for cid in candidates:
        for u, _, d in g.in_edges(cid, data=True):
            if d.get("kind") == "EXPRESSES":
                out.append(u)
    return list(dict.fromkeys(out))


def events_for(g: nx.MultiDiGraph, person: str) -> list[str]:
    pid = person_id(person)
    if pid not in g:
        ql = person.lower()
        pids = [n for n, d in g.nodes(data=True)
                if d.get("kind") == "Person" and ql in (d.get("name") or "").lower()]
    else:
        pids = [pid]
    out: list[str] = []
    for p in pids:
        for u, _, d in g.in_edges(p, data=True):
            if d.get("kind") == "INVOLVES" and g.nodes[u].get("kind") == "Event":
                out.append(u)
    return list(dict.fromkeys(out))


def relation_path(g: nx.MultiDiGraph, a: str, b: str, max_hops: int = 4) -> list[str]:
    """Find a short undirected path between two entity-name queries."""
    src_candidates = _resolve_entity(g, a)
    dst_candidates = _resolve_entity(g, b)
    if not src_candidates or not dst_candidates:
        return []
    ug = g.to_undirected(as_view=True)
    best: list[str] | None = None
    for s in src_candidates:
        for t in dst_candidates:
            try:
                path = nx.shortest_path(ug, s, t)
            except nx.NetworkXNoPath:
                continue
            if len(path) - 1 <= max_hops and (best is None or len(path) < len(best)):
                best = path
    return best or []


def _resolve_entity(g: nx.MultiDiGraph, name: str) -> list[str]:
    slug = _slug(name)
    out: list[str] = []
    for cand in (person_id(slug), place_id(slug), theme_id(slug)):
        if cand in g:
            out.append(cand)
    if out:
        return out
    ql = name.lower()
    for nid, d in g.nodes(data=True):
        if d.get("kind") in ("Person", "Place", "Theme") and ql in (d.get("name") or "").lower():
            out.append(nid)
    return out


def neighborhood(g: nx.MultiDiGraph, node: str, hops: int = 2) -> nx.MultiDiGraph:
    """Return the subgraph within `hops` of `node` (treated as undirected)."""
    if node not in g:
        return nx.MultiDiGraph()
    ug = g.to_undirected(as_view=True)
    nodes = set([node])
    frontier = {node}
    for _ in range(hops):
        nxt = set()
        for n in frontier:
            nxt.update(ug.neighbors(n))
        nodes.update(nxt)
        frontier = nxt
    return g.subgraph(nodes).copy()


def subgraph_for_verses(g: nx.MultiDiGraph, verse_ids: Iterable[str], extra_hops: int = 1) -> nx.MultiDiGraph:
    """Build a subgraph containing the given verses + their immediate context."""
    seeds = [v for v in verse_ids if v in g]
    if not seeds:
        return nx.MultiDiGraph()
    ug = g.to_undirected(as_view=True)
    nodes: set[str] = set(seeds)
    # Always pull in chapter and book ancestors.
    for v in seeds:
        d = g.nodes[v]
        nodes.add(chapter_id(d["book"], d["chapter"]))
        nodes.add(book_id(d["book"]))
    # Plus extra_hops out from verse nodes (gets entities/themes/events).
    frontier = set(seeds)
    for _ in range(extra_hops):
        nxt = set()
        for n in frontier:
            nxt.update(ug.neighbors(n))
        nodes.update(nxt)
        frontier = nxt
    return g.subgraph(nodes).copy()


# ---------- Visualization ----------

_KIND_COLOR = {
    "Book":    "#1f77b4",
    "Chapter": "#aec7e8",
    "Verse":   "#dddddd",
    "Person":  "#d62728",
    "Place":   "#2ca02c",
    "Event":   "#ff7f0e",
    "Theme":   "#9467bd",
}
_KIND_SIZE = {
    "Book":    34,
    "Chapter": 22,
    "Verse":   12,
    "Person":  26,
    "Place":   24,
    "Event":   24,
    "Theme":   22,
}


def render_subgraph(sg: nx.MultiDiGraph, height_px: int = 600) -> str:
    """Render `sg` as standalone HTML using pyvis. Returns HTML string."""
    from pyvis.network import Network

    net = Network(
        height=f"{height_px}px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        cdn_resources="remote",
        notebook=False,
    )
    net.barnes_hut(spring_length=120, gravity=-12000, damping=0.35)

    for nid, d in sg.nodes(data=True):
        kind = d.get("kind", "?")
        label = d.get("label") or d.get("name") or nid
        if kind == "Verse":
            short = (d.get("text") or "")[:140]
            title = f"{d.get('ref') or label}\n\n{short}"
            display = d.get("ref") or label
        else:
            title = f"{kind}: {label}"
            display = label
        net.add_node(
            nid,
            label=display,
            title=title,
            color=_KIND_COLOR.get(kind, "#888888"),
            size=_KIND_SIZE.get(kind, 16),
            shape="dot" if kind != "Verse" else "square",
        )

    for u, v, d in sg.edges(data=True):
        net.add_edge(u, v, title=d.get("kind", ""), arrows="to")

    net.set_options(json.dumps({
        "interaction": {"hover": True, "tooltipDelay": 80},
        "physics": {"stabilization": {"iterations": 120}},
    }))
    return net.generate_html(notebook=False)


# ---------- CLI for ad-hoc inspection ----------

if __name__ == "__main__":
    import argparse, sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="data/graph.json")
    ap.add_argument("--verses", default="data/verses.parquet")
    ap.add_argument("--build-from-verses-only", action="store_true",
                    help="Build a structural-only graph from verses parquet and save.")
    ap.add_argument("--ref", help="Look up a reference, e.g. 'Genesis 1:1' or 'John 3:16-17'.")
    args = ap.parse_args()

    if args.build_from_verses_only:
        df = pd.read_parquet(args.verses)
        g = build_graph(df, extractions=None)
        save_graph(g, Path(args.graph))
        print(f"saved {args.graph}")
        print("counts:", node_counts(g))
        sys.exit(0)

    g = load_graph(Path(args.graph))
    print(f"loaded {args.graph}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
    print("counts:", node_counts(g))
    if args.ref:
        vids = verses_by_ref(g, args.ref)
        for v in vids:
            d = g.nodes[v]
            print(f"  {d.get('ref')}: {d.get('text','')[:90]}")
