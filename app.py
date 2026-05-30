"""
BibleGraphRAG — Streamlit chatbot UI with live graph visualization.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]

import bible_graph as bg
from bible_rag import BibleRAG


ROOT = Path(__file__).parent

st.set_page_config(
    page_title="BibleGraphRAG",
    page_icon="📖",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading verses, embeddings, and graph…")
def get_rag() -> BibleRAG:
    return BibleRAG()


def render_graph(sg, height: int = 520) -> None:
    if sg is None or sg.number_of_nodes() == 0:
        st.info("No graph context available for this answer.")
        return
    html = bg.render_subgraph(sg, height_px=height)
    components.html(html, height=height + 30, scrolling=False)


def explore_panel(rag: BibleRAG) -> None:
    st.subheader("Explore the graph")
    g = rag.graph
    kinds = ["Person", "Place", "Theme"]
    kind = st.selectbox("Node kind", kinds, key="explore_kind")
    options = sorted(
        {d.get("name", n) for n, d in g.nodes(data=True) if d.get("kind") == kind},
        key=str.lower,
    )
    if not options:
        st.caption("(no nodes of this kind yet — run chapter extraction first)")
        return
    pick = st.selectbox(f"{kind}", options, key="explore_pick")
    hops = st.slider("Hops", 1, 3, 2, key="explore_hops")
    seed = bg._resolve_entity(g, pick)
    if not seed:
        st.warning("Couldn't locate that node.")
        return
    sg = bg.neighborhood(g, seed[0], hops=hops)
    st.caption(f"Subgraph: {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges")
    render_graph(sg, height=520)


def chat_panel(rag: BibleRAG) -> None:
    st.subheader("Ask a question")
    if "history" not in st.session_state:
        st.session_state.history = []  # list of dicts: {role, content, answer?}

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("citations"):
                with st.expander(f"📖 {len(msg['citations'])} citations"):
                    for c in msg["citations"]:
                        st.markdown(f"**{c['ref']}** — {c['text']}")

    k = st.session_state.get("retrieval_k", 25)
    q = st.chat_input("e.g. 'Who was Moses?', 'verses about forgiveness', 'Genesis 1:1'")
    if q:
        st.session_state.history.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            with st.spinner(f"Routing intent and retrieving up to {k} verses…"):
                ans = rag.answer(q, k=k)
            st.markdown(f"_(mode: `{ans.mode}` · {len(ans.citations)} citations)_")
            st.markdown(ans.answer)
            if ans.citations:
                with st.expander(f"📖 {len(ans.citations)} citations"):
                    for c in ans.citations:
                        st.markdown(f"**{c.ref}** — {c.text}")
        st.session_state.history.append({
            "role": "assistant",
            "content": ans.answer,
            "citations": [{"ref": c.ref, "text": c.text} for c in ans.citations],
            "mode": ans.mode,
        })
        st.session_state.last_subgraph = ans.subgraph


def main() -> None:
    rag = get_rag()
    counts = bg.node_counts(rag.graph)

    with st.sidebar:
        st.title("📖 BibleGraphRAG")
        st.caption("KJV · Databricks AI Gateway")
        st.markdown("**Graph stats**")
        for k in ("Book", "Chapter", "Verse", "Person", "Place", "Event", "Theme"):
            st.markdown(f"- {k}: `{counts.get(k, 0):,}`")
        st.divider()
        st.session_state.retrieval_k = st.slider(
            "Retrieval depth (verses)", 5, 100,
            st.session_state.get("retrieval_k", 25), step=5,
        )
        st.divider()
        explore_panel(rag)

    left, right = st.columns([2, 1])
    with left:
        chat_panel(rag)
    with right:
        st.subheader("Answer subgraph")
        sg = st.session_state.get("last_subgraph")
        render_graph(sg, height=520)


if __name__ == "__main__":
    main()
