"""
FastAPI wrapper for BibleRAG.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health
    POST /query   body: {"question": str, "k": int=25}
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")
if not os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_API"):
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_API"]

import networkx as nx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from bible_rag import BibleRAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] loading BibleRAG (verses, embeddings, graph)…")
    app.state.rag = BibleRAG()
    g = app.state.rag.graph
    print(f"[api] ready — {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    yield


app = FastAPI(title="BibleGraphRAG API", version="0.1.0", lifespan=lifespan)

# Permissive CORS for local dev; restrict before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(25, ge=1, le=100)


class CitationOut(BaseModel):
    ref: str
    text: str
    verse_id: str


class QueryResponse(BaseModel):
    question: str
    mode: str
    answer: str
    citations: list[CitationOut]
    graph_context: list[str]
    subgraph: dict


@app.get("/health")
def health(request: Request) -> dict:
    rag: BibleRAG = request.app.state.rag
    return {
        "status": "ok",
        "graph_nodes": rag.graph.number_of_nodes(),
        "graph_edges": rag.graph.number_of_edges(),
        "n_verses": int(len(rag.verses)),
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    rag: BibleRAG = request.app.state.rag
    try:
        ans = rag.answer(req.question, k=req.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}")

    if ans.subgraph is not None and ans.subgraph.number_of_nodes() > 0:
        sg_json = nx.node_link_data(ans.subgraph, edges="links")
    else:
        sg_json = {"directed": True, "multigraph": True, "nodes": [], "links": []}

    return QueryResponse(
        question=ans.question,
        mode=ans.mode,
        answer=ans.answer,
        citations=[
            CitationOut(ref=c.ref, text=c.text, verse_id=c.verse_id)
            for c in ans.citations
        ],
        graph_context=ans.graph_context,
        subgraph=sg_json,
    )
