"""
FastAPI wrapper for BibleRAG.

Run locally:
    uvicorn api:app --reload --port 8000

Endpoints:
    GET  /health
    POST /query         body: {"question": str, "k": int=25}  — full JSON
    POST /query/stream  same body  — SSE: mode, citation, graph_context,
                                     subgraph, token, done (or error)
"""

from __future__ import annotations

import json
import logging
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
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from bible_rag import BibleRAG

logger = logging.getLogger("bible_graph_rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] loading BibleRAG (verses, embeddings, graph)…")
    app.state.rag = BibleRAG()
    g = app.state.rag.graph
    print(f"[api] ready — {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")
    yield


app = FastAPI(title="BibleGraphRAG API", version="0.1.0", lifespan=lifespan)

# ALLOWED_ORIGINS is a comma-separated list, e.g.
#   "https://biblegraphrag.vercel.app,https://biblegraphrag-git-main.vercel.app"
# Defaults to the local Next.js dev origin — NOT "*" — so a deployment that
# forgets to set this env var fails closed (blocks browsers) instead of
# opening the API to every origin on the internet.
_origins = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-IP rate limiting on the OpenAI-backed endpoints — /query and
# /query/stream each cost real OpenAI spend, and the API has no auth, so an
# unlimited public endpoint is an open invitation to run up the bill.
# Override via QUERY_RATE_LIMIT, e.g. "30/minute".
QUERY_RATE_LIMIT = os.environ.get("QUERY_RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


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
@limiter.limit(QUERY_RATE_LIMIT)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    rag: BibleRAG = request.app.state.rag
    try:
        ans = rag.answer(req.question, k=req.k)
    except Exception:
        logger.exception("/query failed for question=%r", req.question)
        raise HTTPException(status_code=500, detail="Internal error while answering the question.")

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


def _sse(event: str, payload: object) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/query/stream")
@limiter.limit(QUERY_RATE_LIMIT)
def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    rag: BibleRAG = request.app.state.rag

    def event_gen():
        try:
            for event_name, payload in rag.answer_stream(req.question, k=req.k):
                yield _sse(event_name, payload)
        except Exception:
            logger.exception("/query/stream failed for question=%r", req.question)
            yield _sse("error", {"detail": "Internal error while answering the question."})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
