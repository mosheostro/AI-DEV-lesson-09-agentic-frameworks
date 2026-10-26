"""
The RAG tool's engine (Exercise 3, server side).

Retrieval over the NovaOps IT knowledge base — the same idea as Lessons 6–7,
deliberately shrunk to a local in-memory FAISS index so this lab stays standalone
(no OpenSearch collection to stand up). The MCP server calls search_kb(); the
agent on the other side never sees any of this.

Each KB article is short, so we embed it whole as a single chunk. Titan returns
normalized vectors, so an inner-product index is cosine similarity.
"""

import functools
import json
import os
from pathlib import Path

import boto3
import faiss
import numpy as np
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "it_kb"
REGION = os.environ.get("AWS_REGION", "us-east-1")
EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIM = 1024   # Titan Text Embeddings V2 default width
TOP_K = 3

_bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def embed_text(text: str) -> list[float]:
    """One Titan embedding call. normalize=True so inner product == cosine
    similarity. The index and the query must use the SAME model, or the vector
    geometry is meaningless."""
    resp = _bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        body=json.dumps({"inputText": text, "dimensions": EMBED_DIM, "normalize": True}),
    )
    return json.loads(resp["body"].read())["embedding"]


@functools.lru_cache(maxsize=1)
def _index():
    """Embed every KB article and build the FAISS index. lru_cache makes this run
    once — on the first search, or when warm_index() is called at server startup."""
    docs = [(p.stem, p.read_text(encoding="utf-8")) for p in sorted(KB_DIR.glob("*.md"))]
    vectors = np.array([embed_text(text) for _, text in docs], dtype="float32")
    # Flat inner-product index = exact cosine search. Fine for 15 docs; Lesson 7
    # is where this becomes an approximate HNSW index in a managed store.
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vectors)
    return index, docs


def warm_index() -> int:
    """Force the index to build now (called at server startup). Returns doc count."""
    _, docs = _index()
    return len(docs)


def search_kb(query: str, k: int = TOP_K) -> list[dict]:
    """Return the top-k KB articles most similar to `query`, best first."""
    index, docs = _index()
    qvec = np.array([embed_text(query)], dtype="float32")
    scores, ids = index.search(qvec, k)
    return [
        {"article": docs[i][0], "score": round(float(s), 3), "text": docs[i][1]}
        for s, i in zip(scores[0], ids[0])
    ]
