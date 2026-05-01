"""tools/policy_retrieval_tool.py

Queries the Pinecone index populated by pinecone.ipynb.
Index: ecommerce-doc  (llama-text-embed-v2, integrated embedding)
Namespace: default
Source document: Generic E-Commerce Company_ Master Policy Compendium.docx

All answers returned by this tool carry chunk-level citations so the agent
can surface them in the UI — no invented policy text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from pinecone import Pinecone
from langchain_core.tools import tool

from settings import load_settings

logger = logging.getLogger(__name__)

cfg = load_settings()

# Number of top chunks to retrieve per query
TOP_K = 5


def _pc_index(cfg):
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    return pc.Index(host=cfg.pinecone_index_host)


@tool
def retrieve_policy(query: str) -> str:
    """
    Search the Generic E-Commerce Company Master Policy Compendium stored in
    Pinecone for chunks relevant to the given query.

    Returns cited excerpts so the agent can ground its response in actual
    policy text rather than hallucinated content.

    Args:
        query: Natural-language question about company policy
               (e.g. 'What is the refund window for electronics?').

    Returns:
        JSON string with a list of {chunk_text, chunk_index, source, score}
        objects ordered by relevance.
    """

    try:
        index = _pc_index(cfg)
        results = index.search(
            namespace=cfg.pinecone_namespace,
            query={"inputs": {"text": query}, "top_k": TOP_K},
        )
    except Exception as exc:
        logger.error("Pinecone search error: %s", exc)
        return json.dumps({"error": str(exc)})

    hits: list[dict[str, Any]] = []
    for match in results.get("result", {}).get("hits", []):
        fields = match.get("fields", {})
        hits.append(
            {
                "chunk_text": fields.get("chunk_text", ""),
                "chunk_index": fields.get("chunk_index", -1),
                "source": fields.get("source", "Master Policy Compendium"),
                "score": round(match.get("_score", 0.0), 4),
            }
        )

    if not hits:
        return json.dumps(
            {
                "policy_chunks": [],
                "note": "No relevant policy sections found for the query.",
            }
        )

    return json.dumps({"policy_chunks": hits, "query": query})
