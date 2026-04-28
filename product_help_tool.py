"""tools/product_help_tool.py

Product-oriented help tool: given a product category and optional context,
it (1) retrieves relevant policy chunks from Pinecone and (2) returns
structured guidance for the category with policy citations.

The tool intentionally defers all policy wording to the Pinecone index
so every answer is grounded in the Master Policy Compendium.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from policy_retrieval_tool import retrieve_policy

logger = logging.getLogger(__name__)

# ── Category metadata known at build time ─────────────────────────────────────
# These labels correspond to the 'primary_category' values in the training data.
KNOWN_CATEGORIES = {
    "electronics":  "Electronics & Technology",
    "clothing":     "Apparel & Fashion",
    "beauty":       "Beauty & Personal Care",
    "books":        "Books & Media",
    "home":         "Home & Garden",
    "sports":       "Sports & Outdoors",
    "toys":         "Toys & Games",
}


@tool
def get_product_help(category: str, concern: str = "refund") -> str:
    """
    Return product-category-specific guidance on refunds, returns, or warranty,
    grounded in policy chunks retrieved from Pinecone.

    Args:
        category: Product category string
                  (electronics|clothing|beauty|books|home|sports|toys).
        concern:  What the agent needs help with — e.g. 'refund', 'return window',
                  'warranty', 'damaged item'. Defaults to 'refund'.

    Returns:
        JSON with category metadata and relevant policy excerpts (cited).
    """
    normalised = category.strip().lower()
    display_name = KNOWN_CATEGORIES.get(normalised, category.title())

    query = f"{display_name} {concern} policy"

    # Delegate to Pinecone retrieval — this tool is a focused wrapper
    policy_result_raw = retrieve_policy.invoke({"query": query})

    try:
        policy_result = json.loads(policy_result_raw)
    except json.JSONDecodeError:
        policy_result = {"error": policy_result_raw}

    return json.dumps(
        {
            "category": normalised,
            "category_display": display_name,
            "concern": concern,
            "policy_query": query,
            "policy_result": policy_result,
            "note": (
                "All guidance above is sourced from the Master Policy Compendium. "
                "Do not state policy details not present in policy_result."
            ),
        }
    )
