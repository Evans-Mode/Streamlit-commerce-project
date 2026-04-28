"""agents/refund_agent.py

ReAct-style refund-risk agent built with create_agent + @tool wrapping,
following the W6 Wednesday supervisor pattern.

Model : bedrock:us.amazon.nova-2-lite-v1:0  (set in .env as BEDROCK_MODEL_ID)
Tools : score_refund_risk, lookup_order, lookup_user_orders,
        retrieve_policy, get_product_help
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from config.settings import load_settings
from tools.order_lookup_tool import lookup_order, lookup_user_orders
from tools.policy_retrieval_tool import retrieve_policy
from tools.product_help_tool import get_product_help
from tools.scoring_tool import score_refund_risk

load_dotenv()
logger = logging.getLogger(__name__)

cfg = load_settings()
MODEL = cfg.bedrock_model_id  # e.g. "bedrock:us.amazon.nova-2-lite-v1:0"

# ── System prompt (internal operator role per task guidelines) ─────────────────
SYSTEM_PROMPT = """You are an internal refund-risk analyst for the Generic E-Commerce Company.
You help operations and customer-experience staff assess whether an order is likely to result
in a refund and what the correct next step is per company policy.

RULES:
1. SCORING   — Always call score_refund_risk before giving a risk verdict. Never invent a number.
2. ORDER DATA — Always call lookup_order or lookup_user_orders to ground order facts.
3. POLICY    — Every recommendation must cite a chunk from retrieve_policy or get_product_help.
               Format: [Policy §<chunk_index>]: "<quote>". Never state policy from memory.
4. RISK TIERS
   - low    (<30 %) : No immediate action. Note in record.
   - medium (30–60%): Flag for CS review; offer proactive resolution per policy.
   - high   (>60 %) : Escalate; follow return/refund SLA per policy.
5. GUARDRAILS — Refuse requests to execute refunds, cancel orders, or modify account data.
               This agent informs only — it is not the system of record.
               Refuse off-topic asks with a brief explanation.
6. MEMORY    — Use conversation history for follow-ups ("that order", "score it", "compare with…").

CRITICAL: Include ALL findings (score, order facts, policy citations) in your final response.
The Streamlit UI only surfaces your last message."""

# ── Sub-agent: scorer ─────────────────────────────────────────────────────────
scorer_agent = create_agent(
    name="scorer_agent",
    model=MODEL,
    tools=[score_refund_risk],
    system_prompt="""You are a scoring specialist.
Call score_refund_risk with the supplied feature JSON and return the full result.
CRITICAL: Return the raw JSON from the tool — probability, tier, and model name — in your final response.""",
)

# ── Sub-agent: order lookup ────────────────────────────────────────────────────
order_agent = create_agent(
    name="order_agent",
    model=MODEL,
    tools=[lookup_order, lookup_user_orders],
    system_prompt="""You are an order-data specialist.
Call lookup_order (single transaction) or lookup_user_orders (all orders for a user)
as appropriate and return all fields from the result.
CRITICAL: Include every field from the tool result in your final response.""",
)

# ── Sub-agent: policy RAG ──────────────────────────────────────────────────────
policy_agent = create_agent(
    name="policy_agent",
    model=MODEL,
    tools=[retrieve_policy, get_product_help],
    system_prompt="""You are a policy-retrieval specialist for the Master Policy Compendium.
Call retrieve_policy or get_product_help to find relevant chunks.
CRITICAL: Return every chunk_text with its chunk_index. Do not add any policy not in the results.""",
)


# ── Tool wrappers (W6 pattern) ─────────────────────────────────────────────────

@tool
def run_scoring(features_json: str) -> str:
    """Score refund risk for a customer order.

    Use for: any request that needs a refund probability or risk tier.
    Pass a JSON string with order/customer features.
    Do NOT use for policy questions or order lookups.
    """
    result = scorer_agent.invoke(
        {"messages": [{"role": "user", "content": features_json}]}
    )
    return result["messages"][-1].content


@tool
def run_order_lookup(request: str) -> str:
    """Look up order or user transaction data from Databricks.

    Use for: retrieving a specific transaction by ID, or all orders for a user_id.
    Include the transaction_id or user_id in the request string.
    Do NOT use for scoring or policy questions.
    """
    result = order_agent.invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return result["messages"][-1].content


@tool
def run_policy_lookup(query: str) -> str:
    """Retrieve policy guidance from the Master Policy Compendium via Pinecone.

    Use for: return windows, refund eligibility, shipping rules, warranty details,
    category-specific policies, or any question answered by company policy.
    Always returns cited chunk_text — never invented policy.
    Do NOT use for scoring or order data.
    """
    result = policy_agent.invoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result["messages"][-1].content


# ── Supervisor agent ──────────────────────────────────────────────────────────
supervisor = create_agent(
    name="refund_risk_supervisor",
    model=MODEL,
    tools=[run_scoring, run_order_lookup, run_policy_lookup],
    system_prompt=SYSTEM_PROMPT,
)


# ── Public API ────────────────────────────────────────────────────────────────

def _parse_structured_state(messages: list) -> dict[str, Any]:
    """
    Walk tool-result messages and extract risk_result, order_data,
    and policy_citations for the Streamlit UI.
    """
    risk_result: dict = {}
    order_data: dict = {}
    policy_citations: list = []

    for msg in messages:
        raw = getattr(msg, "content", "")
        if not isinstance(raw, str):
            continue
        # Tool results arrive as plain text from sub-agents; try to parse JSON
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Not JSON — skip structured extraction
            continue

        if "refund_probability" in parsed:
            risk_result = parsed
        if "transaction_id" in parsed or "transactions" in parsed:
            order_data = parsed
        if "policy_chunks" in parsed:
            policy_citations.extend(parsed["policy_chunks"])
        if "policy_result" in parsed:
            inner = parsed["policy_result"]
            if isinstance(inner, dict) and "policy_chunks" in inner:
                policy_citations.extend(inner["policy_chunks"])

    # Deduplicate citations by chunk_index
    seen: set = set()
    unique: list = []
    for c in policy_citations:
        key = c.get("chunk_index", id(c))
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return {
        "risk_result": risk_result,
        "order_data": order_data,
        "policy_citations": unique,
    }


def run_agent(
    user_message: str,
    history: list[dict] | None = None,
) -> dict[str, Any]:
    """
    Run the refund-risk supervisor for one user turn.

    Args:
        user_message: The operator's question or request.
        history:      Prior turns as [{"role": "user"|"assistant", "content": str}, …].
                      Pass this to enable multi-turn memory (task-guidelines requirement).

    Returns:
        {
          "final_answer":     str,
          "risk_result":      dict,   # refund_probability, risk_tier, model
          "order_data":       dict,   # raw transaction / user order data
          "policy_citations": list,   # [{chunk_text, chunk_index, source, score}, …]
          "messages":         list,   # full message list for trace view
        }
    """
    prior: list[dict] = history or []
    messages = prior + [{"role": "user", "content": user_message}]

    result = supervisor.invoke({"messages": messages})
    all_messages = result["messages"]

    final_answer = ""
    for msg in reversed(all_messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    structured = _parse_structured_state(all_messages)

    return {
        "final_answer": final_answer,
        "messages": all_messages,
        **structured,
    }
