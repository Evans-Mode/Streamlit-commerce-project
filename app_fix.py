"""app.py — Internal Refund Risk Console

Business console for operations and customer-experience staff.
Layout : chat interface (agent) + metrics sidebar (risk gauge, order stats).
Memory : full conversation history passed to the agent each turn.
Alerts : banner when score crosses threshold or policy surfaces a hard rule.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import streamlit as st

# ── Path fix (works regardless of CWD on Windows or Mac) ─────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Refund Risk Console",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD = 0.60   # matches agent system-prompt tier
MED_RISK_THRESHOLD  = 0.30

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-low    {background:#d4edda;color:#155724;padding:8px 16px;border-radius:6px;font-weight:700;}
.risk-medium {background:#fff3cd;color:#856404;padding:8px 16px;border-radius:6px;font-weight:700;}
.risk-high   {background:#f8d7da;color:#721c24;padding:8px 16px;border-radius:6px;font-weight:700;}
.alert-high  {background:#f8d7da;color:#721c24;padding:12px 16px;border-radius:6px;
              border-left:5px solid #721c24;margin-bottom:12px;font-weight:600;}
.alert-med   {background:#fff3cd;color:#856404;padding:12px 16px;border-radius:6px;
              border-left:5px solid #856404;margin-bottom:12px;font-weight:600;}
.citation    {background:#f0f4ff;border-left:4px solid #4a6cf7;padding:10px 14px;
              border-radius:4px;margin-bottom:8px;font-size:0.87rem;}
.cite-meta   {color:#666;font-size:0.76rem;margin-top:4px;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "chat_history":    [],   # [{role, content}, …] — fed to agent each turn
    "display_history": [],   # [{role, content}, …] — shown in chat UI
    "last_result":     None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tier(prob: float) -> str:
    if prob >= HIGH_RISK_THRESHOLD:
        return "high"
    if prob >= MED_RISK_THRESHOLD:
        return "medium"
    return "low"


def show_risk_badge(rr: dict) -> None:
    prob = rr.get("refund_probability")
    if prob is None:
        return
    tier = _tier(float(prob))
    label = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}[tier]
    st.markdown(f'<div class="risk-{tier}">{label} — {prob:.1%}</div>', unsafe_allow_html=True)
    st.progress(float(prob))


def show_alert(rr: dict) -> None:
    """Banner above chat when score crosses a threshold — driven from model output."""
    prob = rr.get("refund_probability")
    if prob is None:
        return
    prob = float(prob)
    if prob >= HIGH_RISK_THRESHOLD:
        st.markdown(
            f'<div class="alert-high">⚠ HIGH REFUND RISK ({prob:.1%}) — '
            f'Escalate per return/refund SLA. Do not process changes via this tool.</div>',
            unsafe_allow_html=True,
        )
    elif prob >= MED_RISK_THRESHOLD:
        st.markdown(
            f'<div class="alert-med">⚡ MEDIUM REFUND RISK ({prob:.1%}) — '
            f'Flag for CS review and offer proactive resolution.</div>',
            unsafe_allow_html=True,
        )


def show_citations(citations: list[dict]) -> None:
    if not citations:
        return
    st.caption("Policy citations — sourced verbatim from the Master Policy Compendium")
    for c in citations:
        idx   = c.get("chunk_index", "?")
        text  = c.get("chunk_text", "")
        score = c.get("score", 0.0)
        src   = c.get("source", "Master Policy Compendium")
        st.markdown(
            f'<div class="citation"><b>[Policy §{idx}]</b> {text}'
            f'<div class="cite-meta">Source: {src} | Relevance: {score:.4f}</div></div>',
            unsafe_allow_html=True,
        )


def show_order_table(order_data: dict) -> None:
    if not order_data:
        return
    import pandas as pd
    if "transactions" in order_data:
        df = pd.DataFrame(order_data["transactions"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Purchases",    order_data.get("purchase_count", 0))
        c2.metric("Refunds",      order_data.get("refund_count", 0))
        c3.metric("Refund Rate",  f"{order_data.get('refund_rate', 0):.1%}")
    else:
        st.json(order_data)


# ── Sidebar — metrics area ────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Refund Risk Console")
    st.caption("Internal — operations & customer-experience only")
    st.divider()

    # Risk gauge (persists across turns)
    rr = (st.session_state.last_result or {}).get("risk_result", {})
    if rr:
        st.subheader("Risk Score")
        show_risk_badge(rr)
        st.metric("Probability", f"{rr.get('refund_probability', 0):.1%}")
        st.metric("Tier",        rr.get("risk_tier", "—").upper())
        st.metric("Model",       rr.get("model", "—"))
        st.divider()

    # Order stats (persists across turns)
    od = (st.session_state.last_result or {}).get("order_data", {})
    if od and "transactions" in od:
        st.subheader("Order Stats")
        st.metric("Purchases",   od.get("purchase_count", 0))
        st.metric("Refunds",     od.get("refund_count", 0))
        st.metric("Refund Rate", f"{od.get('refund_rate', 0):.1%}")
        st.divider()

    # Quick-action buttons — pre-fill chat input
    st.subheader("Quick Actions")
    qa = [
        ("Score an Order",       "Look up transaction e0999e81-8b5e-430d-9770-d154c0ef243d, score its refund risk, and advise on next steps per policy."),
        ("Electronics Policy",   "What is the refund and return policy for electronics? Cite exact policy sections."),
        ("User Order History",   "Retrieve all orders for user 1c9b9667, score refund risk on their latest purchase, and summarise per policy."),
        ("High-Risk Protocol",   "What internal steps should we take when an order scores high refund risk? Cite policy."),
    ]
    for label, prompt in qa:
        if st.button(label, use_container_width=True):
            st.session_state["_prefill"] = prompt
            st.rerun()

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.chat_history    = []
        st.session_state.display_history = []
        st.session_state.last_result     = None
        st.rerun()


# ── Main — chat interface ─────────────────────────────────────────────────────
st.title("Refund Risk Assessment")
st.caption(
    "Bedrock Nova 2 Lite · XGBoost via SageMaker · "
    "Policy RAG via Pinecone · Order data via Databricks"
)

# Alert banner above chat (threshold-driven, not LLM-invented)
if st.session_state.last_result:
    show_alert(st.session_state.last_result.get("risk_result", {}))

# Chat history display
for turn in st.session_state.display_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# Policy citations below the last assistant turn
if st.session_state.last_result:
    citations = st.session_state.last_result.get("policy_citations", [])
    if citations:
        with st.expander(f"Policy citations ({len(citations)})", expanded=False):
            show_citations(citations)

# Order data expander
if st.session_state.last_result:
    od = st.session_state.last_result.get("order_data", {})
    if od:
        with st.expander("Order data", expanded=False):
            show_order_table(od)

# Message trace (debug)
if st.session_state.last_result:
    msgs = st.session_state.last_result.get("messages", [])
    if msgs:
        with st.expander("Agent message trace", expanded=False):
            for msg in msgs:
                role = type(msg).__name__.replace("Message", "").lower()
                role = role if role in ("human", "ai") else "assistant"
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
                with st.chat_message(role):
                    st.markdown(content[:2000] + ("…" if len(content) > 2000 else ""))

# ── Chat input ────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", "")
user_input = st.chat_input(
    "Ask about an order, score refund risk, or query policy…",
    key="chat_input",
) or prefill

if user_input:
    # Show user turn
    st.session_state.display_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run agent with full conversation history (memory requirement)
    with st.chat_message("assistant"):
        with st.spinner("Agent working…"):
            try:
                from agents.refund_agent_fix import run_agent

                result = run_agent(
                    user_message=user_input,
                    history=st.session_state.chat_history,   # multi-turn memory
                )
                answer = result["final_answer"] or "No response returned."
                st.markdown(answer)

                # Persist state
                st.session_state.last_result = result
                # Update chat history for next turn (agent memory)
                st.session_state.chat_history.append({"role": "user",      "content": user_input})
                st.session_state.chat_history.append({"role": "assistant",  "content": answer})
                st.session_state.display_history.append({"role": "assistant", "content": answer})

            except Exception:
                err = traceback.format_exc()
                st.error(f"Agent error:\n```\n{err}\n```")

    st.rerun()

# Welcome state (no history yet)
if not st.session_state.display_history:
    st.info(
        "Enter a question below or use a Quick Action from the sidebar.\n\n"
        "The agent will: look up the order from Databricks → score refund risk "
        "via XGBoost → retrieve relevant policy from Pinecone → "
        "produce a grounded recommendation with citations."
    )
