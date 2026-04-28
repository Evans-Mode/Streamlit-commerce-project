"""app.py — Refund Risk Agent (internal use only)"""

from __future__ import annotations

import json
import os
import sys
import traceback

import streamlit as st

# ── Path fix ──────────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Refund Risk Agent — Internal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.risk-low    {background:#d4edda;color:#155724;padding:8px 16px;border-radius:6px;font-weight:700;font-size:1.1rem;}
.risk-medium {background:#fff3cd;color:#856404;padding:8px 16px;border-radius:6px;font-weight:700;font-size:1.1rem;}
.risk-high   {background:#f8d7da;color:#721c24;padding:8px 16px;border-radius:6px;font-weight:700;font-size:1.1rem;}
.alert-high  {background:#f8d7da;color:#721c24;padding:12px 16px;border-radius:6px;border-left:5px solid #721c24;margin-bottom:12px;font-weight:600;}
.alert-med   {background:#fff3cd;color:#856404;padding:12px 16px;border-radius:6px;border-left:5px solid #856404;margin-bottom:12px;font-weight:600;}
.citation-box{background:#f0f4ff;border-left:4px solid #4a6cf7;padding:10px 14px;border-radius:4px;margin-bottom:8px;font-size:0.88rem;}
.citation-meta{color:#555;font-size:0.78rem;margin-top:4px;}
.answer-box  {background:#fafafa;border:1px solid #e0e0e0;padding:16px;border-radius:8px;line-height:1.65;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [
    ("last_result", None),
    ("history", []),
    ("pending_query", ""),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Renderers ─────────────────────────────────────────────────────────────────

def render_risk_badge(rr: dict) -> None:
    prob = rr.get("refund_probability")
    if prob is None:
        return
    tier = rr.get("risk_tier", "medium")
    label = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}.get(tier, tier.upper())
    css = f"risk-{tier}" if tier in ("low", "medium", "high") else "risk-medium"
    st.markdown(f'<div class="{css}">{label} — {float(prob):.1%} refund probability</div>', unsafe_allow_html=True)
    st.progress(float(prob), text=f"Refund probability: {float(prob):.1%}")


def render_alert(rr: dict) -> None:
    prob = rr.get("refund_probability")
    if prob is None:
        return
    prob = float(prob)
    if prob >= 0.60:
        st.markdown(f'<div class="alert-high">HIGH REFUND RISK ({prob:.1%}) — Escalate per return/refund SLA. This tool does not execute refunds.</div>', unsafe_allow_html=True)
    elif prob >= 0.30:
        st.markdown(f'<div class="alert-med">MEDIUM REFUND RISK ({prob:.1%}) — Flag for CS review and offer proactive resolution.</div>', unsafe_allow_html=True)


def render_citations(citations: list) -> None:
    if not citations:
        st.info("No policy citations retrieved for this query.")
        return
    st.subheader("Policy Citations", anchor=False)
    st.caption("Sourced verbatim from the Generic E-Commerce Company Master Policy Compendium.")
    for c in citations:
        idx   = c.get("chunk_index", "?")
        text  = c.get("chunk_text", "")
        score = c.get("score", 0.0)
        src   = c.get("source", "Master Policy Compendium")
        st.markdown(
            f'<div class="citation-box"><b>[Policy §{idx}]</b> {text}'
            f'<div class="citation-meta">Source: {src} | Relevance: {float(score):.4f}</div></div>',
            unsafe_allow_html=True,
        )


def render_order_data(od: dict) -> None:
    if not od:
        return
    if "transactions" in od:
        import pandas as pd
        st.dataframe(pd.DataFrame(od["transactions"]), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Purchases",   od.get("purchase_count", 0))
        c2.metric("Refunds",     od.get("refund_count", 0))
        c3.metric("Refund Rate", f"{od.get('refund_rate', 0):.1%}")
    else:
        st.json(od)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Refund Risk Agent")
    st.caption("Internal — operations & customer experience teams only")
    st.divider()

    # Quick-action buttons write to pending_query (no key= on text_area to avoid conflict)
    st.subheader("Quick Actions")
    qa_cols = st.columns(2)
    if qa_cols[0].button("Score by Order ID", use_container_width=True):
        st.session_state.pending_query = (
            "Look up transaction e0999e81-8b5e-430d-9770-d154c0ef243d, "
            "score its refund risk, and advise on next steps per company policy."
        )
    if qa_cols[1].button("Electronics Policy", use_container_width=True):
        st.session_state.pending_query = (
            "What is the refund and return policy for electronics? "
            "Provide exact policy citations."
        )
    if qa_cols[0].button("User Order History", use_container_width=True):
        st.session_state.pending_query = (
            "Retrieve order history for user 1c9b9667, score refund risk "
            "on their latest purchase, and summarise per policy."
        )
    if qa_cols[1].button("High-Risk Protocol", use_container_width=True):
        st.session_state.pending_query = (
            "What steps should we take for a high-risk refund order? "
            "Cite relevant policy sections."
        )

    st.divider()
    st.subheader("Query")

    # No key= here — value= drives content, avoiding Streamlit widget-state conflicts
    user_query = st.text_area(
        "Enter your question",
        value=st.session_state.pending_query,
        placeholder=(
            "e.g. 'Assess refund risk for transaction e0999e81...'\n"
            "or 'What is the return window for electronics per policy?'"
        ),
        height=160,
    )

    run_btn = st.button("Run Agent", type="primary", use_container_width=True)

    # Consume pending_query after it has been rendered into the textarea
    st.session_state.pending_query = ""

    st.divider()
    if st.session_state.last_result:
        rr = st.session_state.last_result.get("risk_result", {})
        if rr:
            render_risk_badge(rr)

    if st.button("Clear", use_container_width=True):
        st.session_state.last_result = None
        st.session_state.history = []
        st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Refund Risk Assessment")
st.caption("AWS Bedrock Nova Lite 2 · XGBoost via SageMaker · Policy RAG via Pinecone · Orders via Databricks")

# ── Run agent (no st.rerun — result renders immediately in same pass) ──────────
if run_btn:
    query = user_query.strip()
    if not query:
        st.warning("Please enter a question before clicking Run Agent.")
    else:
        with st.spinner("Agent working…"):
            try:
                from agents.refund_agent_fix import run_agent

                result = run_agent(
                    user_message=query,
                    history=[
                        {"role": h["role"], "content": h["content"]}
                        for h in st.session_state.history
                    ],
                )
                st.session_state.last_result = result
                st.session_state.history.append({"role": "user",     "content": query})
                st.session_state.history.append({"role": "assistant", "content": result.get("final_answer", "")})

            except Exception:
                st.error(f"Agent error:\n```\n{traceback.format_exc()}\n```")

# ── Display result ─────────────────────────────────────────────────────────────
if st.session_state.last_result:
    result = st.session_state.last_result
    rr = result.get("risk_result", {})

    if rr:
        render_alert(rr)
        render_risk_badge(rr)
        st.divider()

    st.subheader("Agent Analysis", anchor=False)
    answer = result.get("final_answer") or "No answer returned."
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    st.divider()
    render_citations(result.get("policy_citations", []))

    st.divider()
    with st.expander("Order Data"):
        render_order_data(result.get("order_data", {}))

    with st.expander("Raw Risk Score JSON"):
        st.json(result.get("risk_result", {}))

    with st.expander("Agent Message Trace"):
        for msg in result.get("messages", []):
            role = type(msg).__name__.replace("Message", "").lower()
            role = role if role in ("human", "ai") else "assistant"
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
            with st.chat_message(role):
                st.markdown(content[:2000] + ("…" if len(content) > 2000 else ""))

else:
    st.info(
        "Enter a question in the sidebar and click **Run Agent**.\n\n"
        "The agent will: look up the order from Databricks → score refund risk "
        "via XGBoost → retrieve policy from Pinecone → produce a grounded recommendation."
    )

# ── Session history ───────────────────────────────────────────────────────────
user_turns = [h for h in st.session_state.history if h["role"] == "user"]
if len(user_turns) > 1:
    with st.expander(f"Session History ({len(user_turns)} queries)"):
        for i, turn in enumerate(reversed(user_turns[-10:]), 1):
            st.markdown(f"**{i}.** {turn['content'][:120]}")
