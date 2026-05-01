"""app.py — Refund Risk Agent (internal use only)"""

from __future__ import annotations

import json
import os
import io
import sys
import traceback
import plotly.express as px
import pyarrow.parquet as pq
import pyspark.sql.functions as F
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
import requests

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
.alert-high  {background:#f8d7da;color:#721c24;padding:12px 16px;border-radius:6px;
              border-left:5px solid #721c24;margin-bottom:12px;font-weight:600;}
.alert-med   {background:#fff3cd;color:#856404;padding:12px 16px;border-radius:6px;
              border-left:5px solid #856404;margin-bottom:12px;font-weight:600;}
.citation-box {background:#f0f4ff;border-left:4px solid #4a6cf7;padding:10px 14px;
               border-radius:4px;margin-bottom:8px;font-size:0.88rem;}
.citation-meta{color:#555;font-size:0.78rem;margin-top:4px;}
.answer-box  {background:#fafafa;border:1px solid #e0e0e0;padding:16px;
              border-radius:8px;line-height:1.65;}
[data-testid="stTab"] { font-size: 40px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)
 
# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("last_result", None), ("history", []), ("pending_query", ""), ("agent_input", "")]:
    if k not in st.session_state:
        st.session_state[k] = v
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDERERS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
def render_risk_badge(rr: dict) -> None:
    prob = rr.get("refund_probability")
    if prob is None:
        return
    tier  = rr.get("risk_tier", "medium")
    label = {"low": "LOW RISK", "medium": "MEDIUM RISK", "high": "HIGH RISK"}.get(tier, tier.upper())
    css   = f"risk-{tier}" if tier in ("low", "medium", "high") else "risk-medium"
    st.markdown(
        f'<div class="{css}" style="font-size:24pt;">{label} — {float(prob):.1%} refund probability</div>',
        unsafe_allow_html=True
    )
    st.progress(float(prob), text=f"Refund probability: {float(prob):.1%}")
 
 
def render_alert(rr: dict) -> None:
    prob = rr.get("refund_probability")
    if prob is None:
        return
    prob = float(prob)
    if prob >= 0.60:
        st.markdown(
            f'<div class="alert-high" style="font-size:24pt;">HIGH REFUND RISK ({prob:.1%}) — '
            f'Escalate per return/refund SLA. This tool does not execute refunds.</div>',
            unsafe_allow_html=True,
        )
    elif prob >= 0.30:
        st.markdown(
            f'<div class="alert-med" style="font-size:24pt;">MEDIUM REFUND RISK ({prob:.1%}) — '
            f'Flag for CS review and offer proactive resolution.</div>',
            unsafe_allow_html=True,
        )
 
 
def render_citations(citations: list) -> None:
    if not citations:
        st.info("No policy citations retrieved for this query.")
        return
    st.subheader("Policy Citations", anchor=False)
    st.caption("Sourced from the Generic E-Commerce Company Master Policy Compendium.")
    for c in citations:
        idx   = c.get("chunk_index", "?")
        text  = c.get("chunk_text", "")
        score = c.get("score", 0.0)
        src   = c.get("source", "Master Policy Compendium")
        st.markdown(
            f'<div class="citation-box" style="font-size:24pt;"><b>[Policy §{idx}]</b> {text}'
            f'<div class="citation-meta" style="font-size:24pt;">Source: {src} | Relevance: {float(score):.4f}</div></div>',
            unsafe_allow_html=True,
        )
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
with st.sidebar:
    st.title("Refund Risk Agent")
    st.caption("<span style='font-size:24pt;'>Internal — operations & customer experience teams only</span>", unsafe_allow_html=True)
    st.divider()
 
    st.subheader("Quick Actions")
    qa_cols = st.columns(2)
    if qa_cols[0].button("Score by Order ID", use_container_width=True):
        st.session_state.pending_query = (
            "Look up order ORD-10042, score its refund risk, "
            "and advise on next steps per company policy."
        )
    if qa_cols[1].button("Electronics Policy", use_container_width=True):
        st.session_state.pending_query = (
            "What is the refund and return policy for electronics? "
            "Provide exact policy citations."
        )
    if qa_cols[0].button("User Order History", use_container_width=True):
        st.session_state.pending_query = (
            "Retrieve order history for user USR-9921, score refund risk "
            "on their latest purchase, and summarise per policy."
        )
    if qa_cols[1].button("High-Risk Protocol", use_container_width=True):
        st.session_state.pending_query = (
            "What steps should we take for a high-risk refund order? "
            "Cite relevant policy sections."
        )
 
    st.divider()
    if st.session_state.last_result:
        rr = st.session_state.last_result.get("risk_result", {})
        if rr:
            render_risk_badge(rr)
 
    if st.button("Clear Session", use_container_width=True):
        st.session_state.last_result   = None
        st.session_state.history       = []
        st.session_state.pending_query = ""
        st.rerun()
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN — TWO TABS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
st.markdown("<h1 style='font-size:30pt;'>Refund Risk Assessment</h1>", unsafe_allow_html=True)
st.caption(
    "<span style='font-size:24pt;'>kimi-k2-0905-preview · XGBoost via SageMaker · "
    "Policy RAG via Pinecone · Orders via Databricks</span>",
    unsafe_allow_html=True
)
 
tab_chat, tab_dashboard = st.tabs(["💬 Agent Chat", "📊 Dashboard"])
 
 
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — AGENT CHAT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 
with tab_chat:
    st.markdown("<h2 style='font-size:30pt;'>Ask the Refund Risk Agent</h2>", unsafe_allow_html=True)
 
    # If a sidebar quick-action injected a query, seed the widget key once then clear.
    if st.session_state.pending_query:
        st.session_state["agent_input"] = st.session_state.pending_query
        st.session_state.pending_query  = ""
 
    st.text_area(
        "Enter your question",
        key="agent_input",
        placeholder=(
            "e.g. 'Assess refund risk for order ORD-10042'\n"
            "or 'What is the return window for electronics per policy?'"
        ),
        height=140,
    )
    run_btn = st.button("▶ Run Agent", type="primary")
 
    if run_btn:
        query = st.session_state.agent_input.strip()
        if not query:
            st.warning("<span style='font-size:24pt;'>Please enter a question before clicking Run Agent.</span>", unsafe_allow_html=True)
        else:
            with st.spinner("Agent working…"):
                try:
                    from refund_agent import run_agent
 
                    result = run_agent(
                        user_message=query,
                        history=[
                            {"role": h["role"], "content": h["content"]}
                            for h in st.session_state.history
                        ],
                    )
                    st.session_state.last_result = result
                    st.session_state.history.append({"role": "user",      "content": query})
                    st.session_state.history.append({"role": "assistant",  "content": result.get("final_answer", "")})
 
                except RuntimeError as rte:
                    st.error(f"<span style='font-size:24pt;'>{str(rte)}</span>", unsafe_allow_html=True)
                except Exception:
                    import traceback
                    st.error(f"<span style='font-size:24pt;'>Agent error:\n\n{traceback.format_exc()}\n</span>", unsafe_allow_html=True)
 
    if st.session_state.last_result:
        result = st.session_state.last_result
        rr     = result.get("risk_result", {})
 
        if rr:
            render_alert(rr)
            render_risk_badge(rr)
            st.divider()
 
        st.markdown("<h3 style='font-size:30pt;'>Agent Analysis</h3>", unsafe_allow_html=True)
        answer = result.get("final_answer") or "No answer returned."
        st.markdown(f'<div class="answer-box" style="font-size:24pt;">{answer}</div>', unsafe_allow_html=True)
 
        st.divider()
        render_citations(result.get("policy_citations", []))
 
        # with st.expander("Raw Risk Score JSON"):
        #      st.json(result.get("risk_result", {}))
 
        with st.expander("Agent Message Trace"):
            for msg in result.get("messages", []):
                role    = type(msg).__name__.replace("Message", "").lower()
                role    = role if role in ("human", "ai") else "assistant"
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content, default=str)
                with st.chat_message(role):
                    st.markdown(
                        f"<span style='font-size:24pt;'>{content[:2000]}{'…' if len(content) > 2000 else ''}</span>",
                        unsafe_allow_html=True
                    )
    else:
        st.info(
            "<span style='font-size:24pt;'>Enter a question in the box above and click <b>▶ Run Agent</b>.<br><br>"
            "The agent will: look up the order → score refund risk via XGBoost/SageMaker "
            "→ retrieve policy from Pinecone → produce a grounded recommendation.</span>"
        )
 
    user_turns = [h for h in st.session_state.history if h["role"] == "user"]
    if len(user_turns) > 1:
        with st.expander(f"Session History ({len(user_turns)} queries)"):
            for i, turn in enumerate(reversed(user_turns[-10:]), 1):
                st.markdown(
                    f"<span style='font-size:24pt;'><b>{i}.</b> {turn['content'][:120]}</span>",
                    unsafe_allow_html=True
                )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — DASHBOARD (Databricks Unity Catalog Volumes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_dotenv()
access_token = os.environ.get("access_token")
host = "dbc-38471c53-2db6.cloud.databricks.com"

ANALYSIS_TIME_SERIES = "/Volumes/project_3/datalake/gold_zone/analysis/category_time_series/"
ANALYSIS_TOTALS = "/Volumes/project_3/datalake/gold_zone/analysis/category_totals/"
PRODUCT_TOTALS = "/Volumes/project_3/datalake/gold_zone/analysis/product_totals/"

# --- Global Font Size Config ---
st.markdown("""
<style>
    h1 { font-size: 50px !important; }
    h2 { font-size: 50px !important; }
    h3 { font-size: 50px !important; }
    .stMetric { font-size: 50px !important; }
    .st-ae { font-size: 150px !important; }
</style>
""", unsafe_allow_html=True)

# Plotly global font size
FONT_SIZE = 60

headers  = {"Authorization": f"Bearer {access_token}"}
base_url = f"https://{host}/api/2.0"

def list_volume_files(path):
    """Return all .parquet file paths inside a Unity Catalog Volume directory."""
    resp = requests.get(
        f"{base_url}/fs/directories{path}",
        headers=headers
    )
    resp.raise_for_status()
    entries = resp.json().get("contents", [])
    return [
        e["path"] for e in entries
        if not e.get("is_directory", False) and e["path"].endswith(".parquet")
    ]

def read_parquet_from_volume(file_path):
    """Stream a single parquet file from a Unity Catalog Volume into a DataFrame."""
    url  = f"https://{host}/api/2.0/fs/files{file_path}"
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()
    return pq.read_table(io.BytesIO(resp.content)).to_pandas()

def load_volume(path):
    """Load all parquet files in a Volume path into a single DataFrame."""
    files = list_volume_files(path)
    print(f"  {path}  →  {len(files)} file(s) found")
    if not files:
        return pd.DataFrame()
    return pd.concat(
        [read_parquet_from_volume(f) for f in files],
        ignore_index=True
    )

category_totals = load_volume(ANALYSIS_TOTALS)
category_time_series = load_volume(ANALYSIS_TIME_SERIES)
product_totals = load_volume(PRODUCT_TOTALS)

# --- KPIs ---
total_revenue = category_totals["total_revenue"].sum()
total_completed_orders = category_totals["transaction_count"].sum()
top_category = category_totals.iloc[0]["category"]
top_category_revenue = category_totals.iloc[0]["total_revenue"]

best_selling_product = product_totals.iloc[0]["product_name"]
best_selling_qty = product_totals.iloc[0]["total_quantity_sold"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"<div style='font-size:30px;font-weight:bold;'>Total Revenue</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:30px;color:#000000;'>${total_revenue:,.2f}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='font-size:30px;font-weight:bold;'>Total Completed Orders</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:30px;color:#000000;'>{total_completed_orders:,}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='font-size:30px;font-weight:bold;'>Top-Selling Category</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:30px;color:#000000;'>{top_category}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:30px;'>${top_category_revenue:,.2f}</div>", unsafe_allow_html=True)
with col4:
    st.markdown(f"<div style='font-size:30px;font-weight:bold;'>Best-Selling Product</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:30px;color:#000000;'>{best_selling_product}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:30px;'>{int(best_selling_qty):,} units</div>", unsafe_allow_html=True)


# --- Pie Chart ---
st.subheader("Revenue by Category")
fig_pie = px.pie(category_totals, values="total_revenue", names="category")
fig_pie.update_layout(
    font=dict(size=FONT_SIZE),
    legend=dict(font=dict(size=22)),
)
fig_pie.update_traces(textposition='inside', textinfo='percent+label', textfont_size=14)
st.plotly_chart(fig_pie)
 
# --- Line Graph ---
st.subheader("Revenue Over Time by Category")
# Pivot for line chart
pivot_df = category_time_series.pivot(
    index="date",
    columns="category",
    values="daily_revenue"
).fillna(0)
fig_line = px.line(pivot_df, labels={"value": "Revenue", "date": "Date", "category": "Category"})
fig_line.update_layout(
    font=dict(size=FONT_SIZE),
    legend=dict(font=dict(size=22)),
    xaxis=dict(title_font=dict(size=18), tickfont=dict(size=18)),
    yaxis=dict(title_font=dict(size=18), tickfont=dict(size=18)),
)
st.plotly_chart(fig_line)
 