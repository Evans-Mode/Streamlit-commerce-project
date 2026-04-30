from __future__ import annotations

import json
import os
import time
from typing import Annotated, Literal, TypedDict

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ── Import Modular Tools ─────────────────────────────────────────────────────
from order_lookup_tool import lookup_order
from scoring_tool import score_refund_risk
from policy_retrieval_tool import retrieve_policy
from product_help_tool import get_product_help

load_dotenv()

# Configuration
MODEL = os.environ.get("bedrock_model_id")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PUBLIC ENTRY POINT — Called by app.py
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_agent(user_message: str, history: list[dict] | None = None) -> dict:
    """
    Run the refund risk LangGraph agent and return a structured result dict 
    formatted specifically for the Streamlit UI.
    """
    # 1. Rebuild the message history for LangChain
    messages: list[BaseMessage] = []
    for turn in (history or []):
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
            
    # Add the current user query
    messages.append(HumanMessage(content=user_message))

    # 2. Invoke the compiled LangGraph supervisor
    result = supervisor.invoke({
        "messages": messages,
        "next_agent": "",
        "task_result": "",
        "order_features": {},
        "risk_result": {},
        "policy_citations": [],
    })

    # 3. Extract the final AI response
    final_msg = result["messages"][-1].content if result.get("messages") else "No answer returned."

    # 4. Return the exact payload expected by app.py
    return {
        "final_answer": final_msg,
        "risk_result": result.get("risk_result", {}),
        "policy_citations": result.get("policy_citations", []),
        "messages": result.get("messages", []),
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STATE & UTILS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AgentState(TypedDict):
    messages:        Annotated[list[BaseMessage], add_messages]
    next_agent:      Literal["lookup", "score", "policy", "product", "respond", "__end__"]
    task_result:     str
    order_features:  dict
    risk_result:     dict
    policy_citations: list[dict]

_llm = None
def get_model():
    global _llm
    if _llm is None:
        _llm = init_chat_model(MODEL)
    return _llm

def invoke_with_retry(model, messages, max_retries: int = 1, **kwargs):
    for attempt in range(max_retries):
        try:
            return model.invoke(messages, **kwargs)
        except ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException" and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SYSTEM PROMPTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPERVISOR_SYSTEM = """You are an internal risk operations supervisor for Generic E-Commerce Company.
You coordinate specialist sub-agents to help customer support leads, fraud analysts, and team managers.

Available specialists:
- lookup  : Retrieve order feature row by order_id or user_id from Databricks / CSV
- score   : Call SageMaker XGBoost endpoint and return refund-risk probability + tier
- policy  : Search the Master Policy Compendium via Pinecone RAG; always cite sources
- product : Answer product/category questions (warranties, return windows, category rules)
- respond : Synthesise all gathered results into a final concise answer for the operator

Routing rules (respond with ONLY the specialist name):
- If the request mentions an order ID or user ID and no features have been fetched → lookup
- If features exist in state but no score yet → score
- If the request asks about policy, returns, refunds, shipping eligibility → policy
- If the request asks about product categories, warranties, or return windows → product
- If task_result is populated and ready to synthesise → respond
- If none apply (e.g. off-topic, requests to execute refunds) → respond

NEVER invent model scores. NEVER execute refunds or account changes — only inform.
Keep all answers to internal operator language, not customer-facing. All output should be less
than 200 characters and converted to strings, never raw JSON or code blocks, to ensure clean display in the UI. 
Always cite policy with section numbers when relevant, never stating policy from memory. 
Always return the next_agent as a string, never null or None. 
Always return task_result as a string, never null or None. 
Always return order_features and risk_result as dictionaries, never null or None. 
Always return policy_citations as a list of dictionaries, never null or None. Convert all outputs
to strings """


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# NODES (Connected to Tools)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def supervisor_node(state: AgentState) -> dict:
    """Routes to specialists using the LLM and the strict system prompt."""
    model = get_model()
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    routing_response = invoke_with_retry(
        model,
        [
            {"role": "system", "content": SUPERVISOR_SYSTEM},
            {"role": "user", "content": last_message},
        ],
        max_tokens=10,  # Ensure we only get the single routing word back
    )

    route = routing_response.content.strip().lower().split()[0]
    valid = {"lookup", "score", "policy", "product", "respond"}
    if route not in valid:
        route = "respond"

    # Auto-chain logic: force score next if lookup just finished
    if route == "lookup" and state.get("order_features"):
        route = "score"

    return {"next_agent": route}


def respond_node(state: AgentState) -> dict:
    """Final synthesis for the operator, strictly enforcing the 200-character limit."""
    model = get_model()
    task_result = state.get("task_result", "")
    user_content = state["messages"][-1].content
    
    prompt = f"Original request: {user_content}\n\nSpecialist result: {task_result}"
    
    response = invoke_with_retry(
        model, 
        [
            {
                "role": "system", 
                "content": (
                    "You are an internal operations assistant synthesizing data for operators. "
                    "CRITICAL RULES: "
                    "1. Keep all answers to internal operator language, not customer-facing. "
                    "2. Your final output MUST be less than 200 characters. "
                    "3. Convert all output to raw strings. NEVER use JSON, markdown code blocks, or special formatting. "
                    "4. Do not execute refunds; only inform."
                )
            },
            {"role": "user", "content": prompt}
        ],
        max_tokens=100, # Hard cap to help enforce the 200 character restriction
    )
    
    return {
        "messages": [AIMessage(content=response.content)],
        "next_agent": "__end__"
    }

def order_lookup_node(state: AgentState) -> dict:
    """Uses lookup_order tool to fetch data from Databricks."""
    # Extract ID (simplified for this example)
    last_msg = state["messages"][-1].content
    order_id = last_msg.split("ORD-")[-1].split()[0] if "ORD-" in last_msg else ""
    
    # Call the tool[cite: 4]
    result_json = lookup_order.invoke({"transaction_id": f"ORD-{order_id}"})
    data = json.loads(result_json)
    
    if "error" in data:
        return {"task_result": data["error"], "next_agent": "respond"}
        
    return {
        "order_features": data, 
        "task_result": f"Found order details for {data.get('transaction_id')}",
        "next_agent": "score"
    }

def score_node(state: AgentState) -> dict:
    """Uses score_refund_risk tool for SageMaker XGBoost inference."""
    features = state.get("order_features", {})
    if not features:
        return {"task_result": "No features for scoring.", "next_agent": "respond"}

    # Call the tool[cite: 3]
    score_json = score_refund_risk.invoke({"features_json": json.dumps(features)})
    risk_data = json.loads(score_json)
    
    return {
        "risk_result": risk_data,
        "task_result": f"Risk Probability: {risk_data.get('refund_probability')}",
        "next_agent": "respond"
    }

def policy_node(state: AgentState) -> dict:
    """Uses retrieve_policy tool for Pinecone RAG."""
    query = state["messages"][-1].content
    
    # Call the tool[cite: 5]
    policy_json = retrieve_policy.invoke({"query": query})
    policy_data = json.loads(policy_json)
    
    chunks = policy_data.get("policy_chunks", [])
    summary = "\n".join([c['chunk_text'] for c in chunks[:2]])
    
    return {
        "policy_citations": chunks,
        "task_result": f"Policy found: {summary}",
        "next_agent": "respond"
    }

def product_node(state: AgentState) -> dict:
    """Uses get_product_help for category-specific guidance."""
    last_msg = state["messages"][-1].content
    
    # Determine category (simplified)
    category = "electronics" if "electronic" in last_msg.lower() else "clothing"
    
    # Call the tool[cite: 2]
    help_json = get_product_help.invoke({"category": category})
    help_data = json.loads(help_json)
    
    return {
        "task_result": str(help_data.get("policy_result", {}).get("policy_chunks", "No data")),
        "next_agent": "respond"
    }

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GRAPH CONSTRUCTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

workflow = StateGraph(AgentState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("lookup",     order_lookup_node)
workflow.add_node("score",      score_node)
workflow.add_node("policy",     policy_node)
workflow.add_node("product",    product_node)
workflow.add_node("respond",    respond_node)

workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges("supervisor", lambda x: x["next_agent"])
workflow.add_edge("lookup", "score")
workflow.add_edge("score", "respond")
workflow.add_edge("policy", "respond")
workflow.add_edge("product", "respond")
workflow.add_edge("respond", END)

supervisor = workflow.compile()