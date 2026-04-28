"""tools/order_lookup_tool.py

Fetches order / transaction data from the Databricks Unity Catalog Volume
using the Files API pattern from databricks_parquet.ipynb.

Volume path: /Volumes/project_3/datalake/gold_zone/transaction_fact/
Schema (from notebook head(10)):
  transaction_id, user_id, transaction_type, timestamp, status,
  payment_method, currency, subtotal, tax, total,
  billing_address_id, shipping_address_id
"""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import pyarrow.parquet as pq
import pandas as pd
import requests
from langchain_core.tools import tool

from config.settings import load_settings

logger = logging.getLogger(__name__)


def _databricks_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _list_parquet_files(host: str, token: str, volume_path: str) -> list[str]:
    """List parquet files in a Unity Catalog Volume — mirrors databricks_parquet.ipynb."""
    base_url = f"https://{host}/api/2.0"
    resp = requests.get(
        f"{base_url}/fs/directories{volume_path}",
        headers=_databricks_headers(token),
        timeout=30,
    )
    resp.raise_for_status()
    entries = resp.json().get("contents", [])
    return [
        e["path"]
        for e in entries
        if not e.get("is_directory", False) and e["path"].endswith(".parquet")
    ]


def _read_parquet_from_volume(host: str, token: str, file_path: str) -> pd.DataFrame:
    """Stream one parquet file into a DataFrame — mirrors databricks_parquet.ipynb."""
    url = f"https://{host}/api/2.0/fs/files{file_path}"
    resp = requests.get(
        url, headers=_databricks_headers(token), stream=True, timeout=60
    )
    resp.raise_for_status()
    buffer = io.BytesIO(resp.content)
    return pq.read_table(buffer).to_pandas()


def _load_transactions(cfg) -> pd.DataFrame:
    """Load all parquet files from the volume into one DataFrame."""
    files = _list_parquet_files(cfg.DATABRICKS_HOST, cfg.access_token, cfg.DATABRICKS_VOLUME_PATH)
    if not files:
        return pd.DataFrame()
    dfs = [_read_parquet_from_volume(cfg.DATABRICKS_HOST, cfg.access_token, fp) for fp in files]
    return pd.concat(dfs, ignore_index=True)


@tool
def lookup_order(transaction_id: str) -> str:
    """
    Retrieve full transaction details for a given transaction_id from the
    Databricks Unity Catalog gold zone parquet volume.

    Args:
        transaction_id: UUID string of the transaction to look up.

    Returns:
        JSON string with all transaction fields, or an error message.
    """
    cfg = load_settings()

    try:
        df = _load_transactions(cfg)
    except Exception as exc:
        logger.error("Databricks load error: %s", exc)
        return json.dumps({"error": f"Could not load transactions: {exc}"})

    if df.empty:
        return json.dumps({"error": "No transaction data found in the volume."})

    match = df[df["transaction_id"] == transaction_id]
    if match.empty:
        return json.dumps({"error": f"Transaction '{transaction_id}' not found."})

    record = match.iloc[0].to_dict()
    # Convert non-JSON-serialisable types
    for k, v in record.items():
        if hasattr(v, "isoformat"):
            record[k] = v.isoformat()
        elif pd.isna(v) if not isinstance(v, (str, dict, list)) else False:
            record[k] = None
    return json.dumps(record, default=str)


@tool
def lookup_user_orders(user_id: str, limit: int = 10) -> str:
    """
    Retrieve recent orders for a customer user_id from the Databricks volume.

    Args:
        user_id: Customer user_id string.
        limit:   Max number of most-recent transactions to return (default 10).

    Returns:
        JSON string with a list of transactions and summary stats.
    """
    cfg = load_settings()

    try:
        df = _load_transactions(cfg)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    if df.empty:
        return json.dumps({"error": "No transaction data available."})

    user_df = df[df["user_id"] == user_id].copy()
    if user_df.empty:
        return json.dumps({"error": f"No orders found for user '{user_id}'."})

    user_df["timestamp"] = pd.to_datetime(user_df["timestamp"], errors="coerce")
    user_df = user_df.sort_values("timestamp", ascending=False).head(limit)

    purchases = user_df[user_df["transaction_type"] == "purchase"]
    refunds = user_df[user_df["transaction_type"] == "refund"]

    summary: dict[str, Any] = {
        "user_id": user_id,
        "total_transactions_shown": len(user_df),
        "purchase_count": len(purchases),
        "refund_count": len(refunds),
        "refund_rate": round(len(refunds) / len(user_df), 3) if len(user_df) else 0,
        "total_spend_usd": round(float(purchases["total"].sum()), 2),
        "transactions": json.loads(
            user_df[
                ["transaction_id", "transaction_type", "timestamp", "status",
                 "payment_method", "total"]
            ]
            .to_json(orient="records", date_format="iso")
        ),
    }
    return json.dumps(summary, default=str)
