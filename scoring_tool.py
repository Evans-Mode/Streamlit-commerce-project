"""tools/scoring_tool.py

Calls the SageMaker XGBoost endpoint trained in project1_churn_xgboost.ipynb
to predict refund/churn probability for a given order feature vector.

Feature order must match the training CSV produced by to_xgb_format():
  days_since_last_purchase, total_orders, avg_order_value,
  user_session_count_7d, primary_category (encoded), payment_method (encoded),
  currency (encoded), primary_device (encoded), ...
  (all columns in X_train.columns order, target excluded)

The endpoint was deployed with:
  serializer=CSVSerializer(), deserializer=CSVDeserializer()
It accepts a CSV row (no header) and returns a float probability string.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
from langchain_core.tools import tool

from settings import load_settings

logger = logging.getLogger(__name__)


# ── Label encoders reproduced from project1_churn_xgboost.ipynb ───────────────
# These mirror the LabelEncoder.classes_ lists printed during feature encoding.
# Update if training data categories change.
CATEGORY_ENCODER: dict[str, int] = {
    "beauty": 0,
    "books": 1,
    "clothing": 2,
    "electronics": 3,
    "home": 4,
    "sports": 5,
    "toys": 6,
}
PAYMENT_ENCODER: dict[str, int] = {
    "apple_pay": 0,
    "credit_card": 1,
    "google_pay": 2,
    "paypal": 3,
}
CURRENCY_ENCODER: dict[str, int] = {
    "EUR": 0,
    "GBP": 1,
    "USD": 2,
}
DEVICE_ENCODER: dict[str, int] = {
    "android": 0,
    "desktop": 1,
    "ios": 2,
}

# ── Feature column order (must match X_train.columns from the notebook) ────────
FEATURE_COLUMNS = [
    "days_since_last_purchase",
    "total_orders",
    "avg_order_value",
    "user_session_count_7d",
    "primary_category",   # encoded int
    "payment_method",     # encoded int
    "currency",           # encoded int
    "primary_device",     # encoded int
]


def _build_feature_vector(features: dict[str, Any]) -> str:
    """Encode categorical features and serialise as a CSV row for the endpoint."""
    primary_category = CATEGORY_ENCODER.get(
        str(features.get("primary_category", "")).lower(), 0
    )
    payment_method = PAYMENT_ENCODER.get(
        str(features.get("payment_method", "")).lower(), 3
    )
    currency = CURRENCY_ENCODER.get(
        str(features.get("currency", "USD")).upper(), 2
    )
    primary_device = DEVICE_ENCODER.get(
        str(features.get("primary_device", "desktop")).lower(), 1
    )

    row = [
        features.get("days_since_last_purchase", 0),
        features.get("total_orders", 1),
        features.get("avg_order_value", 0.0),
        features.get("user_session_count_7d", 0),
        primary_category,
        payment_method,
        currency,
        primary_device,
    ]
    return ",".join(str(v) for v in row)


@tool
def score_refund_risk(features_json: str) -> str:
    """
    Score refund / churn risk for a customer order using the SageMaker XGBoost endpoint.

    Args:
        features_json: JSON string with keys:
            days_since_last_purchase (int),
            total_orders (int),
            avg_order_value (float),
            user_session_count_7d (int),
            primary_category (str: beauty|books|clothing|electronics|home|sports|toys),
            payment_method (str: apple_pay|credit_card|google_pay|paypal),
            currency (str: USD|EUR|GBP),
            primary_device (str: android|desktop|ios)

    Returns:
        JSON string with refund_probability (float 0-1) and risk_tier (low/medium/high).
    """
    cfg = load_settings()

    try:
        features = json.loads(features_json)
    except json.JSONDecodeError as exc:
        return json.dumps({"error": f"Invalid JSON: {exc}"})

    csv_payload = _build_feature_vector(features)
    logger.debug("Endpoint payload: %s", csv_payload)

    try:
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=cfg.AWS_REGION,
            aws_access_key_id=cfg.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=cfg.AWS_SECRET_ACCESS_KEY,
        )
        response = runtime.invoke_endpoint(
            EndpointName=cfg.SAGEMAKER_ENDPOINT_NAME,
            ContentType="text/csv",
            Body=csv_payload,
        )
        raw = response["Body"].read().decode("utf-8").strip()
        prob = float(raw.split("\n")[0])
    except Exception as exc:
        logger.error("SageMaker endpoint error: %s", exc)
        return json.dumps({"error": str(exc)})

    if prob < 0.30:
        risk_tier = "low"
    elif prob < 0.60:
        risk_tier = "medium"
    else:
        risk_tier = "high"

    return json.dumps(
        {
            "refund_probability": round(prob, 4),
            "risk_tier": risk_tier,
            "threshold_used": 0.50,
            "model": cfg.SAGEMAKER_ENDPOINT_NAME,
        }
    )
