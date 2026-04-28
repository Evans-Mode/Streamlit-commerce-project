"""config/settings.py — centralised settings loaded from .env"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # ── AWS / SageMaker ───────────────────────────────────────────────────────
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    sagemaker_endpoint_name: str
    s3_bucket: str
    s3_prefix: str

    # ── AWS Bedrock (Nova Lite 2) ─────────────────────────────────────────────
    # bedrock_model_id is read from BEDROCK_MODEL_ID in .env
    # Recommended value: bedrock:us.amazon.nova-lite-v1:0
    bedrock_model_id: str

    # ── LangSmith ─────────────────────────────────────────────────────────────
    # These are read by LangChain automatically from env — stored here for
    # the startup connectivity check in refund_agent.py only.
    langsmith_api_key: str
    langsmith_project: str
    langsmith_endpoint: str

    # ── Pinecone ──────────────────────────────────────────────────────────────
    pinecone_api_key: str
    pinecone_index_host: str
    pinecone_namespace: str
    pinecone_index_name: str

    # ── Databricks ────────────────────────────────────────────────────────────
    databricks_host: str
    databricks_token: str
    databricks_volume_path: str


def load_settings() -> Settings:
    return Settings(
        aws_access_key_id       = os.environ.get("AWS_ACCESS_KEY_ID", ""),
        aws_secret_access_key   = os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        aws_region              = os.environ.get("AWS_REGION", "us-east-1"),
        sagemaker_endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME", ""),
        s3_bucket               = os.environ.get("S3_BUCKET", ""),
        s3_prefix               = os.environ.get("S3_PREFIX", "project"),
        bedrock_model_id        = os.environ.get(
            "BEDROCK_MODEL_ID", "bedrock:us.amazon.nova-lite-v1:0"
        ),
        langsmith_api_key       = os.environ.get("LANGSMITH_API_KEY", ""),
        langsmith_project       = os.environ.get("LANGSMITH_PROJECT", "default"),
        langsmith_endpoint      = os.environ.get(
            "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
        ),
        pinecone_api_key        = os.environ.get("PINECONE_API_KEY", ""),
        pinecone_index_host     = os.environ.get("PINECONE_INDEX_HOST", ""),
        pinecone_namespace      = os.environ.get("PINECONE_NAMESPACE", "default"),
        pinecone_index_name     = os.environ.get("PINECONE_INDEX_NAME", "ecommerce-doc"),
        databricks_host         = os.environ.get("DATABRICKS_HOST", ""),
        databricks_token        = os.environ.get("access_token", ""),
        databricks_volume_path  = os.environ.get(
            "DATABRICKS_VOLUME_PATH",
            "/Volumes/project_3/datalake/gold_zone/transaction_fact/",
        ),
    )
