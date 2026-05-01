# Refund Risk Agent

Internal LangGraph agent on AWS Bedrock that predicts order refund probability using XGBoost, grounds all policy output via Pinecone RAG, and pulls live order data from Databricks.

---

## Architecture

```
Streamlit UI (app.py)
    └── agents/refund_agent.py   [LangGraph ReAct · AWS Bedrock Claude 3.5 Sonnet]
            ├── tools/scoring_tool.py          → AWS SageMaker XGBoost endpoint
            ├── tools/order_lookup_tool.py     → Databricks Files API (Parquet)
            ├── tools/policy_retrieval_tool.py → Pinecone (llama-text-embed-v2)
            └── tools/product_help_tool.py     → Pinecone (category-scoped wrapper)
```

---

## API Keys & Where They Come From

| Env Var | Source |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS IAM — same account used in `project1_churn_xgboost.ipynb` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM |
| `AWS_REGION` | Region where SageMaker endpoint and Bedrock are enabled (e.g. `us-east-1`) |
| `SAGEMAKER_ENDPOINT_NAME` | Printed by the deploy cell in `project1_churn_xgboost.ipynb` (`project-xgb-YYYYMMDD-HHMMSS`) |
| `S3_BUCKET` | `sess.default_bucket()` from `project1_churn_xgboost.ipynb` |
| `BEDROCK_MODEL_ID` | Bedrock model ID — default `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `PINECONE_API_KEY` | Pinecone console → API Keys (used in `pinecone.ipynb`) |
| `PINECONE_INDEX_HOST` | `INDEX_HOST` constant in `pinecone.ipynb` |
| `PINECONE_INDEX_NAME` | `index_name` in `pinecone.ipynb` (`ecommerce-doc`) |
| `PINECONE_NAMESPACE` | `NAMESPACE` in `pinecone.ipynb` (`default`) |
| `DATABRICKS_HOST` | `host` variable in `databricks_parquet.ipynb` |
| `DATABRICKS_TOKEN` | `access_token` from `.env` in `databricks_parquet.ipynb` |
| `DATABRICKS_VOLUME_PATH` | `volume_path` in `databricks_parquet.ipynb` |

---

## Setup

```bash
# 1. Copy and fill in credentials
cp .env.example .env
# Edit .env with your actual values

# 2. Install dependencies
pip install -r requirements.txt

# 3. Ensure the Pinecone index is populated
#    Run pinecone.ipynb against the Master Policy Compendium docx first.

# 4. Ensure the SageMaker endpoint is live
#    Run the deploy cell in project1_churn_xgboost.ipynb first.

# 5. Launch Streamlit
streamlit run app.py
```

---

## Feature Vector Contract

The XGBoost model expects features in this exact column order (matching `X_train.columns` from the notebook):

```
days_since_last_purchase, total_orders, avg_order_value,
user_session_count_7d, primary_category (int), payment_method (int),
currency (int), primary_device (int)
```

Label encoders in `tools/scoring_tool.py` mirror the `LabelEncoder.classes_` printed during training. Update them if the training data categories change.

---

## Policy Citation Guarantee

`retrieve_policy` and `get_product_help` return `chunk_text` verbatim from Pinecone. The agent system prompt enforces that every recommendation must quote a retrieved `chunk_text` with its `chunk_index`. The Streamlit UI displays all citations in a dedicated panel — no policy text is invented.
