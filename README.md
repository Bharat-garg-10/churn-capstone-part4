# Part 4: FastAPI Churn Scoring Service
## D2C Customer Churn Capstone

### Overview
A production-ready REST API that loads a trained XGBoost churn prediction model (from Part 3) and exposes endpoints for single and batch customer churn-risk scoring. Built for integration with an internal CRM tool.

### Project Structure
```
churn-capstone-part4/
├── app/main.py          ← FastAPI application
├── tests/test_api.py    ← API test suite
├── model.pkl            ← Trained XGBoost model
├── requirements.txt     ← Python dependencies
├── monitoring_plan.md   ← Post-deployment monitoring
├── Dockerfile           ← Docker setup
└── README.md
```

### Setup & Installation
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/churn-capstone-part4.git
cd churn-capstone-part4

# 2. Install dependencies
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn app.main:app --reload --port 8000
```

Interactive docs available at: `http://localhost:8000/docs`

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check — confirms API and model status |
| POST | `/predict` | Single customer churn prediction |
| POST | `/batch_predict` | Batch predictions (max 500 customers) |

### Sample Request — POST /predict
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_001",
    "city_tier": 1.0,
    "age_group": 2.0,
    "acquisition_channel": 3.0,
    "loyalty_tier": 2.0,
    "preferred_category": 4.0,
    "marketing_consent": 1.0,
    "recency_days": 150.0,
    "frequency_180d": 1.0,
    "monetary_180d": 400.0,
    "return_rate_180d": 0.5,
    "avg_discount_pct_180d": 0.2,
    "avg_rating_180d": 2.0,
    "category_diversity_180d": 1.0,
    "ticket_count_90d": 6.0,
    "negative_ticket_rate_90d": 1.0,
    "avg_resolution_hours_90d": 48.0,
    "days_since_signup": 300.0,
    "sessions_30d": 0.0,
    "product_views_30d": 0.0,
    "cart_adds_30d": 0.0,
    "wishlist_adds_30d": 0.0,
    "abandoned_carts_30d": 0.0,
    "email_opens_30d": 0.0,
    "campaign_clicks_30d": 0.0,
    "last_visit_days_ago": 60.0
  }'
```

### Sample Response
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.8872,
  "predicted_class": 1,
  "risk_level": "high",
  "risk_explanation": "Churn probability 89%. Key drivers: very high inactivity (90+ days since last order); no recent app/web visits; high support complaint volume; high rate of negative tickets; high product return rate; low purchase frequency in last 6 months."
}
```

### Run Tests
```bash
pytest tests/ -v
```

### Docker
```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

### Model Notes
- Model: XGBoost Binary Classifier
- Format: joblib pickle (`model.pkl`)
- Trained on D2C customer snapshot — see Part 3 repository for full training details
- Decision threshold: 0.30 (tuned for high recall on churners as determined in Part 3)