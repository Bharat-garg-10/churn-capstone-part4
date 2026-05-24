"""
D2C Churn Scoring API
FastAPI service that loads a trained churn model and exposes prediction endpoints.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import joblib
import pandas as pd
import logging
import os

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title       = "Churn Scoring API",
    description = "D2C Customer Churn Risk Prediction Service for internal CRM use.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

# ─────────────────────────────────────────────
# Load Model on Startup
# ─────────────────────────────────────────────
MODEL      = None
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")

@app.on_event("startup")
def load_model():
    global MODEL
    try:
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"Model file not found at {MODEL_PATH}. Predictions will fail.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# ─────────────────────────────────────────────
# Feature Schema — matches XGBoost model from Part 3
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'city_tier', 'age_group', 'acquisition_channel', 'loyalty_tier', 
    'preferred_category', 'marketing_consent', 'recency_days', 'frequency_180d', 
    'monetary_180d', 'return_rate_180d', 'avg_discount_pct_180d', 'avg_rating_180d', 
    'category_diversity_180d', 'ticket_count_90d', 'negative_ticket_rate_90d', 
    'avg_resolution_hours_90d', 'days_since_signup', 'sessions_30d', 'product_views_30d', 
    'cart_adds_30d', 'wishlist_adds_30d', 'abandoned_carts_30d', 'email_opens_30d', 
    'campaign_clicks_30d', 'last_visit_days_ago'
]

class CustomerFeatures(BaseModel):
    customer_id                 : Optional[str] = Field(None, example="CUST_001")
    city_tier                   : float = Field(..., example=1.0)
    age_group                   : float = Field(..., example=2.0)
    acquisition_channel         : float = Field(..., example=3.0)
    loyalty_tier                : float = Field(..., example=2.0)
    preferred_category          : float = Field(..., example=4.0)
    marketing_consent           : float = Field(..., example=1.0)
    recency_days                : float = Field(..., ge=0, example=45.0)
    frequency_180d              : float = Field(..., ge=0, example=8.0)
    monetary_180d               : float = Field(..., ge=0, example=4500.0)
    return_rate_180d            : float = Field(..., ge=0.0, le=1.0, example=0.1)
    avg_discount_pct_180d       : float = Field(..., ge=0.0, le=1.0, example=0.2)
    avg_rating_180d             : float = Field(..., ge=1.0, le=5.0, example=4.5)
    category_diversity_180d     : float = Field(..., ge=1.0, example=3.0)
    ticket_count_90d            : float = Field(..., ge=0, example=2.0)
    negative_ticket_rate_90d    : float = Field(..., ge=0.0, le=1.0, example=0.0)
    avg_resolution_hours_90d    : float = Field(..., ge=0.0, example=24.5)
    days_since_signup           : float = Field(..., ge=0.0, example=365.0)
    sessions_30d                : float = Field(..., ge=0.0, example=12.0)
    product_views_30d           : float = Field(..., ge=0.0, example=45.0)
    cart_adds_30d               : float = Field(..., ge=0.0, example=5.0)
    wishlist_adds_30d           : float = Field(..., ge=0.0, example=2.0)
    abandoned_carts_30d         : float = Field(..., ge=0.0, example=1.0)
    email_opens_30d             : float = Field(..., ge=0.0, example=8.0)
    campaign_clicks_30d         : float = Field(..., ge=0.0, example=3.0)
    last_visit_days_ago         : float = Field(..., ge=0.0, example=5.0)

    @field_validator('monetary_180d')
    @classmethod
    def monetary_not_zero(cls, v):
        if v < 0:
            raise ValueError("monetary_180d must be >= 0")
        return v

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def get_risk_level(prob: float) -> str:
    if prob >= 0.70:
        return "high"
    elif prob >= 0.40:
        return "medium"
    return "low"

def build_explanation(features: dict, prob: float) -> str:
    reasons = []
    if features.get('recency_days', 0) > 90:
        reasons.append("very high inactivity (90+ days since last order)")
    elif features.get('recency_days', 0) > 60:
        reasons.append("moderate inactivity (60-90 days since last order)")
        
    if features.get('last_visit_days_ago', 0) > 30:
        reasons.append("no recent app/web visits")

    if features.get('ticket_count_90d', 0) >= 3:
        reasons.append("high support complaint volume")
        
    if features.get('negative_ticket_rate_90d', 0) > 0.5:
        reasons.append("high rate of negative tickets")

    if features.get('return_rate_180d', 0) > 0.30:
        reasons.append("high product return rate")

    if features.get('frequency_180d', 0) <= 2:
        reasons.append("low purchase frequency in last 6 months")

    if not reasons:
        reasons.append("moderate combined behavioural signals")

    return f"Churn probability {prob:.0%}. Key drivers: {'; '.join(reasons)}."

def run_prediction(customer: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check server logs.")
    features = customer.model_dump(exclude={'customer_id'})
    X = pd.DataFrame([features])[FEATURE_COLS]
    prob       = float(MODEL.predict_proba(X)[0][1])
    pred_class = int(prob >= 0.30)  # threshold from Part 3
    return prob, pred_class, features

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health", tags=["System"], summary="Health check")
def health():
    """
    Returns the API health status and whether the model is loaded.
    """
    return {
        "status"      : "ok",
        "model_loaded": MODEL is not None,
        "model_path"  : MODEL_PATH,
        "api_version" : "1.0.0"
    }


@app.post("/predict", tags=["Prediction"], summary="Single customer churn prediction")
def predict(customer: CustomerFeatures):
    """
    Accepts one customer's feature payload and returns a churn-risk prediction.
    """
    prob, pred_class, features = run_prediction(customer)
    return {
        "customer_id"      : customer.customer_id,
        "churn_probability": round(prob, 4),
        "predicted_class"  : pred_class,
        "risk_level"       : get_risk_level(prob),
        "risk_explanation" : build_explanation(features, prob)
    }


@app.post("/batch_predict", tags=["Prediction"], summary="Batch customer churn predictions")
def batch_predict(customers: List[CustomerFeatures]):
    """
    Accepts a list of customer feature payloads (max 500) and returns predictions for each.
    """
    if len(customers) == 0:
        raise HTTPException(status_code=400, detail="Request body must contain at least 1 customer.")
    if len(customers) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 customers per batch request.")

    predictions = []
    for customer in customers:
        prob, pred_class, features = run_prediction(customer)
        predictions.append({
            "customer_id"      : customer.customer_id,
            "churn_probability": round(prob, 4),
            "predicted_class"  : pred_class,
            "risk_level"       : get_risk_level(prob),
            "risk_explanation" : build_explanation(features, prob)
        })

    high_risk_count = sum(1 for p in predictions if p['risk_level'] == 'high')
    return {
        "count"           : len(predictions),
        "high_risk_count" : high_risk_count,
        "predictions"     : predictions
    }


# ─────────────────────────────────────────────
# Global Error Handler
# ─────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check server logs."}
    )
