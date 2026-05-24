"""
API Test Suite for the Churn Scoring Service.
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# ─────────────────────────────
# Helper payloads (25 features)
# ─────────────────────────────
HIGH_RISK_CUSTOMER = {
    "customer_id": "CUST_HIGH_001",
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
}

LOW_RISK_CUSTOMER = {
    "customer_id": "CUST_LOW_001",
    "city_tier": 1.0,
    "age_group": 2.0,
    "acquisition_channel": 3.0,
    "loyalty_tier": 3.0,
    "preferred_category": 4.0,
    "marketing_consent": 1.0,
    "recency_days": 3.0,
    "frequency_180d": 25.0,
    "monetary_180d": 18000.0,
    "return_rate_180d": 0.0,
    "avg_discount_pct_180d": 0.05,
    "avg_rating_180d": 5.0,
    "category_diversity_180d": 4.0,
    "ticket_count_90d": 0.0,
    "negative_ticket_rate_90d": 0.0,
    "avg_resolution_hours_90d": 0.0,
    "days_since_signup": 600.0,
    "sessions_30d": 40.0,
    "product_views_30d": 150.0,
    "cart_adds_30d": 20.0,
    "wishlist_adds_30d": 5.0,
    "abandoned_carts_30d": 2.0,
    "email_opens_30d": 10.0,
    "campaign_clicks_30d": 5.0,
    "last_visit_days_ago": 1.0
}

# ─────────────────────────────
# Test 1: Health Check
# ─────────────────────────────
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "api_version" in data

# ─────────────────────────────
# Test 2: Predict — High Risk
# ─────────────────────────────
def test_predict_high_risk_customer():
    with TestClient(app) as client:
        response = client.post("/predict", json=HIGH_RISK_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert "predicted_class" in data
        assert "risk_level" in data
        assert "risk_explanation" in data
        assert data["predicted_class"] in [0, 1]
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["low", "medium", "high"]

# ─────────────────────────────
# Test 3: Predict — Low Risk
# ─────────────────────────────
def test_predict_low_risk_customer():
    with TestClient(app) as client:
        response = client.post("/predict", json=LOW_RISK_CUSTOMER)
        assert response.status_code == 200
        data = response.json()
        assert data["churn_probability"] < 0.5
        assert data["risk_level"] in ["low", "medium"]

# ─────────────────────────────
# Test 4: Batch Predict
# ─────────────────────────────
def test_batch_predict():
    with TestClient(app) as client:
        batch = [HIGH_RISK_CUSTOMER, LOW_RISK_CUSTOMER]
        response = client.post("/batch_predict", json=batch)
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
        assert "high_risk_count" in data

# ─────────────────────────────
# Test 5: Batch — Empty List
# ─────────────────────────────
def test_batch_predict_empty_list():
    with TestClient(app) as client:
        response = client.post("/batch_predict", json=[])
        assert response.status_code == 400

# ─────────────────────────────
# Test 6: Input Validation — Missing Required Field
# ─────────────────────────────
def test_predict_missing_required_field():
    with TestClient(app) as client:
        incomplete = {
            "customer_id": "CUST_BAD",
            "recency_days": 45.0
            # Missing remaining 24 fields
        }
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422  # Unprocessable Entity

# ─────────────────────────────
# Test 7: Input Validation — Invalid Values
# ─────────────────────────────
def test_predict_invalid_return_rate():
    with TestClient(app) as client:
        bad_payload = HIGH_RISK_CUSTOMER.copy()
        bad_payload["return_rate_180d"] = 1.5  # Must be 0-1
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

# ─────────────────────────────
# Test 8: Customer ID is Optional
# ─────────────────────────────
def test_predict_without_customer_id():
    with TestClient(app) as client:
        payload = HIGH_RISK_CUSTOMER.copy()
        del payload["customer_id"]
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert response.json()["customer_id"] is None
