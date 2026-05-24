# Monitoring Plan & Responsible Use
## Churn Scoring API — Post-Deployment

---

## 1. Data Drift Monitoring

**What to track**: Distribution of critical input features (e.g. `recency_days`, `negative_ticket_rate_90d`, `last_visit_days_ago`).
**How**: Log every incoming prediction payload to a data warehouse (like BigQuery or Snowflake). Run a weekly cron job to compare feature distributions against the original training baseline (2,401 snapshot customers) using statistical tests (KS test, Population Stability Index - PSI).

**Alert threshold**:
- Population Stability Index (PSI) > 0.2 for any of the top 5 features → investigate
- Mean `recency_days` shifts by more than 20% week-over-week → alert data engineering

**Tool suggestions**: `evidently`, `great_expectations`, or custom pandas scripts in Airflow.

---

## 2. Prediction Distribution

**What to track**: Daily distribution of `churn_probability` and `risk_level` (low/medium/high).
**How**: Append the API response metadata to the request log in the database. Compute daily summaries.

**Alert threshold**:
- % of "high risk" predictions exceeds 35% on any given day → check data pipeline for upstream anomalies.
- % of "high risk" drops below 3% → the model may have stopped discriminating effectively.

**Dashboard**: A BI dashboard (e.g., Tableau or Looker) showing a daily stacked bar chart of risk levels.

---

## 3. Business Outcomes

**What to track**: Ground truth churn labels once available (60 days after the prediction date).
**How**: Join the historical prediction logs (using `customer_id` and prediction date) to actual transaction records 60 days later.

**Metrics to recompute monthly**:
- ROC-AUC, PR-AUC, F1-Score
- False Negative Rate (missed churners — our most expensive error type)

**Retraining trigger**: ROC-AUC drops below 0.70 on the monthly validation set.

---

## 4. API Error Monitoring

**What to track**: HTTP 4XX and 5XX error rates.
**How**: Application-level logging (already configured in `main.py` via Python `logging`). Send these logs to a centralized log aggregator.

**Alert threshold**:
- 5XX errors > 0.5% of requests → page the backend engineering team.
- 422 Validation errors > 5% → check the CRM data pipeline feeding the API; upstream systems might be sending invalid schemas.

**Recommended tool**: Grafana + Prometheus, Datadog, or AWS CloudWatch.

---

## 5. Retraining Triggers

Retrain the model when any of the following occur:
1. **Performance Drop**: ROC-AUC drops below 0.70 on monthly evaluation.
2. **Feature Drift**: Feature PSI > 0.2 for 2+ features simultaneously.
3. **Business Shift**: Major product changes (e.g., introducing a completely new product category, pricing overhaul).
4. **Base Rate Shift**: Actual overall churn rate changes by more than 10% from the training base rate.
5. **Time-Based**: Every 3 months regardless of performance to capture evolving consumer trends.

---

## Responsible Use Guidelines

### The retention team SHOULD:
- Use `churn_probability` to **prioritize** which customers to contact first.
- Read the `risk_explanation` field to **personalize** outreach messaging (e.g., sending a "We miss you" email for high recency vs. a "How can we make things right?" email for high ticket volume).
- Combine model scores with human judgment for borderline cases (0.25 - 0.35 probability).
- Review manual cases where the model score strongly conflicts with known customer context (like VIP customers).

### The retention team SHOULD NOT:
- Automatically send heavy discounts to every high-risk customer without review (can lead to discount abuse).
- Use the score to **deny** any service, benefit, or promotion to a customer.
- Share churn scores externally or with unauthorized teams.
- Treat the score as an absolute guarantee — it is a probability, not a certainty.
- Rely heavily on the score during atypical periods (e.g., major holiday sale events, supply chain disruptions).

### Escalation Process
If a customer has a `churn_probability` > 0.85 AND `monetary_180d` > ₹5,000:
- Flag for personal outreach from a senior CX agent (not an automated email journey).
- Do not immediately send a generic discount — understand the exact reason for their dissatisfaction first.
