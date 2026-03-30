DEMO HF : https://huggingface.co/spaces/melon771/ForexGuard

# ForexGuard — Real-Time Trader Anomaly Detection Engine

> An industry-grade, production-ready anomaly detection system that identifies suspicious trader behaviour across client portal and trading terminal activity in a forex brokerage environment.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture Overview](#architecture-overview)
3. [Tech Stack](#tech-stack)
4. [Feature Engineering](#feature-engineering)
5. [Model Explanation](#model-explanation)
6. [API Reference](#api-reference)
7. [Setup Instructions](#setup-instructions)
8. [Suspicious Behaviour Detection](#suspicious-behaviour-detection)
9. [Assumptions, Trade-offs & Limitations](#assumptions-trade-offs--limitations)

---

## Project Overview

ForexGuard is a real-time anomaly detection engine built to identify suspicious trader and user behaviour in a forex brokerage environment. The system ingests raw events from client portals and trading terminals, engineers behavioural features per user, scores each user using an ensemble of ML models, and generates human-readable alerts for a compliance team — all delivered through a REST API.

The system detects a wide range of anomalous patterns including:

- Login from unusual geographies or at unusual hours
- Bot-like automated trading behaviour
- Deposit → minimal trading → withdrawal cycles (bonus abuse)
- High-frequency small deposits (financial structuring)
- Sudden trade volume spikes (10x personal baseline)
- KYC changes immediately before large withdrawals
- Consistent abnormal profits in short windows (latency arbitrage)
- Multi-IP and multi-device account abuse

---
## Project Structure

```
forexguard/
├── docker-compose.yml          # RabbitMQ + PostgreSQL
├── config.py                   # All connection settings
├── requirements.txt
├── .env                        # Credentials (not committed)
│
├── data/
│   ├── events.csv              # Synthetic dataset (50,000 events)
│   └── dataset.py              # Data generator script
│
├── db/
│   ├── schema.sql              # raw_events + alerts tables
│   └── add_features_table.sql  # user_features table
│
├── pipeline/
│   ├── producer.py             # CSV → RabbitMQ publisher
│   ├── consumer.py             # RabbitMQ → PostgreSQL consumer
│   └── feature_engineer.py     # Builds user_features table
│
├── features/
│   ├── network_features.py     # IP, device, geo signals
│   ├── session_features.py     # Login, timing, bot detection
│   └── trade_features.py       # PnL, volume, financial flows
│
├── models/
│   ├── isolation_forest.py     # Baseline anomaly model
│   ├── lstm_autoencoder.py     # Advanced sequence model
│   ├── ensemble.py             # Combines scores + explainability
│   └── saved/                  # Trained model artifacts (.pkl, .pt)
│
├── api/
│   ├── main.py                 # FastAPI app entry point
│   ├── routes.py               # /score /alerts /health endpoints
│   └── schemas.py              # Pydantic request/response models
│
├── alerts/
│   └── publisher.py            # Formats + publishes to RabbitMQ
│
└── tests/
    └── test_pipeline.py        # End-to-end smoke test
```
## Architecture Overview

```
Raw Events (CSV / Live Stream)
           │
           ▼
    [ RabbitMQ Queue ]
    forex.events exchange
           │
           ▼
    [ Consumer ]
    Deserialises JSON
    Writes to PostgreSQL
    raw_events table
           │
           ▼
    [ Feature Engineering ]
    Per-user behavioural features
    Rolling windows, z-scores,
    deviation scoring
    → user_features table
           │
           ▼
    ┌──────────────────────┐
    │   Isolation Forest   │  ← Baseline model (sklearn)
    │   LSTM Autoencoder   │  ← Advanced model (PyTorch)
    └──────────────────────┘
           │
           ▼
    [ Ensemble Scorer ]
    Weighted combination (60% IF + 40% LSTM)
    SHAP / feature deviation explainability
    Risk labels: low / medium / high
           │
           ▼
    [ FastAPI — /score endpoint ]
    Returns: risk_score, risk_label,
             top_features, alert_sent
           │
           ├──► [ PostgreSQL alerts table ]
           │
           └──► [ RabbitMQ forex.alerts queue ]
                 Human-readable alert messages
                 for compliance team
```


## Tech Stack

| Layer | Tool | Reason |
|---|---|---|
| Message Queue | RabbitMQ | Lightweight, reliable, topic exchange routing, management UI |
| Database | PostgreSQL 15 | ACID compliance, JSONB for alert features, strong indexing |
| ML — Baseline | Isolation Forest (sklearn) | Fast, unsupervised, no labels needed, SHAP-compatible |
| ML — Advanced | LSTM Autoencoder (PyTorch) | Captures temporal sequence anomalies IF misses |
| Feature Engineering | Pandas + NumPy | Rolling windows, z-scores, inter-event timing |
| API | FastAPI + Uvicorn | Async, automatic Swagger docs, Pydantic validation |
| Explainability | SHAP + Feature Deviation | Required by spec — top contributing features per user |
| Experiment Tracking | MLflow | Logs parameters, metrics, model artifacts per run |
| Infrastructure | Docker + Docker Compose | Single command setup, identical environment everywhere |
| Language | Python 3.10 | Type hints, modern async, broad ML library support |

---


### Table: `user_features`

One row per user. Computed by `feature_engineer.py`. Models train on this table.

| Column Group | Columns | Description |
|---|---|---|
| Identity | `user_id`, `computed_at` | User identifier and when features were last computed |
| Event counts | `total_events`, `total_logins`, `total_trades`, `total_deposits`, `total_withdrawals` | Raw activity volumes |
| Login signals | `failed_login_ratio`, `login_hour_mean`, `login_hour_std`, `login_hour_deviation` | Login timing anomalies |
| Network signals | `unique_ips`, `unique_devices`, `unique_countries`, `ip_change_rate`, `max_ip_switch_gap_hours` | Geographic and device anomalies |
| Session signals | `session_duration_mean`, `session_duration_std`, `session_duration_zscore`, `short_session_ratio`, `avg_events_per_session` | Session behaviour |
| Timing signals | `avg_time_between_events`, `min_time_between_events`, `time_delta_std`, `bot_like_timing` | Bot detection |
| Financial signals | `deposit_total`, `withdraw_total`, `deposit_withdraw_ratio`, `net_flow`, `deposit_to_trade_ratio` | Money flow patterns |
| Fraud flags | `large_withdraw_after_dormancy`, `high_freq_small_deposits`, `kyc_change_before_withdraw` | Hard fraud signals |
| Trading signals | `trade_volume_mean`, `trade_volume_std`, `trade_volume_zscore`, `trade_volume_spike`, `pnl_total`, `pnl_volatility`, `win_rate`, `consistent_profit_bursts` | Trading anomalies |
| Instrument | `unique_instruments`, `instrument_concentration` | Concentration risk |
| Leverage | `avg_leverage`, `avg_margin_used`, `margin_usage_zscore` | Leverage abuse |
| Model scores | `isolation_forest_score`, `lstm_reconstruction_error`, `ensemble_risk_score`, `risk_label` | ML outputs |

### Table: `alerts`

Stores every alert generated by the system.

| Column 
|---|
| `id` 
| `user_id`
| `alert_type`
| `risk_score` 
| `top_features`
| `message`
| `created_at`
| `acknowledged`

---

## Feature Engineering

Features are computed per user across their entire event history using three modules:

### `features/network_features.py`

Detects IP, device, and geographic anomalies.

- `unique_ips` — number of distinct IP addresses used. High = multi-location access.
- `unique_devices` — distinct device fingerprints. High = device sharing or spoofing.
- `unique_countries` — distinct countries. High = impossible travel.
- `ip_change_rate` — IP switches per active day. High = rapid geo-jumping.
- `max_ip_switch_gap_hours` — smallest time gap between IP switches. Very low = physically impossible travel.

### `features/session_features.py`

Detects bot-like behaviour, unusual login patterns, and abnormal session activity.

- `failed_login_ratio` — ratio of failed to total login attempts. High = credential stuffing.
- `login_hour_mean` / `login_hour_std` — user's typical login hours.
- `login_hour_deviation` — z-score of latest login hour vs personal baseline. High = unusual time.
- `session_duration_mean` / `session_duration_zscore` — typical and anomalous session lengths.
- `short_session_ratio` — fraction of sessions under 60 seconds. High = scripted/bot access.
- `avg_events_per_session` — events fired per session. Very high = automation.
- `avg_time_between_events` / `min_time_between_events` — inter-event timing.
- `bot_like_timing` — TRUE if any events fired less than 1 second apart.
- `time_delta_std` — standard deviation of inter-event gaps. Very low = robotic regularity.

### `features/trade_features.py`

Detects trading anomalies, financial fraud patterns, and abusive money flows.

- `deposit_withdraw_ratio` — withdrawal / deposit ratio. High = extracting money fast.
- `deposit_to_trade_ratio` — traded volume / deposits. Low = deposit → withdraw without trading (bonus abuse).
- `large_withdraw_after_dormancy` — TRUE if withdrawal follows 14+ day inactivity gap.
- `high_freq_small_deposits` — TRUE if 5+ deposits under $500 within 24 hours (structuring).
- `net_flow` — total deposits minus total withdrawals.
- `trade_volume_mean` / `trade_volume_std` / `trade_volume_zscore` — rolling volume baseline.
- `trade_volume_spike` — TRUE if latest volume exceeds 10x personal mean.
- `pnl_volatility` — standard deviation of per-trade profit. High = erratic / lucky trading.
- `pnl_zscore` — latest PnL vs personal baseline.
- `win_rate` — fraction of profitable trades.
- `consistent_profit_bursts` — TRUE if win rate > 80% in last 10 trades (latency arbitrage signal).
- `instrument_concentration` — 1.0 = all trades in single instrument (concentration risk).
- `avg_leverage` / `margin_usage_zscore` — leverage abuse detection.
- `kyc_change_before_withdraw` — TRUE if KYC updated within 24 hours of a withdrawal.
- `balance_volatility` — std of account balance across history.


## Model Explanation

### Why Isolation Forest (Baseline)?

Isolation Forest was chosen as the baseline model for the following reasons:

**How it works:** It randomly partitions the feature space by selecting a random feature and a random split value. Anomalous points require fewer partitions to isolate because they sit far from the dense normal cluster. The anomaly score is based on average path length across all trees — shorter path = more anomalous.

**Why we chose it over alternatives:**

| Alternative | Why we didn't use it |
|---|---|
| Local Outlier Factor (LOF) | Does not scale well to 500+ users with 34 features — quadratic complexity |
| One-Class SVM | Sensitive to feature scaling, slow on high-dimensional data, hard to explain |
| K-Means clustering | Requires choosing K, assumes spherical clusters, poor at detecting point anomalies |
| DBSCAN | Density threshold is hard to tune without labels, poor performance on mixed-scale features |

**Advantages for our use case:**
- No labels required — purely unsupervised
- Handles high-dimensional feature space (34 features) well
- Fast training (~10 seconds on 500 users)
- Compatible with SHAP for explainability
- `contamination=0.05` directly encodes our assumption that ~5% of users are suspicious

### Why LSTM Autoencoder (Advanced)?

The LSTM Autoencoder was chosen as the advanced model because trader behaviour is inherently sequential — the order and timing of events matters, not just their aggregate statistics.

**How it works:** The encoder LSTM compresses a user's feature sequence into a low-dimensional latent vector. The decoder LSTM attempts to reconstruct the original sequence from this compressed representation. Normal users reconstruct accurately (low error). Anomalous users whose behaviour deviates from the learned normal pattern produce high reconstruction error — that error is the anomaly score.

**Why we chose it over alternatives:**

| Alternative | Why we didn't use it |
|---|---|
| Variational Autoencoder (VAE) | More complex to train, requires careful KL-divergence tuning, overkill for tabular data |
| Transformer-based AE | Requires much larger datasets (50k events is borderline), heavy compute, harder to explain |
| Standard Autoencoder (MLP) | Ignores temporal ordering of events — misses sequence-level anomalies |
| Prophet / ARIMA | Univariate only — cannot handle the multi-feature behavioural profile we need |

**Advantages for our use case:**
- Captures temporal dependencies that Isolation Forest misses entirely
- Reconstruction error is naturally interpretable — high error = unusual sequence
- Works well on per-user aggregate feature vectors when raw event sequences are not available
- PyTorch implementation is lightweight and fast to retrain

### Ensemble Strategy

The final risk score combines both models:

```
ensemble_score = 0.60 × isolation_forest_score + 0.40 × lstm_reconstruction_error
```

**Why 60/40 weighting?**
- Isolation Forest is more reliable on tabular aggregate features
- LSTM adds signal on temporal patterns but is noisier on small per-user histories
- The weighted combination consistently outperforms either model alone

**Risk thresholds:**
- `score >= 0.75` → HIGH risk → alert sent to compliance + RabbitMQ
- `score >= 0.50` → MEDIUM risk → flagged but no immediate alert
- `score < 0.50` → LOW risk → normal behaviour

**Screenshot — Model Scores Output**

RF
<img width="1516" height="291" alt="image" src="https://github.com/user-attachments/assets/157d6ce3-ef98-43ef-a608-d23a4aa7fc1a" />

LSTM
<img width="1446" height="321" alt="image" src="https://github.com/user-attachments/assets/203d15e5-38cf-43fb-af88-b11f77679b26" />

Combined
<img width="1486" height="866" alt="image" src="https://github.com/user-attachments/assets/27554a32-41aa-4ca7-a535-6031441ecf75" />




---

## API Reference

The API is served by FastAPI with automatic interactive documentation.

**Base URL:** `http://localhost:8000`

**Interactive docs:** `http://localhost:8000/docs`

### `GET /health`

Returns system health status.

```json
{
  "status": "ok",
  "db": "ok",
  "rabbitmq": "ok",
  "models_loaded": true
}
```

### `POST /score`

Score a user by their user_id. Returns risk score, label, top contributing features, and whether an alert was sent.

**Request:**
```json
{ "user_id": "92" }
```

**Response:**
```json
{
  "user_id": "92",
  "risk_score": 1.0,
  "risk_label": "high",
  "if_score": 0.95,
  "lstm_score": 0.88,
  "top_features": [
    {
      "feature": "trade_volume_spike",
      "description": "Sudden trade volume spike (10x baseline)",
      "value": 1.0,
      "importance": 2.0
    },
    {
      "feature": "kyc_change_before_withdraw",
      "description": "KYC change within 24h of withdrawal",
      "value": 1.0,
      "importance": 2.0
    },
    {
      "feature": "pnl_volatility",
      "description": "High PnL volatility",
      "value": 55.93,
      "importance": 10.0
    }
  ],
  "alert_sent": true,
  "scored_at": "2026-03-29T08:35:01Z"
}
```

### `GET /alerts`

Returns recent alerts. Supports `limit` and `min_score` query params.

```
GET /alerts?limit=20&min_score=0.75
```

**Screenshot — /Score Response**

<img width="2020" height="678" alt="image" src="https://github.com/user-attachments/assets/fc1baa07-fc76-47a2-a443-4eaf4a5c2428" />

**Screenshot — /alerts Response**

<img width="2332" height="649" alt="image" src="https://github.com/user-attachments/assets/9a5385b9-d72f-4c8d-9207-9db1954aae09" />




## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker Desktop
- Git

### Step 1 — Clone and install dependencies

```bash
git clone <your-repo-url>
cd forexguard
pip install -r requirements.txt
```

### Step 2 — Start infrastructure

```bash
docker compose up -d
docker compose ps   # both should show "healthy"
```

This starts:
- PostgreSQL on port 5432
- RabbitMQ on port 5672 (management UI at http://localhost:15672 — login: guest/guest)

### Step 3 — Create database tables

```bash
docker exec -i forexguard_pg psql -U forexguard -d forexguard < db/schema.sql
docker exec -i forexguard_pg psql -U forexguard -d forexguard < db/add_features_table.sql
```

### Step 4 — Ingest the dataset

Open two terminals:

**Terminal 1 — start consumer:**
```bash
python pipeline/consumer.py --batch-size 200
```

**Terminal 2 — run producer:**
```bash
python pipeline/producer.py --csv data/events.csv
```

Wait for the producer to finish (progress bar hits 100%).

### Step 5 — Run feature engineering

```bash
python pipeline/feature_engineer.py
```

Verify:
```bash
docker exec -it forexguard_pg psql -U forexguard -d forexguard -c "SELECT COUNT(*) FROM user_features;"
```

### Step 6 — Train models

```bash
python models/isolation_forest.py --explain
python models/lstm_autoencoder.py --epochs 30
python models/ensemble.py --top 15
```

Model files are saved to `models/saved/`.

### Step 7 — Start the API

```bash
cd api
python -m uvicorn main:app --reload --port 8000
```

### Step 8 — Test

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:8000/health"

# Score a user
Invoke-RestMethod -Uri "http://localhost:8000/score" -Method POST -ContentType "application/json" -Body '{"user_id": "92"}'

# View alerts
Invoke-RestMethod -Uri "http://localhost:8000/alerts"
```

Or open `http://localhost:8000/docs` for the interactive Swagger UI.


## Suspicious Behaviour Detection

The system is designed to detect the following categories of anomalies:

### Login & Access Anomalies
- Simultaneous logins from multiple IPs
- Rapid IP switching across geographies (impossible travel)
- Login from unusual hours vs personal baseline
- High failed login attempts followed by success
- New device fingerprint detected

### Financial Behaviour Anomalies
- Large withdrawal after 14+ day dormancy period
- Deposit → minimal trading → withdrawal (bonus abuse cycle)
- High-frequency small deposits under $500 (structuring / money laundering signal)
- High withdrawal-to-deposit ratio

### Trading Behaviour Anomalies
- Trade volume spike greater than 10x personal baseline
- Single-instrument concentration (all trades in one pair)
- Consistent profit rate above 80% in short windows (latency arbitrage)
- Abnormal PnL volatility

### Behavioural Anomalies
- Bot-like event timing (sub-second inter-event gaps)
- Unusually short sessions (under 60 seconds)
- High events-per-session ratio (automation signal)

### Account Risk Patterns
- KYC profile update within 24 hours of a withdrawal
- Multiple failed logins followed by successful access
- Rapid account balance volatility

---

## Assumptions, Trade-offs & Limitations

### Assumptions

- **5% contamination rate** — We assume approximately 5% of users exhibit genuinely anomalous behaviour. This is encoded in `IsolationForest(contamination=0.05)` and influences the anomaly score distribution.
- **Synthetic dataset** — The 50,000 events were generated synthetically using realistic distributions. Real-world performance may differ as real fraud patterns are more subtle and adaptive.
- **Stateless scoring** — The `/score` endpoint uses pre-computed features from `user_features`. It does not recompute features in real-time on every API call. This trades latency for throughput.
- **User-level granularity** — Anomaly detection operates at the per-user aggregate level. Event-level real-time scoring would require a streaming feature store (e.g. Redis) which is beyond this prototype scope.
- **No ground truth labels** — The system is fully unsupervised. There are no labelled fraud cases to validate against. Performance is assessed qualitatively by inspecting top-flagged users and their feature profiles.

### Trade-offs

| Decision | Trade-off |
|---|---|
| Pre-computed features vs real-time features | Faster API response but scores are only as fresh as the last `feature_engineer.py` run |
| Isolation Forest as baseline | Fast and explainable but assumes anomalies are globally rare — misses local cluster anomalies |
| LSTM on aggregate features vs raw event sequences | Practical given dataset size but loses fine-grained temporal signal |
| 60/40 ensemble weighting | Empirically set — would need labelled data and cross-validation to optimise properly |
| RabbitMQ over Kafka | Simpler setup and sufficient for this volume — Kafka would be preferred at 10M+ events/day |
| Batch feature engineering vs streaming | Simpler and reliable — streaming feature computation (Faust / Spark Streaming) would be needed for true real-time |

### Limitations

- **No model retraining pipeline** — Models are trained once. In production, models should retrain weekly on fresh data as user behaviour evolves.
- **No feedback loop** — The system has no mechanism for compliance analysts to mark alerts as true/false positives, which would enable supervised fine-tuning over time.
- **Single-node deployment** — The current architecture runs on a single machine. A production system would require horizontal scaling of the API and consumer layers.
- **SHAP on Isolation Forest is approximate** — SHAP TreeExplainer on Isolation Forest produces approximate explanations. For high-stakes decisions, additional explainability methods should be applied.
- **Graph-level anomalies not implemented** — Section 8.5 of the spec (collusion rings, mirror trades, withdrawal clustering across accounts) requires graph neural networks or network analysis tools (NetworkX / PyG) which are beyond this prototype scope.
- **No authentication on API** — The `/score` and `/alerts` endpoints are unprotected. Production deployment must add OAuth2 / API key authentication.

---

### FastAPI Swagger UI
<img width="2227" height="628" alt="image" src="https://github.com/user-attachments/assets/b5e3eb17-66ef-4b08-af94-ce907c7eff51" />

### HF screenshots

<img width="2030" height="1319" alt="image" src="https://github.com/user-attachments/assets/c478ccfa-5130-411c-b6cf-322c3b5fd09c" />

<img width="2480" height="667" alt="image" src="https://github.com/user-attachments/assets/58c4614e-67ca-4eae-88d2-21a1fa97cb95" />



---

