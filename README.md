DEMO HF : https://huggingface.co/spaces/melon771/ForexGuard
# ForexGuard — Real-Time Trader Anomaly Detection Engine

A production-grade anomaly detection system for forex brokerages. It watches client portal and trading terminal activity in real time, builds a behavioural profile per trader, and flags suspicious patterns before they become compliance problems.

---

## How it works

Raw events flow through RabbitMQ into PostgreSQL. A feature engineering pipeline builds a per-user behavioural profile. Two ML models — an Isolation Forest and an LSTM Autoencoder — score each user independently, and an ensemble combines their outputs into a single risk label. High-risk users trigger human-readable alerts that land in both the database and a compliance queue.

Patterns it catches: login from unusual geographies, bot-like trading, bonus abuse cycles (deposit → no trading → withdraw), financial structuring, sudden volume spikes, KYC changes before large withdrawals, and latency arbitrage.

---

## Docs

| | |
|---|---|
| [Architecture & Tech Stack](docs/arch.md) | System flow, component choices, project layout |
| [Database Schema](docs/schema.md) | `raw_events`, `user_features`, `alerts` table definitions |
| [Feature Engineering](docs/features.md) | Every behavioural signal the system computes, and why |
| [ML Models](docs/models.md) | Isolation Forest vs LSTM Autoencoder, ensemble strategy, risk thresholds |
| [API Reference](docs/api.md) | `/score`, `/alerts`, `/health` — request/response examples |
| [Detection Catalogue](docs/detections.md) | Full list of anomaly categories the system covers |
| [Setup Guide](docs/setup.md) | Step-by-step local setup from clone to live API |
| [Design Decisions](docs/decisions.md) | Assumptions made, trade-offs taken, known limitations |

---

## Quickstart

Prerequisites: Python 3.10+, Docker Desktop, Git.

```bash
git clone <your-repo-url>
cd forexguard
pip install -r requirements.txt

# Spin up Postgres + RabbitMQ
docker compose up -d

# Create tables
docker exec -i forexguard_pg psql -U forexguard -d forexguard < db/schema.sql
docker exec -i forexguard_pg psql -U forexguard -d forexguard < db/add_features_table.sql

# Ingest data (run both in parallel)
python pipeline/consumer.py --batch-size 200   # terminal 1
python pipeline/producer.py --csv data/events.csv  # terminal 2

# Build features, train models
python pipeline/feature_engineer.py
python models/isolation_forest.py --explain
python models/lstm_autoencoder.py --epochs 30
python models/ensemble.py --top 15

# Start the API
cd api && python -m uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive Swagger UI, or see [docs/setup.md](docs/setup.md) for the full walkthrough with verification steps.

---

## Stack

RabbitMQ · PostgreSQL 15 · Isolation Forest (sklearn) · LSTM Autoencoder (PyTorch) · FastAPI · SHAP · MLflow · Docker
