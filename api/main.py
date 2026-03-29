"""
api/main.py
-----------
FastAPI application entry point.
Models are loaded ONCE on startup and stored in app.state
so every request reuses them without reloading from disk.

Usage:
    uvicorn api.main:app --reload --port 8000
    
Then open: http://localhost:8000/docs  (auto-generated Swagger UI)
"""

import logging
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg2
import pika
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [api] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent.parent / "models" / "saved"


# ── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all resources once on startup, clean up on shutdown."""
    log.info("Starting ForexGuard API …")

    # ── PostgreSQL ────────────────────────────────────────────
    try:
        app.state.pg_conn = psycopg2.connect(config.PG_DSN)
        app.state.pg_conn.autocommit = False
        log.info("PostgreSQL connected")
    except Exception as exc:
        log.error(f"PostgreSQL connection failed: {exc}")
        app.state.pg_conn = None

    # ── RabbitMQ ──────────────────────────────────────────────
    try:
        creds  = pika.PlainCredentials(config.RABBIT_USER, config.RABBIT_PASS)
        params = pika.ConnectionParameters(
            host=config.RABBIT_HOST,
            port=config.RABBIT_PORT,
            credentials=creds,
            heartbeat=600,
        )
        app.state.rabbit_conn    = pika.BlockingConnection(params)
        app.state.rabbit_channel = app.state.rabbit_conn.channel()
        # Declare the alerts queue
        app.state.rabbit_channel.queue_declare(
            queue="forex.alerts", durable=True
        )
        log.info("RabbitMQ connected")
    except Exception as exc:
        log.warning(f"RabbitMQ connection failed (alerts won't publish): {exc}")
        app.state.rabbit_conn    = None
        app.state.rabbit_channel = None

    # ── Load models ───────────────────────────────────────────
    app.state.models_loaded = False
    try:
        if_model_path  = SAVED_DIR / "isolation_forest.pkl"
        if_scaler_path = SAVED_DIR / "if_scaler.pkl"

        if if_model_path.exists() and if_scaler_path.exists():
            with open(if_model_path,  "rb") as f:
                app.state.if_model  = pickle.load(f)
            with open(if_scaler_path, "rb") as f:
                app.state.if_scaler = pickle.load(f)
            app.state.models_loaded = True
            log.info("Isolation Forest loaded")
        else:
            log.warning("Model files not found — run models/isolation_forest.py first")
    except Exception as exc:
        log.error(f"Model loading failed: {exc}")

    log.info("ForexGuard API ready")
    yield

    # ── Shutdown cleanup ──────────────────────────────────────
    log.info("Shutting down …")
    if app.state.pg_conn:
        app.state.pg_conn.close()
    if app.state.rabbit_conn and not app.state.rabbit_conn.is_closed:
        app.state.rabbit_conn.close()
    log.info("Connections closed")


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ForexGuard — Anomaly Detection API",
    description=(
        "Real-time trader anomaly detection engine. "
        "Submit a user_id to get their risk score, contributing features, "
        "and alert status."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow all origins for demo purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)