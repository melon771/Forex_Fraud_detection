"""
api/routes.py
-------------
All FastAPI route handlers.
The app state (DB conn, RabbitMQ channel, loaded models) is passed
via FastAPI's dependency injection through request.app.state.
"""

import logging
from datetime import datetime, timezone

import psycopg2.extras
from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    ScoreRequest, ScoreResponse,
    AlertResponse, HealthResponse,
    FeatureContribution,
)
from alerts.publisher import publish_alert
from models.ensemble import get_top_features, EXPLAIN_FEATURES

import pandas as pd

log = logging.getLogger(__name__)
router = APIRouter()

ALERT_THRESHOLD = 0.75   # score above this triggers an alert

# ── Feature columns needed for scoring ───────────────────────────────────────
SCORE_COLS = [
    "isolation_forest_score", "lstm_reconstruction_error",
    "ensemble_risk_score", "risk_label",
    "trade_volume_spike", "bot_like_timing",
    "large_withdraw_after_dormancy", "high_freq_small_deposits",
    "kyc_change_before_withdraw", "consistent_profit_bursts",
    "ip_change_rate", "deposit_withdraw_ratio",
    "login_hour_deviation", "session_duration_zscore",
    "trade_volume_zscore", "margin_usage_zscore",
    "pnl_volatility", "unique_ips",
]


def fetch_user_scores(conn, user_id: str) -> dict | None:
    """Pull pre-computed scores + feature values for a user."""
    cols = ", ".join(SCORE_COLS)
    sql  = f"SELECT {cols} FROM user_features WHERE user_id = %s"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (user_id,))
        row = cur.fetchone()
    if not row:
        return None
    # Convert to plain dict with floats
    result = {}
    for k, v in row.items():
        if v is None:
            result[k] = None
        elif hasattr(v, "__float__"):
            result[k] = float(v)
        elif isinstance(v, bool):
            result[k] = v
        else:
            result[k] = v
    return result


# ── /health ───────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    state = request.app.state

    # Check DB
    db_status = "ok"
    try:
        with state.pg_conn.cursor() as cur:
            cur.execute("SELECT 1")
    except Exception:
        db_status = "error"

    # Check RabbitMQ
    rmq_status = "ok"
    try:
        if state.rabbit_conn.is_closed:
            rmq_status = "closed"
    except Exception:
        rmq_status = "error"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        db=db_status,
        rabbitmq=rmq_status,
        models_loaded=state.models_loaded,
    )


# ── /score ────────────────────────────────────────────────────────────────────

@router.post("/score", response_model=ScoreResponse)
def score_user(body: ScoreRequest, request: Request):
    state   = request.app.state
    user_id = body.user_id

    # ── Fetch pre-computed features + scores ──────────────────
    row = fetch_user_scores(state.pg_conn, user_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"User '{user_id}' not found in user_features. "
                   "Run feature_engineer.py first."
        )

    ensemble_score = row.get("ensemble_risk_score")
    if ensemble_score is None:
        raise HTTPException(
            status_code=422,
            detail=f"No ensemble score for user '{user_id}'. "
                   "Run models/ensemble.py first."
        )

    risk_label = row.get("risk_label") or (
        "high"   if ensemble_score >= 0.75 else
        "medium" if ensemble_score >= 0.50 else
        "low"
    )

    # ── Get top contributing features ─────────────────────────
    row_series   = pd.Series(row)
    raw_features = get_top_features(row_series, top_n=5)
    top_features = [
        FeatureContribution(**f) for f in raw_features
    ]

    # ── Publish alert if high risk ────────────────────────────
    alert_sent = False
    if ensemble_score >= ALERT_THRESHOLD:
        try:
            publish_alert(
                rabbit_channel=state.rabbit_channel,
                pg_conn=state.pg_conn,
                user_id=user_id,
                risk_score=ensemble_score,
                top_features=[f.dict() for f in top_features],
            )
            alert_sent = True
        except Exception as exc:
            log.warning(f"Alert publish failed for {user_id}: {exc}")

    return ScoreResponse(
        user_id=user_id,
        risk_score=round(ensemble_score, 6),
        risk_label=risk_label,
        if_score=row.get("isolation_forest_score"),
        lstm_score=row.get("lstm_reconstruction_error"),
        top_features=top_features,
        alert_sent=alert_sent,
        scored_at=datetime.now(timezone.utc),
    )


# ── /alerts ───────────────────────────────────────────────────────────────────

@router.get("/alerts", response_model=list[AlertResponse])
def get_alerts(
    request: Request,
    limit: int = 20,
    min_score: float = 0.0,
):
    state = request.app.state
    sql = """
        SELECT id, user_id, alert_type, risk_score, message, created_at, acknowledged
        FROM alerts
        WHERE risk_score >= %s
        ORDER BY created_at DESC
        LIMIT %s
    """
    with state.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql, (min_score, limit))
        rows = cur.fetchall()

    return [
        AlertResponse(
            id=r["id"],
            user_id=r["user_id"],
            alert_type=r["alert_type"],
            risk_score=float(r["risk_score"]),
            message=r["message"],
            created_at=r["created_at"],
            acknowledged=r["acknowledged"],
        )
        for r in rows
    ]