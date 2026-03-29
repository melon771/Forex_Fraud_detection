"""
alerts/publisher.py
-------------------
Formats human-readable alert messages and publishes them
to the forex.alerts RabbitMQ queue.
Also writes alerts to the PostgreSQL alerts table.
"""

import json
import logging
from datetime import datetime, timezone

import pika
import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

ALERTS_QUEUE    = "forex.alerts"
ALERTS_EXCHANGE = "forex.events"
ALERTS_ROUTING  = "forex.alert"


def format_alert_message(user_id: str, risk_score: float,
                          top_features: list[dict]) -> str:
    """Produce a human-readable alert string for the compliance team."""
    label = "HIGH RISK" if risk_score >= 0.75 else "MEDIUM RISK"
    feat_lines = []
    for f in top_features[:3]:
        feat_lines.append(f"  - {f['description']} (value={f['value']})")
    feats_str = "\n".join(feat_lines) if feat_lines else "  - No specific features flagged"

    return (
        f"[ForexGuard Alert] {label} — User: {user_id}\n"
        f"Ensemble Risk Score: {risk_score:.4f}\n"
        f"Top contributing factors:\n{feats_str}\n"
        f"Timestamp: {datetime.now(timezone.utc).isoformat()}"
    )


def publish_alert(
    rabbit_channel,
    pg_conn,
    user_id: str,
    risk_score: float,
    top_features: list[dict],
    alert_type: str = "anomaly_detected",
):
    """
    1. Format the alert message
    2. Write to PostgreSQL alerts table
    3. Publish to RabbitMQ forex.alerts queue
    """
    message = format_alert_message(user_id, risk_score, top_features)

    # ── Write to DB ───────────────────────────────────────────
    try:
        sql = """
            INSERT INTO alerts (user_id, alert_type, risk_score, top_features, message)
            VALUES (%s, %s, %s, %s, %s)
        """
        with pg_conn.cursor() as cur:
            cur.execute(sql, (
                user_id,
                alert_type,
                risk_score,
                json.dumps(top_features),
                message,
            ))
        pg_conn.commit()
        log.info(f"Alert saved to DB for user {user_id}")
    except Exception as exc:
        log.error(f"Failed to save alert to DB: {exc}")
        pg_conn.rollback()

    # ── Publish to RabbitMQ ───────────────────────────────────
    try:
        payload = json.dumps({
            "user_id":      user_id,
            "alert_type":   alert_type,
            "risk_score":   risk_score,
            "top_features": top_features,
            "message":      message,
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        })

        rabbit_channel.basic_publish(
            exchange=ALERTS_EXCHANGE,
            routing_key=ALERTS_ROUTING,
            body=payload,
            properties=pika.BasicProperties(
                delivery_mode=2,
                content_type="application/json",
            ),
        )
        log.info(f"Alert published to RabbitMQ for user {user_id} (score={risk_score:.4f})")
    except Exception as exc:
        log.error(f"Failed to publish alert to RabbitMQ: {exc}")