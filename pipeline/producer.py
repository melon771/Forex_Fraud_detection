"""
producer.py
-----------
Reads the synthetic CSV dataset and publishes every row as a JSON
message to RabbitMQ on the forex.events exchange.

Usage:
    python pipeline/producer.py --csv data/events.csv
    python pipeline/producer.py --csv data/events.csv --delay 5
                                                        ↑ ms between messages
                                                          (simulate real-time)
"""

import argparse
import json
import time
import uuid
import sys
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pika
from tqdm import tqdm

# allow `import config` from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── helpers ──────────────────────────────────────────────────────────────────

def build_connection() -> pika.BlockingConnection:
    creds = pika.PlainCredentials(config.RABBIT_USER, config.RABBIT_PASS)
    params = pika.ConnectionParameters(
        host=config.RABBIT_HOST,
        port=config.RABBIT_PORT,
        credentials=creds,
        heartbeat=600,
        blocked_connection_timeout=300,
    )
    return pika.BlockingConnection(params)


def setup_exchange(channel: pika.adapters.blocking_connection.BlockingChannel):
    """Declare exchange (idempotent — safe to call every run)."""
    channel.exchange_declare(
        exchange=config.EXCHANGE_NAME,
        exchange_type=config.EXCHANGE_TYPE,
        durable=True,
    )


def sanitize_row(row: dict) -> dict:
    """
    Convert pandas NaN / Timestamp / numpy types to JSON-serialisable
    Python primitives.
    """
    clean = {}
    for k, v in row.items():
        if pd.isna(v) if not isinstance(v, (list, dict)) else False:
            clean[k] = None
        elif isinstance(v, pd.Timestamp):
            clean[k] = v.isoformat()
        elif hasattr(v, "item"):          # numpy scalar → python scalar
            clean[k] = v.item()
        else:
            clean[k] = v
    return clean


def publish_batch(
    channel: pika.adapters.blocking_connection.BlockingChannel,
    rows: list[dict],
    delay_ms: int,
):
    for row in rows:
        row["event_id"] = str(uuid.uuid4())   # dedup key for consumer
        row["published_at"] = datetime.now(timezone.utc).isoformat()

        body = json.dumps(sanitize_row(row))

        channel.basic_publish(
            exchange=config.EXCHANGE_NAME,
            routing_key=config.ROUTING_KEY,
            body=body,
            properties=pika.BasicProperties(
                delivery_mode=2,              # persistent (survives broker restart)
                content_type="application/json",
                message_id=row["event_id"],
            ),
        )

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ForexGuard event producer")
    parser.add_argument("--csv",   required=True, help="Path to events CSV")
    parser.add_argument("--delay", type=int, default=config.PUBLISH_DELAY_MS,
                        help="Delay between messages in ms (0 = full speed)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only publish first N rows (for testing)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        sys.exit(1)

    # ── load CSV ──────────────────────────────────────────────
    print(f"[producer] Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    if args.limit:
        df = df.head(args.limit)

    total = len(df)
    print(f"[producer] {total:,} rows loaded")

    # ── connect ───────────────────────────────────────────────
    print(f"[producer] Connecting to RabbitMQ at "
          f"{config.RABBIT_HOST}:{config.RABBIT_PORT} …")
    connection = build_connection()
    channel    = connection.channel()
    channel.confirm_delivery()          # publisher confirms — no silent drops
    setup_exchange(channel)
    print(f"[producer] Exchange '{config.EXCHANGE_NAME}' ready")

    # ── publish ───────────────────────────────────────────────
    published = 0
    errors    = 0
    batch_size = config.PUBLISH_BATCH_SIZE

    with tqdm(total=total, unit="msg", desc="Publishing") as pbar:
        for start in range(0, total, batch_size):
            chunk = df.iloc[start : start + batch_size]
            rows  = chunk.to_dict(orient="records")
            try:
                publish_batch(channel, rows, args.delay)
                published += len(rows)
            except Exception as exc:
                errors += len(rows)
                print(f"\n[producer] Batch error at row {start}: {exc}")
            pbar.update(len(rows))

    connection.close()
    print(f"\n[producer] Done — published {published:,} | errors {errors}")


if __name__ == "__main__":
    main()