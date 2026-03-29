"""
consumer.py
-----------
Listens on the forex.raw.store queue, batches messages, and bulk-inserts
into the raw_events PostgreSQL table.

Usage:
    python pipeline/consumer.py
    python pipeline/consumer.py --batch-size 500
"""

import argparse
import json
import signal
import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone

import pika
import psycopg2
import psycopg2.extras

print("FILE IS RUNNING")

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [consumer] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── DB columns we actually insert (in order) ─────────────────────────────────
INSERT_COLUMNS = [
    "event_id", "user_id", "timestamp", "event_type",
    "account_balance", "account_age_days", "kyc_status",
    "ip_address", "device_id", "country",
    "login_success", "failed_login_attempts", "login_hour",
    "trade_volume", "deposit_amount", "withdraw_amount",
    "balance_before", "balance_after", "pnl",
    "instrument", "trade_direction", "leverage", "margin_used",
    "session_duration", "time_since_last_event",
    "rabbitmq_queue",
]

INSERT_SQL = f"""
    INSERT INTO raw_events ({', '.join(INSERT_COLUMNS)})
    VALUES ({', '.join(['%s'] * len(INSERT_COLUMNS))})
    ON CONFLICT (event_id) DO NOTHING
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def connect_db() -> psycopg2.extensions.connection:
    conn = psycopg2.connect(config.PG_DSN)
    conn.autocommit = False
    log.info("PostgreSQL connected")
    return conn


def connect_rabbit() -> tuple[pika.BlockingConnection,
                               pika.adapters.blocking_connection.BlockingChannel]:
    creds  = pika.PlainCredentials(config.RABBIT_USER, config.RABBIT_PASS)
    params = pika.ConnectionParameters(
        host=config.RABBIT_HOST,
        port=config.RABBIT_PORT,
        credentials=creds,
        heartbeat=600,
        blocked_connection_timeout=300,
    )
    conn    = pika.BlockingConnection(params)
    channel = conn.channel()

    # declare exchange + queue + binding (idempotent)
    channel.exchange_declare(
        exchange=config.EXCHANGE_NAME,
        exchange_type=config.EXCHANGE_TYPE,
        durable=True,
    )
    channel.queue_declare(
        queue=config.QUEUE_NAME,
        durable=True,           # survives broker restart
        arguments={"x-max-priority": 0},
    )
    channel.queue_bind(
        queue=config.QUEUE_NAME,
        exchange=config.EXCHANGE_NAME,
        routing_key=config.ROUTING_KEY,
    )
    channel.basic_qos(prefetch_count=config.CONSUMER_PREFETCH)
    log.info(f"RabbitMQ connected — queue '{config.QUEUE_NAME}'")
    return conn, channel


def row_from_message(body: bytes) -> dict:
    data = json.loads(body)
    row = {col: data.get(col) for col in INSERT_COLUMNS
           if col != "rabbitmq_queue"} | {"rabbitmq_queue": config.QUEUE_NAME}
    
    # Cast integer 0/1 from CSV → proper Python bool for Postgres BOOLEAN column
    if row.get("login_success") is not None:
        row["login_success"] = bool(int(row["login_success"]))
    
    return row


def flush_batch(pg_conn, batch: list[dict]) -> int:
    """Bulk-insert a batch of rows. Returns number of rows inserted."""
    if not batch:
        return 0
    rows = [tuple(row[col] for col in INSERT_COLUMNS) for row in batch]
    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, INSERT_SQL, rows, page_size=200)
    pg_conn.commit()
    return len(rows)


# ── consumer state ────────────────────────────────────────────────────────────

class Consumer:
    def __init__(self, batch_size: int):
        self.batch_size  = batch_size
        self._buffer     : list[dict]  = []
        self._delivery_tags: list[int] = []
        self._total_stored = 0
        self._total_dupes  = 0
        self._pg_conn    = None
        self._rb_conn    = None
        self._channel    = None

    def start(self):
        print("STARTED CONSUMER")   # add this
        self._pg_conn             = connect_db()
        self._rb_conn, self._channel = connect_rabbit()

        # graceful shutdown on Ctrl-C / SIGTERM
        signal.signal(signal.SIGINT,  self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        self._channel.basic_consume(
            queue=config.QUEUE_NAME,
            on_message_callback=self._on_message,
        )
        log.info(f"Waiting for messages (batch={self.batch_size}) …  Ctrl-C to stop")
        self._channel.start_consuming()

    def _on_message(self, channel, method, properties, body):
        try:
            row = row_from_message(body)
            self._buffer.append(row)
            self._delivery_tags.append(method.delivery_tag)
        except Exception as exc:
            log.warning(f"Parse error, NACKing message: {exc}")
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return

        if len(self._buffer) >= self.batch_size:
            self._flush()

    def _flush(self):
        if not self._buffer:
            return
        try:
            stored = flush_batch(self._pg_conn, self._buffer)
            dupes  = len(self._buffer) - stored
            # ACK all messages in this batch
            self._channel.basic_ack(
                delivery_tag=self._delivery_tags[-1],
                multiple=True,
            )
            self._total_stored += stored
            self._total_dupes  += dupes
            log.info(
                f"Flushed {stored} rows  "
                f"(+{dupes} dupes skipped)  "
                f"total={self._total_stored:,}"
            )
        except Exception as exc:
            log.error(f"DB flush failed: {exc} — NACKing batch")
            self._pg_conn.rollback()
            for tag in self._delivery_tags:
                self._channel.basic_nack(delivery_tag=tag, requeue=True)
        finally:
            self._buffer.clear()
            self._delivery_tags.clear()

    def _shutdown(self, *_):
        log.info("Shutdown signal — flushing remaining buffer …")
        self._flush()
        self._channel.stop_consuming()
        if self._rb_conn and not self._rb_conn.is_closed:
            self._rb_conn.close()
        if self._pg_conn and not self._pg_conn.closed:
            self._pg_conn.close()
        log.info(f"Graceful shutdown complete. "
                 f"Stored: {self._total_stored:,}  Dupes: {self._total_dupes:,}")
        sys.exit(0)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ForexGuard event consumer")
    parser.add_argument("--batch-size", type=int,
                        default=config.DB_BATCH_SIZE,
                        help="Rows to buffer before DB insert")
    args = parser.parse_args()

    consumer = Consumer(batch_size=args.batch_size)
    consumer.start()


if __name__ == "__main__":
    main()