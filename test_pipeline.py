"""
test_pipeline.py
----------------
Quick sanity check — publishes 10 rows, waits 2s, checks DB count.
Run AFTER docker-compose up and consumer is running.
 
    python test_pipeline.py --csv data/events.csv
"""
 
import argparse, json, time, uuid, sys
from pathlib import Path
import pandas as pd
import pika
import psycopg2
 
sys.path.insert(0, str(Path(__file__).parent))
import config
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
 
    df = pd.read_csv(args.csv).head(10)
 
    # ── publish 10 rows ──────────────────────────────────────
    creds   = pika.PlainCredentials(config.RABBIT_USER, config.RABBIT_PASS)
    conn    = pika.BlockingConnection(pika.ConnectionParameters(
        config.RABBIT_HOST, config.RABBIT_PORT, credentials=creds))
    channel = conn.channel()
    channel.exchange_declare(config.EXCHANGE_NAME, config.EXCHANGE_TYPE, durable=True)
 
    ids = []
    for _, row in df.iterrows():
        eid = str(uuid.uuid4())
        ids.append(eid)
        body = row.to_dict()
        body["event_id"] = eid
        for k, v in body.items():
            if hasattr(v, "item"):
                body[k] = v.item()
            elif pd.isna(v) if not isinstance(v, (list,dict)) else False:
                body[k] = None
 
        channel.basic_publish(
            exchange=config.EXCHANGE_NAME,
            routing_key=config.ROUTING_KEY,
            body=json.dumps(body, default=str),
            properties=pika.BasicProperties(delivery_mode=2),
        )
    conn.close()
    print(f"[test] Published 10 rows. Waiting 3s for consumer…")
    time.sleep(3)
 
    # ── check DB ─────────────────────────────────────────────
    pg = psycopg2.connect(config.PG_DSN)
    with pg.cursor() as cur:
        placeholders = ",".join(["%s"] * len(ids))
        cur.execute(f"SELECT COUNT(*) FROM raw_events WHERE event_id IN ({placeholders})", ids)
        count = cur.fetchone()[0]
    pg.close()
 
    print(f"[test] Found {count}/10 rows in DB")
    if count == 10:
        print("[test] ✓ PASS — pipeline is working end-to-end")
    else:
        print("[test] ✗ FAIL — check consumer logs")
        sys.exit(1)
 
if __name__ == "__main__":
    main()