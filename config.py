import os
 
# ── RabbitMQ ─────────────────────────────────────────────────
RABBIT_HOST     = os.getenv("RABBIT_HOST",     "127.0.0.1")
RABBIT_PORT     = int(os.getenv("RABBIT_PORT", "5672"))
RABBIT_USER     = os.getenv("RABBIT_USER",     "guest")
RABBIT_PASS     = os.getenv("RABBIT_PASS",     "guest")
 
EXCHANGE_NAME   = "forex.events"
EXCHANGE_TYPE   = "topic"           # topic lets us route by event_type later
ROUTING_KEY     = "forex.raw"       # producer publishes here
QUEUE_NAME      = "forex.raw.store" # consumer binds to this
 
# ── PostgreSQL ────────────────────────────────────────────────
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT     = int(os.getenv("PG_PORT", "5432"))
PG_DB       = os.getenv("PG_DB",       "forexguard")
PG_USER     = os.getenv("PG_USER",     "forexguard")
PG_PASS     = os.getenv("PG_PASS",     "forexguard")
 
PG_DSN = (
    f"host={PG_HOST} port={PG_PORT} "
    f"dbname={PG_DB} user={PG_USER} password={PG_PASS}"
)
 
# ── Producer tuning ───────────────────────────────────────────
PUBLISH_BATCH_SIZE  = 500   # rows read from CSV at a time
PUBLISH_DELAY_MS    = 0     # set to e.g. 10 to simulate real-time throttle
 
# ── Consumer tuning ──────────────────────────────────────────
CONSUMER_PREFETCH   = 100   # messages fetched before ACK
DB_BATCH_SIZE       = 200   # rows inserted per executemany
 