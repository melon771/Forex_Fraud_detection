"""
pipeline/feature_engineer.py
-----------------------------
Reads raw_events from Postgres, computes per-user feature vectors
using all feature modules, and writes to the user_features table.

Usage:
    python pipeline/feature_engineer.py           # process all users
    python pipeline/feature_engineer.py --limit 100   # first 100 users (testing)
    python pipeline/feature_engineer.py --user U_001  # single user
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from features.network_features import compute_network_features
from features.session_features  import compute_session_features
from features.trade_features    import compute_trade_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [feature_eng] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── SQL ───────────────────────────────────────────────────────────────────────

FETCH_USERS_SQL = "SELECT DISTINCT user_id FROM raw_events ORDER BY user_id"

FETCH_USER_SQL  = """
    SELECT *
    FROM raw_events
    WHERE user_id = %s
    ORDER BY timestamp ASC
"""

UPSERT_SQL = """
    INSERT INTO user_features ({cols})
    VALUES ({placeholders})
    ON CONFLICT (user_id) DO UPDATE SET
        {updates},
        computed_at = EXCLUDED.computed_at
"""


# ── Feature vector builder ────────────────────────────────────────────────────

def build_feature_vector(user_id: str, df: pd.DataFrame) -> dict:
    """
    Combines all feature modules into one flat dict for a single user.
    df must be sorted by timestamp ascending.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    feats = {"user_id": user_id, "computed_at": datetime.now(timezone.utc)}

    # ── Base counts ───────────────────────────────────────────
    feats["total_events"]     = len(df)
    feats["account_age_days"] = int(df["account_age_days"].dropna().iloc[-1]) \
                                if len(df["account_age_days"].dropna()) > 0 else 0

    # ── Sub-module features ───────────────────────────────────
    feats.update(compute_network_features(df))
    feats.update(compute_session_features(df))
    feats.update(compute_trade_features(df))

    # ── Placeholders for model scores (filled later by models) ─
    feats.setdefault("isolation_forest_score",   None)
    feats.setdefault("lstm_reconstruction_error", None)
    feats.setdefault("ensemble_risk_score",       None)
    feats.setdefault("risk_label",                None)

    # ── Convert numpy bools/ints to Python native types ───────
    for k, v in feats.items():
        if isinstance(v, (np.bool_,)):
            feats[k] = bool(v)
        elif isinstance(v, (np.integer,)):
            feats[k] = int(v)
        elif isinstance(v, (np.floating,)):
            feats[k] = float(v)
        elif isinstance(v, float) and np.isnan(v):
            feats[k] = None

    return feats


# ── DB helpers ────────────────────────────────────────────────────────────────

def upsert_features(conn, feature_rows: list[dict]):
    if not feature_rows:
        return
    cols = list(feature_rows[0].keys())
    ph   = ", ".join(["%s"] * len(cols))
    # Exclude user_id AND computed_at from the UPDATE clause
    ups  = ", ".join([f"{c} = EXCLUDED.{c}" for c in cols
                      if c not in ("user_id", "computed_at")])
    sql  = UPSERT_SQL.format(
        cols=", ".join(cols),
        placeholders=ph,
        updates=ups,
    )
    rows = [tuple(r[c] for c in cols) for r in feature_rows]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=100)
    conn.commit()

def fetch_user_df(conn, user_id: str) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(FETCH_USER_SQL, (user_id,))
        rows = cur.fetchall()
    df = pd.DataFrame(rows)
    # Convert Decimal → float cleanly
    from decimal import Decimal
    for col in df.columns:
        if len(df) > 0 and isinstance(df[col].iloc[0], Decimal):
            df[col] = df[col].astype(float)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ForexGuard feature engineering")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N users (for testing)")
    parser.add_argument("--user",  type=str, default=None,
                        help="Process a single user_id")
    parser.add_argument("--batch", type=int, default=50,
                        help="Users to process before writing to DB")
    args = parser.parse_args()

    conn = psycopg2.connect(config.PG_DSN)
    log.info("Connected to PostgreSQL")

    # ── Get user list ─────────────────────────────────────────
    if args.user:
        user_ids = [args.user]
    else:
        with conn.cursor() as cur:
            cur.execute(FETCH_USERS_SQL)
            user_ids = [r[0] for r in cur.fetchall()]
        if args.limit:
            user_ids = user_ids[: args.limit]

    total = len(user_ids)
    log.info(f"Processing {total} users …")

    processed = 0
    errors    = 0
    batch_buf = []

    for i, uid in enumerate(user_ids, 1):
        try:
            df = fetch_user_df(conn, uid)
            if df.empty:
                continue
            fv = build_feature_vector(uid, df)
            batch_buf.append(fv)
        except Exception as exc:
            log.warning(f"Error on user {uid}: {exc}")
            errors += 1
            continue

        # Flush batch to DB
        if len(batch_buf) >= args.batch:
            upsert_features(conn, batch_buf)
            processed += len(batch_buf)
            log.info(f"  [{i}/{total}] Written {processed} users so far …")
            batch_buf.clear()

    # Flush remainder
    if batch_buf:
        upsert_features(conn, batch_buf)
        processed += len(batch_buf)

    conn.close()
    log.info(f"Done — {processed} users processed, {errors} errors")
    log.info("Run: SELECT user_id, ensemble_risk_score FROM user_features LIMIT 10;")


if __name__ == "__main__":
    main()