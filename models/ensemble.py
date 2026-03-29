"""
models/ensemble.py
------------------
Combines Isolation Forest + LSTM scores into a final ensemble risk score.
Also computes top contributing features per user for explainability.

Usage:
    python models/ensemble.py           # score all users
    python models/ensemble.py --top 20  # show top 20 risky users
"""

import argparse
import sys
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ensemble] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAVED_DIR = Path(__file__).parent / "saved"

# Weights for ensemble — IF catches point anomalies, LSTM catches sequence anomalies
IF_WEIGHT   = 0.6
LSTM_WEIGHT = 0.4


# ── Load scores from DB ───────────────────────────────────────────────────────

def load_scores(conn) -> pd.DataFrame:
    query = """
        SELECT user_id,
               isolation_forest_score,
               lstm_reconstruction_error,
               total_trades, unique_ips, pnl_volatility,
               trade_volume_spike, bot_like_timing,
               large_withdraw_after_dormancy, high_freq_small_deposits,
               kyc_change_before_withdraw, consistent_profit_bursts,
               deposit_withdraw_ratio, login_hour_deviation,
               ip_change_rate, session_duration_zscore,
               trade_volume_zscore, margin_usage_zscore
        FROM user_features
        WHERE isolation_forest_score IS NOT NULL
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)

    from decimal import Decimal
    for col in df.columns:
        if len(df) > 0 and df[col].notna().any():
            sample = df[col].dropna().iloc[0]
            if isinstance(sample, Decimal):
                df[col] = df[col].astype(float)

    return df


# ── Ensemble scoring ──────────────────────────────────────────────────────────

def compute_ensemble(df: pd.DataFrame) -> np.ndarray:
    if_scores   = df["isolation_forest_score"].fillna(0).values
    lstm_scores = df["lstm_reconstruction_error"].fillna(0).values

    ensemble = IF_WEIGHT * if_scores + LSTM_WEIGHT * lstm_scores

    # Normalise to [0, 1]
    ensemble = (ensemble - ensemble.min()) / (ensemble.max() - ensemble.min() + 1e-8)
    return ensemble


# ── Feature explanation ───────────────────────────────────────────────────────

EXPLAIN_FEATURES = [
    ("trade_volume_spike",            "Sudden trade volume spike (10x baseline)"),
    ("bot_like_timing",               "Bot-like event timing (sub-second)"),
    ("large_withdraw_after_dormancy", "Large withdrawal after dormancy period"),
    ("high_freq_small_deposits",      "High-frequency small deposits (structuring)"),
    ("kyc_change_before_withdraw",    "KYC change within 24h of withdrawal"),
    ("consistent_profit_bursts",      "Consistently profitable in short bursts"),
    ("ip_change_rate",                "Rapid IP switching across sessions"),
    ("deposit_withdraw_ratio",        "High withdrawal-to-deposit ratio"),
    ("login_hour_deviation",          "Unusual login hour vs personal baseline"),
    ("session_duration_zscore",       "Abnormal session duration"),
    ("trade_volume_zscore",           "Trade volume far from personal baseline"),
    ("margin_usage_zscore",           "Unusual margin usage"),
    ("pnl_volatility",                "High PnL volatility"),
    ("unique_ips",                    "Multiple distinct IP addresses"),
]


def get_top_features(row: pd.Series, top_n: int = 3) -> list[dict]:
    """
    Returns top N contributing features for a user as a list of dicts.
    Boolean flags are weighted heavily; numeric features by z-score magnitude.
    """
    contributions = []

    for feat, description in EXPLAIN_FEATURES:
        if feat not in row.index:
            continue
        val = row[feat]
        if pd.isna(val):
            continue

        # Boolean flags — if True, high contribution
        if feat in [
            "trade_volume_spike", "bot_like_timing",
            "large_withdraw_after_dormancy", "high_freq_small_deposits",
            "kyc_change_before_withdraw", "consistent_profit_bursts",
        ]:
            score = float(bool(val)) * 2.0
        else:
            score = min(abs(float(val)), 10.0)  # cap at 10 to avoid user_id bleed

        if score > 0:
            contributions.append({
                "feature":     feat,
                "description": description,
                "value":       round(float(val), 4),
                "importance":  round(score, 4),
            })

    contributions.sort(key=lambda x: x["importance"], reverse=True)
    return contributions[:top_n]


# ── Write ensemble scores to DB ───────────────────────────────────────────────

def write_ensemble_scores(conn, df: pd.DataFrame, scores: np.ndarray):
    sql = """
        UPDATE user_features
        SET ensemble_risk_score = %s,
            risk_label = CASE
                WHEN %s >= 0.75 THEN 'high'
                WHEN %s >= 0.50 THEN 'medium'
                ELSE 'low'
            END
        WHERE user_id = %s
    """
    rows = [(float(s), float(s), float(s), uid)
            for uid, s in zip(df["user_id"], scores)]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
    conn.commit()
    log.info(f"Ensemble scores written for {len(rows)} users")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15,
                        help="Show top N risky users")
    args = parser.parse_args()

    conn = psycopg2.connect(config.PG_DSN)
    log.info("Connected to PostgreSQL")

    df = load_scores(conn)
    log.info(f"Loaded scores for {len(df)} users")

    if len(df) == 0:
        log.error("No scores found — run isolation_forest.py and lstm_autoencoder.py first")
        return

    # ── Compute ensemble ──────────────────────────────────────
    scores = compute_ensemble(df)
    df["ensemble_score"] = scores

    write_ensemble_scores(conn, df, scores)

    # ── Risk breakdown ────────────────────────────────────────
    high   = (scores >= 0.75).sum()
    medium = ((scores >= 0.50) & (scores < 0.75)).sum()
    low    = (scores < 0.50).sum()
    log.info(f"\nRisk breakdown — HIGH: {high}  MEDIUM: {medium}  LOW: {low}")

    # ── Top risky users with explanations ─────────────────────
    top_users = df.nlargest(args.top, "ensemble_score")
    log.info(f"\nTop {args.top} highest risk users:")
    log.info(f"{'User ID':<15} {'IF Score':<12} {'LSTM Score':<12} {'Ensemble':<10} Top features")
    log.info("-" * 80)

    for _, row in top_users.iterrows():
        top_feats = get_top_features(row)
        feat_str  = " | ".join([f["description"] for f in top_feats])
        log.info(
            f"{row['user_id']:<15} "
            f"{float(row['isolation_forest_score']):<12.4f} "
            f"{float(row['lstm_reconstruction_error']):<12.4f} "
            f"{row['ensemble_score']:<10.4f} "
            f"{feat_str}"
        )

    conn.close()
    log.info("\nEnsemble scoring complete — ready for FastAPI")


if __name__ == "__main__":
    main()