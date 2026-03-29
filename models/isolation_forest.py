"""
models/isolation_forest.py
---------------------------
Trains an Isolation Forest on the user_features table.
Writes anomaly scores back to user_features.isolation_forest_score.
Also saves the trained model + feature list to models/saved/.

Usage:
    python models/isolation_forest.py           # train + score all users
    python models/isolation_forest.py --explain  # show top SHAP features
"""

import argparse
import sys
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [isolation_forest] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature columns used for training ────────────────────────────────────────
# Only numeric columns — booleans converted to int, categoricals excluded
FEATURE_COLS = [
    # login / network
    "failed_login_ratio",
    "login_hour_std",
    "login_hour_deviation",
    "unique_ips",
    "unique_devices",
    "unique_countries",
    "ip_change_rate",
    "max_ip_switch_gap_hours",
    # session
    "session_duration_mean",
    "session_duration_zscore",
    "short_session_ratio",
    "avg_events_per_session",
    "avg_time_between_events",
    "min_time_between_events",
    "time_delta_std",
    # financial
    "deposit_withdraw_ratio",
    "deposit_to_trade_ratio",
    "net_flow",
    # trading
    "trade_volume_mean",
    "trade_volume_zscore",
    "pnl_volatility",
    "pnl_zscore",
    "win_rate",
    "instrument_concentration",
    "avg_leverage",
    "margin_usage_zscore",
    # boolean signals → cast to int (0/1)
    "large_withdraw_after_dormancy",
    "high_freq_small_deposits",
    "trade_volume_spike",
    "consistent_profit_bursts",
    "bot_like_timing",
    "kyc_change_before_withdraw",
    # account
    "account_age_days",
    "balance_volatility",
]

SAVED_DIR   = Path(__file__).parent / "saved"
MODEL_PATH  = SAVED_DIR / "isolation_forest.pkl"
SCALER_PATH = SAVED_DIR / "if_scaler.pkl"
FEATURES_PATH = SAVED_DIR / "if_features.pkl"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(conn) -> pd.DataFrame:
    query = f"SELECT user_id, {', '.join(FEATURE_COLS)} FROM user_features"
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        rows = cur.fetchall()
    df = pd.DataFrame(rows)

    # Convert Decimal → float
    from decimal import Decimal
    for col in df.columns:
        if len(df) > 0 and isinstance(df[col].dropna().iloc[0] if df[col].notna().any() else 0, Decimal):
            df[col] = df[col].astype(float)

    # Cast boolean columns to int
    bool_cols = [
        "large_withdraw_after_dormancy", "high_freq_small_deposits",
        "trade_volume_spike", "consistent_profit_bursts",
        "bot_like_timing", "kyc_change_before_withdraw",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(int)

    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train(df: pd.DataFrame):
    log.info(f"Training on {len(df)} users, {len(FEATURE_COLS)} features …")

    X = df[FEATURE_COLS].copy()

    # Fill any remaining NaNs with column median
    X = X.fillna(X.median())

    # Scale features — IF works better on normalised data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination=0.05 means we expect ~5% anomalous users
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function returns higher = more normal
    # We invert and normalise to [0, 1] where 1 = most anomalous
    raw_scores = model.decision_function(X_scaled)
    scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())

    log.info(f"Score stats — min: {scores.min():.4f}  max: {scores.max():.4f}  mean: {scores.mean():.4f}")
    log.info(f"Flagged as anomalous (score > 0.7): {(scores > 0.7).sum()} users")

    return model, scaler, scores


# ── SHAP explainability ───────────────────────────────────────────────────────

def compute_shap(model, scaler, df: pd.DataFrame, top_n: int = 5):
    """
    Isolation Forest doesn't support SHAP natively.
    We use feature importance via mean depth — higher = more anomalous contribution.
    Returns a dict of {user_id: [(feature, importance), ...]}
    """
    try:
        import shap
        X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
        X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        # shap_values shape: (n_users, n_features)
        explanations = {}
        for i, uid in enumerate(df["user_id"]):
            importances = list(zip(FEATURE_COLS, np.abs(shap_values[i])))
            importances.sort(key=lambda x: x[1], reverse=True)
            explanations[uid] = importances[:top_n]
        log.info("SHAP explanations computed successfully")
        return explanations
    except ImportError:
        log.warning("shap not installed — run: pip install shap")
        return {}
    except Exception as exc:
        log.warning(f"SHAP failed: {exc} — falling back to feature deviation scoring")
        return compute_feature_deviation(model, scaler, df, top_n)


def compute_feature_deviation(model, scaler, df: pd.DataFrame, top_n: int = 5):
    """
    Fallback explainability: for each user, find which features deviate
    most from the population mean (in std units).
    """
    X = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLS)

    explanations = {}
    for i, uid in enumerate(df["user_id"]):
        row = X_scaled.iloc[i]
        deviations = [(col, abs(float(row[col]))) for col in FEATURE_COLS]
        deviations.sort(key=lambda x: x[1], reverse=True)
        explanations[uid] = deviations[:top_n]
    return explanations


# ── Write scores back to DB ───────────────────────────────────────────────────

def write_scores(conn, user_ids: list, scores: np.ndarray):
    sql = """
        UPDATE user_features
        SET isolation_forest_score = %s,
            risk_label = CASE
                WHEN %s >= 0.75 THEN 'high'
                WHEN %s >= 0.50 THEN 'medium'
                ELSE 'low'
            END
        WHERE user_id = %s
    """
    rows = [(float(s), float(s), float(s), uid) for uid, s in zip(user_ids, scores)]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
    conn.commit()
    log.info(f"Scores written for {len(rows)} users")


# ── Save / load ───────────────────────────────────────────────────────────────

def save_artifacts(model, scaler):
    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH,    "wb") as f: pickle.dump(model,  f)
    with open(SCALER_PATH,   "wb") as f: pickle.dump(scaler, f)
    with open(FEATURES_PATH, "wb") as f: pickle.dump(FEATURE_COLS, f)
    log.info(f"Model saved → {MODEL_PATH}")


def load_artifacts():
    with open(MODEL_PATH,    "rb") as f: model  = pickle.load(f)
    with open(SCALER_PATH,   "rb") as f: scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: feats  = pickle.load(f)
    return model, scaler, feats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explain", action="store_true",
                        help="Print top anomalous users with feature explanations")
    parser.add_argument("--top", type=int, default=10,
                        help="How many top anomalous users to show")
    args = parser.parse_args()

    conn = psycopg2.connect(config.PG_DSN)
    log.info("Connected to PostgreSQL")

    # ── Load features ─────────────────────────────────────────
    df = load_features(conn)
    log.info(f"Loaded {len(df)} users from user_features")

    if len(df) == 0:
        log.error("No users found — run feature_engineer.py first")
        return

    # ── Train ─────────────────────────────────────────────────
    model, scaler, scores = train(df)

    # ── Save ──────────────────────────────────────────────────
    save_artifacts(model, scaler)

    # ── Write scores to DB ────────────────────────────────────
    write_scores(conn, df["user_id"].tolist(), scores)

    # ── Show top anomalous users ──────────────────────────────
    df["if_score"] = scores
    top_users = df.nlargest(args.top, "if_score")[["user_id", "if_score"]]
    log.info(f"\nTop {args.top} most anomalous users:")
    for _, row in top_users.iterrows():
        log.info(f"  {row['user_id']}  score={row['if_score']:.4f}")

    if args.explain:
        log.info("\nComputing feature explanations …")
        explanations = compute_shap(model, scaler, df)
        top_ids = top_users["user_id"].tolist()
        for uid in top_ids:
            if uid in explanations:
                feats_str = ", ".join([f"{f}={v:.3f}" for f, v in explanations[uid]])
                log.info(f"  {uid}: {feats_str}")

    conn.close()
    log.info("Isolation Forest complete")


if __name__ == "__main__":
    main()