"""
models/lstm_autoencoder.py
---------------------------
LSTM Autoencoder for sequence-based anomaly detection.
Each user's events are treated as a time series.
High reconstruction error = anomalous behaviour pattern.

Usage:
    python models/lstm_autoencoder.py           # train + score
    python models/lstm_autoencoder.py --epochs 20
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
    format="%(asctime)s [lstm_ae] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAVED_DIR  = Path(__file__).parent / "saved"
MODEL_PATH = SAVED_DIR / "lstm_autoencoder.pt"
SCALER_PATH = SAVED_DIR / "lstm_scaler.pkl"

# Features used as the time-series input
# These are the per-user aggregate features that capture behaviour over time
SEQUENCE_FEATURES = [
    "failed_login_ratio",
    "login_hour_deviation",
    "unique_ips",
    "ip_change_rate",
    "session_duration_zscore",
    "short_session_ratio",
    "avg_time_between_events",
    "deposit_withdraw_ratio",
    "trade_volume_zscore",
    "pnl_volatility",
    "win_rate",
    "instrument_concentration",
    "margin_usage_zscore",
    "bot_like_timing",
    "trade_volume_spike",
    "kyc_change_before_withdraw",
]

INPUT_DIM  = len(SEQUENCE_FEATURES)
HIDDEN_DIM = 32
LATENT_DIM = 16


# ── Model definition ──────────────────────────────────────────────────────────

def build_model():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        log.error("PyTorch not installed. Run: pip install torch")
        sys.exit(1)


class LSTMAutoencoder:
    """Wrapper that handles both PyTorch model and numpy interface."""

    def __init__(self, input_dim, hidden_dim, latent_dim):
        torch, nn = build_model()
        self.torch = torch
        self.nn    = nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.LSTM(input_dim,  hidden_dim, batch_first=True)
                self.hidden_to_latent = nn.Linear(hidden_dim, latent_dim)
                self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
                self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
                self.output_layer = nn.Linear(hidden_dim, input_dim)

            def forward(self, x):
                # x: (batch, seq_len, input_dim)
                _, (h, _) = self.encoder(x)
                h = h.squeeze(0)                        # (batch, hidden)
                z = self.hidden_to_latent(h)            # (batch, latent)
                h_dec = self.latent_to_hidden(z)        # (batch, hidden)
                h_dec = h_dec.unsqueeze(0)              # (1, batch, hidden)
                seq_len = x.shape[1]
                # Repeat latent vector as decoder input
                dec_input = z.unsqueeze(1).repeat(1, seq_len, 1)
                dec_input = self.latent_to_hidden(dec_input)
                out, _ = self.decoder(dec_input, (h_dec, torch.zeros_like(h_dec)))
                out = self.output_layer(out)
                return out                              # (batch, seq_len, input_dim)

        self.model = _Model()

    def fit(self, X: np.ndarray, epochs: int = 30, lr: float = 1e-3):
        """
        X: (n_users, seq_len, input_dim) — each user is one sequence.
        For our case seq_len=1 since we have per-user aggregates not raw events.
        """
        torch = self.torch
        nn    = self.nn

        tensor = torch.FloatTensor(X)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.model(tensor)
            loss   = criterion(output, tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                log.info(f"  Epoch {epoch+1}/{epochs}  loss={loss.item():.6f}")

        self.model.eval()

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            tensor = torch.FloatTensor(X)
            output = self.model(tensor)
            errors = ((output - tensor) ** 2).mean(dim=(1, 2)).numpy()
        return errors

    def save(self):
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        self.torch.save(self.model.state_dict(), MODEL_PATH)
        log.info(f"LSTM model saved → {MODEL_PATH}")

    def load(self):
        self.model.load_state_dict(self.torch.load(MODEL_PATH))
        self.model.eval()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_features(conn) -> pd.DataFrame:
    query = f"SELECT user_id, {', '.join(SEQUENCE_FEATURES)} FROM user_features"
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

    bool_cols = ["bot_like_timing", "trade_volume_spike", "kyc_change_before_withdraw"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(float)

    return df


# ── Write scores to DB ────────────────────────────────────────────────────────

def write_scores(conn, user_ids: list, errors: np.ndarray):
    # Normalise reconstruction error to [0, 1]
    scores = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)

    sql = """
        UPDATE user_features
        SET lstm_reconstruction_error = %s
        WHERE user_id = %s
    """
    rows = [(float(s), uid) for uid, s in zip(user_ids, scores)]
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, sql, rows, page_size=200)
    conn.commit()
    log.info(f"LSTM scores written for {len(rows)} users")
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--top",    type=int, default=10)
    args = parser.parse_args()

    conn = psycopg2.connect(config.PG_DSN)
    log.info("Connected to PostgreSQL")

    df = load_features(conn)
    log.info(f"Loaded {len(df)} users")

    # ── Prepare input ─────────────────────────────────────────
    from sklearn.preprocessing import StandardScaler
    X_raw = df[SEQUENCE_FEATURES].fillna(0).values.astype(np.float32)

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Reshape to (n_users, seq_len=1, input_dim)
    X_seq = X_scaled.reshape(len(df), 1, INPUT_DIM)

    # ── Train ─────────────────────────────────────────────────
    ae = LSTMAutoencoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
    log.info(f"Training LSTM Autoencoder for {args.epochs} epochs …")
    ae.fit(X_seq, epochs=args.epochs)
    ae.save()

    # Save scaler
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    # ── Score ─────────────────────────────────────────────────
    errors = ae.reconstruction_error(X_seq)
    scores = write_scores(conn, df["user_id"].tolist(), errors)

    # ── Show top anomalous users ──────────────────────────────
    df["lstm_score"] = scores
    top = df.nlargest(args.top, "lstm_score")[["user_id", "lstm_score"]]
    log.info(f"\nTop {args.top} most anomalous users (LSTM):")
    for _, row in top.iterrows():
        log.info(f"  {row['user_id']}  score={row['lstm_score']:.4f}")

    conn.close()
    log.info("LSTM Autoencoder complete")


if __name__ == "__main__":
    main()