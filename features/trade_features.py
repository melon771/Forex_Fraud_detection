"""
features/trade_features.py
---------------------------
Computes trading, PnL, deposit/withdrawal anomaly signals per user.
"""

import pandas as pd
import numpy as np


def compute_trade_features(df: pd.DataFrame) -> dict:
    """
    Input : all raw_events rows for a single user, sorted by timestamp.
    Output: dict of trade + financial feature values.
    """
    feats = {}

    # ── Event type subsets ────────────────────────────────────
    trade_df    = df[df["event_type"] == "trade"].copy()
    deposit_df  = df[df["event_type"] == "deposit"].copy()
    withdraw_df = df[df["event_type"] == "withdrawal"].copy()

    feats["total_trades"]      = len(trade_df)
    feats["total_deposits"]    = len(deposit_df)
    feats["total_withdrawals"] = len(withdraw_df)

    # ── Financial flows ───────────────────────────────────────
    deposit_total  = deposit_df["deposit_amount"].fillna(0).sum()
    withdraw_total = withdraw_df["withdraw_amount"].fillna(0).sum()

    feats["deposit_total"]  = round(float(deposit_total), 4)
    feats["withdraw_total"] = round(float(withdraw_total), 4)
    feats["net_flow"]       = round(float(deposit_total - withdraw_total), 4)

    # Withdraw/deposit ratio — high value = money flowing out fast
    feats["deposit_withdraw_ratio"] = round(
        float(withdraw_total / deposit_total) if deposit_total > 0 else 0.0, 4
    )

    # Deposit → trade ratio — low value = deposit then withdraw without trading
    if deposit_total > 0 and len(trade_df) > 0:
        traded_volume = trade_df["trade_volume"].fillna(0).sum()
        feats["deposit_to_trade_ratio"] = round(
            float(traded_volume / deposit_total), 4
        )
    else:
        feats["deposit_to_trade_ratio"] = 0.0

    # Large withdrawal after dormancy (14+ day gap before a withdrawal)
    feats["large_withdraw_after_dormancy"] = False
    if len(withdraw_df) > 0 and len(df) > 1:
        for _, wrow in withdraw_df.iterrows():
            prev_events = df[df["timestamp"] < wrow["timestamp"]]
            if len(prev_events) > 0:
                gap_days = (
                    wrow["timestamp"] - prev_events["timestamp"].max()
                ).total_seconds() / 86400
                if gap_days >= 14:
                    feats["large_withdraw_after_dormancy"] = True
                    break

    # High-frequency small deposits (structuring signal)
    # More than 5 deposits in 24h window, each < $500
    feats["high_freq_small_deposits"] = False
    if len(deposit_df) >= 5:
        deposit_df = deposit_df.sort_values("timestamp")
        deposit_df["rolling_count"] = (
            deposit_df.set_index("timestamp")
            .rolling("24h")["deposit_amount"]
            .count()
            .values
        )
        small = deposit_df[deposit_df["deposit_amount"] < 500]
        if small["rolling_count"].max() >= 5:
            feats["high_freq_small_deposits"] = True

    # ── Trading behaviour ─────────────────────────────────────
    if len(trade_df) > 0:
        vols = trade_df["trade_volume"].dropna()
        if len(vols) > 1:
            vol_mean = vols.mean()
            vol_std  = vols.std()
            feats["trade_volume_mean"] = round(float(vol_mean), 4)
            feats["trade_volume_std"]  = round(float(vol_std), 4)

            latest_vol = vols.iloc[-1]
            std = vol_std if vol_std > 0 else 1.0
            feats["trade_volume_zscore"] = round(float((latest_vol - vol_mean) / std), 4)
            feats["trade_volume_spike"]  = bool(latest_vol > 10 * vol_mean)
        else:
            feats["trade_volume_mean"]   = round(float(vols.mean()), 4) if len(vols) else 0.0
            feats["trade_volume_std"]    = 0.0
            feats["trade_volume_zscore"] = 0.0
            feats["trade_volume_spike"]  = False

        # ── PnL ───────────────────────────────────────────────
        pnl = trade_df["pnl"].dropna()
        feats["pnl_total"] = round(float(pnl.sum()), 4)
        if len(pnl) > 1:
            pnl_mean = pnl.mean()
            pnl_std  = pnl.std()
            feats["pnl_volatility"] = round(float(pnl_std), 4)
            std = pnl_std if pnl_std > 0 else 1.0
            feats["pnl_zscore"] = round(
                float((pnl.iloc[-1] - pnl_mean) / std), 4
            )
        else:
            feats["pnl_volatility"] = 0.0
            feats["pnl_zscore"]     = 0.0

        feats["win_rate"] = round(float((pnl > 0).sum() / max(len(pnl), 1)), 4)

        # Consistent profit bursts — win rate > 80% in last 10 trades
        recent_pnl = pnl.iloc[-10:] if len(pnl) >= 10 else pnl
        feats["consistent_profit_bursts"] = bool(
            (recent_pnl > 0).mean() > 0.80 and len(recent_pnl) >= 5
        )

        # ── Instrument diversity ──────────────────────────────
        instruments = trade_df["instrument"].dropna()
        n_unique = instruments.nunique()
        feats["unique_instruments"] = int(n_unique)

        # Concentration: 1.0 = all in one instrument (high risk signal)
        if n_unique > 0 and len(instruments) > 0:
            top_share = instruments.value_counts().iloc[0] / len(instruments)
            feats["instrument_concentration"] = round(float(top_share), 4)
        else:
            feats["instrument_concentration"] = 1.0

        # ── Leverage and margin ───────────────────────────────
        leverage = trade_df["leverage"].dropna()
        margin   = trade_df["margin_used"].dropna()

        feats["avg_leverage"]  = round(float(leverage.mean()), 4) if len(leverage) else 0.0
        feats["avg_margin_used"] = round(float(margin.mean()), 4) if len(margin) else 0.0

        if len(margin) > 1:
            m_mean = margin.mean()
            m_std  = margin.std()
            std = m_std if m_std > 0 else 1.0
            feats["margin_usage_zscore"] = round(
                float((margin.iloc[-1] - m_mean) / std), 4
            )
        else:
            feats["margin_usage_zscore"] = 0.0

    else:
        # No trades at all
        for key in [
            "trade_volume_mean", "trade_volume_std", "trade_volume_zscore",
            "pnl_total", "pnl_volatility", "pnl_zscore", "win_rate",
            "avg_leverage", "avg_margin_used", "margin_usage_zscore",
            "instrument_concentration",
        ]:
            feats[key] = 0.0
        feats["trade_volume_spike"]        = False
        feats["consistent_profit_bursts"]  = False
        feats["unique_instruments"]        = 0

    # ── KYC change before withdrawal (fraud signal) ───────────
    feats["kyc_change_before_withdraw"] = False
    kyc_df = df[df["event_type"] == "kyc_update"]
    if len(kyc_df) > 0 and len(withdraw_df) > 0:
        for _, wrow in withdraw_df.iterrows():
            window_start = wrow["timestamp"] - pd.Timedelta(hours=24)
            kyc_in_window = kyc_df[
                (kyc_df["timestamp"] >= window_start) &
                (kyc_df["timestamp"] <= wrow["timestamp"])
            ]
            if len(kyc_in_window) > 0:
                feats["kyc_change_before_withdraw"] = True
                break

    # ── Balance volatility ────────────────────────────────────
    balance = df["account_balance"].dropna()
    feats["balance_volatility"] = round(float(balance.std()), 4) if len(balance) > 1 else 0.0

    return feats