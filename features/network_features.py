"""
features/network_features.py
-----------------------------
Computes IP, device, and geography-based anomaly signals per user.
"""

import pandas as pd
import numpy as np


def compute_network_features(df: pd.DataFrame) -> dict:
    """
    Input : all raw_events rows for a single user, sorted by timestamp.
    Output: dict of network feature values.
    """
    feats = {}

    # ── Unique counts ─────────────────────────────────────────
    feats["unique_ips"]       = df["ip_address"].nunique()
    feats["unique_devices"]   = df["device_id"].nunique()
    feats["unique_countries"] = df["country"].nunique()

    # ── IP change rate (switches per active day) ──────────────
    if len(df) > 1:
        days_active = max(
            (df["timestamp"].max() - df["timestamp"].min()).total_seconds()
            / 86400,
            1,
        )
        # count each consecutive IP change
        ip_switches = (df["ip_address"] != df["ip_address"].shift()).sum() - 1
        feats["ip_change_rate"] = round(max(ip_switches, 0) / days_active, 4)
    else:
        feats["ip_change_rate"] = 0.0

    # ── Largest time gap between consecutive IP switches ──────
    # A very short gap = rapid geo-jump (suspicious)
    ip_change_mask = df["ip_address"] != df["ip_address"].shift()
    if ip_change_mask.sum() > 1:
        switch_times = df.loc[ip_change_mask, "timestamp"]
        gaps_hours = switch_times.diff().dropna().dt.total_seconds() / 3600
        feats["max_ip_switch_gap_hours"] = round(gaps_hours.min(), 4)  # min = fastest switch
    else:
        feats["max_ip_switch_gap_hours"] = 9999.0  # no switches = no concern

    return feats