"""
features/session_features.py
-----------------------------
Computes session duration, login timing, and bot-detection signals.
"""

import pandas as pd
import numpy as np


def compute_session_features(df: pd.DataFrame) -> dict:
    """
    Input : all raw_events rows for a single user, sorted by timestamp.
    Output: dict of session + login feature values.
    """
    feats = {}

    # ── Login behaviour ───────────────────────────────────────
    login_df = df[df["event_type"] == "login"].copy()
    feats["total_logins"] = len(login_df)

    if len(login_df) > 0:
        total_attempts = login_df["failed_login_attempts"].fillna(0).sum() + len(login_df)
        feats["failed_login_ratio"] = round(
            login_df["failed_login_attempts"].fillna(0).sum() / max(total_attempts, 1), 4
        )

        hours = login_df["login_hour"].dropna()
        if len(hours) > 0:
            feats["login_hour_mean"] = round(hours.mean(), 4)
            feats["login_hour_std"]  = round(hours.std() if len(hours) > 1 else 0.0, 4)

            # z-score of the LATEST login hour vs user's own history
            latest_hour = hours.iloc[-1]
            std = feats["login_hour_std"] if feats["login_hour_std"] > 0 else 1.0
            feats["login_hour_deviation"] = round(
                (latest_hour - feats["login_hour_mean"]) / std, 4
            )
        else:
            feats["login_hour_mean"]      = 12.0
            feats["login_hour_std"]       = 0.0
            feats["login_hour_deviation"] = 0.0
    else:
        feats["failed_login_ratio"]   = 0.0
        feats["login_hour_mean"]      = 12.0
        feats["login_hour_std"]       = 0.0
        feats["login_hour_deviation"] = 0.0

    # ── Session duration ──────────────────────────────────────
    durations = df["session_duration"].dropna()
    if len(durations) > 1:
        dur_mean = durations.mean()
        dur_std  = durations.std()
        feats["session_duration_mean"] = round(dur_mean, 4)
        feats["session_duration_std"]  = round(dur_std, 4)

        latest_dur = durations.iloc[-1]
        std = dur_std if dur_std > 0 else 1.0
        feats["session_duration_zscore"] = round((latest_dur - dur_mean) / std, 4)

        # Short session = < 60 seconds (possible bot or scripted access)
        feats["short_session_ratio"] = round(
            (durations < 60).sum() / len(durations), 4
        )
    else:
        feats["session_duration_mean"]   = float(durations.mean()) if len(durations) else 0.0
        feats["session_duration_std"]    = 0.0
        feats["session_duration_zscore"] = 0.0
        feats["short_session_ratio"]     = 0.0

    # ── Events per session (proxy for automation) ─────────────
    # Count distinct session groups via login events as session starts
    n_sessions = max(feats["total_logins"], 1)
    feats["avg_events_per_session"] = round(len(df) / n_sessions, 4)

    # ── Inter-event timing (bot detection) ────────────────────
    if len(df) > 1:
        time_deltas = df["timestamp"].diff().dropna().dt.total_seconds()
        feats["avg_time_between_events"] = round(time_deltas.mean(), 4)
        feats["min_time_between_events"] = round(time_deltas.min(), 4)
        feats["time_delta_std"]          = round(time_deltas.std(), 4)

        # Bot signal: events firing faster than 1 second apart
        feats["bot_like_timing"] = bool(time_deltas.min() < 1.0)
    else:
        feats["avg_time_between_events"] = 0.0
        feats["min_time_between_events"] = 0.0
        feats["time_delta_std"]          = 0.0
        feats["bot_like_timing"]         = False

    return feats