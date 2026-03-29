-- ============================================================
--  ForexGuard — user_features table
--  One row per user, updated every time feature engineering runs.
--  Isolation Forest + LSTM train directly on this table.
-- ============================================================

CREATE TABLE IF NOT EXISTS user_features (
    user_id                     VARCHAR(64) PRIMARY KEY,
    computed_at                 TIMESTAMPTZ DEFAULT NOW(),

    -- ── Event counts ──────────────────────────────────────────
    total_events                INTEGER,
    total_logins                INTEGER,
    total_trades                INTEGER,
    total_deposits              INTEGER,
    total_withdrawals           INTEGER,

    -- ── Login behaviour ───────────────────────────────────────
    failed_login_ratio          NUMERIC(8,4),   -- failed / total logins
    login_hour_mean             NUMERIC(8,4),   -- avg hour of day
    login_hour_std              NUMERIC(8,4),   -- std of login hour
    login_hour_deviation        NUMERIC(8,4),   -- z-score of latest login hour
    unique_ips                  INTEGER,        -- distinct IPs used
    unique_devices              INTEGER,        -- distinct device IDs
    unique_countries            INTEGER,        -- distinct countries
    ip_change_rate              NUMERIC(8,4),   -- ip switches per day
    max_ip_switch_gap_hours     NUMERIC(8,4),   -- largest geo-jump time gap

    -- ── Session behaviour ─────────────────────────────────────
    session_duration_mean       NUMERIC(12,4),
    session_duration_std        NUMERIC(12,4),
    session_duration_zscore     NUMERIC(8,4),   -- latest vs historical
    short_session_ratio         NUMERIC(8,4),   -- sessions < 60s / total
    avg_events_per_session      NUMERIC(8,4),

    -- ── Financial behaviour ───────────────────────────────────
    deposit_total               NUMERIC(18,4),
    withdraw_total              NUMERIC(18,4),
    deposit_withdraw_ratio      NUMERIC(8,4),   -- withdraw / deposit (high = suspicious)
    large_withdraw_after_dormancy BOOLEAN,      -- withdraw after 14+ day gap
    deposit_to_trade_ratio      NUMERIC(8,4),   -- low = deposit→withdraw abuse
    high_freq_small_deposits    BOOLEAN,        -- structuring signal
    net_flow                    NUMERIC(18,4),  -- deposit_total - withdraw_total

    -- ── Trading behaviour ─────────────────────────────────────
    trade_volume_mean           NUMERIC(18,4),
    trade_volume_std            NUMERIC(18,4),
    trade_volume_zscore         NUMERIC(8,4),   -- latest vs historical baseline
    trade_volume_spike          BOOLEAN,        -- latest > 10x mean
    pnl_total                   NUMERIC(18,4),
    pnl_volatility              NUMERIC(18,4),  -- std of per-trade PnL
    pnl_zscore                  NUMERIC(8,4),
    win_rate                    NUMERIC(8,4),   -- profitable trades / total
    consistent_profit_bursts    BOOLEAN,        -- high win rate in short window
    unique_instruments          INTEGER,
    instrument_concentration    NUMERIC(8,4),   -- 1 = all in one instrument
    avg_leverage                NUMERIC(8,4),
    avg_margin_used             NUMERIC(8,4),
    margin_usage_zscore         NUMERIC(8,4),

    -- ── Inter-event timing ────────────────────────────────────
    avg_time_between_events     NUMERIC(12,4),  -- seconds
    min_time_between_events     NUMERIC(12,4),  -- very low = bot signal
    time_delta_std              NUMERIC(12,4),
    bot_like_timing             BOOLEAN,        -- min delta < 1 second

    -- ── Composite risk signals ────────────────────────────────
    kyc_change_before_withdraw  BOOLEAN,        -- KYC update within 24h of withdraw
    account_age_days            INTEGER,
    balance_volatility          NUMERIC(18,4),  -- std of account balance

    -- ── Final anomaly scores (filled by models) ───────────────
    isolation_forest_score      NUMERIC(8,6),
    lstm_reconstruction_error   NUMERIC(12,6),
    ensemble_risk_score         NUMERIC(8,6),
    risk_label                  VARCHAR(16)     -- low / medium / high
);

CREATE INDEX IF NOT EXISTS idx_user_features_risk
    ON user_features (ensemble_risk_score DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS idx_user_features_computed
    ON user_features (computed_at DESC);