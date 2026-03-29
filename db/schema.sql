-- ============================================================
--  ForexGuard  —  raw events table
--  Mirrors the CSV schema. Consumer writes every event here.
-- ============================================================
 
CREATE TABLE IF NOT EXISTS raw_events (
    -- identity
    id                  BIGSERIAL PRIMARY KEY,
    event_id            UUID UNIQUE,          -- dedup key set by producer
    user_id             VARCHAR(64)  NOT NULL,
    timestamp           TIMESTAMPTZ  NOT NULL,
    event_type          VARCHAR(64)  NOT NULL,
 
    -- account
    account_balance     NUMERIC(18,4),
    account_age_days    INTEGER,
    kyc_status          VARCHAR(32),
 
    -- network
    ip_address          VARCHAR(45),
    device_id           VARCHAR(128),
    country             VARCHAR(64),
 
    -- login
    login_success       BOOLEAN,
    failed_login_attempts INTEGER,
    login_hour          SMALLINT,
 
    -- financial
    trade_volume        NUMERIC(18,4),
    deposit_amount      NUMERIC(18,4),
    withdraw_amount     NUMERIC(18,4),
    balance_before      NUMERIC(18,4),
    balance_after       NUMERIC(18,4),
    pnl                 NUMERIC(18,4),
 
    -- trading
    instrument          VARCHAR(32),
    trade_direction     VARCHAR(8),
    leverage            NUMERIC(8,2),
    margin_used         NUMERIC(18,4),
 
    -- session
    session_duration    INTEGER,              -- seconds
 
    -- derived / enriched later
    time_since_last_event NUMERIC(12,2),      -- seconds
 
    -- pipeline meta
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    rabbitmq_queue      VARCHAR(64)
);
 
-- Indexes for feature engineering queries
CREATE INDEX IF NOT EXISTS idx_raw_events_user_ts    ON raw_events (user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_events_event_type ON raw_events (event_type);
CREATE INDEX IF NOT EXISTS idx_raw_events_timestamp  ON raw_events (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_raw_events_ip         ON raw_events (ip_address);
 
-- ============================================================
--  Alerts table — written by the alert engine (Day 2)
-- ============================================================
 
CREATE TABLE IF NOT EXISTS alerts (
    id              BIGSERIAL PRIMARY KEY,
    user_id         VARCHAR(64)  NOT NULL,
    alert_type      VARCHAR(64),
    risk_score      NUMERIC(5,4),
    top_features    JSONB,
    message         TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    acknowledged    BOOLEAN DEFAULT FALSE
);
 
CREATE INDEX IF NOT EXISTS idx_alerts_user_id    ON alerts (user_id);
CREATE INDEX IF NOT EXISTS idx_alerts_risk_score ON alerts (risk_score DESC);