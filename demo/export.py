import psycopg2
import pandas as pd
import sys
sys.path.insert(0, '.')
import config

conn = psycopg2.connect(config.PG_DSN)
df = pd.read_sql("""
    SELECT user_id, ensemble_risk_score, risk_label,
           isolation_forest_score, lstm_reconstruction_error,
           trade_volume_spike, bot_like_timing,
           large_withdraw_after_dormancy, high_freq_small_deposits,
           kyc_change_before_withdraw, consistent_profit_bursts,
           ip_change_rate, deposit_withdraw_ratio,
           login_hour_deviation, pnl_volatility, unique_ips,
           trade_volume_zscore, session_duration_zscore,
           margin_usage_zscore
    FROM user_features
    ORDER BY ensemble_risk_score DESC
""", conn)
conn.close()
df.to_csv('demo/user_features.csv', index=False)
print(f'Exported {len(df)} users to demo/user_features.csv')