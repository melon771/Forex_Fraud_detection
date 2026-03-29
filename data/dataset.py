import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

NUM_USERS = 500
NUM_EVENTS = 50000

event_types = ['login', 'trade', 'deposit', 'withdraw']
countries = ['IN', 'US', 'UK', 'SG', 'AE']
instruments = ['EURUSD', 'GBPUSD', 'BTCUSD', 'XAUUSD']

data = []

start_time = datetime.now() - timedelta(days=30)

# initialize user profiles
user_profiles = {}

for user in range(1, NUM_USERS + 1):
    user_profiles[user] = {
        "balance": np.random.uniform(1000, 10000),
        "account_age_days": random.randint(10, 1000),
        "kyc_status": random.choice([0, 1]),
    }

for _ in range(NUM_EVENTS):
    user_id = random.randint(1, NUM_USERS)
    profile = user_profiles[user_id]

    timestamp = start_time + timedelta(seconds=random.randint(0, 30*24*3600))
    event_type = random.choice(event_types)

    ip_address = f"192.168.{random.randint(0,255)}.{random.randint(0,255)}"
    device_id = f"device_{random.randint(1,1000)}"
    country = random.choice(countries)

    login_hour = timestamp.hour
    login_success = np.random.choice([0,1], p=[0.1, 0.9])
    failed_login_attempts = random.randint(0, 5) if login_success == 0 else 0

    balance_before = profile["balance"]

    trade_volume = 0
    deposit_amount = 0
    withdraw_amount = 0
    pnl = 0

    if event_type == 'trade':
        trade_volume = np.random.exponential(scale=100)
        pnl = np.random.normal(loc=0, scale=50)
        profile["balance"] += pnl

    elif event_type == 'deposit':
        deposit_amount = np.random.exponential(scale=500)
        profile["balance"] += deposit_amount

    elif event_type == 'withdraw':
        withdraw_amount = np.random.exponential(scale=400)
        withdraw_amount = min(withdraw_amount, profile["balance"])
        profile["balance"] -= withdraw_amount

    balance_after = profile["balance"]

    session_duration = np.random.randint(10, 3600)
    instrument = random.choice(instruments)
    trade_direction = random.choice(['buy', 'sell'])
    leverage = random.choice([1, 5, 10, 20])
    margin_used = trade_volume * leverage * 0.01

    data.append([
        user_id, timestamp, event_type,
        profile["account_age_days"], profile["kyc_status"],
        ip_address, device_id, country,
        login_hour, login_success, failed_login_attempts,
        trade_volume, deposit_amount, withdraw_amount,
        balance_before, balance_after, pnl,
        instrument, trade_direction, leverage, margin_used,
        session_duration
    ])

df = pd.DataFrame(data, columns=[
    'user_id', 'timestamp', 'event_type',
    'account_age_days', 'kyc_status',
    'ip_address', 'device_id', 'country',
    'login_hour', 'login_success', 'failed_login_attempts',
    'trade_volume', 'deposit_amount', 'withdraw_amount',
    'balance_before', 'balance_after', 'pnl',
    'instrument', 'trade_direction', 'leverage', 'margin_used',
    'session_duration'
])

# sort by time
df.sort_values(by='timestamp', inplace=True)

# -----------------------------
# 🚨 INJECT SMART ANOMALIES
# -----------------------------

# 1. Large withdrawal anomaly
for _ in range(300):
    idx = random.randint(0, len(df)-1)
    df.loc[idx, 'withdraw_amount'] = df['withdraw_amount'].mean() * 15

# 2. Trade spike anomaly
for _ in range(300):
    idx = random.randint(0, len(df)-1)
    df.loc[idx, 'trade_volume'] *= 10

# 3. IP switching anomaly
for _ in range(200):
    user = random.randint(1, NUM_USERS)
    df.loc[df['user_id'] == user, 'ip_address'] = f"10.0.{random.randint(0,255)}.{random.randint(0,255)}"

# 4. Failed login burst
for _ in range(200):
    idx = random.randint(0, len(df)-1)
    df.loc[idx, 'failed_login_attempts'] = random.randint(5, 10)

# 5. Suspicious profit burst
for _ in range(200):
    idx = random.randint(0, len(df)-1)
    df.loc[idx, 'pnl'] *= 10

df.to_csv("forex_data_v2.csv", index=False)

print("🔥 Advanced dataset generated: forex_data_v2.csv")