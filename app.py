"""
demo/app.py
-----------
ForexGuard demo — Gradio UI with Gemini LLM risk summaries.
Runs on HuggingFace Spaces (no Postgres or RabbitMQ needed).
Data is pre-baked from user_features.csv exported from the main system.
"""

import os
import json
import pandas as pd
import gradio as gr
import google.generativeai as genai

# ── Load pre-baked data ───────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "user_features.csv")
df = pd.read_csv(CSV_PATH)

# Fill NaN with 0 for display
df = df.fillna(0)

# Boolean columns
BOOL_COLS = [
    "trade_volume_spike", "bot_like_timing",
    "large_withdraw_after_dormancy", "high_freq_small_deposits",
    "kyc_change_before_withdraw", "consistent_profit_bursts",
]
for col in BOOL_COLS:
    if col in df.columns:
        df[col] = df[col].astype(bool)

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    gemini_model = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_risk_color(label: str) -> str:
    return {"high": "#e53e3e", "medium": "#dd6b20", "low": "#38a169"}.get(label, "#718096")


def get_top_features(row: pd.Series) -> list[dict]:
    signals = [
        ("trade_volume_spike",            "Trade volume spike (10x baseline)",     2.0  if row.get("trade_volume_spike") else 0),
        ("bot_like_timing",               "Bot-like event timing",                 2.0  if row.get("bot_like_timing") else 0),
        ("large_withdraw_after_dormancy", "Large withdrawal after dormancy",        2.0  if row.get("large_withdraw_after_dormancy") else 0),
        ("high_freq_small_deposits",      "High-frequency small deposits",          2.0  if row.get("high_freq_small_deposits") else 0),
        ("kyc_change_before_withdraw",    "KYC change before withdrawal",           2.0  if row.get("kyc_change_before_withdraw") else 0),
        ("consistent_profit_bursts",      "Consistent profit bursts",               2.0  if row.get("consistent_profit_bursts") else 0),
        ("pnl_volatility",                "High PnL volatility",                   min(abs(float(row.get("pnl_volatility", 0))), 10)),
        ("ip_change_rate",                "Rapid IP switching",                    min(abs(float(row.get("ip_change_rate", 0))), 10)),
        ("trade_volume_zscore",           "Trade volume deviation",                min(abs(float(row.get("trade_volume_zscore", 0))), 10)),
        ("login_hour_deviation",          "Unusual login hour",                    min(abs(float(row.get("login_hour_deviation", 0))), 10)),
        ("deposit_withdraw_ratio",        "High withdrawal-to-deposit ratio",      min(abs(float(row.get("deposit_withdraw_ratio", 0))), 10)),
        ("margin_usage_zscore",           "Unusual margin usage",                  min(abs(float(row.get("margin_usage_zscore", 0))), 10)),
        ("session_duration_zscore",       "Abnormal session duration",             min(abs(float(row.get("session_duration_zscore", 0))), 10)),
        ("unique_ips",                    "Multiple distinct IP addresses",        min(abs(float(row.get("unique_ips", 0))), 10)),
    ]
    signals.sort(key=lambda x: x[2], reverse=True)
    return [{"feature": s[0], "description": s[1], "importance": round(s[2], 3)}
            for s in signals if s[2] > 0][:5]


def generate_llm_summary(user_id: str, risk_score: float,
                          risk_label: str, top_features: list[dict]) -> str:
    if not gemini_model:
        return "⚠️ Gemini API key not configured. Add GEMINI_API_KEY to HF Space secrets."

    feat_lines = "\n".join([f"- {f['description']} (importance: {f['importance']})"
                             for f in top_features])

    prompt = f"""You are a senior compliance analyst at a forex brokerage.
Analyse this trader risk profile and write a 3-sentence professional summary
for the compliance team. Be specific about the signals, their implications,
and recommend a clear action (investigate / monitor / clear).

User ID: {user_id}
Risk Score: {risk_score:.4f} ({risk_label.upper()} RISK)
Top anomaly signals:
{feat_lines}

Write exactly 3 sentences. Be direct and professional. No bullet points."""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as exc:
        return f"LLM summary failed: {exc}"


# ── Main scoring function ─────────────────────────────────────────────────────

def score_user(user_id: str, gemini_key_input: str):
    user_id = str(user_id).strip()

    # Allow runtime API key input
    global gemini_model
    key = gemini_key_input.strip() or GEMINI_KEY
    if key and not gemini_model:
        genai.configure(api_key=key)
        gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    row = df[df["user_id"].astype(str) == user_id]
    if row.empty:
        return (
            "❌ User not found",
            "",
            None,
            "User ID not found in the dataset.",
        )

    row = row.iloc[0]
    risk_score = float(row.get("ensemble_risk_score", 0))
    risk_label = str(row.get("risk_label", "low"))
    if_score   = float(row.get("isolation_forest_score", 0))
    lstm_score = float(row.get("lstm_reconstruction_error", 0))

    top_features = get_top_features(row)
    color        = get_risk_color(risk_label)

    # ── Score card HTML ───────────────────────────────────────
    score_html = f"""
    <div style="font-family: sans-serif; padding: 20px; border-radius: 12px;
                border: 2px solid {color}; background: #1a1a2e; color: white;">
        <h2 style="margin: 0 0 12px; color: {color};">
            User {user_id} — {risk_label.upper()} RISK
        </h2>
        <div style="display: flex; gap: 24px; margin-bottom: 16px;">
            <div style="text-align: center;">
                <div style="font-size: 36px; font-weight: bold; color: {color};">
                    {risk_score:.3f}
                </div>
                <div style="font-size: 12px; color: #aaa;">Ensemble Score</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #7ec8e3;">
                    {if_score:.3f}
                </div>
                <div style="font-size: 12px; color: #aaa;">Isolation Forest</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: #b5a4e3;">
                    {lstm_score:.3f}
                </div>
                <div style="font-size: 12px; color: #aaa;">LSTM Autoencoder</div>
            </div>
        </div>
        <div style="margin-top: 12px;">
            <div style="font-size: 13px; font-weight: bold; color: #ccc;
                        margin-bottom: 8px;">Top Contributing Signals</div>
            {''.join([
                f'<div style="display:flex; justify-content:space-between; '
                f'padding: 6px 10px; margin-bottom: 4px; border-radius: 6px; '
                f'background: rgba(255,255,255,0.05);">'
                f'<span style="color:#eee; font-size:13px;">{f["description"]}</span>'
                f'<span style="color:{color}; font-weight:bold; font-size:13px;">'
                f'{f["importance"]:.2f}</span></div>'
                for f in top_features
            ])}
        </div>
    </div>
    """

    # ── Boolean flags table ───────────────────────────────────
    flags = {
        "Trade volume spike":       bool(row.get("trade_volume_spike", False)),
        "Bot-like timing":          bool(row.get("bot_like_timing", False)),
        "Dormancy withdrawal":      bool(row.get("large_withdraw_after_dormancy", False)),
        "Structuring deposits":     bool(row.get("high_freq_small_deposits", False)),
        "KYC before withdrawal":    bool(row.get("kyc_change_before_withdraw", False)),
        "Profit burst":             bool(row.get("consistent_profit_bursts", False)),
    }
    flags_data = [[k, "🔴 YES" if v else "🟢 NO"] for k, v in flags.items()]

    # ── LLM summary ───────────────────────────────────────────
    llm_summary = generate_llm_summary(user_id, risk_score, risk_label, top_features)

    return score_html, llm_summary, flags_data, ""


def get_high_risk_users():
    high = df[df["risk_label"] == "high"].copy()
    high = high.sort_values("ensemble_risk_score", ascending=False).head(20)
    return high[["user_id", "ensemble_risk_score", "risk_label",
                 "isolation_forest_score", "lstm_reconstruction_error"]].values.tolist()


# ── Gradio UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="ForexGuard — Anomaly Detection",
    theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"),
    css=".gradio-container { max-width: 1100px !important; }"
) as demo:

    gr.Markdown("""
    # ForexGuard — Real-Time Trader Anomaly Detection
    **AI/ML system for detecting suspicious trader behaviour in forex brokerage environments.**
    Powered by Isolation Forest + LSTM Autoencoder ensemble with Gemini LLM risk summaries.
    """)

    with gr.Tabs():

        # ── Tab 1: Score a user ───────────────────────────────
        with gr.TabItem("Score a User"):
            with gr.Row():
                with gr.Column(scale=2):
                    user_input = gr.Textbox(
                        label="Enter User ID",
                        placeholder="e.g. 92",
                        info="Enter any user ID from the dataset"
                    )
                    gemini_key = gr.Textbox(
                        label="Gemini API Key (optional)",
                        placeholder="AIza...",
                        type="password",
                        info="Get free key at aistudio.google.com"
                    )
                    score_btn = gr.Button("Analyse User", variant="primary")

                with gr.Column(scale=3):
                    score_display = gr.HTML(label="Risk Assessment")

            llm_output = gr.Textbox(
                label="Gemini AI Risk Summary",
                lines=4,
                interactive=False
            )
            flags_table = gr.Dataframe(
                headers=["Signal", "Status"],
                label="Fraud Signal Flags",
                interactive=False,
            )
            error_msg = gr.Textbox(visible=False)

            score_btn.click(
                fn=score_user,
                inputs=[user_input, gemini_key],
                outputs=[score_display, llm_output, flags_table, error_msg],
            )

            gr.Examples(
                examples=[["92"], ["1"], ["100"], ["250"], ["400"]],
                inputs=user_input,
            )

        # ── Tab 2: High risk users ────────────────────────────
        with gr.TabItem("High Risk Users"):
            gr.Markdown("### Top 20 highest risk users in the dataset")
            refresh_btn = gr.Button("Refresh", variant="secondary")
            risk_table = gr.Dataframe(
                headers=["User ID", "Ensemble Score", "Risk Label", "IF Score", "LSTM Score"],
                label="High Risk Users",
                interactive=False,
                value=get_high_risk_users(),
            )
            refresh_btn.click(fn=get_high_risk_users, outputs=risk_table)

        # ── Tab 3: About ──────────────────────────────────────
        with gr.TabItem("About"):
            gr.Markdown("""
            ## How ForexGuard Works

            ### Pipeline
            1. Raw events (logins, trades, deposits, withdrawals) are ingested via **RabbitMQ**
            2. A consumer writes events to **PostgreSQL**
            3. **Feature engineering** computes 34 behavioural signals per user
            4. **Isolation Forest** scores point anomalies (baseline model)
            5. **LSTM Autoencoder** scores sequence anomalies (advanced model)
            6. An **ensemble** combines both scores (60% IF + 40% LSTM)
            7. High-risk users trigger alerts published back to RabbitMQ

            ### Models
            - **Isolation Forest** — unsupervised, no labels needed, fast inference, SHAP-compatible
            - **LSTM Autoencoder** — captures temporal behaviour patterns IF misses
            - **Ensemble** — weighted combination, risk threshold at 0.75 for HIGH alerts

            ### Anomaly Signals Detected
            - Bot-like trading automation (sub-second event timing)
            - Rapid IP switching across geographies
            - Deposit → withdraw cycles (bonus abuse)
            - High-frequency small deposits (financial structuring)
            - Trade volume spikes (10x personal baseline)
            - KYC changes before large withdrawals
            - Consistent abnormal profit bursts (latency arbitrage)

            ### Tech Stack
            RabbitMQ • PostgreSQL • Isolation Forest (sklearn) •
            LSTM Autoencoder (PyTorch) • FastAPI • Gradio • Gemini AI
            """)

demo.launch()