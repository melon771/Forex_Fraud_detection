"""
alerts/llm_summary.py
----------------------
Uses Gemini API to generate a human-readable compliance alert
from anomaly detection data.
"""

import logging
import google.generativeai as genai

from secrets import GEMINI_API_KEY

log = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

MODEL = genai.GenerativeModel("gemini-1.5-pro")  # or gemini-1.5-flash


def generate_llm_summary(
    user_id: str,
    risk_score: float,
    risk_label: str,
    if_score: float,
    lstm_score: float,
    top_features: list[dict],
) -> str:
    """
    Generate a compliance-ready alert summary using Gemini.
    Falls back to rule-based summary if API fails.
    """

    features_text = "\n".join([
        f"- {f['description']}: {f['value']} (importance: {f['importance']:.2f})"
        for f in top_features
    ])

    prompt = f"""
You are a compliance analyst at a forex brokerage.
Our anomaly detection system has flagged a trader.

DETECTED DATA:
- User ID: {user_id}
- Risk Level: {risk_label.upper()} ({risk_score:.2%})
- Isolation Forest Score: {if_score:.4f}
- LSTM Sequence Score: {lstm_score:.4f}
- Top anomalous signals:
{features_text}

Write a 3-4 sentence professional alert that:
1. Clearly states risk level
2. Explains suspicious behavior
3. Suggests action (freeze / review / monitor)
4. Mentions confidence level

No bullet points. Paragraph only.
"""

    try:
        response = MODEL.generate_content(prompt)
        summary = response.text.strip()
        log.info(f"Gemini summary generated for user {user_id}")
        return summary

    except Exception as exc:
        log.warning(f"Gemini API failed: {exc} — using fallback")
        return _fallback_summary(user_id, risk_score, risk_label, top_features)


def _fallback_summary(
    user_id: str,
    risk_score: float,
    risk_label: str,
    top_features: list[dict],
) -> str:
    """Fallback summary if Gemini fails."""
    top = top_features[0]["description"] if top_features else "unusual activity"

    action = (
        "Immediate account review and potential freeze recommended."
        if risk_score >= 0.85 else
        "Manual compliance review recommended within 24 hours."
        if risk_score >= 0.65 else
        "Flag for monitoring over the next 48 hours."
    )

    return (
        f"FOREXGUARD ALERT — {risk_label.upper()} RISK: Trader {user_id} has been flagged "
        f"with a risk score of {risk_score:.2%}. Primary signal: {top}. "
        f"{action} Confidence: Isolation Forest + LSTM Autoencoder ensemble."
    )