# File: core/utils/notifier.py

import os
import requests
import time
import smtplib
from typing import Optional
from core.utils.logger import get_logger

logger = get_logger(__name__)


def send_telegram_alert(message: str, bot_token: Optional[str] = None, chat_id: Optional[str] = None, retries: int = 3) -> None:
    """
    Send an HTML-formatted Telegram message with retry logic.

    Args:
        bot_token (str): Telegram bot token.
        chat_id (str): Telegram chat ID.
        message (str): HTML-formatted message content.
        retries (int): Number of retry attempts.
    """
    bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    if not bot_token or not chat_id:
        print("Telegram credentials not configured.")
        return
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return  # Successfully sent; exit the function
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Attempt {attempt}/{retries} failed to send Telegram message: {e}")
            if attempt < retries:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("All retry attempts failed.")

    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code != 200:
            print(f"Telegram error: {response.text}")
    except Exception as e:
        print(f"Telegram send failed: {e}")

def send_email_alert(subject: str, body: str, to_email: str, smtp_server: str, smtp_port: int, login: str, password: str) -> None:
    """
    Send an email alert via SMTP.

    Args:
        subject: Email subject.
        body: Email content.
        to_email: Recipient address.
        smtp_server: SMTP host.
        smtp_port: SMTP port.
        login: Sender username.
        password: Sender password.
    """
    message = f"Subject: {subject}\n\n{body}"

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(login, password)
            server.sendmail(login, to_email, message)
    except Exception as e:
        print(f"Email alert failed: {e}")
