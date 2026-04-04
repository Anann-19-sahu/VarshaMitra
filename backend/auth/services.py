import logging
import random
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

from auth.config import AuthSettings

logger = logging.getLogger("varsha_mitra.auth")


def generate_otp_code() -> str:
    if AuthSettings.ENV == "dev":
        return "123456"
    max_value = 10 ** AuthSettings.OTP_LENGTH - 1
    min_value = 10 ** (AuthSettings.OTP_LENGTH - 1)
    return str(random.randint(min_value, max_value))


def otp_expiry_time() -> datetime:
    return datetime.now(timezone.utc) + timedelta(minutes=AuthSettings.OTP_EXPIRE_MINUTES)


def send_email_otp(email: str, otp: str) -> tuple[bool, str]:
    if AuthSettings.ENV == "dev":
        logger.warning(f"[AUTH][DEV] OTP for {email}: {otp}")
        return True, "dev-mode"

    if not (
        AuthSettings.SMTP_HOST
        and AuthSettings.SMTP_USER
        and AuthSettings.SMTP_PASSWORD
        and AuthSettings.SMTP_FROM_EMAIL
    ):
        return False, "SMTP credentials are not configured"

    msg = MIMEText(
        f"Your verification OTP is {otp}. It expires in {AuthSettings.OTP_EXPIRE_MINUTES} minutes."
    )
    msg["Subject"] = "VarshaMitra OTP Verification"
    msg["From"] = AuthSettings.SMTP_FROM_EMAIL
    msg["To"] = email

    try:
        with smtplib.SMTP(AuthSettings.SMTP_HOST, AuthSettings.SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(AuthSettings.SMTP_USER, AuthSettings.SMTP_PASSWORD)
            server.sendmail(AuthSettings.SMTP_FROM_EMAIL, [email], msg.as_string())
        return True, "sent"
    except Exception as exc:
        logger.error(f"[AUTH] Failed to send OTP email: {exc}")
        return False, "Failed to send OTP email"
