import os
from dotenv import load_dotenv

load_dotenv()


class AuthSettings:
    ENV = os.getenv("ENV", "dev").strip().lower()
    DATABASE_URL = os.getenv("AUTH_DATABASE_URL") or os.getenv("DATABASE_URL", "sqlite:///./varsha_auth.db")

    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
    JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

    OTP_EXPIRE_MINUTES = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))
    OTP_LENGTH = int(os.getenv("OTP_LENGTH", "6"))

    SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "").strip()
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
    SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER).strip()

    @classmethod
    def is_prod(cls) -> bool:
        return cls.ENV == "prod"
