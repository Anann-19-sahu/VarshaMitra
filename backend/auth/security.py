from datetime import datetime, timedelta, timezone
from jose import JWTError, jwt
from passlib.context import CryptContext

from auth.config import AuthSettings


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> tuple[str, int]:
    expire_delta = timedelta(minutes=AuthSettings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = datetime.now(timezone.utc) + expire_delta
    payload = {"sub": subject, "exp": expires_at}
    token = jwt.encode(payload, AuthSettings.JWT_SECRET_KEY, algorithm=AuthSettings.JWT_ALGORITHM)
    return token, int(expire_delta.total_seconds())


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, AuthSettings.JWT_SECRET_KEY, algorithms=[AuthSettings.JWT_ALGORITHM])
    except JWTError as exc:
        raise ValueError("Invalid or expired token") from exc
