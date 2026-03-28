from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from auth.database import Base, engine, get_db
from auth.models import OTPVerification, User
from auth.schemas import (
    ApiResponse,
    LoginRequest,
    SignupRequest,
    TokenResponse,
    UserResponse,
    VerifyOtpRequest,
)
from auth.security import create_access_token, decode_token, hash_password, verify_password
from auth.services import generate_otp_code, otp_expiry_time, send_email_otp


router = APIRouter(prefix="/api/v2/auth", tags=["auth-v2"])
bearer = HTTPBearer(auto_error=False)

# Safe to call on startup/import; creates auth tables if missing.
Base.metadata.create_all(bind=engine)


def response_ok(message: str, data=None):
    return ApiResponse(success=True, message=message, data=data).model_dump()


def response_error(message: str, error: str):
    return ApiResponse(success=False, message=message, error=error).model_dump()


@router.post("/signup", response_model=ApiResponse)
def signup(payload: SignupRequest, db: Session = Depends(get_db)):
    try:
        email = payload.email.lower().strip()
        existing = db.query(User).filter(User.email == email).first()

        if existing and existing.is_verified:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=response_error("Signup failed", "User already exists"),
            )

        if existing and not existing.is_verified:
            existing.password_hash = hash_password(payload.password)
            existing.full_name = payload.full_name
            existing.is_active = False
        else:
            db.add(
                User(
                    email=email,
                    full_name=payload.full_name,
                    password_hash=hash_password(payload.password),
                    is_active=False,
                    is_verified=False,
                )
            )
        db.commit()

        db.query(OTPVerification).filter(
            OTPVerification.email == email,
            OTPVerification.is_used == False,  # noqa: E712
            OTPVerification.purpose == "signup",
        ).update({"is_used": True})

        otp = generate_otp_code()
        db.add(
            OTPVerification(
                email=email,
                otp_code=otp,
                purpose="signup",
                is_used=False,
                expires_at=otp_expiry_time(),
            )
        )
        db.commit()

        sent, send_msg = send_email_otp(email, otp)
        if not sent:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=response_error("OTP delivery failed", send_msg),
            )

        return response_ok("Signup initiated. Verify OTP sent to your email.", {"email": email})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response_error("Signup failed", str(exc)),
        ) from exc


@router.post("/verify-otp", response_model=ApiResponse)
def verify_otp(payload: VerifyOtpRequest, db: Session = Depends(get_db)):
    try:
        email = payload.email.lower().strip()
        otp_row = (
            db.query(OTPVerification)
            .filter(
                OTPVerification.email == email,
                OTPVerification.otp_code == payload.otp.strip(),
                OTPVerification.purpose == "signup",
                OTPVerification.is_used == False,  # noqa: E712
            )
            .order_by(OTPVerification.created_at.desc())
            .first()
        )
        if not otp_row:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response_error("OTP verification failed", "Invalid OTP"),
            )

        now = datetime.now(timezone.utc)
        expires_at = otp_row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < now:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=response_error("OTP verification failed", "OTP expired"),
            )

        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=response_error("OTP verification failed", "User not found"),
            )

        otp_row.is_used = True
        user.is_verified = True
        user.is_active = True
        db.commit()
        return response_ok("OTP verified. Account is now active.", {"email": email})
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response_error("OTP verification failed", str(exc)),
        ) from exc


@router.post("/login", response_model=ApiResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    try:
        email = payload.email.lower().strip()
        user = db.query(User).filter(User.email == email).first()
        if not user or not verify_password(payload.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=response_error("Login failed", "Invalid email or password"),
            )
        if not user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=response_error("Login failed", "Account is not verified"),
            )

        token, expires_in = create_access_token(subject=user.email)
        data = TokenResponse(access_token=token, expires_in=expires_in).model_dump()
        return response_ok("Login successful", data)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=response_error("Login failed", str(exc)),
        ) from exc


@router.get("/me", response_model=ApiResponse)
def me(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=response_error("Unauthorized", "Missing bearer token"),
        )
    try:
        payload = decode_token(credentials.credentials)
        email = payload.get("sub", "").lower().strip()
        user = db.query(User).filter(User.email == email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=response_error("Unauthorized", "User not found"),
            )
        return response_ok("Current user fetched", UserResponse.model_validate(user).model_dump())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=response_error("Unauthorized", str(exc)),
        ) from exc
