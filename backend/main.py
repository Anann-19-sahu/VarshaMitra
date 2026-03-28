"""
VarshaMitra – Pre-Monsoon Flood Intelligence Platform
FastAPI Backend Server
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Optional, List, Dict, Any
import asyncpg
import json
import io
import datetime
import asyncio
import logging
import hashlib
import secrets
import os
import smtplib
from email.mime.text import MIMEText
from pydantic import BaseModel
import httpx
from weather_service import WeatherService
from elevation_service import ElevationService
from drainage_service import DrainageService
from disaster_service import DisasterService
from auth.router import router as auth_router

logger = logging.getLogger("varsha_mitra")

# ─── App Init ─────────────────────────────────────────────
app = FastAPI(
    title="VarshaMitra Flood Intelligence API",
    description="Pre-Monsoon Flood Risk Assessment for Indian Cities",
    version="2.0.0"
)
app.include_router(auth_router)

# ─── Weather Service (singleton, shared across requests) ──────
weather_svc   = WeatherService()
elevation_svc = ElevationService()
drainage_svc  = DrainageService()
disaster_svc  = DisasterService()

# ─── Serve Frontend Static Files ───────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main frontend HTML at http://localhost:8000"""
    html_file = FRONTEND_DIR / "VarshaMitra.html"
    if html_file.exists():
        return FileResponse(str(html_file))
    return {"service": "VarshaMitra Flood Intelligence Platform", "version": "2.0.0", "status": "operational"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

# ─── Pydantic Models ────────────────────────────────────────
class WardFloodScore(BaseModel):
    ward_id: int
    ward_name: str
    city: str
    lat: float
    lng: float
    score: int
    risk_class: str
    rainfall_mm: float
    elevation_m: float
    drainage_pct: float
    flood_events: int
    population: int
    ai_report: Optional[str] = None
    computed_at: str

class DrainReport(BaseModel):
    ward_name: str
    city: str
    description: str
    reporter_name: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class AlertPayload(BaseModel):
    ward_id: int
    severity: str
    message: str
    channels: List[str] = ["dashboard"]

class MLPredictionRequest(BaseModel):
    ward_id: int
    rainfall_mm: float
    elevation_m: float
    drainage_pct: float
    flood_events: int

class CitizenSignupRequest(BaseModel):
    username: str
    password: str
    gmail: str
    phone: str

class CitizenOtpVerifyRequest(BaseModel):
    signup_id: str
    email_otp: str
    sms_otp: str

class CitizenLoginRequest(BaseModel):
    identifier: str
    password: str

class AuthorityLoginRequest(BaseModel):
    authority_id: str
    password: str

# ─── DB Connection ──────────────────────────────────────────
DATABASE_URL = "postgresql://varsha:monsoon2024@localhost:5432/varsha_mitra"

async def get_db():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

# ─── Auth Storage / RBAC ──────────────────────────────────────
AUTH_DATA_DIR = Path(__file__).parent / "data"
AUTH_STORE_FILE = AUTH_DATA_DIR / "auth_store.json"
AUTH_LOCK = asyncio.Lock()
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}
SESSION_TTL_HOURS = 12
OTP_TTL_MINUTES = 10
ENV = os.getenv("ENV", "dev").strip().lower()
ALLOW_DEV_OTP_FALLBACK = os.getenv("ALLOW_DEV_OTP_FALLBACK", "false").lower() == "true"
DEV_DUMMY_OTP = os.getenv("DEV_DUMMY_OTP", "123456")

PREALLOTTED_AUTHORITIES = {
    "AUTH-BBMP-001": {"name": "BBMP CONTROL ROOM", "password": "BBMP@2026"},
    "AUTH-BMC-002": {"name": "BMC FLOOD CELL", "password": "BMC@2026"},
    "AUTH-DMA-003": {"name": "STATE DMA OPS", "password": "DMA@2026"},
}

def _hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(8)
    digest = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return f"{salt}${digest}"

def _verify_password(password: str, password_hash: str) -> bool:
    try:
        salt, existing = password_hash.split("$", 1)
    except ValueError:
        return False
    probe = hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()
    return secrets.compare_digest(probe, existing)

def _default_authorities():
    return {
        aid: {"name": meta["name"], "password_hash": _hash_password(meta["password"])}
        for aid, meta in PREALLOTTED_AUTHORITIES.items()
    }

def _ensure_auth_store():
    AUTH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not AUTH_STORE_FILE.exists():
        payload = {
            "citizens": [],
            "pending_otps": [],
            "authorities": _default_authorities()
        }
        AUTH_STORE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def _read_auth_store() -> dict:
    _ensure_auth_store()
    try:
        data = json.loads(AUTH_STORE_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data.setdefault("citizens", [])
    data.setdefault("pending_otps", [])
    data.setdefault("authorities", _default_authorities())
    return data

def _write_auth_store(data: dict) -> None:
    _ensure_auth_store()
    AUTH_STORE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _issue_session(role: str, subject: str, display_name: str) -> dict:
    token = secrets.token_urlsafe(32)
    now = datetime.datetime.utcnow()
    exp = now + datetime.timedelta(hours=SESSION_TTL_HOURS)
    ACTIVE_SESSIONS[token] = {
        "role": role,
        "subject": subject,
        "display_name": display_name,
        "issued_at": now.isoformat(),
        "expires_at": exp.isoformat(),
    }
    allowed_tabs = ["citizen"] if role == "citizen" else ["map", "analytics", "wards", "citizen"]
    return {
        "access_token": token,
        "token_type": "bearer",
        "role": role,
        "display_name": display_name,
        "allowed_tabs": allowed_tabs,
        "expires_at": exp.isoformat()
    }

def _cleanup_expired_sessions() -> None:
    now = datetime.datetime.utcnow()
    expired = []
    for token, session in ACTIVE_SESSIONS.items():
        try:
            exp = datetime.datetime.fromisoformat(session["expires_at"])
        except Exception:
            expired.append(token)
            continue
        if exp <= now:
            expired.append(token)
    for token in expired:
        ACTIVE_SESSIONS.pop(token, None)

def _send_email_otp(gmail: str, otp: str) -> (bool, str):
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM_EMAIL", smtp_user).strip()

    if not (smtp_host and smtp_user and smtp_password and smtp_from):
        return False, "SMTP credentials are not configured"

    subject = "VarshaMitra Citizen OTP Verification"
    body = (
        f"Your VarshaMitra Email OTP is: {otp}\n\n"
        f"This OTP expires in {OTP_TTL_MINUTES} minutes.\n"
        "If you did not request this, please ignore this email."
    )
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = gmail

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_from, [gmail], msg.as_string())
        return True, "sent"
    except Exception as exc:
        logger.error(f"[AUTH] Email OTP send failed: {exc}")
        return False, "Failed to send email OTP"

async def _send_sms_otp(phone: str, otp: str) -> (bool, str):
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_number = os.getenv("TWILIO_FROM_NUMBER", "").strip()
    country_code = os.getenv("DEFAULT_COUNTRY_CODE", "+91").strip()

    if not (sid and token and from_number):
        return False, "Twilio credentials are not configured"

    to_number = phone if phone.startswith("+") else f"{country_code}{phone}"
    url = f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json"
    data = {
        "To": to_number,
        "From": from_number,
        "Body": f"VarshaMitra SMS OTP: {otp}. Valid for {OTP_TTL_MINUTES} minutes.",
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(url, data=data, auth=(sid, token))
        if res.status_code >= 300:
            logger.error(f"[AUTH] Twilio SMS OTP failed: {res.status_code} {res.text[:250]}")
            return False, "Failed to send SMS OTP"
        return True, "sent"
    except Exception as exc:
        logger.error(f"[AUTH] SMS OTP send failed: {exc}")
        return False, "Failed to send SMS OTP"

def _is_dev_mode() -> bool:
    return ENV != "prod"

def _generate_otps() -> (str, str):
    """
    Central OTP generation strategy:
    - dev: deterministic dummy OTP
    - prod: random 6-digit OTPs
    """
    if _is_dev_mode():
        return DEV_DUMMY_OTP, DEV_DUMMY_OTP
    return f"{secrets.randbelow(900000) + 100000}", f"{secrets.randbelow(900000) + 100000}"

async def _deliver_or_stub_otps(username: str, gmail: str, phone: str, email_otp: str, sms_otp: str) -> (bool, str, str):
    """
    Deliver OTPs via providers in production mode, or bypass in dev mode.
    Returns (ok, email_msg, sms_msg).
    """
    if _is_dev_mode():
        logger.warning(
            f"[AUTH][DEV-BYPASS] OTP sending bypassed for {username}. "
            f"Dummy Email OTP={email_otp}, Dummy SMS OTP={sms_otp}, gmail={gmail}, phone={phone}"
        )
        return True, "dev-bypass", "dev-bypass"

    email_ok, email_msg = _send_email_otp(gmail, email_otp)
    sms_ok, sms_msg = await _send_sms_otp(phone, sms_otp)
    if email_ok and sms_ok:
        return True, email_msg, sms_msg

    if ALLOW_DEV_OTP_FALLBACK:
        logger.warning(
            f"[AUTH] OTP providers failed; fallback enabled for {username}. "
            f"email={email_msg}, sms={sms_msg}, emailOtp={email_otp}, smsOtp={sms_otp}"
        )
        return True, email_msg, sms_msg

    return False, email_msg, sms_msg

# ─── In-Memory Fallback Data ────────────────────────────────
# Used when PostgreSQL is not available (standalone / dev mode)
FALLBACK_WARDS = [
    # Mumbai wards
    {"id": 1,  "name": "Dharavi",       "city": "Mumbai",  "lat": 19.0422, "lng": 72.8521, "population": 850000,  "drainage_density": 25, "rainfall_mm": 210, "elevation_m": 8,   "drainage_pct": 25, "flood_events_5yr": 4, "score": 18, "risk_class": "RED_ALERT"},
    {"id": 2,  "name": "Kurla",         "city": "Mumbai",  "lat": 19.0726, "lng": 72.8794, "population": 420000,  "drainage_density": 35, "rainfall_mm": 195, "elevation_m": 12,  "drainage_pct": 35, "flood_events_5yr": 3, "score": 32, "risk_class": "RED_ALERT"},
    {"id": 3,  "name": "Sion",          "city": "Mumbai",  "lat": 19.0446, "lng": 72.8612, "population": 310000,  "drainage_density": 30, "rainfall_mm": 200, "elevation_m": 10,  "drainage_pct": 28, "flood_events_5yr": 4, "score": 22, "risk_class": "RED_ALERT"},
    {"id": 4,  "name": "Chembur",       "city": "Mumbai",  "lat": 19.0623, "lng": 72.9007, "population": 350000,  "drainage_density": 40, "rainfall_mm": 185, "elevation_m": 18,  "drainage_pct": 38, "flood_events_5yr": 2, "score": 42, "risk_class": "WATCH_ZONE"},
    {"id": 5,  "name": "Borivali",      "city": "Mumbai",  "lat": 19.1879, "lng": 72.8484, "population": 520000,  "drainage_density": 55, "rainfall_mm": 170, "elevation_m": 45,  "drainage_pct": 55, "flood_events_5yr": 1, "score": 62, "risk_class": "WATCH_ZONE"},
    {"id": 6,  "name": "Mulund",        "city": "Mumbai",  "lat": 19.1059, "lng": 72.9276, "population": 280000,  "drainage_density": 42, "rainfall_mm": 175, "elevation_m": 22,  "drainage_pct": 42, "flood_events_5yr": 2, "score": 48, "risk_class": "WATCH_ZONE"},
    {"id": 7,  "name": "Worli",         "city": "Mumbai",  "lat": 19.0544, "lng": 72.8402, "population": 180000,  "drainage_density": 65, "rainfall_mm": 160, "elevation_m": 35,  "drainage_pct": 60, "flood_events_5yr": 1, "score": 72, "risk_class": "SAFE_ZONE"},
    {"id": 8,  "name": "Powai",         "city": "Mumbai",  "lat": 19.0860, "lng": 72.9081, "population": 250000,  "drainage_density": 50, "rainfall_mm": 165, "elevation_m": 50,  "drainage_pct": 52, "flood_events_5yr": 1, "score": 68, "risk_class": "WATCH_ZONE"},
    # Pune wards
    {"id": 9,  "name": "Shivajinagar",  "city": "Pune",    "lat": 18.5188, "lng": 73.8567, "population": 195000,  "drainage_density": 60, "rainfall_mm": 112, "elevation_m": 550, "drainage_pct": 60, "flood_events_5yr": 1, "score": 82, "risk_class": "SAFE_ZONE"},
    {"id": 10, "name": "Kothrud",       "city": "Pune",    "lat": 18.5590, "lng": 73.9040, "population": 310000,  "drainage_density": 55, "rainfall_mm": 120, "elevation_m": 530, "drainage_pct": 55, "flood_events_5yr": 1, "score": 78, "risk_class": "SAFE_ZONE"},
    {"id": 11, "name": "Sinhagad Road", "city": "Pune",    "lat": 18.5074, "lng": 73.8077, "population": 280000,  "drainage_density": 45, "rainfall_mm": 130, "elevation_m": 560, "drainage_pct": 48, "flood_events_5yr": 2, "score": 72, "risk_class": "SAFE_ZONE"},
    {"id": 12, "name": "Hadapsar",      "city": "Pune",    "lat": 18.4689, "lng": 73.8614, "population": 350000,  "drainage_density": 38, "rainfall_mm": 135, "elevation_m": 545, "drainage_pct": 40, "flood_events_5yr": 2, "score": 65, "risk_class": "WATCH_ZONE"},
    {"id": 13, "name": "Wagholi",       "city": "Pune",    "lat": 18.5089, "lng": 73.9260, "population": 210000,  "drainage_density": 32, "rainfall_mm": 140, "elevation_m": 540, "drainage_pct": 35, "flood_events_5yr": 2, "score": 60, "risk_class": "WATCH_ZONE"},
    {"id": 14, "name": "Pimpri",        "city": "Pune",    "lat": 18.6298, "lng": 73.7997, "population": 400000,  "drainage_density": 50, "rainfall_mm": 110, "elevation_m": 555, "drainage_pct": 52, "flood_events_5yr": 1, "score": 80, "risk_class": "SAFE_ZONE"},
    {"id": 15, "name": "Kondhwa",       "city": "Pune",    "lat": 18.4900, "lng": 73.8900, "population": 230000,  "drainage_density": 40, "rainfall_mm": 125, "elevation_m": 548, "drainage_pct": 42, "flood_events_5yr": 2, "score": 68, "risk_class": "WATCH_ZONE"},
    {"id": 16, "name": "Deccan",        "city": "Pune",    "lat": 18.5308, "lng": 73.8475, "population": 180000,  "drainage_density": 62, "rainfall_mm": 108, "elevation_m": 560, "drainage_pct": 62, "flood_events_5yr": 0, "score": 85, "risk_class": "SAFE_ZONE"},
    # Delhi wards
    {"id": 17, "name": "Yamuna Bazar",  "city": "Delhi",   "lat": 28.7041, "lng": 77.2993, "population": 220000,  "drainage_density": 28, "rainfall_mm": 72,  "elevation_m": 210, "drainage_pct": 28, "flood_events_5yr": 3, "score": 55, "risk_class": "WATCH_ZONE"},
    {"id": 18, "name": "Karol Bagh",    "city": "Delhi",   "lat": 28.7381, "lng": 77.2712, "population": 310000,  "drainage_density": 35, "rainfall_mm": 68,  "elevation_m": 215, "drainage_pct": 35, "flood_events_5yr": 2, "score": 65, "risk_class": "WATCH_ZONE"},
    {"id": 19, "name": "Rohini",        "city": "Delhi",   "lat": 28.7534, "lng": 77.1859, "population": 450000,  "drainage_density": 42, "rainfall_mm": 65,  "elevation_m": 220, "drainage_pct": 45, "flood_events_5yr": 1, "score": 75, "risk_class": "SAFE_ZONE"},
    {"id": 20, "name": "Laxmi Nagar",   "city": "Delhi",   "lat": 28.6716, "lng": 77.2952, "population": 380000,  "drainage_density": 32, "rainfall_mm": 70,  "elevation_m": 208, "drainage_pct": 32, "flood_events_5yr": 3, "score": 52, "risk_class": "WATCH_ZONE"},
    {"id": 21, "name": "Dwarka",        "city": "Delhi",   "lat": 28.7495, "lng": 77.0639, "population": 600000,  "drainage_density": 58, "rainfall_mm": 62,  "elevation_m": 225, "drainage_pct": 58, "flood_events_5yr": 0, "score": 82, "risk_class": "SAFE_ZONE"},
    {"id": 22, "name": "Mehrauli",      "city": "Delhi",   "lat": 28.5921, "lng": 77.0460, "population": 150000,  "drainage_density": 48, "rainfall_mm": 66,  "elevation_m": 230, "drainage_pct": 50, "flood_events_5yr": 1, "score": 78, "risk_class": "SAFE_ZONE"},
    {"id": 23, "name": "Shahdara",      "city": "Delhi",   "lat": 28.6244, "lng": 77.3048, "population": 340000,  "drainage_density": 30, "rainfall_mm": 75,  "elevation_m": 205, "drainage_pct": 30, "flood_events_5yr": 4, "score": 45, "risk_class": "WATCH_ZONE"},
    {"id": 24, "name": "Seelampur",     "city": "Delhi",   "lat": 28.6271, "lng": 77.2961, "population": 290000,  "drainage_density": 25, "rainfall_mm": 78,  "elevation_m": 203, "drainage_pct": 25, "flood_events_5yr": 4, "score": 38, "risk_class": "RED_ALERT"},
    # Chennai wards
    {"id": 25, "name": "Tondiarpet",    "city": "Chennai", "lat": 13.1651, "lng": 80.2650, "population": 200000,  "drainage_density": 20, "rainfall_mm": 145, "elevation_m": 4,   "drainage_pct": 20, "flood_events_5yr": 5, "score": 15, "risk_class": "RED_ALERT"},
    {"id": 26, "name": "Mylapore",      "city": "Chennai", "lat": 13.1227, "lng": 80.2889, "population": 180000,  "drainage_density": 22, "rainfall_mm": 140, "elevation_m": 5,   "drainage_pct": 22, "flood_events_5yr": 5, "score": 20, "risk_class": "RED_ALERT"},
    {"id": 27, "name": "T.Nagar",       "city": "Chennai", "lat": 13.1190, "lng": 80.2470, "population": 250000,  "drainage_density": 35, "rainfall_mm": 135, "elevation_m": 8,   "drainage_pct": 32, "flood_events_5yr": 4, "score": 28, "risk_class": "RED_ALERT"},
    {"id": 28, "name": "Velachery",     "city": "Chennai", "lat": 12.9785, "lng": 80.2209, "population": 210000,  "drainage_density": 15, "rainfall_mm": 150, "elevation_m": 3,   "drainage_pct": 15, "flood_events_5yr": 6, "score": 12, "risk_class": "RED_ALERT"},
    {"id": 29, "name": "Tambaram",      "city": "Chennai", "lat": 12.9249, "lng": 80.1000, "population": 320000,  "drainage_density": 38, "rainfall_mm": 130, "elevation_m": 15,  "drainage_pct": 38, "flood_events_5yr": 3, "score": 38, "risk_class": "RED_ALERT"},
    {"id": 30, "name": "Adyar",         "city": "Chennai", "lat": 13.0011, "lng": 80.2565, "population": 190000,  "drainage_density": 28, "rainfall_mm": 142, "elevation_m": 6,   "drainage_pct": 28, "flood_events_5yr": 5, "score": 18, "risk_class": "RED_ALERT"},
    {"id": 31, "name": "Anna Nagar",    "city": "Chennai", "lat": 13.0530, "lng": 80.2209, "population": 270000,  "drainage_density": 45, "rainfall_mm": 125, "elevation_m": 12,  "drainage_pct": 42, "flood_events_5yr": 3, "score": 42, "risk_class": "WATCH_ZONE"},
    {"id": 32, "name": "Kilpauk",       "city": "Chennai", "lat": 13.0857, "lng": 80.2101, "population": 160000,  "drainage_density": 40, "rainfall_mm": 128, "elevation_m": 10,  "drainage_pct": 40, "flood_events_5yr": 3, "score": 38, "risk_class": "RED_ALERT"},
    # Kolkata wards
    {"id": 33, "name": "Tiljala",       "city": "Kolkata", "lat": 22.5382, "lng": 88.3980, "population": 195000,  "drainage_density": 20, "rainfall_mm": 215, "elevation_m": 6,   "drainage_pct": 20, "flood_events_5yr": 5, "score": 12, "risk_class": "RED_ALERT"},
    {"id": 34, "name": "Park Circus",   "city": "Kolkata", "lat": 22.5507, "lng": 88.3888, "population": 170000,  "drainage_density": 25, "rainfall_mm": 200, "elevation_m": 7,   "drainage_pct": 25, "flood_events_5yr": 4, "score": 22, "risk_class": "RED_ALERT"},
    {"id": 35, "name": "Behala",        "city": "Kolkata", "lat": 22.4942, "lng": 88.3019, "population": 380000,  "drainage_density": 30, "rainfall_mm": 190, "elevation_m": 8,   "drainage_pct": 30, "flood_events_5yr": 4, "score": 28, "risk_class": "RED_ALERT"},
    {"id": 36, "name": "Jadavpur",      "city": "Kolkata", "lat": 22.4982, "lng": 88.3728, "population": 250000,  "drainage_density": 28, "rainfall_mm": 195, "elevation_m": 7,   "drainage_pct": 28, "flood_events_5yr": 4, "score": 25, "risk_class": "RED_ALERT"},
    {"id": 37, "name": "Salt Lake",     "city": "Kolkata", "lat": 22.5847, "lng": 88.3990, "population": 300000,  "drainage_density": 45, "rainfall_mm": 180, "elevation_m": 5,   "drainage_pct": 42, "flood_events_5yr": 3, "score": 35, "risk_class": "RED_ALERT"},
    {"id": 38, "name": "New Town",      "city": "Kolkata", "lat": 22.5958, "lng": 88.4005, "population": 220000,  "drainage_density": 55, "rainfall_mm": 170, "elevation_m": 6,   "drainage_pct": 52, "flood_events_5yr": 2, "score": 45, "risk_class": "WATCH_ZONE"},
    {"id": 39, "name": "Dum Dum",       "city": "Kolkata", "lat": 22.5803, "lng": 88.4217, "population": 260000,  "drainage_density": 35, "rainfall_mm": 185, "elevation_m": 9,   "drainage_pct": 35, "flood_events_5yr": 3, "score": 32, "risk_class": "RED_ALERT"},
    {"id": 40, "name": "Baranagar",     "city": "Kolkata", "lat": 22.6511, "lng": 88.3974, "population": 200000,  "drainage_density": 40, "rainfall_mm": 175, "elevation_m": 10,  "drainage_pct": 40, "flood_events_5yr": 3, "score": 38, "risk_class": "RED_ALERT"},
]

FALLBACK_CITIES = [
    {"name": "Mumbai",  "state": "Maharashtra", "population": 20667656},
    {"name": "Pune",    "state": "Maharashtra", "population": 6629347},
    {"name": "Delhi",   "state": "NCT Delhi",   "population": 32941309},
    {"name": "Chennai", "state": "Tamil Nadu",  "population": 10971108},
    {"name": "Kolkata", "state": "West Bengal",  "population": 14850066},
]

# Realistic IMD-like monthly rainfall (mm) per city
FALLBACK_MONTHLY_RAINFALL = {
    "Mumbai":  [0.5, 1.2, 0.3, 1.8, 18.5, 520.4, 840.2, 610.5, 340.1, 65.3, 12.8, 2.1],
    "Pune":    [1.1, 0.8, 2.5, 12.4, 35.2, 135.6, 185.3, 120.8, 125.4, 75.2, 18.6, 4.5],
    "Delhi":   [18.2, 15.5, 12.8, 8.5, 22.1, 65.4, 210.5, 245.3, 120.6, 15.2, 4.8, 8.1],
    "Chennai": [25.8, 8.5, 5.2, 15.6, 52.3, 45.1, 80.5, 120.2, 118.4, 265.5, 350.2, 140.8],
    "Kolkata": [12.5, 22.8, 32.1, 48.5, 135.2, 280.4, 345.6, 325.1, 285.4, 145.3, 22.5, 5.8],
}

def _get_fallback_wards(city: str = None, risk_class: str = None,
                        min_score: int = None, max_score: int = None):
    """Return filtered copy of fallback ward data."""
    import copy
    wards = copy.deepcopy(FALLBACK_WARDS)
    now = datetime.datetime.utcnow().isoformat()
    for w in wards:
        w["computed_at"] = now
    if city:
        wards = [w for w in wards if w["city"] == city]
    if risk_class:
        wards = [w for w in wards if w["risk_class"] == risk_class]
    if min_score is not None:
        wards = [w for w in wards if w["score"] >= min_score]
    if max_score is not None:
        wards = [w for w in wards if w["score"] <= max_score]
    return wards

def _get_fallback_ward(ward_id: int):
    """Return a single fallback ward by ID."""
    for w in FALLBACK_WARDS:
        if w["id"] == ward_id:
            import copy
            ward = copy.deepcopy(w)
            ward["computed_at"] = datetime.datetime.utcnow().isoformat()
            return ward
    return None

# ─── Startup ────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Pre-warm weather cache."""
    try:
        await weather_svc.fetch_all_cities()
    except Exception:
        pass

# ─── Scoring Engine ──────────────────────────────────────────
def compute_flood_readiness_score(
    rainfall_mm: float,
    elevation_m: float,
    drainage_pct: float,
    flood_events: int,
    weights: dict = None
) -> dict:
    if weights is None:
        weights = {"rainfall": 0.40, "elevation": 0.30, "drainage": 0.20, "history": 0.10}

    MAX_RAIN, MIN_RAIN = 350.0, 50.0
    MAX_ELEV, MIN_ELEV = 700.0, 2.0
    MAX_DRAIN, MIN_DRAIN = 90.0, 10.0
    MAX_HIST, MIN_HIST = 8.0, 0.0

    rain_risk = (rainfall_mm - MIN_RAIN) / (MAX_RAIN - MIN_RAIN)
    elev_risk = 1.0 - (elevation_m - MIN_ELEV) / (MAX_ELEV - MIN_ELEV)
    drain_risk = 1.0 - (drainage_pct - MIN_DRAIN) / (MAX_DRAIN - MIN_DRAIN)
    hist_risk = (flood_events - MIN_HIST) / (MAX_HIST - MIN_HIST)

    risk_components = {
        "rainfall":  max(0.0, min(1.0, rain_risk)),
        "elevation": max(0.0, min(1.0, elev_risk)),
        "drainage":  max(0.0, min(1.0, drain_risk)),
        "history":   max(0.0, min(1.0, hist_risk)),
    }

    composite_risk = (
        risk_components["rainfall"]  * weights["rainfall"]  +
        risk_components["elevation"] * weights["elevation"] +
        risk_components["drainage"]  * weights["drainage"]  +
        risk_components["history"]   * weights["history"]
    )

    readiness = round((1.0 - composite_risk) * 100)
    readiness = max(5, min(95, readiness))

    if readiness <= 40:
        risk_class = "RED_ALERT"
    elif readiness <= 70:
        risk_class = "WATCH_ZONE"
    else:
        risk_class = "SAFE_ZONE"

    return {
        "score": readiness,
        "risk_class": risk_class,
        "components": risk_components,
        "composite_risk": round(composite_risk, 4),
    }

# ─── Routes ──────────────────────────────────────────────────

# ─── Authentication APIs (Citizen + Authority + RBAC) ───────
@app.post("/api/v1/auth/citizen/signup")
async def citizen_signup(payload: CitizenSignupRequest):
    username = payload.username.strip()
    password = payload.password.strip()
    gmail = payload.gmail.strip().lower()
    phone = payload.phone.strip()

    if not username or not password or not gmail or not phone:
        raise HTTPException(status_code=400, detail="username, password, gmail, and phone are required")
    if not (3 <= len(username) <= 30):
        raise HTTPException(status_code=400, detail="username must be 3-30 characters")
    if not gmail.endswith("@gmail.com"):
        raise HTTPException(status_code=400, detail="Only Gmail addresses are accepted")
    if not phone.isdigit() or len(phone) != 10:
        raise HTTPException(status_code=400, detail="phone must be exactly 10 digits")

    email_otp, sms_otp = _generate_otps()
    signup_id = secrets.token_hex(8)
    now = datetime.datetime.utcnow()
    expires_at = (now + datetime.timedelta(minutes=OTP_TTL_MINUTES)).isoformat()

    async with AUTH_LOCK:
        store = _read_auth_store()
        citizens = store.get("citizens", [])
        pending = store.get("pending_otps", [])
        if any(c["username"].lower() == username.lower() or c["gmail"] == gmail for c in citizens):
            raise HTTPException(status_code=409, detail="Citizen already exists")

        # Remove stale or duplicate pending requests for same user/email.
        filtered_pending = []
        for p in pending:
            if p.get("username", "").lower() == username.lower() or p.get("gmail") == gmail:
                continue
            filtered_pending.append(p)

        filtered_pending.append({
            "signup_id": signup_id,
            "username": username,
            "gmail": gmail,
            "phone": phone,
            "password_hash": _hash_password(password),
            "email_otp": email_otp,
            "sms_otp": sms_otp,
            "created_at": now.isoformat(),
            "expires_at": expires_at,
        })
        store["pending_otps"] = filtered_pending
        _write_auth_store(store)

    delivered_ok, email_msg, sms_msg = await _deliver_or_stub_otps(
        username=username,
        gmail=gmail,
        phone=phone,
        email_otp=email_otp,
        sms_otp=sms_otp,
    )
    if not delivered_ok:
        async with AUTH_LOCK:
            store = _read_auth_store()
            store["pending_otps"] = [p for p in store.get("pending_otps", []) if p.get("signup_id") != signup_id]
            _write_auth_store(store)
        raise HTTPException(
            status_code=502,
            detail=f"OTP delivery failed. email={email_msg}; sms={sms_msg}"
        )

    logger.info(f"[AUTH] OTP issued for citizen signup: {username} ({gmail}, {phone})")
    return {
        "message": "OTP sent to email and SMS channels",
        "signup_id": signup_id,
        "otp_channels": ["email", "sms"],
        "expires_at": expires_at,
    }

@app.post("/api/v1/auth/citizen/verify-otp")
async def citizen_verify_otp(payload: CitizenOtpVerifyRequest):
    signup_id = payload.signup_id.strip()
    email_otp = payload.email_otp.strip()
    sms_otp = payload.sms_otp.strip()
    now = datetime.datetime.utcnow()

    async with AUTH_LOCK:
        store = _read_auth_store()
        pending = store.get("pending_otps", [])
        match = next((p for p in pending if p.get("signup_id") == signup_id), None)
        if not match:
            raise HTTPException(status_code=404, detail="Pending signup not found")

        expires_at = datetime.datetime.fromisoformat(match["expires_at"])
        if expires_at < now:
            store["pending_otps"] = [p for p in pending if p.get("signup_id") != signup_id]
            _write_auth_store(store)
            raise HTTPException(status_code=400, detail="OTP expired. Please signup again")

        if match.get("email_otp") != email_otp or match.get("sms_otp") != sms_otp:
            raise HTTPException(status_code=400, detail="OTP verification failed")

        citizens = store.get("citizens", [])
        if any(c["username"].lower() == match["username"].lower() or c["gmail"] == match["gmail"] for c in citizens):
            store["pending_otps"] = [p for p in pending if p.get("signup_id") != signup_id]
            _write_auth_store(store)
            raise HTTPException(status_code=409, detail="Citizen already exists")

        citizens.append({
            "username": match["username"],
            "gmail": match["gmail"],
            "phone": match["phone"],
            "password_hash": match["password_hash"],
            "is_active": True,
            "created_at": now.isoformat()
        })
        store["citizens"] = citizens
        store["pending_otps"] = [p for p in pending if p.get("signup_id") != signup_id]
        _write_auth_store(store)

    return {"message": "Citizen account activated", "username": match["username"], "role": "citizen"}

@app.post("/api/v1/auth/citizen/login")
async def citizen_login(payload: CitizenLoginRequest):
    identifier = payload.identifier.strip().lower()
    password = payload.password.strip()
    if not identifier or not password:
        raise HTTPException(status_code=400, detail="identifier and password are required")

    async with AUTH_LOCK:
        store = _read_auth_store()
        citizens = store.get("citizens", [])
        user = next(
            (
                c for c in citizens
                if c.get("is_active") and (
                    c.get("username", "").lower() == identifier or c.get("gmail", "").lower() == identifier
                )
            ),
            None
        )
    if not user or not _verify_password(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid citizen credentials")
    _cleanup_expired_sessions()
    return _issue_session(role="citizen", subject=user["username"], display_name=user["username"].upper())

@app.post("/api/v1/auth/authority/login")
async def authority_login(payload: AuthorityLoginRequest):
    authority_id = payload.authority_id.strip().upper()
    password = payload.password.strip()
    if not authority_id or not password:
        raise HTTPException(status_code=400, detail="authority_id and password are required")

    async with AUTH_LOCK:
        store = _read_auth_store()
        authorities = store.get("authorities", {})
        authority = authorities.get(authority_id)
    if not authority:
        raise HTTPException(status_code=401, detail="Unknown authority ID")
    if not _verify_password(password, authority.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid authority credentials")
    _cleanup_expired_sessions()
    return _issue_session(role="authority", subject=authority_id, display_name=authority.get("name", authority_id))

@app.get("/api/v1/auth/session")
async def validate_session(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = credentials.credentials
    _cleanup_expired_sessions()
    session = ACTIVE_SESSIONS.get(token)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    role = session["role"]
    return {
        "role": role,
        "display_name": session["display_name"],
        "allowed_tabs": ["citizen"] if role == "citizen" else ["map", "analytics", "wards", "citizen"],
        "expires_at": session["expires_at"]
    }

@app.get("/api/v1/status")
async def root():
    return {
        "service": "VarshaMitra Flood Intelligence Platform",
        "version": "2.0.0",
        "status": "operational",
        "data_sources": ["Open-Meteo", "OpenTopoData SRTM30m", "OSM", "NDMA"],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# ─── Elevation Endpoints ───────────────────────────────────────

@app.get("/api/v1/elevation")
async def get_elevation(lat: float, lng: float):
    return await elevation_svc.get_elevation(lat, lng)

@app.post("/api/v1/elevation/batch")
async def get_elevation_batch(locations: List[dict]):
    coords = [(float(loc["lat"]), float(loc["lng"])) for loc in locations[:100]]
    return await elevation_svc.get_elevation_batch(coords)

@app.get("/api/v1/wards/{ward_id}/elevation")
async def get_ward_elevation(ward_id: int):
    # Try DB first, fall back to in-memory
    ward = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT w.name, w.city,
                   ST_Y(w.centroid::geometry) as lat,
                   ST_X(w.centroid::geometry) as lng
            FROM wards w WHERE w.id = $1
        """
        row = await conn.fetchrow(query, ward_id)
        await conn.close()
        if row:
            ward = dict(row)
    except Exception:
        pass

    if not ward:
        fb = _get_fallback_ward(ward_id)
        if not fb:
            raise HTTPException(status_code=404, detail="Ward not found")
        ward = {"name": fb["name"], "city": fb["city"], "lat": fb["lat"], "lng": fb["lng"]}

    elev_data = await elevation_svc.get_elevation(float(ward["lat"]), float(ward["lng"]))
    return {
        "ward_id": ward_id,
        "ward_name": ward["name"],
        "city": ward["city"],
        "elevation_m": elev_data["elevation_m"],
        "low_lying_pct": elev_data["low_lying_pct"],
        "min_elevation": round(elev_data["elevation_m"] * 0.6, 2),
        "source": elev_data["source"],
        "dataset": "srtm30m (OpenTopoData)",
        "cache_stats": elevation_svc.get_cache_stats(),
    }

@app.get("/api/v1/elevation/cache-stats")
async def elevation_cache_stats():
    return elevation_svc.get_cache_stats()


@app.get("/api/wards", response_model=List[dict])
@app.get("/api/v1/wards", response_model=List[dict])
async def get_all_wards(
    city: Optional[str] = None,
    risk_class: Optional[str] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
):
    """Get all ward flood scores with optional filters, using live rainfall and live drainage."""
    live_weather = await weather_svc.fetch_all_cities()
    drainage_by_city = {}

    # Try DB first
    wards = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT w.id, w.name, w.city, ST_Y(w.centroid::geometry) as lat,
                   ST_X(w.centroid::geometry) as lng, w.population, w.drainage_density,
                   fs.score, fs.risk_class, fs.rainfall_mm, fs.elevation_m,
                   fs.drainage_pct, fs.flood_events_5yr, fs.computed_at
            FROM wards w
            JOIN flood_scores fs ON w.id = fs.ward_id
            WHERE fs.computed_at = (SELECT MAX(computed_at) FROM flood_scores WHERE ward_id = w.id)
        """
        params = []
        if city:
            query += f" AND w.city = ${len(params)+1}"; params.append(city)
        if risk_class:
            query += f" AND fs.risk_class = ${len(params)+1}"; params.append(risk_class)
        if min_score is not None:
            query += f" AND fs.score >= ${len(params)+1}"; params.append(min_score)
        if max_score is not None:
            query += f" AND fs.score <= ${len(params)+1}"; params.append(max_score)
        query += " ORDER BY fs.score ASC"
        rows = await conn.fetch(query, *params)
        await conn.close()
        wards = [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(f"[API] /api/v1/wards DB unavailable: {exc}. Using fallback data.")

    # Fall back to in-memory data
    if wards is None:
        wards = _get_fallback_wards(city=city, risk_class=None, min_score=min_score, max_score=max_score)

    # Inject live weather + recompute scores
    for w in wards:
        city_name = w.get("city", "")
        if city_name and city_name not in drainage_by_city:
            try:
                drainage_by_city[city_name] = await drainage_svc.fetch_drainage_data(city_name)
            except Exception:
                drainage_by_city[city_name] = {}

        drainage_data = drainage_by_city.get(city_name, {})
        live_drainage = drainage_data.get("drainage_score", w.get("drainage_pct", 50))
        w["drainage_pct"] = round(float(live_drainage), 2)
        w["drainage_density"] = round(float(drainage_data.get("drainage_density", live_drainage)), 2)
        w["drainage_source"] = drainage_data.get("source", "fallback")
        w["drainage_updated_at"] = drainage_data.get("updated_at")

        if city_name in live_weather:
            live_rain = live_weather[city_name]["rainfall_mm"]
            w["rainfall_mm"] = live_rain
            w["weather_source"] = live_weather[city_name]["source"]
            w["weather_updated_at"] = live_weather[city_name]["updated_at"]
            flood_events = int(w.get("flood_events_5yr", w.get("flood_events", 0)))
            w["flood_events"] = flood_events

            rescored = compute_flood_readiness_score(
                rainfall_mm=float(live_rain),
                elevation_m=float(w.get("elevation_m", 50)),
                drainage_pct=float(live_drainage),
                flood_events=flood_events,
            )
            w["score"] = rescored["score"]
            w["risk_class"] = rescored["risk_class"]

    if risk_class:
        wards = [w for w in wards if w["risk_class"] == risk_class]

    return sorted(wards, key=lambda w: w["score"])


@app.get("/api/v1/drainage/network")
async def get_drainage_network(city: str):
    data = await drainage_svc.fetch_drainage_data(city)
    return {
        "city": data["city"],
        "drainage_score": data["drainage_score"],
        "drainage_density": data["drainage_density"],
        "total_length_m": data["total_length_m"],
        "line_count": data["line_count"],
        "source": data["source"],
        "updated_at": data["updated_at"],
        "geojson": data["geojson"],
    }

@app.get("/api/v1/wards/{ward_id}")
async def get_ward(ward_id: int):
    """Get detailed flood data for a specific ward (DB or fallback)."""
    ward = None
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT w.id, w.name, w.city, ST_Y(w.centroid::geometry) as lat,
                   ST_X(w.centroid::geometry) as lng, w.population, w.drainage_density,
                   fs.score, fs.risk_class, fs.rainfall_mm, fs.elevation_m,
                   fs.drainage_pct, fs.flood_events_5yr, fs.computed_at
            FROM wards w
            JOIN flood_scores fs ON w.id = fs.ward_id
            WHERE fs.computed_at = (SELECT MAX(computed_at) FROM flood_scores WHERE ward_id = w.id)
              AND w.id = $1
        """
        row = await conn.fetchrow(query, ward_id)
        await conn.close()
        if row:
            ward = dict(row)
    except Exception as exc:
        logger.warning(f"[API] /api/v1/wards/{ward_id} DB unavailable: {exc}. Using fallback.")

    if not ward:
        ward = _get_fallback_ward(ward_id)
        if not ward:
            raise HTTPException(status_code=404, detail="Ward not found")

    city_name = ward.get("city", "")
    if city_name:
        live_rain_data = await weather_svc.fetch_city_weather(city_name)
        try:
            live_drainage_data = await drainage_svc.fetch_drainage_data(city_name)
        except Exception:
            live_drainage_data = {}

        flood_events = int(ward.get("flood_events_5yr", ward.get("flood_events", 0)))
        ward["flood_events"] = flood_events

        live_drainage = live_drainage_data.get("drainage_score", ward.get("drainage_pct", 50))
        ward["drainage_pct"] = round(float(live_drainage), 2)
        ward["drainage_density"] = round(float(live_drainage_data.get("drainage_density", live_drainage)), 2)
        ward["drainage_source"] = live_drainage_data.get("source", "fallback")

        rescored = compute_flood_readiness_score(
            rainfall_mm=float(live_rain_data.get("rainfall_mm", ward.get("rainfall_mm", 150))),
            elevation_m=float(ward.get("elevation_m", 50)),
            drainage_pct=float(live_drainage),
            flood_events=flood_events,
        )

        ward["rainfall_mm"] = live_rain_data.get("rainfall_mm", ward.get("rainfall_mm", 150))
        ward["weather_source"] = live_rain_data.get("source", None)
        ward["weather_updated_at"] = live_rain_data.get("updated_at", None)
        ward["score"] = rescored["score"]
        ward["risk_class"] = rescored["risk_class"]

    return ward

@app.post("/api/v1/wards/{ward_id}/score")
async def compute_ward_score(ward_id: int, background_tasks: BackgroundTasks):
    return {"status": "queued", "ward_id": ward_id, "task": "score_computation"}

@app.post("/api/v1/score/compute")
async def compute_score_endpoint(req: MLPredictionRequest):
    result = compute_flood_readiness_score(
        rainfall_mm=req.rainfall_mm,
        elevation_m=req.elevation_m,
        drainage_pct=req.drainage_pct,
        flood_events=req.flood_events
    )
    return {
        "ward_id": req.ward_id,
        **result,
        "computed_at": datetime.datetime.utcnow().isoformat()
    }

@app.get("/api/v1/wards/{ward_id}/report")
async def generate_ai_report(ward_id: int):
    import httpx
    ward = await get_ward(ward_id)
    prompt = f"""You are a flood risk analyst for the Indian government (NDMA).
Generate a concise professional flood risk assessment report for:

Ward: {ward['name']}, {ward['city']}
Flood Readiness Score: {ward['score']}/100 ({ward['risk_class']})
Rainfall: {ward['rainfall_mm']}mm
Elevation: {ward['elevation_m']}m
Drainage Capacity: {ward['drainage_pct']}%
Historical Flood Events (5yr): {ward.get('flood_events', 0)}

Provide:
1. Risk summary (2 sentences)
2. Key vulnerabilities (3 bullet points)
3. Recommended actions (3 bullet points)
4. 72-hour outlook

Keep it under 200 words. Professional tone."""

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": "llama3.2", "prompt": prompt, "stream": False}
            )
            data = response.json()
            return {"ward_id": ward_id, "report": data.get("response", ""), "model": "llama3.2"}
    except Exception:
        return {
            "ward_id": ward_id,
            "report": _fallback_report(ward),
            "model": "rule-based-fallback"
        }

@app.post("/api/v1/drain-reports")
async def submit_drain_report(report: DrainReport, background_tasks: BackgroundTasks):
    ticket_id = f"DR{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    return {
        "status": "submitted",
        "ticket_id": ticket_id,
        "message": f"Report submitted. Ticket: {ticket_id}",
        "ward": report.ward_name,
        "estimated_response": "24-48 hours"
    }

@app.post("/api/v1/alerts")
async def create_alert(alert: AlertPayload):
    return {
        "status": "sent",
        "ward_id": alert.ward_id,
        "severity": alert.severity,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "recipients": 0,
        "channels": alert.channels
    }

@app.get("/api/v1/alerts/active")
async def get_active_alerts():
    # Try DB first
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT fa.id, w.name AS ward, w.city AS city,
                   fa.severity, fa.title, fa.message,
                   fa.issued_at, fa.expires_at
            FROM flood_alerts fa
            JOIN wards w ON w.id = fa.ward_id
            WHERE fa.active = TRUE
            ORDER BY fa.issued_at DESC LIMIT 20
        """
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(r) for r in rows]
    except Exception:
        # No active alerts if DB is unavailable
        return []

@app.get("/api/v1/weather/live")
async def get_live_weather():
    return await weather_svc.fetch_all_cities()

@app.get("/api/v1/alerts/auto")
async def get_auto_alerts():
    """Auto-generate flood alerts based on live-rescored ward data."""
    # Use the same live-rescored ward data as the map markers
    wards = await get_all_wards()
    auto_alerts = []
    alert_id = 100
    now_iso = datetime.datetime.utcnow().isoformat() + "Z"

    for ward in wards:
        score = ward.get("score", 50)
        rc = ward.get("risk_class", "SAFE_ZONE")
        city_name = ward.get("city", "")
        live_rain = ward.get("rainfall_mm", 0)
        weather_src = ward.get("weather_source", "fallback")

        if rc == "RED_ALERT":
            severity = "CRITICAL"
        elif rc == "WATCH_ZONE":
            severity = "WARNING"
        else:
            continue

        auto_alerts.append({
            "id": alert_id,
            "ward": ward.get("name"),
            "city": city_name,
            "severity": severity,
            "score": score,
            "risk_class": rc,
            "rainfall_mm": live_rain,
            "weather_source": weather_src,
            "message": (
                f"{'⛔ CRITICAL' if severity == 'CRITICAL' else '⚠ WARNING'}: "
                f"{ward.get('name')}, {city_name} — Score {score}/100. "
                f"Live rainfall: {live_rain:.1f}mm."
            ),
            "generated_at": now_iso,
        })
        alert_id += 1

    return sorted(auto_alerts, key=lambda a: a["score"])

@app.get("/api/v1/analytics/summary")
async def analytics_summary():
    """Platform-wide analytics summary."""
    # Try DB first
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT COUNT(*) AS total_wards, COUNT(DISTINCT city) AS cities,
                   COUNT(*) FILTER (WHERE risk_class = 'RED_ALERT') AS red_alert,
                   COUNT(*) FILTER (WHERE risk_class = 'WATCH_ZONE') AS watch_zone,
                   COUNT(*) FILTER (WHERE risk_class = 'SAFE_ZONE') AS safe_zone,
                   ROUND(AVG(score)::numeric, 1) AS avg_readiness_score,
                   MAX(rainfall_mm) AS peak_rainfall_mm,
                   COUNT(*) FILTER (WHERE drainage_pct < 30) AS drainage_alerts,
                   SUM(population) AS total_population
            FROM latest_ward_scores
        """
        row = await conn.fetchrow(query)
        await conn.close()
        return dict(row)
    except Exception:
        pass

    # Fallback: derive from live-rescored data (same as /api/v1/wards)
    # This ensures risk summary counts match the map markers
    wards = await get_all_wards()
    cities = set(w.get("city", "") for w in wards)
    red = sum(1 for w in wards if w.get("risk_class") == "RED_ALERT")
    watch = sum(1 for w in wards if w.get("risk_class") == "WATCH_ZONE")
    safe = sum(1 for w in wards if w.get("risk_class") == "SAFE_ZONE")
    avg_score = round(sum(w.get("score", 0) for w in wards) / len(wards), 1) if wards else 0
    peak_rain = max((w.get("rainfall_mm", 0) for w in wards), default=0)
    drain_alerts = sum(1 for w in wards if w.get("drainage_pct", 50) < 30)
    total_pop = sum(w.get("population", 0) for w in wards)

    return {
        "total_wards": len(wards),
        "cities": len(cities),
        "red_alert": red,
        "watch_zone": watch,
        "safe_zone": safe,
        "avg_readiness_score": avg_score,
        "peak_rainfall_mm": peak_rain,
        "drainage_alerts": drain_alerts,
        "total_population": total_pop,
        "by_city": [
            {
                "city": c,
                "total_wards": sum(1 for w in wards if w.get("city") == c),
                "red_alert": sum(1 for w in wards if w.get("city") == c and w.get("risk_class") == "RED_ALERT"),
                "watch_zone": sum(1 for w in wards if w.get("city") == c and w.get("risk_class") == "WATCH_ZONE"),
                "safe_zone": sum(1 for w in wards if w.get("city") == c and w.get("risk_class") == "SAFE_ZONE"),
                "avg_score": round(sum(w.get("score", 0) for w in wards if w.get("city") == c) / max(1, sum(1 for w in wards if w.get("city") == c)), 1),
                "avg_rainfall_mm": round(sum(w.get("rainfall_mm", 0) for w in wards if w.get("city") == c) / max(1, sum(1 for w in wards if w.get("city") == c)), 1),
            }
            for c in sorted(cities)
        ],
        "source": "live-rescored",
    }

@app.get("/api/v1/analytics/rainfall")
async def rainfall_analytics(city: Optional[str] = None):
    """Monthly rainfall totals for charting."""
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    # Try DB first
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        if city:
            query = """
                SELECT rs.city AS city, EXTRACT(MONTH FROM ro.observed_at)::int AS month,
                       SUM(ro.rainfall_mm)::float AS rainfall_mm
                FROM rainfall_observations ro
                JOIN rainfall_stations rs ON rs.id = ro.station_id
                WHERE rs.city = $1
                GROUP BY rs.city, month ORDER BY month ASC
            """
            rows = await conn.fetch(query, city)
        else:
            query = """
                SELECT rs.city AS city, EXTRACT(MONTH FROM ro.observed_at)::int AS month,
                       SUM(ro.rainfall_mm)::float AS rainfall_mm
                FROM rainfall_observations ro
                JOIN rainfall_stations rs ON rs.id = ro.station_id
                GROUP BY rs.city, month ORDER BY rs.city ASC, month ASC
            """
            rows = await conn.fetch(query)
        await conn.close()

        cities_found = sorted({r["city"] for r in rows})
        data = {c: [0] * 12 for c in cities_found}
        for r in rows:
            data[r["city"]][int(r["month"]) - 1] = round(float(r["rainfall_mm"]), 2)
        return {"months": months, "data": data}
    except Exception:
        pass

    # Fallback: use static IMD-like data
    if city and city in FALLBACK_MONTHLY_RAINFALL:
        return {"months": months, "data": {city: FALLBACK_MONTHLY_RAINFALL[city]}}
    return {"months": months, "data": FALLBACK_MONTHLY_RAINFALL}

@app.get("/api/v1/cities")
async def list_cities():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT c.name, c.state, c.population,
                   COUNT(lws.id) AS total_wards,
                   COUNT(lws.id) FILTER (WHERE lws.risk_class = 'RED_ALERT') AS red_alert,
                   ROUND(AVG(lws.score)::numeric, 1) AS avg_score,
                   ROUND(AVG(lws.rainfall_mm)::numeric, 1) AS avg_rainfall_mm
            FROM cities c
            LEFT JOIN latest_ward_scores lws ON lws.city = c.name
            GROUP BY c.name, c.state, c.population
            ORDER BY c.name ASC
        """
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(r) for r in rows]
    except Exception:
        # Fallback
        result = []
        for c in FALLBACK_CITIES:
            city_wards = [w for w in FALLBACK_WARDS if w["city"] == c["name"]]
            result.append({
                "name": c["name"],
                "state": c["state"],
                "population": c["population"],
                "total_wards": len(city_wards),
                "red_alert": sum(1 for w in city_wards if w["risk_class"] == "RED_ALERT"),
                "avg_score": round(sum(w["score"] for w in city_wards) / max(1, len(city_wards)), 1),
                "avg_rainfall_mm": round(sum(w["rainfall_mm"] for w in city_wards) / max(1, len(city_wards)), 1),
            })
        return result

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

# ─── Helpers ─────────────────────────────────────────────────
def _fallback_report(ward: dict) -> str:
    return (
        f"Ward {ward['name']} ({ward['city']}) has a Flood Readiness Score of "
        f"{ward['score']}/100, classified as {ward['risk_class']}. "
        f"Rainfall intensity of {ward['rainfall_mm']}mm exceeds safe thresholds. "
        f"Terrain elevation of {ward['elevation_m']}m indicates high vulnerability. "
        f"Drainage capacity at {ward['drainage_pct']}% requires immediate attention. "
        f"Recommended: Deploy emergency pumps and issue evacuation advisory."
    )
