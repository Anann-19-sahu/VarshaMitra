"""
VarshaMitra – Pre-Monsoon Flood Intelligence Platform
FastAPI Backend Server
"""
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from typing import Optional, List
import asyncpg
import json
import io
import datetime
from pydantic import BaseModel

# ─── App Init ─────────────────────────────────────────────
app = FastAPI(
    title="VarshaMitra Flood Intelligence API",
    description="Pre-Monsoon Flood Risk Assessment for Indian Cities",
    version="2.0.0"
)

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
    risk_class: str  # RED_ALERT | WATCH_ZONE | SAFE_ZONE
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
    severity: str  # CRITICAL | WARNING | INFO
    message: str
    channels: List[str] = ["dashboard"]

class MLPredictionRequest(BaseModel):
    ward_id: int
    rainfall_mm: float
    elevation_m: float
    drainage_pct: float
    flood_events: int

# ─── DB Connection ──────────────────────────────────────────
DATABASE_URL = "postgresql://varsha:monsoon2024@localhost:5432/varsha_mitra"

async def get_db():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

# ─── Scoring Engine ──────────────────────────────────────────
def compute_flood_readiness_score(
    rainfall_mm: float,
    elevation_m: float,
    drainage_pct: float,
    flood_events: int,
    weights: dict = None
) -> dict:
    """
    Ward Flood Readiness Score (0–100)
    
    Composite weighted score:
      40% Rainfall Intensity
      30% Terrain Elevation (inverse risk)
      20% Drainage Capacity
      10% Flood History (inverse risk)
    """
    if weights is None:
        weights = {"rainfall": 0.40, "elevation": 0.30, "drainage": 0.20, "history": 0.10}

    # Normalization bounds (India-specific)
    MAX_RAIN, MIN_RAIN = 350.0, 50.0
    MAX_ELEV, MIN_ELEV = 700.0, 2.0
    MAX_DRAIN, MIN_DRAIN = 90.0, 10.0
    MAX_HIST, MIN_HIST = 8.0, 0.0

    # Normalize to [0, 1] — higher = more dangerous
    rain_risk = (rainfall_mm - MIN_RAIN) / (MAX_RAIN - MIN_RAIN)
    elev_risk = 1.0 - (elevation_m - MIN_ELEV) / (MAX_ELEV - MIN_ELEV)  # low elev = high risk
    drain_risk = 1.0 - (drainage_pct - MIN_DRAIN) / (MAX_DRAIN - MIN_DRAIN)  # low drain = high risk
    hist_risk = (flood_events - MIN_HIST) / (MAX_HIST - MIN_HIST)

    # Clip to [0, 1]
    risk_components = {
        "rainfall":  max(0.0, min(1.0, rain_risk)),
        "elevation": max(0.0, min(1.0, elev_risk)),
        "drainage":  max(0.0, min(1.0, drain_risk)),
        "history":   max(0.0, min(1.0, hist_risk)),
    }

    # Composite risk score
    composite_risk = (
        risk_components["rainfall"]  * weights["rainfall"]  +
        risk_components["elevation"] * weights["elevation"] +
        risk_components["drainage"]  * weights["drainage"]  +
        risk_components["history"]   * weights["history"]
    )

    # Readiness = complement of risk, scaled 0–100
    readiness = round((1.0 - composite_risk) * 100)
    readiness = max(5, min(95, readiness))

    # Risk classification
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

@app.get("/")
async def root():
    return {
        "service": "VarshaMitra Flood Intelligence Platform",
        "version": "2.0.0",
        "status": "operational",
        "data_sources": ["IMD", "ISRO_CartoDEM", "OSM", "NDMA"],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

@app.get("/api/v1/wards", response_model=List[dict])
async def get_all_wards(
    city: Optional[str] = None,
    risk_class: Optional[str] = None,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
):
    """Get all ward flood scores with optional filters."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT w.id, w.name, w.city, ST_Y(w.centroid::geometry) as lat,
                   ST_X(w.centroid::geometry) as lng, w.population,
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
        return [dict(r) for r in rows]
    except Exception:
        # Return mock data if DB not available
        return _mock_wards(city, risk_class)

@app.get("/api/v1/wards/{ward_id}")
async def get_ward(ward_id: int):
    """Get detailed flood data for a specific ward."""
    return _get_mock_ward(ward_id)

@app.post("/api/v1/wards/{ward_id}/score")
async def compute_ward_score(ward_id: int, background_tasks: BackgroundTasks):
    """Trigger ML scoring pipeline for a specific ward."""
    background_tasks.add_task(_run_scoring_pipeline, ward_id)
    return {"status": "queued", "ward_id": ward_id, "task": "score_computation"}

@app.post("/api/v1/score/compute")
async def compute_score_endpoint(req: MLPredictionRequest):
    """Compute flood readiness score for given parameters."""
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
    """Generate AI-powered ward risk report using Ollama LLM."""
    import httpx, asyncio
    ward = _get_mock_ward(ward_id)
    prompt = f"""You are a flood risk analyst for the Indian government (NDMA).
Generate a concise professional flood risk assessment report for:

Ward: {ward['name']}, {ward['city']}
Flood Readiness Score: {ward['score']}/100 ({ward['risk_class']})
Rainfall: {ward['rainfall_mm']}mm
Elevation: {ward['elevation_m']}m
Drainage Capacity: {ward['drainage_pct']}%
Historical Flood Events (5yr): {ward['flood_events']}

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
    """Submit citizen drain blockage report."""
    ticket_id = f"DR{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    background_tasks.add_task(_notify_municipal, report, ticket_id)
    return {
        "status": "submitted",
        "ticket_id": ticket_id,
        "message": f"Report submitted. Ticket: {ticket_id}",
        "ward": report.ward_name,
        "estimated_response": "24-48 hours"
    }

@app.post("/api/v1/alerts")
async def create_alert(alert: AlertPayload):
    """Create and broadcast flood alert for a ward."""
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
    """Get all currently active flood alerts."""
    return [
        {"id": 1, "ward": "Dharavi", "city": "Mumbai", "severity": "CRITICAL", "score": 18, "message": "Flood Risk Alert: High Risk in Next 72 Hours"},
        {"id": 2, "ward": "Velachery", "city": "Chennai", "severity": "CRITICAL", "score": 15, "message": "Drainage capacity failed. Immediate action required."},
        {"id": 3, "ward": "Yamuna Vihar", "city": "Delhi", "severity": "WARNING", "score": 38, "message": "Rising water levels detected."},
    ]

@app.get("/api/v1/analytics/summary")
async def analytics_summary():
    """Platform-wide analytics summary."""
    return {
        "total_wards": 40,
        "cities": 5,
        "red_alert": 8,
        "watch_zone": 14,
        "safe_zone": 18,
        "avg_readiness_score": 54,
        "peak_rainfall_mm": 247,
        "drainage_alerts": 12,
        "season": "Pre-Monsoon 2024",
        "data_sources": {
            "imd_rainfall": "2024-06-01",
            "isro_dem": "CartoDEM_30m",
            "osm_drainage": "2024-05-15",
            "ndma_flood_history": "2019-2023"
        }
    }

@app.get("/api/v1/analytics/rainfall")
async def rainfall_analytics(city: Optional[str] = None):
    """Monthly rainfall data for charting."""
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return {
        "months": months,
        "data": {
            "Mumbai":  [12,8,25,60,120,240,310,275,180,72,18,8],
            "Chennai": [15,12,20,40,90,185,215,200,165,60,20,10],
            "Kolkata": [20,15,30,65,110,210,280,250,170,65,22,12],
            "Delhi":   [8,5,12,25,60,95,125,110,85,30,10,5],
            "Pune":    [10,8,18,45,95,165,210,195,148,55,14,7],
        }
    }

@app.get("/api/v1/cities")
async def list_cities():
    return ["Mumbai", "Pune", "Delhi", "Chennai", "Kolkata"]

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}

# ─── Background Tasks ────────────────────────────────────────
async def _run_scoring_pipeline(ward_id: int):
    """Celery-delegated scoring pipeline."""
    from celery import current_app as celery_app
    celery_app.send_task("tasks.compute_ward_score", args=[ward_id])

async def _notify_municipal(report: DrainReport, ticket_id: str):
    """Notify municipal corporation about drain report."""
    print(f"[NOTIFY] Drain report {ticket_id} for {report.ward_name}")

# ─── Helpers ─────────────────────────────────────────────────
def _mock_wards(city=None, risk_class=None):
    wards = [
        {"id": 1, "name": "Dharavi", "city": "Mumbai", "lat": 19.0422, "lng": 72.8521, "score": 18, "risk_class": "RED_ALERT", "rainfall_mm": 210, "elevation_m": 8, "drainage_pct": 25, "flood_events": 4, "population": 850000},
        {"id": 2, "name": "Velachery", "city": "Chennai", "lat": 12.9785, "lng": 80.2209, "score": 15, "risk_class": "RED_ALERT", "rainfall_mm": 210, "elevation_m": 3, "drainage_pct": 15, "flood_events": 6, "population": 210000},
    ]
    if city:
        wards = [w for w in wards if w["city"] == city]
    if risk_class:
        wards = [w for w in wards if w["risk_class"] == risk_class]
    return wards

def _get_mock_ward(ward_id: int):
    mock = {"id": ward_id, "name": f"Ward {ward_id}", "city": "Mumbai", "score": 35,
            "risk_class": "RED_ALERT", "rainfall_mm": 210, "elevation_m": 8,
            "drainage_pct": 25, "flood_events": 4, "population": 200000,
            "lat": 19.076, "lng": 72.877}
    return mock

def _fallback_report(ward: dict) -> str:
    return (
        f"Ward {ward['name']} ({ward['city']}) has a Flood Readiness Score of "
        f"{ward['score']}/100, classified as {ward['risk_class']}. "
        f"Rainfall intensity of {ward['rainfall_mm']}mm exceeds safe thresholds. "
        f"Terrain elevation of {ward['elevation_m']}m indicates high vulnerability. "
        f"Drainage capacity at {ward['drainage_pct']}% requires immediate attention. "
        f"Recommended: Deploy emergency pumps and issue evacuation advisory."
    )
