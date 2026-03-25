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
from typing import Optional, List
import asyncpg
import json
import io
import datetime
import asyncio
from pydantic import BaseModel
from weather_service import WeatherService
from elevation_service import ElevationService
from drainage_service import DrainageService
from disaster_service import DisasterService

# ─── App Init ─────────────────────────────────────────────
app = FastAPI(
    title="VarshaMitra Flood Intelligence API",
    description="Pre-Monsoon Flood Risk Assessment for Indian Cities",
    version="2.0.0"
)

# ─── Weather Service (singleton, shared across requests) ──────
weather_svc   = WeatherService()
elevation_svc = ElevationService()
drainage_svc  = DrainageService()
disaster_svc  = DisasterService()

@app.on_event("startup")
async def startup_event():
    """Pre-warm weather cache and elevation cache for all ward centroids."""
    try:
        await weather_svc.fetch_all_cities()
    except Exception:
        pass
    # Pre-warm elevation for all 40 ward centroids in one batch call
    try:
        ward_coords = [
            (19.0422,72.8521),(19.0726,72.8794),(19.0446,72.8612),(19.0623,72.9007),
            (19.1879,72.8484),(19.1059,72.9276),(19.0544,72.8402),(19.0860,72.9081),
            (18.5188,73.8567),(18.5590,73.9040),(18.5074,73.8077),(18.4689,73.8614),
            (18.5089,73.9260),(18.6298,73.7997),(18.4900,73.8900),(18.5308,73.8475),
            (28.7041,77.2993),(28.7381,77.2712),(28.7534,77.1859),(28.6716,77.2952),
            (28.7495,77.0639),(28.5921,77.0460),(28.6244,77.3048),(28.6271,77.2961),
            (13.1651,80.2650),(13.1227,80.2889),(13.1190,80.2470),(12.9785,80.2209),
            (12.9249,80.1000),(13.0011,80.2565),(13.0530,80.2209),(13.0857,80.2101),
            (22.5382,88.3980),(22.5507,88.3888),(22.4942,88.3019),(22.4982,88.3728),
            (22.5847,88.3990),(22.5958,88.4005),(22.5803,88.4217),(22.6511,88.3974),
        ]
        await elevation_svc.get_elevation_batch(ward_coords)
        logger.info(f"[Startup] Elevation cache pre-warmed for {len(ward_coords)} ward centroids")
    except Exception as exc:
        logger.warning(f"[Startup] Elevation pre-warm failed: {exc}")

import logging
logger = logging.getLogger("varsha_mitra")

# ─── Serve Frontend Static Files ───────────────────────────
# This lets you open http://localhost:8000 instead of file:// URLs
# which avoids all CORS/fetch errors in the browser.
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

@app.get("/api/v1/status")
async def root():
    return {
        "service": "VarshaMitra Flood Intelligence Platform",
        "version": "2.0.0",
        "status": "operational",
        "data_sources": ["Open-Meteo", "OpenTopoData SRTM30m", "OSM", "NDMA"],
        "timestamp": datetime.datetime.utcnow().isoformat()
    }

# ─── Elevation Endpoints (OpenTopoData SRTM30m) ───────────────

@app.get("/api/v1/elevation")
async def get_elevation(lat: float, lng: float):
    """
    Fetch SRTM30m terrain elevation for any lat/lng coordinate.
    Powered by OpenTopoData (free, no API key). Cached 24h.
    Example: /api/v1/elevation?lat=19.0422&lng=72.8521
    """
    return await elevation_svc.get_elevation(lat, lng)

@app.post("/api/v1/elevation/batch")
async def get_elevation_batch(locations: List[dict]):
    """
    Batch elevation lookup for up to 100 coordinates.
    Body: [{"lat": float, "lng": float}, ...]
    """
    coords = [(float(loc["lat"]), float(loc["lng"])) for loc in locations[:100]]
    return await elevation_svc.get_elevation_batch(coords)

@app.get("/api/v1/wards/{ward_id}/elevation")
async def get_ward_elevation(ward_id: int):
    """
    Live SRTM30m elevation + ML features for a ward centroid.
    Replaces local GeoTIFF/rasterio processing with OpenTopoData API.
    """
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT w.name, w.city,
                   ST_Y(w.centroid::geometry) as lat,
                   ST_X(w.centroid::geometry) as lng
            FROM wards w
            WHERE w.id = $1
        """
        row = await conn.fetchrow(query, ward_id)
        await conn.close()
        if not row:
            raise HTTPException(status_code=404, detail="Ward not found")
        ward = dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        # Don't hard-fail the UI if Postgres is unavailable; frontend can fall back to cached data.
        logger.warning(f"[API] /api/wards DB query failed: {exc}")
        return []

    elev_data = await elevation_svc.get_elevation(float(ward["lat"]), float(ward["lng"]))
    elevation_m   = elev_data["elevation_m"]
    low_lying_pct = elev_data["low_lying_pct"]
    return {
        "ward_id":       ward_id,
        "ward_name":     ward["name"],
        "city":          ward["city"],
        "elevation_m":   elevation_m,
        "low_lying_pct": low_lying_pct,
        "min_elevation": round(elevation_m * 0.6, 2),
        "source":        elev_data["source"],
        "dataset":       "srtm30m (OpenTopoData)",
        "cache_stats":   elevation_svc.get_cache_stats(),
    }

@app.get("/api/v1/elevation/cache-stats")
async def elevation_cache_stats():
    """Return elevation cache statistics."""
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
    # Fetch live weather for all cities (uses cache if fresh)
    live_weather = await weather_svc.fetch_all_cities()
    drainage_by_city = {}

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
        # Keep UI usable if Postgres is down; frontend can fall back to cached/static data.
        logger.warning(f"[API] /api/v1/wards DB query failed: {exc}")
        return []

    # Fetch flood history counts from ReliefWeb once per city for this request
    unique_cities = sorted({w.get("city") for w in wards if w.get("city")})
    flood_by_city: dict[str, dict] = {}
    if unique_cities:
        relief_results = await asyncio.gather(
            *[disaster_svc.fetch_reliefweb_history(c) for c in unique_cities],
            return_exceptions=True,
        )
        for c, res in zip(unique_cities, relief_results):
            if isinstance(res, Exception):
                flood_by_city[c] = {"flood_events_5yr": int(0)}
            else:
                flood_by_city[c] = res

    # ── Inject live rainfall + recompute scores ────────────────
    for w in wards:
        city_name = w.get("city", "")
        if city_name and city_name not in drainage_by_city:
            drainage_by_city[city_name] = await drainage_svc.fetch_drainage_data(city_name)

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

            flood_events = int(
                flood_by_city.get(city_name, {}).get(
                    "flood_events_5yr",
                    w.get("flood_events_5yr", 0),
                )
            )
            w["flood_events"] = flood_events

            # Recompute score with live rainfall
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
    """
    Return city-level OSM drainage network as GeoJSON for map overlays.
    """
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
    """
    Get detailed flood data for a specific ward.
    Uses Postgres baseline + live Weather + live Drainage + live ReliefWeb flood counts.
    """
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
        if not row:
            raise HTTPException(status_code=404, detail="Ward not found")
        ward = dict(row)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

    city_name = ward.get("city", "")
    if city_name:
        live_rain_data = await weather_svc.fetch_city_weather(city_name)
        live_drainage_data = await drainage_svc.fetch_drainage_data(city_name)
        flood_data = await disaster_svc.fetch_reliefweb_history(city_name)

        flood_events = int(flood_data.get("flood_events_5yr", ward.get("flood_events_5yr", 0)))
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
    ward = await get_ward(ward_id)
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
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT
                fa.id,
                w.name AS ward,
                w.city AS city,
                fa.severity,
                lws.score,
                fa.message
            FROM flood_alerts fa
            JOIN wards w ON w.id = fa.ward_id
            LEFT JOIN latest_ward_scores lws ON lws.id = w.id
            WHERE fa.active = TRUE
            ORDER BY fa.issued_at DESC
            LIMIT 20
        """
        rows = await conn.fetch(query)
        await conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning(f"[API] /api/v1/alerts/active DB query failed: {exc}")
        return []

@app.get("/api/v1/weather/live")
async def get_live_weather():
    """
    Live rainfall + 72-hour forecast for all 5 monitored cities.
    Data sourced from Open-Meteo (free, no API key required).
    Results are cached for 30 minutes.
    """
    return await weather_svc.fetch_all_cities()

@app.get("/api/v1/alerts/auto")
async def get_auto_alerts():
    """
    Auto-generate flood alerts based on live rainfall data.
    Any ward whose recomputed score falls to RED_ALERT (<=40) is returned.
    Frontend should poll this endpoint every 5 minutes.
    """
    live_weather = await weather_svc.fetch_all_cities()
    auto_alerts = []
    alert_id = 100

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT
                w.id,
                w.name,
                w.city,
                lws.elevation_m,
                lws.drainage_pct,
                lws.flood_events_5yr
            FROM wards w
            JOIN latest_ward_scores lws ON lws.id = w.id
            ORDER BY w.city, w.id
        """
        rows = await conn.fetch(query)
        await conn.close()
    except Exception as exc:
        logger.warning(f"[API] /api/v1/alerts/auto DB query failed: {exc}")
        return []

    wards = [dict(r) for r in rows]

    unique_cities = sorted({w.get("city") for w in wards if w.get("city")})
    drainage_by_city: dict[str, dict] = {}
    flood_by_city: dict[str, dict] = {}

    if unique_cities:
        drainage_results = await asyncio.gather(
            *[drainage_svc.fetch_drainage_data(c) for c in unique_cities],
            return_exceptions=True,
        )
        for c, res in zip(unique_cities, drainage_results):
            drainage_by_city[c] = res if not isinstance(res, Exception) else {}

        flood_results = await asyncio.gather(
            *[disaster_svc.fetch_reliefweb_history(c) for c in unique_cities],
            return_exceptions=True,
        )
        for c, res in zip(unique_cities, flood_results):
            flood_by_city[c] = res if not isinstance(res, Exception) else {}

    for ward in wards:
        city_name = ward.get("city", "")
        if city_name not in live_weather:
            continue

        live_rain = live_weather[city_name]["rainfall_mm"]
        drainage_score = drainage_by_city.get(city_name, {}).get("drainage_score", ward.get("drainage_pct", 50))
        flood_events = int(flood_by_city.get(city_name, {}).get("flood_events_5yr", ward.get("flood_events_5yr", 0)))

        rescored = compute_flood_readiness_score(
            rainfall_mm=live_rain,
            elevation_m=float(ward.get("elevation_m", 50)),
            drainage_pct=float(drainage_score),
            flood_events=flood_events,
        )
        score = rescored["score"]
        risk_class = rescored["risk_class"]

        if risk_class == "RED_ALERT":
            severity = "CRITICAL"
        elif risk_class == "WATCH_ZONE":
            severity = "WARNING"
        else:
            continue  # Only alert on RED and WATCH

        auto_alerts.append({
            "id": alert_id,
            "ward": ward.get("name"),
            "city": city_name,
            "severity": severity,
            "score": score,
            "risk_class": risk_class,
            "rainfall_mm": live_rain,
            "weather_source": live_weather[city_name]["source"],
            "message": (
                f"{'⛔ CRITICAL' if severity == 'CRITICAL' else '⚠ WARNING'}: "
                f"{ward.get('name')}, {city_name} — Score {score}/100. "
                f"Live rainfall: {live_rain:.1f}mm."
            ),
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        })
        alert_id += 1

    return sorted(auto_alerts, key=lambda a: a["score"])

@app.get("/api/v1/analytics/summary")
async def analytics_summary():
    """Platform-wide analytics summary (derived from latest computed scores)."""
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        query = """
            SELECT
                COUNT(*) AS total_wards,
                COUNT(DISTINCT city) AS cities,
                COUNT(*) FILTER (WHERE risk_class = 'RED_ALERT') AS red_alert,
                COUNT(*) FILTER (WHERE risk_class = 'WATCH_ZONE') AS watch_zone,
                COUNT(*) FILTER (WHERE risk_class = 'SAFE_ZONE') AS safe_zone,
                ROUND(AVG(score)::numeric, 1) AS avg_readiness_score,
                MAX(rainfall_mm) AS peak_rainfall_mm,
                COUNT(*) FILTER (WHERE drainage_pct < 30) AS drainage_alerts
            FROM latest_ward_scores
        """
        row = await conn.fetchrow(query)
        await conn.close()
        return dict(row)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

@app.get("/api/v1/analytics/rainfall")
async def rainfall_analytics(city: Optional[str] = None):
    """Monthly rainfall totals for charting (from `rainfall_observations`)."""
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_map = {i: months[i - 1] for i in range(1, 13)}

    try:
        conn = await asyncpg.connect(DATABASE_URL)
        if city:
            query = """
                SELECT
                    rs.city AS city,
                    EXTRACT(MONTH FROM ro.observed_at)::int AS month,
                    SUM(ro.rainfall_mm)::float AS rainfall_mm
                FROM rainfall_observations ro
                JOIN rainfall_stations rs ON rs.id = ro.station_id
                WHERE rs.city = $1
                GROUP BY rs.city, month
                ORDER BY month ASC
            """
            rows = await conn.fetch(query, city)
        else:
            query = """
                SELECT
                    rs.city AS city,
                    EXTRACT(MONTH FROM ro.observed_at)::int AS month,
                    SUM(ro.rainfall_mm)::float AS rainfall_mm
                FROM rainfall_observations ro
                JOIN rainfall_stations rs ON rs.id = ro.station_id
                GROUP BY rs.city, month
                ORDER BY rs.city ASC, month ASC
            """
            rows = await conn.fetch(query)
        await conn.close()

        cities = sorted({r["city"] for r in rows})
        data = {c: [0] * 12 for c in cities}
        for r in rows:
            m = int(r["month"])
            idx = m - 1
            data[r["city"]][idx] = round(float(r["rainfall_mm"]), 2)

        return {"months": months, "data": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

@app.get("/api/v1/cities")
async def list_cities():
    try:
        conn = await asyncpg.connect(DATABASE_URL)
        rows = await conn.fetch("SELECT name FROM cities ORDER BY name ASC")
        await conn.close()
        return [r["name"] for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database query failed: {exc}")

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
def _fallback_report(ward: dict) -> str:
    return (
        f"Ward {ward['name']} ({ward['city']}) has a Flood Readiness Score of "
        f"{ward['score']}/100, classified as {ward['risk_class']}. "
        f"Rainfall intensity of {ward['rainfall_mm']}mm exceeds safe thresholds. "
        f"Terrain elevation of {ward['elevation_m']}m indicates high vulnerability. "
        f"Drainage capacity at {ward['drainage_pct']}% requires immediate attention. "
        f"Recommended: Deploy emergency pumps and issue evacuation advisory."
    )
