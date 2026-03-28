# 🌊 VarshaMitra – Pre-Monsoon Flood Intelligence Platform

> Government-grade flood prediction intelligence for Indian cities.
> Analyzes IMD rainfall, ISRO CartoDEM elevation, OSM drainage, and NDMA flood history to generate **Ward Flood Readiness Scores (0–100)**.

---

## 📁 Project Structure

```
varsha-mitra/
├── frontend/
│   ├── VarshaMitra.html    # Main HTML interface
│   └── VarshaMitra.css     # Styling and themes
├── backend/
│   ├── main.py             # FastAPI REST API server
│   ├── weather_service.py  # IMD rainfall data integration
│   ├── elevation_service.py # SRTM30m elevation data
│   ├── drainage_service.py # OSM drainage network analysis
│   ├── disaster_service.py # NDMA flood history data
│   ├── scoring_engine.py   # Rule-based flood scoring
│   ├── Dockerfile          # Container definition
│   └── requirements.txt    # Python dependencies
├── ml/
│   └── scoring_engine.py   # XGBoost + GeoPandas ML pipeline
├── db/
│   └── schema.sql          # PostgreSQL + PostGIS schema
└── docker-compose.yml      # Full stack deployment
```

---

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────────┐
                    │           VarshaMitra Platform           │
                    └─────────────────────────────────────────┘
                                        │
        ┌───────────────┬───────────────┼───────────────┬───────────────┐
        │               │               │               │               │
   ┌─────────┐   ┌─────────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐
   │Frontend │   │  FastAPI    │  │PostgreSQL│  │  Redis   │  │   Ollama    │
   │(HTML/CSS│◄──│  REST API   │◄─│+ PostGIS │  │ (Cache)  │  │ LLaMA 3.2  │
   │+ JS)    │   │  /api/v1/   │  │ Spatial  │  │ Broker   │  │ AI Reports │
   └─────────┘   └─────────────┘  └──────────┘  └──────────┘  └─────────────┘
                        │
              ┌─────────┴──────────┐
              │   ML Scoring       │
              │  Engine (Python)   │
              │                    │
              │ ┌───────────────┐  │
              │ │  XGBoost v2   │  │
              │ │  GeoPandas    │  │
              │ │  Rasterio DEM │  │
              │ └───────────────┘  │
              └────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Docker + Docker Compose
- 8GB+ RAM recommended
- Optional: NVIDIA GPU for Ollama LLM acceleration

### 1. Clone & Setup
```bash
git clone <repository-url>
cd varsha-mitra
```

### 2. Launch Full Stack
```bash
docker compose up -d
```

### 3. Initialize Database
The database schema is automatically applied on first startup via Docker entrypoint.

### 4. Access Services
| Service | URL | Description |
|---------|-----|-------------|
| Frontend Dashboard | http://localhost | Main web interface |
| API Documentation | http://localhost:8000/docs | FastAPI interactive docs |
| API Health Check | http://localhost:8000/api/v1/health | Service status |
| Ollama API | http://localhost:11434 | Local LLM endpoint |
| Flower (Celery Monitor) | http://localhost:5555 | Task queue monitoring |

### 5. Optional: Train ML Model
```bash
# Access the running API container
docker exec -it varsha_api bash

# Run ML training (requires data files)
cd ml && python scoring_engine.py
```
| Celery Monitor | http://localhost:5555 |
| Ollama API | http://localhost:11434 |

---

## 🖥️ Frontend Features

### Web Dashboard (HTML/CSS/JavaScript)
- **Interactive Map Interface** — City ward visualization with risk overlays
- **Real-time Risk Monitoring** — Live flood readiness scores
- **Analytics Dashboard** — Charts and statistics
- **AI Report Generation** — Ollama-powered ward assessments
- **Alert Management** — Active flood alerts display
- **Citizen Portal** — Public access for ward information

### API-First Design
The frontend consumes the REST API endpoints. The current implementation serves static HTML/CSS files via FastAPI, with JavaScript handling dynamic content and API calls.

### Development Status
- ✅ Backend API fully implemented
- ✅ Database schema and services
- 🚧 Frontend interface (HTML/CSS framework ready)
- ✅ Docker deployment configuration

---

## 📊 Scoring Engine

| Factor | Weight | Data Source |
|--------|--------|-------------|
| Rainfall Intensity | **40%** | IMD Monthly Rainfall |
| Terrain Elevation | **30%** | ISRO CartoDEM 30m |
| Drainage Capacity | **20%** | OpenStreetMap |
| Flood History | **10%** | NDMA Hazard Dataset |

**Score → Risk Classification:**
- 🔴 **0–40**: RED ALERT — Immediate action required
- 🟡 **41–70**: WATCH ZONE — Active monitoring needed
- 🟢 **71–100**: SAFE ZONE — Standard preparedness

---

## 🤖 ML Pipeline

### Current Implementation
The system uses a **rule-based scoring engine** with configurable weights, providing real-time flood risk assessment. The XGBoost ML pipeline is available for advanced modeling.

### Scoring Algorithm
```python
# Rule-based flood readiness calculation
def compute_flood_readiness_score(
    rainfall_mm: float,
    elevation_m: float,
    drainage_pct: float,
    flood_events: int
) -> dict:
    # Weights: rainfall(40%), elevation(30%), drainage(20%), history(10%)
    # Returns score 0-100 and risk classification
```

### XGBoost Model (Optional)
The `ml/scoring_engine.py` provides:
- **GeoPandas** for spatial data processing
- **Rasterio** for DEM analysis
- **XGBoost** classifier for flood probability prediction
- **Scikit-learn** pipeline with cross-validation

**Training Requirements**: Ward boundary shapefiles, historical flood data, and elevation rasters.

---

## 🗄️ API Endpoints

### Core Endpoints
```
GET  /api/v1/status                    # Platform status and data sources
GET  /api/v1/health                    # Health check
GET  /api/v1/wards                     # All wards with live flood scores
GET  /api/v1/wards/{id}                # Single ward details
GET  /api/v1/wards/{id}/report         # AI-generated flood report
POST /api/v1/score/compute             # Compute custom flood score
```

### Data Services
```
GET  /api/v1/elevation                  # Elevation data for coordinates
POST /api/v1/elevation/batch            # Batch elevation queries
GET  /api/v1/wards/{id}/elevation       # Ward elevation analysis
GET  /api/v1/drainage/network           # OSM drainage network data
GET  /api/v1/weather/live               # Live weather data
```

### Analytics & Monitoring
```
GET  /api/v1/analytics/summary          # Platform-wide statistics
GET  /api/v1/analytics/rainfall         # Monthly rainfall analytics
GET  /api/v1/cities                     # City information
GET  /api/v1/alerts/active              # Active flood alerts
GET  /api/v1/alerts/auto                # Auto-generated alerts
POST /api/v1/alerts                     # Create new alert
```

### Citizen Services
```
POST /api/v1/drain-reports              # Submit drain blockage report
POST /api/v1/auth/citizen/signup        # Citizen signup (email+sms OTP dispatch)
POST /api/v1/auth/citizen/verify-otp    # Activate citizen account with dual OTP
POST /api/v1/auth/citizen/login         # Citizen login (RBAC token)
POST /api/v1/auth/authority/login       # Authority login (pre-allotted ID)
GET  /api/v1/auth/session               # Validate active auth token
```

---

## Authentication Provider Setup

Set these environment variables before starting backend for real OTP delivery:

```
# Email OTP via SMTP
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-sender@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=your-sender@gmail.com

# SMS OTP via Twilio
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+1xxxxxxxxxx
DEFAULT_COUNTRY_CODE=+91

# Optional local development override
ALLOW_DEV_OTP_FALLBACK=false
```

Notes:
- With `ALLOW_DEV_OTP_FALLBACK=false`, signup fails if either SMS or Email OTP delivery fails.
- OTP values are no longer returned in API responses.
- For local testing without providers, set `ALLOW_DEV_OTP_FALLBACK=true` and read generated OTPs from backend logs.

---

## 🧰 Backend Requirements

```
fastapi                    # Web framework
uvicorn[standard]         # ASGI server
pydantic                  # Data validation
asyncpg                   # PostgreSQL async driver
psycopg2-binary           # PostgreSQL sync driver
httpx                     # HTTP client for external APIs
requests                  # HTTP requests
geopandas                 # Geospatial data processing
rasterio                  # Raster data processing
xgboost                   # ML model
scikit-learn              # ML utilities
pandas                    # Data manipulation
numpy                     # Numerical computing
joblib                    # Model serialization
shapely                   # Geometry operations
sqlalchemy                # ORM for GeoPandas
```

---

## 🔒 Security

- **API Authentication**: HTTP Bearer token authentication (framework ready)
- **Role-based Access**: Planned for `ADMIN`, `MUNICIPAL`, `CITIZEN` roles
- **CORS Configuration**: Configured for cross-origin requests
- **Input Validation**: Pydantic models for all API inputs
- **Database Security**: PostgreSQL with proper user roles and permissions

**Note**: Authentication is currently disabled for development/demo purposes. Enable in production by implementing JWT token validation.

---

## 📡 Data Sources

| Dataset | Provider | Format | Resolution | API/Service |
|---------|----------|--------|-----------|-------------|
| Rainfall | Open-Meteo | JSON | Hourly/Daily | open-meteo.com |
| Elevation DEM | OpenTopoData SRTM30m | JSON | 30m | opentopodata.org |
| Drainage Network | OpenStreetMap | GeoJSON | Vector | overpass-api.de |
| Flood History | ReliefWeb/NDMA | JSON | Event-level | reliefweb.int |
| Ward Boundaries | Municipal GIS | Shapefile | Polygon | Database stored |

**Note**: The system uses fallback data for development/demo when external APIs are unavailable.

---

## 🌐 Production Deployment

### Docker Compose (Recommended)
The `docker-compose.yml` provides a complete production setup with:
- **FastAPI API Server** (Port 8000)
- **PostgreSQL + PostGIS** (Port 5432)
- **Redis Cache/Broker** (Port 6379)
- **Ollama LLM Server** (Port 11434)
- **Nginx Reverse Proxy** (Ports 80/443)
- **Celery Flower Monitor** (Port 5555)

```bash
# Production deployment
docker compose up -d

# Scale API servers if needed
docker compose up -d --scale api=3
```

### Environment Variables
```bash
DATABASE_URL=postgresql://varsha:monsoon2024@db:5432/varsha_mitra
REDIS_URL=redis://redis:6379/0
OLLAMA_URL=http://ollama:11434
SECRET_KEY=your-secret-key-here
ENVIRONMENT=production
```

### SSL/TLS Setup
Configure SSL certificates in the nginx service volume mounts for HTTPS support.

---

## 📞 Emergency Contacts
- **NDRF Helpline**: 011-24363260
- **NDMA**: 1078
- **IMD Weather**: 1800-180-1717

---

*Built for NDMA · IMD · Municipal Corporations of India*
*VarshaMitra v2.0 — Pre-Monsoon 2026*
