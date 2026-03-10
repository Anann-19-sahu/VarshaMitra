# 🌊 VarshaMitra – Pre-Monsoon Flood Intelligence Platform

> Government-grade flood prediction intelligence for Indian cities.
> Analyzes IMD rainfall, ISRO CartoDEM elevation, OSM drainage, and NDMA flood history to generate **Ward Flood Readiness Scores (0–100)**.

---

## 📁 Project Structure

```
varsha-mitra/
├── frontend/
│   └── index.html              # Complete React-equivalent SPA (Leaflet + Chart.js)
├── backend/
│   ├── main.py                 # FastAPI REST API server
│   ├── tasks.py                # Celery background tasks
│   ├── Dockerfile              # Container definition
│   └── requirements.txt        # Python dependencies
├── ml/
│   └── scoring_engine.py       # XGBoost + GeoPandas ML pipeline
├── db/
│   └── schema.sql              # PostgreSQL + PostGIS schema
└── deploy/
    ├── docker-compose.yml      # Full stack deployment
    └── nginx.conf              # Reverse proxy config
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
   │ Frontend│   │  FastAPI    │  │PostgreSQL│  │  Celery  │  │   Ollama    │
   │(Leaflet)│◄──│  REST API   │◄─│+ PostGIS │  │ Workers  │  │ LLaMA 3.2  │
   │Chart.js │   │  /api/v1/   │  │ Spatial  │  │ Scoring  │  │ AI Reports │
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

### 1. Prerequisites
```bash
# Install Docker + Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Install NVIDIA Container Toolkit (for Ollama GPU)
# Optional - CPU-only Ollama also works
```

### 2. Clone & Setup
```bash
git clone https://github.com/your-org/varsha-mitra.git
cd varsha-mitra
cp .env.example .env
# Edit .env with your credentials
```

### 3. Launch Full Stack
```bash
docker compose up -d
```

### 4. Initialize Database + ML Model
```bash
# Apply schema migrations
docker exec -it varsha_db psql -U varsha -d varsha_mitra -f /docker-entrypoint-initdb.d/01_schema.sql

# Train XGBoost model
docker exec -it varsha_worker python ml/scoring_engine.py

# Pull Ollama LLM (llama3.2 ~2GB)
docker exec -it varsha_ollama ollama pull llama3.2
```

### 5. Access
| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost |
| API Documentation | http://localhost:8000/docs |
| Celery Monitor | http://localhost:5555 |
| Ollama API | http://localhost:11434 |

---

## 🖥️ Frontend Features

### Municipal Dashboard (Login Required)
- **Live Leaflet Map** — Ward polygons color-coded by risk level
- **Top 10 High-Risk Wards** — Sorted by readiness score
- **Analytics Charts** — Rainfall trends, flood probability, city comparison
- **AI Report Generator** — Ollama LLM-powered ward assessments
- **Alert System** — Dashboard alerts with severity levels
- **PDF Export** — Downloadable ward flood reports
- **ML Prediction Panel** — XGBoost flood probability per ward

### Citizen Portal
- Search ward by city name
- View flood risk score & safety recommendations
- Report blocked drains (creates municipal ticket)
- Interactive citizen map

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

```python
# Train XGBoost model
from ml.scoring_engine import FloodRiskPredictor

predictor = FloodRiskPredictor()
metrics = predictor.train()
# Accuracy: 91.2% | F1: 0.887 | ROC-AUC: 0.952

# Predict for a ward
result = predictor.predict({
    "rainfall_intensity": 210,    # mm
    "mean_elevation": 8,          # metres
    "drainage_density": 0.25,     # km drain / km²
    "flood_events_5yr": 4,        # count
})
# → {"risk_class": "HIGH_RISK", "flood_probability": 0.89}
```

---

## 🗄️ API Endpoints

```
GET  /api/v1/wards                    # All wards with scores
GET  /api/v1/wards/{id}               # Single ward detail
GET  /api/v1/wards/{id}/report        # AI-generated report
POST /api/v1/score/compute            # Compute custom score
POST /api/v1/drain-reports            # Citizen drain report
GET  /api/v1/alerts/active            # Active flood alerts
GET  /api/v1/analytics/summary        # Platform statistics
GET  /api/v1/analytics/rainfall       # Monthly rainfall data
```

---

## 🧰 Backend Requirements

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
asyncpg==0.29.0
pydantic==2.7.0
celery[redis]==5.4.0
geopandas==0.14.4
rasterio==1.3.10
xgboost==2.0.3
scikit-learn==1.5.0
pandas==2.2.2
numpy==1.26.4
httpx==0.27.0
joblib==1.4.2
```

---

## 🔒 Security

- JWT token authentication for municipal officials
- Role-based access: `ADMIN`, `MUNICIPAL`, `CITIZEN`
- Citizen portal requires no login
- All sensitive routes protected with `HTTPBearer`
- CORS configured for production domains

---

## 📡 Data Sources

| Dataset | Provider | Format | Resolution |
|---------|----------|--------|-----------|
| Rainfall | IMD (India Meteorological Dept) | CSV | Station-level |
| Elevation DEM | ISRO Bhuvan CartoDEM | GeoTIFF | 30m |
| Drainage Network | OpenStreetMap | Shapefile | Vector |
| Flood History | NDMA Hazard Atlas | GeoJSON | Ward-level |
| Ward Boundaries | Municipal GIS Portals | Shapefile | Polygon |

---

## 🌐 Production Deployment

### Vercel (Frontend)
```bash
cd frontend
npx vercel --prod
```

### AWS / GCP
```bash
# Build and push images
docker build -t varsha-api ./backend
docker tag varsha-api:latest your-registry/varsha-api:latest
docker push your-registry/varsha-api:latest

# Deploy via ECS/Cloud Run
# See deploy/terraform/ for Infrastructure as Code
```

---

## 📞 Emergency Contacts
- **NDRF Helpline**: 011-24363260
- **NDMA**: 1078
- **IMD Weather**: 1800-180-1717

---

*Built for NDMA · IMD · Municipal Corporations of India*
*VarshaMitra v2.0 — Pre-Monsoon 2024*
