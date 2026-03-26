"""
VarshaMitra – ML Scoring Engine
Flood Risk Prediction using XGBoost + Scikit-learn
Geospatial Processing with GeoPandas + Rasterio
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import Point, mapping
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import logging
from pathlib import Path
from typing import Tuple, Optional
import warnings
import os
import asyncio
import asyncpg
import sys
warnings.filterwarnings('ignore')

logger = logging.getLogger("varsha_mitra.ml")

# ─── Paths ───────────────────────────────────────────────────
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Allow importing backend services when running this script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.append(str(_BACKEND_DIR))
from disaster_service import fetch_reliefweb_history  # noqa: E402

# ─── ETL PIPELINE ────────────────────────────────────────────

class FloodDataETL:
    """
    Extract-Transform-Load pipeline for:
    - IMD Rainfall data (CSV)
    - ISRO CartoDEM elevation (GeoTIFF raster)
    - OSM Drainage network (Shapefile)
    - NDMA Flood history (GeoJSON)
    - Ward boundaries (Shapefile)
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir

    def load_ward_boundaries(self, city: str) -> gpd.GeoDataFrame:
        """
        Load ward boundaries for a city directly from PostGIS (`wards` table).

        Expects:
        - `wards.boundary` (GEOGRAPHY(POLYGON, 4326))
        - `wards.name`, `wards.city`, `wards.id`
        """
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://varsha:monsoon2024@localhost:5432/varsha_mitra",
        )

        # `geopandas.read_postgis` uses SQLAlchemy; import lazily.
        try:
            from sqlalchemy import create_engine  # type: ignore

            engine = create_engine(database_url)
            sql = """
                SELECT
                    id AS ward_id,
                    name AS ward_name,
                    city,
                    boundary::geometry AS geometry
                FROM wards
                WHERE city = %s
            """
            gdf = gpd.read_postgis(sql, engine, params=[city])
            gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            logger.info(f"[ETL] Loaded {len(gdf)} ward boundaries for {city} from PostGIS")
            return gdf.to_crs("EPSG:4326")
        except Exception as exc:
            raise RuntimeError(f"Failed to load ward boundaries for {city} from PostGIS: {exc}")

    def load_imd_rainfall(self) -> pd.DataFrame:
        """
        Load IMD monthly rainfall dataset from CSV.

        Raises
        ------
        FileNotFoundError
            If the CSV is absent. Download the station-level monthly dataset
            from https://imdpune.gov.in and place it at:
            data/rainfall/imd_monthly_rainfall.csv

            Expected columns: date (YYYY-MM-DD), city, rainfall_mm, stations
        ValueError
            If required columns are missing from the CSV.
        RuntimeError
            If the CSV exists but cannot be parsed.
        """
        path = self.data_dir / "rainfall" / "imd_monthly_rainfall.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"IMD rainfall data not found at '{path}'. "
                "Download the station-level monthly CSV from "
                "https://imdpune.gov.in and place it at "
                "data/rainfall/imd_monthly_rainfall.csv"
            )
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            logger.error(f"[ETL] Failed to parse IMD rainfall CSV at '{path}': {exc}", exc_info=True)
            raise RuntimeError(
                f"IMD rainfall CSV exists at '{path}' but could not be parsed: {exc}"
            ) from exc

        missing = {"date", "city", "rainfall_mm"} - set(df.columns)
        if missing:
            raise ValueError(
                f"IMD rainfall CSV at '{path}' is missing required columns: {missing}. "
                "Expected at minimum: date (YYYY-MM-DD), city, rainfall_mm."
            )

        df["month"] = pd.to_datetime(df["date"]).dt.month
        df["year"]  = pd.to_datetime(df["date"]).dt.year
        logger.info(f"[ETL] Loaded {len(df)} IMD rainfall rows from '{path}'")
        return df

    def load_elevation_dem(self, ward_geometry, lat: float = None, lng: float = None) -> dict:
        """
        Fetch elevation from OpenTopoData SRTM30m API (replaces local GeoTIFF).

        If lat/lng are provided, queries the OpenTopoData API for the centroid.
        Falls back to synthetic values if the API is unavailable.

        The local rasterio/CartoDEM path is preserved as a secondary fallback
        for offline/Docker environments that have the .tif file.
        """
        import requests

        # ── 1. Try OpenTopoData API (primary source) ──────────────────
        if lat is not None and lng is not None:
            try:
                url = f"https://api.opentopodata.org/v1/srtm30m?locations={lat},{lng}"
                resp = requests.get(url, timeout=8)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results and results[0].get("elevation") is not None:
                    elev = float(results[0]["elevation"])
                    logger.info(f"[ETL] OpenTopoData elevation ({lat},{lng}): {elev:.1f}m")
                    return {
                        "mean_elevation": elev,
                        "min_elevation":  round(elev * 0.6, 2),
                        "std_elevation":  round(elev * 0.15, 2),
                        "low_lying_pct":  self._estimate_low_lying_pct(elev),
                        "source":         "opentopodata_srtm30m",
                    }
            except Exception as exc:
                logger.warning(f"[ETL] OpenTopoData API failed: {exc}")

        # ── 2. Try local GeoTIFF (offline fallback) ───────────────────
        dem_path = self.data_dir / "dem" / "india_cartodem_30m.tif"
        if dem_path.exists():
            try:
                with rasterio.open(dem_path) as src:
                    geom = [mapping(ward_geometry)]
                    out_image, _ = rio_mask(src, geom, crop=True, nodata=-9999)
                    valid_pixels = out_image[out_image != -9999]
                    if len(valid_pixels) > 0:
                        return {
                            "mean_elevation": float(np.mean(valid_pixels)),
                            "min_elevation":  float(np.min(valid_pixels)),
                            "std_elevation":  float(np.std(valid_pixels)),
                            "low_lying_pct":  float(np.sum(valid_pixels < 10) / len(valid_pixels) * 100),
                            "source":         "local_cartodem",
                        }
            except Exception as e:
                logger.warning(f"[ETL] Local DEM extraction failed: {e}")

        # ── 3. Deterministic fallback ─────────────────────────────────
        # Use latitude-band heuristics (same approach as ElevationService)
        # rather than random values so repeated calls return stable features.
        if lat is not None:
            if 12.5 <= lat <= 13.5:
                synthetic_elev = 6.0    # Chennai coast
            elif 18.4 <= lat <= 18.7:
                synthetic_elev = 550.0  # Pune plateau
            elif 19.0 <= lat <= 19.3:
                synthetic_elev = 10.0   # Mumbai coast
            elif 22.4 <= lat <= 22.7:
                synthetic_elev = 7.0    # Kolkata delta
            elif 28.5 <= lat <= 29.0:
                synthetic_elev = 210.0  # Delhi plain
            else:
                synthetic_elev = 50.0   # generic mid-range
        else:
            synthetic_elev = 50.0
        return {
            "mean_elevation": synthetic_elev,
            "min_elevation":  round(synthetic_elev * 0.6, 2),
            "std_elevation":  round(synthetic_elev * 0.15, 2),
            "low_lying_pct":  self._estimate_low_lying_pct(synthetic_elev),
            "source":         "synthetic_deterministic",
        }

    @staticmethod
    def _estimate_low_lying_pct(elevation_m: float) -> float:
        """Estimate % of terrain below 10m based on centroid elevation."""
        if elevation_m <= 0:   return 95.0
        if elevation_m <= 5:   return 80.0
        if elevation_m <= 10:  return 55.0
        if elevation_m <= 20:  return 30.0
        if elevation_m <= 50:  return 15.0
        if elevation_m <= 100: return 5.0
        return 1.0

    def load_drainage_network(self, ward_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Compute drainage density for each ward from the OSM drainage shapefile.

        Drainage density = total drain length (m) / ward area (km²)

        Raises
        ------
        FileNotFoundError
            If the shapefile is absent. Export OSM waterway/drain ways for
            India from https://download.geofabrik.de or via Overpass and
            place the result at data/drainage/india_drains_osm.shp
        RuntimeError
            If the shapefile exists but cannot be read or spatially joined.
        """
        drain_path = self.data_dir / "drainage" / "india_drains_osm.shp"
        if not drain_path.exists():
            raise FileNotFoundError(
                f"OSM drainage shapefile not found at '{drain_path}'. "
                "Export India waterway/drain ways from "
                "https://download.geofabrik.de or via the Overpass API "
                "and place the shapefile at data/drainage/india_drains_osm.shp"
            )
        try:
            drains = gpd.read_file(drain_path).to_crs("EPSG:32643")
        except Exception as exc:
            logger.error(f"[ETL] Failed to read drainage shapefile at '{drain_path}': {exc}", exc_info=True)
            raise RuntimeError(
                f"OSM drainage shapefile at '{drain_path}' could not be read: {exc}"
            ) from exc

        try:
            wards_proj = ward_gdf.to_crs("EPSG:32643")
            joined = gpd.sjoin(drains, wards_proj, how="inner", predicate="intersects")
            drain_lengths = joined.groupby("index_right").geometry.apply(
                lambda g: g.length.sum()
            ).reset_index()
            drain_lengths.columns = ["ward_idx", "drain_length_m"]
            ward_areas = wards_proj.geometry.area / 1e6  # km²
            drain_lengths["area_km2"] = ward_areas[drain_lengths["ward_idx"]].values
            drain_lengths["drainage_density"] = (
                drain_lengths["drain_length_m"] / (drain_lengths["area_km2"] * 1000)
            )
            logger.info(
                f"[ETL] Computed drainage density for {len(drain_lengths)} wards "
                f"from '{drain_path}'"
            )
            return drain_lengths
        except Exception as exc:
            logger.error(f"[ETL] Drainage spatial join failed: {exc}", exc_info=True)
            raise RuntimeError(
                f"Drainage density computation failed during spatial join: {exc}"
            ) from exc

    def calculate_drainage_score(self, osm_data: dict) -> float:
        """
        Calculate drainage capacity score (0–100) from Overpass OSM way geometry.
        Higher total drain length -> higher score.
        """
        def _haversine_m(lat1, lon1, lat2, lon2):
            r = 6371000.0
            d_lat = np.radians(lat2 - lat1)
            d_lon = np.radians(lon2 - lon1)
            a = (
                np.sin(d_lat / 2) ** 2
                + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2) ** 2
            )
            return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        total_length_m = 0.0
        for el in osm_data.get("elements", []):
            if el.get("type") != "way":
                continue
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            for i in range(1, len(geom)):
                p1, p2 = geom[i - 1], geom[i]
                total_length_m += _haversine_m(p1["lat"], p1["lon"], p2["lat"], p2["lon"])

        # Heuristic normalization: 0m -> 0 score, 300km+ -> 100 score
        return float(np.clip((total_length_m / 300000.0) * 100.0, 0.0, 100.0))

    def load_flood_history(self) -> gpd.GeoDataFrame:
        """
        Load NDMA historical flood event data from GeoJSON.

        Raises
        ------
        FileNotFoundError
            If the GeoJSON is absent. Download the NDMA Flood Hazard Atlas from
            https://ndma.gov.in/Resources/flood-hazard-atlas and place it at:
            data/flood_history/ndma_flood_hazard.geojson

            Expected fields: geometry (Point/Polygon), year, severity, city
        RuntimeError
            If the GeoJSON exists but cannot be read.
        """
        path = self.data_dir / "flood_history" / "ndma_flood_hazard.geojson"
        if not path.exists():
            raise FileNotFoundError(
                f"NDMA flood hazard data not found at '{path}'. "
                "Download the Flood Hazard Atlas from "
                "https://ndma.gov.in/Resources/flood-hazard-atlas "
                "and place it at data/flood_history/ndma_flood_hazard.geojson"
            )
        try:
            gdf = gpd.read_file(path)
            logger.info(f"[ETL] Loaded {len(gdf)} NDMA flood history records from '{path}'")
            return gdf
        except Exception as exc:
            logger.error(f"[ETL] Failed to read NDMA flood GeoJSON at '{path}': {exc}", exc_info=True)
            raise RuntimeError(
                f"NDMA flood GeoJSON at '{path}' could not be read: {exc}"
            ) from exc

    def build_features(self, wards: gpd.GeoDataFrame, city: str) -> pd.DataFrame:
        """
        Spatial join and feature engineering pipeline.
        Returns DataFrame with all flood risk features per ward.

        All four data dimensions are loaded once and then joined onto
        each ward row — no np.random values are introduced here.
        """
        logger.info(f"Building features for {city}...")

        # ── 1. Rainfall ───────────────────────────────────────────────
        rainfall_df = self.load_imd_rainfall()
        city_rain = rainfall_df[rainfall_df.city == city]
        peak_rainfall = float(city_rain["rainfall_mm"].max()) if not city_rain.empty else 180.0

        # ── 2. Drainage (loaded once for the entire ward set) ─────────
        # load_drainage_network returns a DataFrame indexed by ward_idx with
        # a `drainage_density` column derived from real OSM geometry (or the
        # shapefile fallback).  We build a dict keyed by ward positional index
        # so the per-ward loop below can look values up in O(1).
        drainage_df = self.load_drainage_network(wards)
        drainage_by_idx: dict[int, float] = {}
        if not drainage_df.empty and "ward_idx" in drainage_df.columns:
            for _, drow in drainage_df.iterrows():
                drainage_by_idx[int(drow["ward_idx"])] = float(drow["drainage_density"])

        # ── 3. Flood history from ReliefWeb ───────────────────────────
        try:
            flood_payload = asyncio.run(fetch_reliefweb_history(city))
            flood_events_5yr = int(flood_payload.get("flood_events_5yr", 0))
        except RuntimeError:
            # Already inside a running event loop (e.g. called from async context).
            flood_events_5yr = 0

        # ── 4. Per-ward feature assembly ──────────────────────────────
        features = []
        for pos_idx, (idx, ward) in enumerate(wards.iterrows()):
            # Elevation: use ward centroid lat/lng when available so the
            # OpenTopoData API path in load_elevation_dem() is exercised.
            centroid = ward.geometry.centroid
            elev = self.load_elevation_dem(
                ward.geometry,
                lat=centroid.y,
                lng=centroid.x,
            )

            # Drainage: use the value loaded by load_drainage_network().
            # Fall back to the neutral midpoint only when the loader
            # returned nothing for this ward (e.g. no OSM ways intersect).
            drainage_density = drainage_by_idx.get(pos_idx, 0.5)

            feat = {
                "ward_id": idx,
                "city": city,
                # Rainfall: use the real peak value with no synthetic noise.
                "rainfall_intensity": peak_rainfall,
                "mean_elevation": elev["mean_elevation"],
                "min_elevation": elev["min_elevation"],
                "low_lying_pct": elev["low_lying_pct"],
                # Drainage: sourced from load_drainage_network(), not random.
                "drainage_density": drainage_density,
                "drainage_capacity_pct": drainage_density * 100,
                # Flood history: sourced from ReliefWeb, same value for all
                # wards in the city (city-level granularity from the API).
                "flood_events_5yr": flood_events_5yr,
                "area_km2": ward.geometry.area * 12321,  # approx deg² → km²
            }
            features.append(feat)

        return pd.DataFrame(features)

# ─── XGBOOST MODEL ───────────────────────────────────────────

class FloodRiskPredictor:
    """
    XGBoost-based flood risk classification model.
    Predicts HIGH_RISK (1) vs LOW_RISK (0) for each ward.
    Features: rainfall, elevation, drainage, flood_history

    Training data priority
    ----------------------
    1. Caller supplies X, y directly              → use as-is
    2. PostgreSQL `flood_scores` table has ≥ MIN_REAL_SAMPLES rows
                                                  → load_real_training_data()
    3. Neither source is available                → generate_training_data()
                                                    (SYNTHETIC LAST RESORT —
                                                     logged loudly as a warning)

    The `training_data_source` key in the metrics dict returned by train()
    always records which path was taken so callers can surface it in dashboards.
    Predictions made by a synthetically-trained model are flagged with
    `trained_on_synthetic=True` in every predict() response.
    """

    FEATURES = [
        "rainfall_intensity",
        "mean_elevation",
        "min_elevation",
        "low_lying_pct",
        "drainage_density",
        "flood_events_5yr",
        "area_km2",
    ]
    MODEL_PATH = MODEL_DIR / "xgboost_flood_risk.pkl"

    # Minimum labelled rows from the DB before we trust real data over synthetic.
    # Below this threshold the DB sample is too small for a reliable 80/20 split.
    MIN_REAL_SAMPLES = 100

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        # Tracks which data source was used so predict() can surface a warning.
        self._training_data_source: str = "unknown"

    # ------------------------------------------------------------------
    # Data source 1 (preferred): real labelled rows from PostgreSQL
    # ------------------------------------------------------------------

    def load_real_training_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Load labelled training data from the `flood_scores` table.

        Each historical score row already has all four feature dimensions
        (rainfall_mm, elevation_m, drainage_pct, flood_events_5yr) and a
        rule-based readiness score.  We derive the binary label from
        risk_class so the XGBoost model learns from *observed* Indian city
        conditions, not from random sampling.

        Returns (X, y) where y=1 means HIGH_RISK (RED_ALERT or WATCH_ZONE).

        Raises
        ------
        RuntimeError  if the DB is unreachable or fewer than MIN_REAL_SAMPLES
                      rows are available (caller falls through to synthetic).
        """
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://varsha:monsoon2024@localhost:5432/varsha_mitra",
        )

        async def _query() -> list[dict]:
            conn = await asyncpg.connect(database_url)
            try:
                rows = await conn.fetch(
                    """
                    SELECT
                        fs.rainfall_mm          AS rainfall_intensity,
                        fs.elevation_m          AS mean_elevation,
                        -- min_elevation proxy: 60 % of mean (same ratio used in ETL)
                        fs.elevation_m * 0.6    AS min_elevation,
                        -- low_lying_pct: re-derive from elevation using the same
                        -- step-function used everywhere else in the codebase
                        CASE
                            WHEN fs.elevation_m <= 0   THEN 95.0
                            WHEN fs.elevation_m <= 5   THEN 80.0
                            WHEN fs.elevation_m <= 10  THEN 55.0
                            WHEN fs.elevation_m <= 20  THEN 30.0
                            WHEN fs.elevation_m <= 50  THEN 15.0
                            WHEN fs.elevation_m <= 100 THEN  5.0
                            ELSE 1.0
                        END                     AS low_lying_pct,
                        -- drainage_density: convert 0-100 pct score → 0-1 density
                        fs.drainage_pct / 100.0 AS drainage_density,
                        fs.flood_events_5yr,
                        -- area_km2: pull from wards table; default 10 if missing
                        COALESCE(w.area_km2, 10.0) AS area_km2,
                        fs.risk_class
                    FROM flood_scores fs
                    JOIN wards w ON w.id = fs.ward_id
                    -- Only rows where all four core features are non-null
                    WHERE fs.rainfall_mm    IS NOT NULL
                      AND fs.elevation_m   IS NOT NULL
                      AND fs.drainage_pct  IS NOT NULL
                      AND fs.flood_events_5yr IS NOT NULL
                    ORDER BY fs.computed_at DESC
                    """
                )
                return [dict(r) for r in rows]
            finally:
                await conn.close()

        try:
            rows = asyncio.run(_query())
        except RuntimeError:
            # Already inside a running event loop.
            raise RuntimeError("Cannot run DB query: already inside an event loop.")

        if len(rows) < self.MIN_REAL_SAMPLES:
            raise RuntimeError(
                f"Only {len(rows)} labelled rows in flood_scores "
                f"(need ≥ {self.MIN_REAL_SAMPLES}). Falling through to synthetic."
            )

        df = pd.DataFrame(rows)
        # Binary label: RED_ALERT or WATCH_ZONE → HIGH_RISK (1), else LOW_RISK (0)
        y = (df["risk_class"].isin(["RED_ALERT", "WATCH_ZONE"])).astype(int).values
        X = df[self.FEATURES]
        logger.info(
            f"[Predictor] Loaded {len(df)} real training rows from PostgreSQL "
            f"(HIGH_RISK: {y.sum()}, LOW_RISK: {(1-y).sum()})"
        )
        return X, y

    # ------------------------------------------------------------------
    # Data source 2 (last resort): fully synthetic samples
    # ------------------------------------------------------------------

    def generate_training_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        *** SYNTHETIC LAST RESORT — no real data was available. ***

        Generates random feature vectors and labels them with the same
        weighted formula used by the rule-based scorer.  A model trained on
        this data learns the rule-based formula and provides ZERO additional
        ML value — it will produce nearly identical results to calling
        compute_readiness_score() directly.

        This path exists only to keep the system functional when the database
        is empty (e.g. fresh deployment, CI environment).  It must NEVER be
        used in production once real flood_scores rows are available.

        The returned tuple is tagged so train() can record and surface the
        `training_data_source="synthetic"` warning in metrics and predictions.
        """
        _SYNTHETIC_WARNING = (
            "SYNTHETIC TRAINING DATA IN USE — XGBoost model is being trained on "
            "randomly generated samples labelled by the rule-based formula. "
            "The model adds no ML value over the rule-based scorer. "
            "Populate the flood_scores table with real ward observations and "
            "retrain to obtain a meaningful ML model."
        )
        logger.warning(_SYNTHETIC_WARNING)
        warnings.warn(_SYNTHETIC_WARNING, UserWarning, stacklevel=3)

        np.random.seed(42)

        # Pull real flood-event counts from ReliefWeb so at least one dimension
        # (flood history) is anchored to real data even in the synthetic path.
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://varsha:monsoon2024@localhost:5432/varsha_mitra",
        )

        async def _fetch_city_flood_counts() -> Tuple[list, dict]:
            conn = await asyncpg.connect(database_url)
            try:
                rows = await conn.fetch(
                    "SELECT DISTINCT city FROM wards WHERE city IS NOT NULL"
                )
                cities = [r["city"] for r in rows if r.get("city")]
                counts: dict[str, int] = {}
                for c in cities:
                    payload = await fetch_reliefweb_history(c)
                    counts[c] = int(payload.get("flood_events_5yr", 0))
                return cities, counts
            finally:
                await conn.close()

        try:
            cities, flood_counts = asyncio.run(_fetch_city_flood_counts())
        except RuntimeError:
            cities, flood_counts = [], {}

        # If we cannot reach the DB at all, use hard-coded city baseline counts
        # so the synthetic labels are at least marginally realistic.
        if not cities:
            logger.warning(
                "[Predictor] DB unreachable during synthetic data generation; "
                "using hard-coded flood-event baselines."
            )
            cities = ["Mumbai", "Pune", "Delhi", "Chennai", "Kolkata"]
            flood_counts = {"Mumbai": 4, "Pune": 2, "Delhi": 3, "Chennai": 5, "Kolkata": 4}

        city_choices = np.random.choice(cities, size=n_samples)
        flood_events = [int(flood_counts.get(c, 0)) for c in city_choices]

        X = pd.DataFrame({
            "rainfall_intensity": np.random.uniform(50, 350, n_samples),
            "mean_elevation":     np.random.uniform(2, 700, n_samples),
            "min_elevation":      np.random.uniform(1, 200, n_samples),
            "low_lying_pct":      np.random.uniform(0, 90,  n_samples),
            "drainage_density":   np.random.uniform(0.05, 0.95, n_samples),
            "flood_events_5yr":   flood_events,
            "area_km2":           np.random.uniform(1, 50, n_samples),
        })

        # Labels derived from the rule-based formula — this is the tautology
        # that makes the synthetic path worthless for real ML.
        risk_score = (
            (X["rainfall_intensity"] - 50) / 300 * 0.40
            + (1 - (X["mean_elevation"] - 2) / 698) * 0.30
            + (1 - X["drainage_density"]) * 0.20
            + X["flood_events_5yr"] / 7 * 0.10
        )
        y = (risk_score > 0.5).astype(int)
        return X, y

    # ------------------------------------------------------------------
    # Training orchestrator — priority chain with explicit source tagging
    # ------------------------------------------------------------------

    def train(self, X: pd.DataFrame = None, y: np.ndarray = None) -> dict:
        """
        Train XGBoost classifier and return evaluation metrics.

        Data source priority
        --------------------
        1. Caller-supplied X, y                  → source = "caller_supplied"
        2. PostgreSQL flood_scores (≥ MIN_REAL_SAMPLES rows)
                                                 → source = "database"
        3. generate_training_data() [last resort] → source = "synthetic"
                                                    (warning emitted)

        The resolved source is recorded in metrics["training_data_source"]
        and stored in self._training_data_source for use by predict().
        """
        if X is not None and y is not None:
            data_source = "caller_supplied"
            logger.info("[Predictor] Training on caller-supplied data.")
        else:
            # ── Try real DB data first ─────────────────────────────────
            try:
                logger.info(
                    "[Predictor] Attempting to load real training data from PostgreSQL…"
                )
                X, y = self.load_real_training_data()
                data_source = "database"
                logger.info(
                    f"[Predictor] Using {len(X)} real rows from PostgreSQL for training."
                )
            except Exception as db_exc:
                # ── Fall through to synthetic — LAST RESORT ────────────
                logger.warning(
                    f"[Predictor] Real data unavailable ({db_exc}). "
                    "Falling back to SYNTHETIC training data. "
                    "Model will NOT add ML value over the rule-based scorer."
                )
                X, y = self.generate_training_data()
                data_source = "synthetic"

        self._training_data_source = data_source

        X_train, X_test, y_train, y_test = train_test_split(
            X[self.FEATURES], y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=50,
        )

        y_pred  = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(self.model, X[self.FEATURES], y, cv=5, scoring="f1")

        metrics = {
            "accuracy":             round(accuracy_score(y_test, y_pred), 4),
            "f1_score":             round(f1_score(y_test, y_pred), 4),
            "roc_auc":              round(roc_auc_score(y_test, y_proba), 4),
            "cv_f1_mean":           round(cv_scores.mean(), 4),
            "cv_f1_std":            round(cv_scores.std(), 4),
            "n_train":              len(X_train),
            "n_test":               len(X_test),
            # Explicit provenance tag — always present so callers can gate on it.
            "training_data_source": data_source,
        }

        if data_source == "synthetic":
            metrics["synthetic_warning"] = (
                "Model trained on synthetic data. Metrics reflect rule-based "
                "formula fit, not real predictive skill. Retrain with real data."
            )

        self.is_trained = True
        logger.info(f"[Predictor] Model trained. source={data_source} metrics={metrics}")

        joblib.dump(
            {
                "model":                self.model,
                "scaler":               self.scaler,
                "features":             self.FEATURES,
                "training_data_source": data_source,
            },
            self.MODEL_PATH,
        )
        logger.info(f"[Predictor] Model saved to {self.MODEL_PATH}")

        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, ward_features: dict) -> dict:
        """
        Predict flood risk for a single ward.

        If the model was trained on synthetic data, every prediction carries
        `trained_on_synthetic=True` as a first-class flag so API consumers
        and the dashboard can display an appropriate caveat.
        """
        if not self.is_trained:
            self.load_model()

        X = pd.DataFrame([{f: ward_features.get(f, 0) for f in self.FEATURES}])

        if self.model is None:
            risk = self._rule_based_risk(ward_features)
            return {
                "risk_class":          "HIGH_RISK" if risk > 0.5 else "LOW_RISK",
                "flood_probability":   round(risk, 4),
                "method":              "rule_based",
                "trained_on_synthetic": None,  # no model was used
            }

        prob = float(self.model.predict_proba(X)[0][1])
        cls  = "HIGH_RISK" if prob > 0.5 else "LOW_RISK"
        importance = dict(zip(self.FEATURES, self.model.feature_importances_))

        result = {
            "risk_class":          cls,
            "flood_probability":   round(prob, 4),
            "confidence":          round(abs(prob - 0.5) * 2, 4),
            "feature_importance":  {k: round(float(v), 4) for k, v in importance.items()},
            "method":              "xgboost_v2",
            "training_data_source": self._training_data_source,
            # Explicit boolean flag — easy for callers to check without string parsing.
            "trained_on_synthetic": self._training_data_source == "synthetic",
        }

        if result["trained_on_synthetic"]:
            result["synthetic_warning"] = (
                "This prediction was made by a model trained on synthetic data. "
                "It reflects the rule-based scoring formula, not learned patterns. "
                "Retrain once real flood_scores rows are available."
            )

        return result

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def load_model(self):
        """Load saved model from disk, restoring training_data_source."""
        if self.MODEL_PATH.exists():
            saved = joblib.load(self.MODEL_PATH)
            self.model  = saved["model"]
            self.scaler = saved.get("scaler", StandardScaler())
            self._training_data_source = saved.get("training_data_source", "unknown")
            self.is_trained = True
            logger.info(
                f"[Predictor] Model loaded from disk. "
                f"training_data_source={self._training_data_source}"
            )
            if self._training_data_source == "synthetic":
                logger.warning(
                    "[Predictor] Loaded model was trained on SYNTHETIC data. "
                    "Retrain with real data for meaningful ML predictions."
                )
        else:
            logger.warning("[Predictor] No saved model found. Training new model…")
            self.train()

    def get_feature_importance(self) -> dict:
        """Return feature importance scores."""
        if self.model is None:
            return {}
        return dict(zip(self.FEATURES, self.model.feature_importances_))

    def _rule_based_risk(self, features: dict) -> float:
        """Fallback rule-based risk estimation (used when model is None)."""
        rain_risk  = (features.get("rainfall_intensity", 150) - 50) / 300
        elev_risk  = 1 - (features.get("mean_elevation", 50) - 2) / 698
        drain_risk = 1 - features.get("drainage_density", 0.5)
        hist_risk  = features.get("flood_events_5yr", 2) / 7
        return rain_risk * 0.4 + elev_risk * 0.3 + drain_risk * 0.2 + hist_risk * 0.1


# ─── COMPOSITE SCORING ENGINE ────────────────────────────────

class FloodScoringEngine:
    """
    Complete flood readiness scoring pipeline.
    Combines rule-based composite scoring with XGBoost ML predictions.
    """

    def __init__(self):
        self.etl = FloodDataETL()
        self.predictor = FloodRiskPredictor()
        self.scaler = MinMaxScaler(feature_range=(0, 100))

    def compute_readiness_score(
        self,
        rainfall_mm: float,
        elevation_m: float,
        drainage_pct: float = None,
        flood_events: int = 0,
        osm_data: Optional[dict] = None,
        weights: Optional[dict] = None
    ) -> dict:
        """
        Compute Ward Flood Readiness Score (0–100).

        Score composition:
          40% → Rainfall intensity risk
          30% → Terrain elevation vulnerability
          20% → Drainage infrastructure capacity
          10% → Historical flood frequency
        """
        if weights is None:
            weights = {"rainfall": 0.40, "elevation": 0.30, "drainage": 0.20, "history": 0.10}

        # Replace any manual drainage_pct with computed drainage when OSM is available.
        if osm_data is not None:
            drainage_pct = self.etl.calculate_drainage_score(osm_data)
        elif drainage_pct is None:
            drainage_pct = 50.0

        # Normalize risk factors to [0,1]
        rain_risk   = np.clip((rainfall_mm - 50) / (350 - 50), 0, 1)
        elev_risk   = np.clip(1 - (elevation_m - 2) / (700 - 2), 0, 1)
        drain_risk  = np.clip(1 - (drainage_pct - 10) / (90 - 10), 0, 1)
        hist_risk   = np.clip(flood_events / 8, 0, 1)

        composite_risk = (
            rain_risk  * weights["rainfall"] +
            elev_risk  * weights["elevation"] +
            drain_risk * weights["drainage"] +
            hist_risk  * weights["history"]
        )

        readiness = int(round((1 - composite_risk) * 100))
        readiness = max(5, min(95, readiness))

        if readiness <= 40:
            risk_class = "RED_ALERT"
            risk_color = "#ff453a"
        elif readiness <= 70:
            risk_class = "WATCH_ZONE"
            risk_color = "#ff9f0a"
        else:
            risk_class = "SAFE_ZONE"
            risk_color = "#32d74b"

        return {
            "score": readiness,
            "risk_class": risk_class,
            "risk_color": risk_color,
            "components": {
                "rainfall_risk": round(float(rain_risk), 3),
                "elevation_risk": round(float(elev_risk), 3),
                "drainage_risk": round(float(drain_risk), 3),
                "history_risk": round(float(hist_risk), 3),
            },
            "composite_risk": round(float(composite_risk), 4),
            "weights_used": weights,
        }

    def score_all_wards(self, wards_df: pd.DataFrame) -> pd.DataFrame:
        """Score all wards in a DataFrame."""
        results = []
        for _, row in wards_df.iterrows():
            score_result = self.compute_readiness_score(
                rainfall_mm=row.get("rainfall_intensity", 150),
                elevation_m=row.get("mean_elevation", 50),
                drainage_pct=row.get("drainage_capacity_pct", 50),
                flood_events=int(row.get("flood_events_5yr", 2))
            )
            # Add XGBoost prediction
            ml_result = self.predictor.predict({
                "rainfall_intensity": row.get("rainfall_intensity", 150),
                "mean_elevation": row.get("mean_elevation", 50),
                "min_elevation": row.get("min_elevation", 20),
                "low_lying_pct": row.get("low_lying_pct", 30),
                "drainage_density": row.get("drainage_density", 0.5),
                "flood_events_5yr": int(row.get("flood_events_5yr", 2)),
                "area_km2": row.get("area_km2", 10),
            })
            results.append({
                "ward_id": row.get("ward_id"),
                "city": row.get("city"),
                **score_result,
                "ml_risk_class": ml_result["risk_class"],
                "ml_flood_probability": ml_result.get("flood_probability", 0.5),
            })
        return pd.DataFrame(results)


# ─── MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("VarshaMitra – Flood Risk ML Pipeline")
    print("=" * 60)

    engine = FloodScoringEngine()

    # Train XGBoost
    print("\n[1] Training XGBoost Flood Risk Model...")
    metrics = engine.predictor.train()
    print(f"  Data source : {metrics['training_data_source'].upper()}")
    if metrics.get("synthetic_warning"):
        print(f"\n  *** WARNING: {metrics['synthetic_warning']} ***\n")
    print(f"  Accuracy : {metrics['accuracy']:.1%}")
    print(f"  F1 Score : {metrics['f1_score']:.1%}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.1%}")
    print(f"  CV F1    : {metrics['cv_f1_mean']:.1%} ± {metrics['cv_f1_std']:.1%}")

    print("\n[2] Feature Importance (XGBoost):")
    importance = engine.predictor.get_feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f} {bar}")

    print("\n✓ ML Pipeline complete.")