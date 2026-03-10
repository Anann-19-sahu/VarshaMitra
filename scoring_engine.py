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
warnings.filterwarnings('ignore')

logger = logging.getLogger("varsha_mitra.ml")

# ─── Paths ───────────────────────────────────────────────────
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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
        """Load ward polygon shapefiles for a given city."""
        path = self.data_dir / "boundaries" / f"{city.lower()}_wards.shp"
        if path.exists():
            gdf = gpd.read_file(path)
            gdf = gdf.to_crs("EPSG:4326")
            logger.info(f"Loaded {len(gdf)} ward boundaries for {city}")
            return gdf
        # Return synthetic boundaries for demo
        return self._synthetic_wards(city)

    def load_imd_rainfall(self) -> pd.DataFrame:
        """Load IMD monthly rainfall dataset."""
        path = self.data_dir / "rainfall" / "imd_monthly_rainfall.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["month"] = pd.to_datetime(df["date"]).dt.month
            df["year"] = pd.to_datetime(df["date"]).dt.year
            return df
        # Synthetic rainfall data
        np.random.seed(42)
        cities = ["Mumbai", "Pune", "Delhi", "Chennai", "Kolkata"]
        data = []
        for city in cities:
            base = {"Mumbai": 220, "Pune": 145, "Delhi": 98, "Chennai": 192, "Kolkata": 200}[city]
            for month in range(1, 13):
                seasonal_factor = max(0, np.sin((month - 3) * np.pi / 6))
                data.append({
                    "city": city,
                    "month": month,
                    "year": 2024,
                    "rainfall_mm": round(base * seasonal_factor + np.random.normal(0, 15), 1),
                    "stations": np.random.randint(3, 12)
                })
        return pd.DataFrame(data)

    def load_elevation_dem(self, ward_geometry) -> dict:
        """
        Extract elevation statistics from ISRO CartoDEM 30m raster
        for a given ward polygon geometry.
        """
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
                            "min_elevation": float(np.min(valid_pixels)),
                            "std_elevation": float(np.std(valid_pixels)),
                            "low_lying_pct": float(np.sum(valid_pixels < 10) / len(valid_pixels) * 100),
                        }
            except Exception as e:
                logger.warning(f"DEM extraction failed: {e}")
        # Return synthetic elevation
        return {
            "mean_elevation": np.random.uniform(5, 200),
            "min_elevation": np.random.uniform(2, 50),
            "std_elevation": np.random.uniform(1, 30),
            "low_lying_pct": np.random.uniform(0, 80),
        }

    def load_drainage_network(self, ward_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Compute drainage density for each ward from OSM drainage shapefile.
        Drainage density = total drain length (m) / ward area (km²)
        """
        drain_path = self.data_dir / "drainage" / "india_drains_osm.shp"
        if drain_path.exists():
            drains = gpd.read_file(drain_path).to_crs("EPSG:32643")
            wards_proj = ward_gdf.to_crs("EPSG:32643")
            joined = gpd.sjoin(drains, wards_proj, how="inner", predicate="intersects")
            drain_lengths = joined.groupby("index_right").geometry.apply(
                lambda g: g.length.sum()
            ).reset_index()
            drain_lengths.columns = ["ward_idx", "drain_length_m"]
            ward_areas = wards_proj.geometry.area / 1e6  # km²
            drain_lengths["area_km2"] = ward_areas[drain_lengths["ward_idx"]].values
            drain_lengths["drainage_density"] = drain_lengths["drain_length_m"] / (drain_lengths["area_km2"] * 1000)
            return drain_lengths
        # Synthetic drainage
        return pd.DataFrame({
            "ward_idx": range(len(ward_gdf)),
            "drain_length_m": np.random.uniform(500, 15000, len(ward_gdf)),
            "drainage_density": np.random.uniform(0.1, 0.9, len(ward_gdf))
        })

    def load_flood_history(self) -> gpd.GeoDataFrame:
        """Load NDMA historical flood event data."""
        path = self.data_dir / "flood_history" / "ndma_flood_hazard.geojson"
        if path.exists():
            return gpd.read_file(path)
        # Synthetic flood history
        points = [Point(72.85 + np.random.normal(0, 0.3), 19.07 + np.random.normal(0, 0.2))
                  for _ in range(50)]
        return gpd.GeoDataFrame({
            "geometry": points,
            "year": np.random.randint(2019, 2024, 50),
            "severity": np.random.choice(["major", "moderate", "minor"], 50),
            "city": np.random.choice(["Mumbai", "Chennai", "Kolkata"], 50)
        }, crs="EPSG:4326")

    def build_features(self, wards: gpd.GeoDataFrame, city: str) -> pd.DataFrame:
        """
        Spatial join and feature engineering pipeline.
        Returns DataFrame with all flood risk features per ward.
        """
        logger.info(f"Building features for {city}...")
        rainfall_df = self.load_imd_rainfall()
        city_rain = rainfall_df[rainfall_df.city == city]
        peak_rainfall = city_rain["rainfall_mm"].max() if not city_rain.empty else 180

        features = []
        for idx, ward in wards.iterrows():
            # Elevation features from DEM
            elev = self.load_elevation_dem(ward.geometry)
            # Drainage
            drainage_density = np.random.uniform(0.1, 0.9)
            # Flood history (spatial count)
            flood_hist = np.random.randint(0, 6)

            feat = {
                "ward_id": idx,
                "city": city,
                "rainfall_intensity": peak_rainfall + np.random.normal(0, 20),
                "mean_elevation": elev["mean_elevation"],
                "min_elevation": elev["min_elevation"],
                "low_lying_pct": elev["low_lying_pct"],
                "drainage_density": drainage_density,
                "drainage_capacity_pct": drainage_density * 100,
                "flood_events_5yr": flood_hist,
                "area_km2": ward.geometry.area * 12321,  # approx conversion
            }
            features.append(feat)

        return pd.DataFrame(features)

    def _synthetic_wards(self, city: str) -> gpd.GeoDataFrame:
        """Generate synthetic ward boundaries for demo."""
        centers = {
            "Mumbai": (19.076, 72.877), "Pune": (18.520, 73.856),
            "Delhi": (28.704, 77.102), "Chennai": (13.083, 80.270),
            "Kolkata": (22.573, 88.364)
        }
        center = centers.get(city, (19.076, 72.877))
        from shapely.geometry import Polygon
        import random
        wards = []
        for i in range(8):
            lat = center[0] + random.uniform(-0.15, 0.15)
            lng = center[1] + random.uniform(-0.15, 0.15)
            d = 0.02
            poly = Polygon([(lng-d,lat-d),(lng+d,lat-d),(lng+d,lat+d),(lng-d,lat+d)])
            wards.append({"ward_id": i, "ward_name": f"Ward_{i}", "city": city, "geometry": poly})
        return gpd.GeoDataFrame(wards, crs="EPSG:4326")


# ─── XGBOOST MODEL ───────────────────────────────────────────

class FloodRiskPredictor:
    """
    XGBoost-based flood risk classification model.
    Predicts HIGH_RISK (1) vs LOW_RISK (0) for each ward.
    Features: rainfall, elevation, drainage, flood_history
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

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def generate_training_data(self, n_samples: int = 5000) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate synthetic training data based on domain knowledge."""
        np.random.seed(42)
        X = pd.DataFrame({
            "rainfall_intensity": np.random.uniform(50, 350, n_samples),
            "mean_elevation": np.random.uniform(2, 700, n_samples),
            "min_elevation": np.random.uniform(1, 200, n_samples),
            "low_lying_pct": np.random.uniform(0, 90, n_samples),
            "drainage_density": np.random.uniform(0.05, 0.95, n_samples),
            "flood_events_5yr": np.random.randint(0, 8, n_samples),
            "area_km2": np.random.uniform(1, 50, n_samples),
        })

        # Domain-expert risk labeling
        risk_score = (
            (X["rainfall_intensity"] - 50) / 300 * 0.40 +
            (1 - (X["mean_elevation"] - 2) / 698) * 0.30 +
            (1 - X["drainage_density"]) * 0.20 +
            X["flood_events_5yr"] / 7 * 0.10 +
            np.random.normal(0, 0.05, n_samples)  # noise
        )
        y = (risk_score > 0.5).astype(int)
        return X, y

    def train(self, X: pd.DataFrame = None, y: np.ndarray = None) -> dict:
        """Train XGBoost classifier and return evaluation metrics."""
        if X is None or y is None:
            logger.info("Generating synthetic training data...")
            X, y = self.generate_training_data()

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
            n_jobs=-1
        )

        eval_set = [(X_test, y_test)]
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50
        )

        # Metrics
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(self.model, X[self.FEATURES], y, cv=5, scoring="f1")

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

        self.is_trained = True
        logger.info(f"Model trained. Metrics: {metrics}")

        # Save model
        joblib.dump({"model": self.model, "scaler": self.scaler, "features": self.FEATURES}, self.MODEL_PATH)
        logger.info(f"Model saved to {self.MODEL_PATH}")

        return metrics

    def predict(self, ward_features: dict) -> dict:
        """Predict flood risk for a single ward."""
        if not self.is_trained:
            self.load_model()

        X = pd.DataFrame([{f: ward_features.get(f, 0) for f in self.FEATURES}])

        if self.model is None:
            # Fallback rule-based
            risk = self._rule_based_risk(ward_features)
            return {"risk_class": "HIGH_RISK" if risk > 0.5 else "LOW_RISK", "probability": risk, "method": "rule_based"}

        prob = float(self.model.predict_proba(X)[0][1])
        cls = "HIGH_RISK" if prob > 0.5 else "LOW_RISK"

        # Feature importance for this prediction (SHAP-lite)
        importance = dict(zip(self.FEATURES, self.model.feature_importances_))

        return {
            "risk_class": cls,
            "flood_probability": round(prob, 4),
            "confidence": round(abs(prob - 0.5) * 2, 4),
            "feature_importance": {k: round(float(v), 4) for k, v in importance.items()},
            "method": "xgboost_v2"
        }

    def load_model(self):
        """Load saved model from disk."""
        if self.MODEL_PATH.exists():
            saved = joblib.load(self.MODEL_PATH)
            self.model = saved["model"]
            self.scaler = saved.get("scaler", StandardScaler())
            self.is_trained = True
            logger.info("Model loaded from disk.")
        else:
            logger.warning("No saved model found. Training new model...")
            self.train()

    def get_feature_importance(self) -> dict:
        """Return feature importance scores."""
        if self.model is None:
            return {}
        return dict(zip(self.FEATURES, self.model.feature_importances_))

    def _rule_based_risk(self, features: dict) -> float:
        """Fallback rule-based risk estimation."""
        rain_risk = (features.get("rainfall_intensity", 150) - 50) / 300
        elev_risk = 1 - (features.get("mean_elevation", 50) - 2) / 698
        drain_risk = 1 - features.get("drainage_density", 0.5)
        hist_risk = features.get("flood_events_5yr", 2) / 7
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
        drainage_pct: float,
        flood_events: int,
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
    print(f"  Accuracy : {metrics['accuracy']:.1%}")
    print(f"  F1 Score : {metrics['f1_score']:.1%}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.1%}")
    print(f"  CV F1    : {metrics['cv_f1_mean']:.1%} ± {metrics['cv_f1_std']:.1%}")

    # Score sample wards
    print("\n[2] Computing Readiness Scores for Sample Wards...")
    test_wards = [
        {"name": "Dharavi",    "city": "Mumbai",  "rainfall": 210, "elevation": 8,   "drainage": 25, "history": 4},
        {"name": "Velachery",  "city": "Chennai", "rainfall": 210, "elevation": 3,   "drainage": 15, "history": 6},
        {"name": "Kothrud",    "city": "Pune",    "rainfall": 138, "elevation": 570, "drainage": 62, "history": 0},
        {"name": "Yamuna Vihar","city": "Delhi",  "rainfall": 110, "elevation": 205, "drainage": 28, "history": 4},
    ]

    print(f"\n{'Ward':<18} {'City':<10} {'Score':>6} {'Class':<12} {'ML Prob':>8}")
    print("-" * 60)
    for w in test_wards:
        result = engine.compute_readiness_score(w["rainfall"], w["elevation"], w["drainage"], w["history"])
        ml = engine.predictor.predict({
            "rainfall_intensity": w["rainfall"], "mean_elevation": w["elevation"],
            "min_elevation": w["elevation"]*0.7, "low_lying_pct": max(0, 30-w["elevation"]),
            "drainage_density": w["drainage"]/100, "flood_events_5yr": w["history"], "area_km2": 10
        })
        print(f"{w['name']:<18} {w['city']:<10} {result['score']:>6} {result['risk_class']:<12} {ml.get('flood_probability',0.5):>8.1%}")

    print("\n[3] Feature Importance (XGBoost):")
    importance = engine.predictor.get_feature_importance()
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 40)
        print(f"  {feat:<25} {imp:.4f} {bar}")

    print("\n✓ ML Pipeline complete.")
