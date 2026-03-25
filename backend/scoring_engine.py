"""
VarshaMitra – Backend Scoring Engine

Lightweight scoring logic used to convert OSM drainage geometry into a
0–100 drainage capacity score, and then combine it into a readiness score.

Note: the project also contains a heavier ML pipeline under `ml/scoring_engine.py`.
This module exists to match the backend integration steps that expect
`backend/scoring_engine.py` to provide `FloodDataETL`.
"""

import math
import datetime
from typing import Optional


class FloodDataETL:
    @staticmethod
    def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0  # earth radius in meters
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def calculate_drainage_score(self, osm_data: dict) -> float:
        """
        Calculate drainage capacity score (0–100) from Overpass OSM data.

        Score increases with total length of drainage-related OSM ways.
        """
        if not osm_data:
            return 0.0

        total_length_m = 0.0
        for el in osm_data.get("elements", []):
            if el.get("type") != "way":
                continue
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            for i in range(1, len(geom)):
                p1, p2 = geom[i - 1], geom[i]
                total_length_m += self._haversine_m(p1["lat"], p1["lon"], p2["lat"], p2["lon"])

        # Heuristic normalization: 0m -> 0 score, 300km+ -> 100 score.
        return max(0.0, min(100.0, (total_length_m / 300000.0) * 100.0))


class FloodScoringEngine:
    def __init__(self):
        self.etl = FloodDataETL()

    def compute_readiness_score(
        self,
        rainfall_mm: float,
        elevation_m: float,
        drainage_pct: Optional[float],
        flood_events: int,
        osm_data: Optional[dict] = None,
        weights: Optional[dict] = None,
    ) -> dict:
        """
        Ward Flood Readiness Score (0–100).

        If `osm_data` is provided, drainage_pct is *replaced* by a computed
        drainage score derived from OSM geometry.
        """
        if weights is None:
            weights = {"rainfall": 0.40, "elevation": 0.30, "drainage": 0.20, "history": 0.10}

        if osm_data is not None:
            drainage_pct = self.etl.calculate_drainage_score(osm_data)

        if drainage_pct is None:
            drainage_pct = 50.0

        # Normalize risk factors to [0, 1] — higher = more dangerous
        rain_risk = (rainfall_mm - 50.0) / (350.0 - 50.0)
        elev_risk = 1.0 - (elevation_m - 2.0) / (700.0 - 2.0)
        drain_risk = 1.0 - (drainage_pct - 10.0) / (90.0 - 10.0)
        hist_risk = flood_events / 8.0

        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, x))

        rain_risk = clamp01(rain_risk)
        elev_risk = clamp01(elev_risk)
        drain_risk = clamp01(drain_risk)
        hist_risk = clamp01(hist_risk)

        composite_risk = (
            rain_risk * weights["rainfall"]
            + elev_risk * weights["elevation"]
            + drain_risk * weights["drainage"]
            + hist_risk * weights["history"]
        )

        readiness = int(round((1.0 - composite_risk) * 100))
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
            "computed_at": datetime.datetime.utcnow().isoformat() + "Z",
            "weights_used": weights,
        }

