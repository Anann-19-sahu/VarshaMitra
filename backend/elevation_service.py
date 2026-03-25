"""
VarshaMitra – Elevation Service
Fetches terrain elevation data from OpenTopoData SRTM30m API (free, no API key).
https://api.opentopodata.org/v1/srtm30m?locations={lat},{lng}

Features:
- In-memory LRU-style cache keyed by rounded coordinates (2 decimal places)
- Batch endpoint support for up to 100 locations per request
- Derives ML features: mean_elevation, low_lying_pct from the returned value
- Falls back gracefully on API errors
"""

import httpx
import asyncio
import datetime
import logging
import math
from typing import Optional

logger = logging.getLogger("varsha_mitra.elevation")

# ─── Config ──────────────────────────────────────────────────
OPENTOPODATA_URL  = "https://api.opentopodata.org/v1/srtm30m"
CACHE_TTL_SECONDS = 86400   # 24 hours — terrain doesn't change
MAX_CACHE_SIZE    = 10_000  # entries
COORD_PRECISION   = 2       # round lat/lng to 2dp (~1km grid)


class ElevationService:
    """
    Async elevation lookup via OpenTopoData SRTM30m (30-metre resolution).
    Results are cached permanently in-memory for the session (terrain is static).
    """

    def __init__(self):
        # cache key: "lat,lng"  →  {"elevation_m": float, "fetched_at": datetime}
        self._cache: dict[str, dict] = {}

    def _cache_key(self, lat: float, lng: float) -> str:
        return f"{round(lat, COORD_PRECISION)},{round(lng, COORD_PRECISION)}"

    def _get_cached(self, lat: float, lng: float) -> Optional[float]:
        entry = self._cache.get(self._cache_key(lat, lng))
        if not entry:
            return None
        age = (datetime.datetime.utcnow() - entry["fetched_at"]).total_seconds()
        if age > CACHE_TTL_SECONDS:
            return None
        return entry["elevation_m"]

    def _set_cache(self, lat: float, lng: float, elevation_m: float):
        if len(self._cache) >= MAX_CACHE_SIZE:
            # Evict oldest entry
            oldest = min(self._cache, key=lambda k: self._cache[k]["fetched_at"])
            del self._cache[oldest]
        self._cache[self._cache_key(lat, lng)] = {
            "elevation_m": elevation_m,
            "fetched_at":  datetime.datetime.utcnow(),
        }

    async def get_elevation(self, lat: float, lng: float) -> dict:
        """
        Fetch elevation for a single coordinate.

        Returns:
            {
              "lat": float,
              "lng": float,
              "elevation_m": float,       # SRTM30m elevation above sea level
              "low_lying_pct": float,     # estimated % of surrounding terrain below 10m
              "source": "opentopodata" | "cache" | "fallback",
              "dataset": "srtm30m"
            }
        """
        cached = self._get_cached(lat, lng)
        if cached is not None:
            return self._build_result(lat, lng, cached, source="cache")

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    OPENTOPODATA_URL,
                    params={"locations": f"{lat},{lng}"}
                )
                resp.raise_for_status()
                data = resp.json()

            results = data.get("results", [])
            if not results or results[0].get("elevation") is None:
                raise ValueError("No elevation in response")

            elevation_m = float(results[0]["elevation"])
            self._set_cache(lat, lng, elevation_m)
            logger.info(f"[ElevationService] ({lat},{lng}) → {elevation_m:.1f}m from OpenTopoData")
            return self._build_result(lat, lng, elevation_m, source="opentopodata")

        except Exception as exc:
            logger.warning(f"[ElevationService] API failed for ({lat},{lng}): {exc}. Using fallback.")
            return self._fallback(lat, lng)

    async def get_elevation_batch(self, coordinates: list[tuple[float, float]]) -> list[dict]:
        """
        Fetch elevation for up to 100 coordinates in a single API call.
        Uses cache where available; only hits API for uncached coordinates.

        Args:
            coordinates: list of (lat, lng) tuples

        Returns:
            list of elevation dicts in the same order as input
        """
        results = [None] * len(coordinates)
        uncached_indices = []
        uncached_coords  = []

        # Phase 1: check cache
        for i, (lat, lng) in enumerate(coordinates):
            cached = self._get_cached(lat, lng)
            if cached is not None:
                results[i] = self._build_result(lat, lng, cached, source="cache")
            else:
                uncached_indices.append(i)
                uncached_coords.append((lat, lng))

        if not uncached_coords:
            return results

        # Phase 2: batch-fetch from API
        locations_str = "|".join(f"{lat},{lng}" for lat, lng in uncached_coords)
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    OPENTOPODATA_URL,
                    params={"locations": locations_str}
                )
                resp.raise_for_status()
                api_results = resp.json().get("results", [])

            for idx, api_idx in enumerate(uncached_indices):
                lat, lng = uncached_coords[idx]
                if idx < len(api_results) and api_results[idx].get("elevation") is not None:
                    elev = float(api_results[idx]["elevation"])
                    self._set_cache(lat, lng, elev)
                    results[api_idx] = self._build_result(lat, lng, elev, source="opentopodata")
                else:
                    results[api_idx] = self._fallback(lat, lng)

        except Exception as exc:
            logger.warning(f"[ElevationService] Batch API failed: {exc}")
            for idx, api_idx in enumerate(uncached_indices):
                lat, lng = uncached_coords[idx]
                results[api_idx] = self._fallback(lat, lng)

        return results

    def _build_result(self, lat: float, lng: float, elevation_m: float, source: str) -> dict:
        """Build the standard result dict including derived ML features."""
        return {
            "lat":           lat,
            "lng":           lng,
            "elevation_m":   round(elevation_m, 2),
            "low_lying_pct": self._estimate_low_lying_pct(elevation_m),
            "source":        source,
            "dataset":       "srtm30m",
        }

    def _estimate_low_lying_pct(self, elevation_m: float) -> float:
        """
        Estimate % of surrounding terrain that is low-lying (<10m).
        This is a proxy used when we only have the centroid elevation.
        Low elevation → higher low_lying_pct (flood risk increases).
        """
        if elevation_m <= 0:
            return 95.0
        elif elevation_m <= 5:
            return 80.0
        elif elevation_m <= 10:
            return 55.0
        elif elevation_m <= 20:
            return 30.0
        elif elevation_m <= 50:
            return 15.0
        elif elevation_m <= 100:
            return 5.0
        else:
            return 1.0

    def _fallback(self, lat: float, lng: float) -> dict:
        """
        Fallback: estimate elevation from known city altitude ranges.
        Mumbai/Chennai/Kolkata coast → ~5–15m, Delhi → ~210m, Pune → ~550m
        """
        # crude lookup by latitude band
        if 12.5 <= lat <= 13.5:   # Chennai
            elev = 6.0
        elif 18.4 <= lat <= 18.7: # Pune
            elev = 550.0
        elif 19.0 <= lat <= 19.3: # Mumbai
            elev = 10.0
        elif 22.4 <= lat <= 22.7: # Kolkata
            elev = 7.0
        elif 28.5 <= lat <= 29.0: # Delhi
            elev = 210.0
        else:
            elev = 50.0

        return {
            "lat":           lat,
            "lng":           lng,
            "elevation_m":   elev,
            "low_lying_pct": self._estimate_low_lying_pct(elev),
            "source":        "fallback",
            "dataset":       "srtm30m",
        }

    def get_cache_stats(self) -> dict:
        return {
            "cached_locations": len(self._cache),
            "max_cache_size":   MAX_CACHE_SIZE,
            "ttl_hours":        CACHE_TTL_SECONDS / 3600,
        }
