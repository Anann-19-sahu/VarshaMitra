"""
VarshaMitra – Drainage Infrastructure Service
Fetches OSM waterway/drainage lines from Overpass API for a city.

Features:
- Async Overpass calls via httpx
- Redis-backed TTL cache (30 minutes), with in-memory fallback
- Graceful fallback when Overpass is unavailable
"""

import datetime
import json
import logging
import math
import os
from typing import Optional

import httpx

logger = logging.getLogger("varsha_mitra.drainage")

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_TTL_SECONDS = 1800  # 30 minutes
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

FALLBACK_DRAINAGE = {
    "Mumbai": 28.0,
    "Pune": 56.0,
    "Delhi": 34.0,
    "Chennai": 24.0,
    "Kolkata": 30.0,
}


class DrainageService:
    def __init__(self):
        self._memory_cache: dict[str, dict] = {}
        self._redis = None

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as redis  # type: ignore

            client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)
            await client.ping()
            self._redis = client
            return self._redis
        except Exception as exc:
            logger.warning(f"[DrainageService] Redis unavailable: {exc}. Using in-memory cache.")
            self._redis = None
            return None

    @staticmethod
    def _cache_key(city_name: str) -> str:
        return f"drainage:{city_name.strip().lower()}"

    def _get_memory_cache(self, city_name: str) -> Optional[dict]:
        entry = self._memory_cache.get(self._cache_key(city_name))
        if not entry:
            return None
        if datetime.datetime.utcnow() >= entry["expires_at"]:
            return None
        return entry["data"]

    def _set_memory_cache(self, city_name: str, data: dict):
        self._memory_cache[self._cache_key(city_name)] = {
            "data": data,
            "expires_at": datetime.datetime.utcnow() + datetime.timedelta(seconds=CACHE_TTL_SECONDS),
        }

    @staticmethod
    def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371000.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _compute_total_length_m(self, overpass_data: dict) -> float:
        total = 0.0
        for el in overpass_data.get("elements", []):
            if el.get("type") != "way":
                continue
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            for i in range(1, len(geom)):
                p1 = geom[i - 1]
                p2 = geom[i]
                total += self._haversine_meters(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
        return total

    def _to_geojson(self, overpass_data: dict) -> dict:
        features = []
        for el in overpass_data.get("elements", []):
            if el.get("type") != "way":
                continue
            geom = el.get("geometry") or []
            if len(geom) < 2:
                continue
            coords = [[p["lon"], p["lat"]] for p in geom]
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "osm_id": el.get("id"),
                        "waterway": (el.get("tags") or {}).get("waterway", "unknown"),
                    },
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def _normalize_score(self, total_length_m: float) -> float:
        # Heuristic normalization for city-level network availability.
        # 0m => 0 score, >= 300km => 100 score.
        return max(0.0, min(100.0, (total_length_m / 300000.0) * 100.0))

    async def fetch_drainage_data(self, city_name: str) -> dict:
        city_name = city_name.strip()
        if not city_name:
            return self._fallback("Unknown", reason="empty city")

        redis_client = await self._get_redis()
        key = self._cache_key(city_name)

        if redis_client is not None:
            try:
                payload = await redis_client.get(key)
                if payload:
                    cached = json.loads(payload)
                    cached["source"] = "redis_cache"
                    return cached
            except Exception as exc:
                logger.warning(f"[DrainageService] Redis read failed: {exc}")

        mem_cached = self._get_memory_cache(city_name)
        if mem_cached:
            return {**mem_cached, "source": "memory_cache"}

        query = (
            f'[out:json][timeout:25]; area[name="{city_name}"]; '
            'way["waterway"~"drain|canal|stream"](area); out geom;'
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(OVERPASS_URL, data={"data": query})
                resp.raise_for_status()
                overpass_data = resp.json()

            total_length_m = self._compute_total_length_m(overpass_data)
            drainage_score = round(self._normalize_score(total_length_m), 2)
            geojson = self._to_geojson(overpass_data)
            payload = {
                "city": city_name,
                "drainage_score": drainage_score,
                "drainage_density": drainage_score,
                "total_length_m": round(total_length_m, 2),
                "line_count": len(geojson.get("features", [])),
                "geojson": geojson,
                "source": "overpass",
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            }

            self._set_memory_cache(city_name, payload)
            if redis_client is not None:
                try:
                    await redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(payload))
                except Exception as exc:
                    logger.warning(f"[DrainageService] Redis write failed: {exc}")

            return payload
        except Exception as exc:
            logger.warning(f"[DrainageService] Overpass failed for {city_name}: {exc}. Using fallback.")
            return self._fallback(city_name, reason=str(exc))

    def _fallback(self, city_name: str, reason: str = "") -> dict:
        score = FALLBACK_DRAINAGE.get(city_name, 35.0)
        return {
            "city": city_name,
            "drainage_score": score,
            "drainage_density": score,
            "total_length_m": 0.0,
            "line_count": 0,
            "geojson": {"type": "FeatureCollection", "features": []},
            "source": "fallback",
            "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "fallback_reason": reason,
        }


# ─── Module-level convenience function (matches requested API) ──────────────
_default_service = DrainageService()


async def fetch_drainage_data(city_name: str) -> dict:
    """
    Convenience wrapper around the singleton drainage service.

    This is the function name requested in the integration steps.
    """
    return await _default_service.fetch_drainage_data(city_name)
