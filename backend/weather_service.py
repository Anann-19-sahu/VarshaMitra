"""
VarshaMitra – Live Weather Service
Fetches real rainfall data from Open-Meteo API (free, no API key needed)
for the 5 Indian cities monitored by the platform.

Caches results for 30 minutes to avoid rate limiting.
Falls back gracefully to static baseline values if API is unavailable.
"""

import httpx
import asyncio
import datetime
import logging
from typing import Optional

logger = logging.getLogger("varsha_mitra.weather")

# ─── City Coordinates ────────────────────────────────────────
# One representative lat/lng per city (wards inherit city rainfall)
CITY_COORDS = {
    "Mumbai":  {"lat": 19.0760, "lng": 72.8777},
    "Pune":    {"lat": 18.5204, "lng": 73.8567},
    "Delhi":   {"lat": 28.7041, "lng": 77.1025},
    "Chennai": {"lat": 13.0827, "lng": 80.2707},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639},
}

# ─── Fallback static baseline rainfall (mm) ──────────────────
# Used when Open-Meteo is unreachable
FALLBACK_RAINFALL = {
    "Mumbai":  168.0,
    "Pune":    112.0,
    "Delhi":   72.0,
    "Chennai": 145.0,
    "Kolkata": 158.0,
}

# ─── Open-Meteo endpoint ─────────────────────────────────────
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

CACHE_TTL_SECONDS = 1800  # 30 minutes


class WeatherService:
    """
    Fetches live precipitation data from Open-Meteo for Indian cities.
    Results are cached for 30 minutes per city to stay within free tier limits.
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}   # city → {"data": ..., "expires_at": datetime}

    def _is_cache_valid(self, city: str) -> bool:
        entry = self._cache.get(city)
        if not entry:
            return False
        return datetime.datetime.utcnow() < entry["expires_at"]

    def _set_cache(self, city: str, data: dict):
        self._cache[city] = {
            "data": data,
            "expires_at": datetime.datetime.utcnow() + datetime.timedelta(seconds=CACHE_TTL_SECONDS),
        }

    async def fetch_city_weather(self, city: str) -> dict:
        """
        Fetch live weather for a city.
        Returns:
            {
              "city": str,
              "rainfall_mm": float,         # today's accumulated precipitation
              "rainfall_24h_mm": float,     # last 24h sum (from daily API)
              "forecast_72h": [float, float, float],  # next 3 days precipitation_sum in mm
              "source": "open_meteo" | "cache" | "fallback",
              "updated_at": str (ISO datetime),
            }
        """
        if self._is_cache_valid(city):
            entry = self._cache[city]["data"]
            return {**entry, "source": "cache"}

        coords = CITY_COORDS.get(city)
        if not coords:
            return self._fallback(city, reason="unknown city")

        params = {
            "latitude":      coords["lat"],
            "longitude":     coords["lng"],
            "daily":         "precipitation_sum",
            "hourly":        "precipitation",
            "forecast_days": 4,          # today + 3 forecast days
            "timezone":      "Asia/Kolkata",
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(OPEN_METEO_URL, params=params)
                resp.raise_for_status()
                raw = resp.json()

            # ── Parse daily precipitation ──────────────────────
            daily_precip = raw.get("daily", {}).get("precipitation_sum", [])
            # daily_precip[0] = today, [1], [2], [3] = forecast days
            today_rain   = float(daily_precip[0]) if len(daily_precip) > 0 and daily_precip[0] is not None else FALLBACK_RAINFALL[city]
            forecast_72h = [
                float(daily_precip[i]) if len(daily_precip) > i and daily_precip[i] is not None else 0.0
                for i in range(1, 4)
            ]

            # ── Parse hourly to get last 24h accumulated sum ───
            hourly_precip = raw.get("hourly", {}).get("precipitation", [])
            # Open-Meteo returns hourly for all forecast_days; first 24 entries = today
            last_24h_sum = round(sum(
                h for h in hourly_precip[:24] if h is not None
            ), 2)

            data = {
                "city":           city,
                "rainfall_mm":    round(today_rain, 2),    # scoring uses this
                "rainfall_24h_mm": last_24h_sum,
                "forecast_72h":   [round(f, 2) for f in forecast_72h],
                "source":         "open_meteo",
                "updated_at":     datetime.datetime.utcnow().isoformat() + "Z",
            }
            self._set_cache(city, data)
            logger.info(f"[WeatherService] {city}: {today_rain:.1f}mm rainfall fetched from Open-Meteo")
            return data

        except Exception as exc:
            logger.warning(f"[WeatherService] Open-Meteo failed for {city}: {exc}. Using fallback.")
            return self._fallback(city, reason=str(exc))

    def _fallback(self, city: str, reason: str = "") -> dict:
        return {
            "city":            city,
            "rainfall_mm":     FALLBACK_RAINFALL.get(city, 150.0),
            "rainfall_24h_mm": FALLBACK_RAINFALL.get(city, 150.0),
            "forecast_72h":    [0.0, 0.0, 0.0],
            "source":          "fallback",
            "updated_at":      datetime.datetime.utcnow().isoformat() + "Z",
            "fallback_reason": reason,
        }

    async def fetch_all_cities(self) -> dict[str, dict]:
        """Fetch weather for all 5 cities concurrently."""
        results = await asyncio.gather(
            *[self.fetch_city_weather(city) for city in CITY_COORDS],
            return_exceptions=True
        )
        output = {}
        for city, result in zip(CITY_COORDS.keys(), results):
            if isinstance(result, Exception):
                output[city] = self._fallback(city, reason=str(result))
            else:
                output[city] = result
        return output

    def get_rainfall_for_scoring(self, city: str) -> float:
        """
        Synchronous helper to get cached rainfall for a city.
        Returns the cached or fallback rainfall_mm value.
        Used by scoring logic in main.py.
        """
        entry = self._cache.get(city)
        if entry:
            return entry["data"].get("rainfall_mm", FALLBACK_RAINFALL.get(city, 150.0))
        return FALLBACK_RAINFALL.get(city, 150.0)

    def get_forecast_for_city(self, city: str) -> list:
        """Return cached 72-h forecast list for a city."""
        entry = self._cache.get(city)
        if entry:
            return entry["data"].get("forecast_72h", [0.0, 0.0, 0.0])
        return [0.0, 0.0, 0.0]
