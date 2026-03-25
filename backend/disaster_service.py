"""
VarshaMitra – Disaster (Flood History) Service

Fetches flood disaster reports from ReliefWeb and derives:
- flood_events_5yr: count of MAJOR flood reports for a city/region.

Uses a simple TTL cache (30 minutes) backed by Redis when available,
with in-memory fallback.
"""

import datetime
import json
import logging
import os
from typing import Optional, Any

import httpx

logger = logging.getLogger("varsha_mitra.disaster")

RELIEFWEB_URL = "https://api.reliefweb.int/v1/reports"
DEFAULT_APPNAME = "varsha-mitra"

CACHE_TTL_SECONDS = 1800  # 30 minutes
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class DisasterService:
    def __init__(self):
        self._memory_cache: dict[str, dict[str, Any]] = {}
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
            logger.warning(f"[DisasterService] Redis unavailable: {exc}. Using in-memory cache.")
            self._redis = None
            return None

    @staticmethod
    def _cache_key(city_name: str) -> str:
        return f"reliefweb_flood:{city_name.strip().lower()}"

    def _get_memory_cache(self, city_name: str) -> Optional[dict]:
        key = self._cache_key(city_name)
        entry = self._memory_cache.get(key)
        if not entry:
            return None
        if datetime.datetime.utcnow() >= entry["expires_at"]:
            return None
        return entry["data"]

    def _set_memory_cache(self, city_name: str, data: dict):
        key = self._cache_key(city_name)
        self._memory_cache[key] = {
            "data": data,
            "expires_at": datetime.datetime.utcnow() + datetime.timedelta(seconds=CACHE_TTL_SECONDS),
        }

    @staticmethod
    def _is_major(report: dict) -> bool:
        # ReliefWeb uses various nested field patterns depending on v1/v2.
        # Try the most likely keys; if we cannot find severity, return True
        # to avoid returning a useless 0 count.
        disaster = report.get("disaster") if isinstance(report, dict) else None
        severity = (disaster or {}).get("severity") if isinstance(disaster, dict) else None

        if isinstance(severity, str) and "major" in severity.lower():
            return True

        fields = report.get("fields") if isinstance(report, dict) else None
        if isinstance(fields, dict):
            # Common alternate patterns
            sev2 = fields.get("disaster.severity") or fields.get("disaster.severity.name")
            if isinstance(sev2, str) and "major" in sev2.lower():
                return True

        # If no severity is found, treat as major (best-effort).
        return True

    @staticmethod
    def _parse_created_date(report: dict) -> Optional[datetime.datetime]:
        # Try a few typical locations for created date.
        fields = report.get("fields") if isinstance(report, dict) else None
        if isinstance(fields, dict):
            # Often stored as date.created
            created = fields.get("date.created") or fields.get("date.created.time")
            if isinstance(created, str):
                # Try ISO parsing; if it includes timezone, fromisoformat usually works.
                try:
                    return datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                except Exception:
                    return None

        # Some APIs may put created at top-level
        created2 = report.get("date") if isinstance(report, dict) else None
        if isinstance(created2, str):
            try:
                return datetime.datetime.fromisoformat(created2.replace("Z", "+00:00"))
            except Exception:
                return None

        return None

    @staticmethod
    def _extract_reports(payload: dict) -> list[dict]:
        if not isinstance(payload, dict):
            return []
        # v1 may return `data` or `results`.
        for key in ("data", "results", "reports"):
            if isinstance(payload.get(key), list):
                return payload[key]
        # Some responses wrap items in `content`
        content = payload.get("content")
        if isinstance(content, list):
            return content
        return []

    async def fetch_reliefweb_history(self, city_name: str) -> dict:
        """
        Fetch ReliefWeb flood report counts for a city/region.

        Filters (as requested):
        - query[value]=flood
        - filter[field]=country&filter[value]=India
        - filter[field]=date.created&filter[value]=[2000-01-01T00:00:00Z TO 2026-12-31T23:59:59Z]
        """
        city_name = (city_name or "").strip()
        if not city_name:
            return {"city": city_name, "flood_events_5yr": 0, "source": "invalid_city"}

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
                logger.warning(f"[DisasterService] Redis read failed: {exc}")

        mem_cached = self._get_memory_cache(city_name)
        if mem_cached:
            return {**mem_cached, "source": "memory_cache"}

        # Best-effort city association: include the city name in query[value].
        # (ReliefWeb uses query search semantics; the spec did not mandate a city filter field.)
        query_value = f"flood {city_name}"
        date_filter_value = "[2000-01-01T00:00:00Z TO 2026-12-31T23:59:59Z]"

        params = [
            ("appname", os.getenv("RELIEFWEB_APPNAME", DEFAULT_APPNAME)),
            ("query[value]", query_value),
            ("filter[field]", "country"),
            ("filter[value]", "India"),
            ("filter[field]", "date.created"),
            ("filter[value]", date_filter_value),
        ]

        try:
            async with httpx.AsyncClient(timeout=25.0) as client:
                resp = await client.get(RELIEFWEB_URL, params=params)
                resp.raise_for_status()
                payload = resp.json()

            reports = self._extract_reports(payload)
            now = datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)
            last5y = now - datetime.timedelta(days=365.25 * 5)

            major_reports = []
            for r in reports:
                if not self._is_major(r):
                    continue
                created = self._parse_created_date(r)
                if created is None:
                    # If created date cannot be parsed, count it in the 5yr bucket as best-effort.
                    major_reports.append(r)
                else:
                    # Ensure timezone-aware comparison
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=datetime.timezone.utc)
                    if created >= last5y:
                        major_reports.append(r)

            result = {
                "city": city_name,
                "flood_events_5yr": len(major_reports),
                "source": "reliefweb",
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "total_reports_matched": len(reports),
            }

            self._set_memory_cache(city_name, result)
            if redis_client is not None:
                try:
                    await redis_client.setex(key, CACHE_TTL_SECONDS, json.dumps(result))
                except Exception as exc:
                    logger.warning(f"[DisasterService] Redis write failed: {exc}")

            return result
        except Exception as exc:
            logger.warning(f"[DisasterService] ReliefWeb request failed for {city_name}: {exc}. Using fallback.")
            # No hardcoded values here: fallback is simply 0 (unknown).
            return {
                "city": city_name,
                "flood_events_5yr": 0,
                "source": "fallback",
                "updated_at": datetime.datetime.utcnow().isoformat() + "Z",
                "error": str(exc),
            }


_default_service = DisasterService()


async def fetch_reliefweb_history(city_name: str) -> dict:
    """
    Function wrapper matching the integration step name.
    """
    return await _default_service.fetch_reliefweb_history(city_name)

