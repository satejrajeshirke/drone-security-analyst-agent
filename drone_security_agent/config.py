"""Configuration loading (YAML -> typed dataclasses with safe defaults)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Optional

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


@dataclass
class Zone:
    name: str
    lat: float
    lon: float
    restricted: bool = False


@dataclass
class RuleConfig:
    night_start: time = time(22, 0)
    night_end: time = time(6, 0)
    business_start: time = time(7, 0)
    business_end: time = time(19, 0)
    loitering_min_dwell_s: int = 120
    visit_gap_s: int = 600
    repeat_vehicle_entries: int = 2
    unattended_package_s: int = 1800
    alert_cooldown_s: int = 600
    low_battery_pct: float = 25
    critical_battery_pct: float = 15
    max_altitude_m: float = 120
    geofence_radius_m: float = 400
    whitelisted_plates: list[str] = field(default_factory=list)
    suspicious_keywords: list[str] = field(default_factory=list)

    @staticmethod
    def _in_window(t: time, start: time, end: time) -> bool:
        if start <= end:
            return start <= t < end
        return t >= start or t < end  # wraps midnight

    def is_night(self, ts: datetime) -> bool:
        return self._in_window(ts.time(), self.night_start, self.night_end)

    def is_business_hours(self, ts: datetime) -> bool:
        return self._in_window(ts.time(), self.business_start, self.business_end)


@dataclass
class AppConfig:
    property_name: str
    dock_lat: float
    dock_lon: float
    zones: dict[str, Zone]
    rules: RuleConfig

    def restricted_zones(self) -> set[str]:
        return {z.name for z in self.zones.values() if z.restricted}

    def zone_for_position(self, lat: float, lon: float) -> str:
        """Nearest-zone geofence lookup (prototype-grade: nearest centroid)."""
        best, best_d = "unknown", math.inf
        for z in self.zones.values():
            d = (z.lat - lat) ** 2 + (z.lon - lon) ** 2
            if d < best_d:
                best, best_d = z.name, d
        return best


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    prop = raw.get("property", {})
    zones = {
        name: Zone(name=name, lat=z["lat"], lon=z["lon"], restricted=bool(z.get("restricted", False)))
        for name, z in prop.get("zones", {}).items()
    }
    r = raw.get("rules", {})
    rc = RuleConfig()
    if "night_hours" in r:
        rc.night_start = _parse_hhmm(r["night_hours"]["start"])
        rc.night_end = _parse_hhmm(r["night_hours"]["end"])
    if "business_hours" in r:
        rc.business_start = _parse_hhmm(r["business_hours"]["start"])
        rc.business_end = _parse_hhmm(r["business_hours"]["end"])
    for key in (
        "loitering_min_dwell_s", "visit_gap_s", "repeat_vehicle_entries", "unattended_package_s",
        "alert_cooldown_s", "low_battery_pct", "critical_battery_pct", "max_altitude_m",
        "geofence_radius_m", "whitelisted_plates", "suspicious_keywords",
    ):
        if key in r:
            setattr(rc, key, r[key])
    rc.whitelisted_plates = [p.upper().replace(" ", "") for p in rc.whitelisted_plates]
    rc.suspicious_keywords = [k.lower() for k in rc.suspicious_keywords]
    dock = prop.get("dock", {"lat": 0.0, "lon": 0.0})
    return AppConfig(
        property_name=prop.get("name", "Property"),
        dock_lat=dock["lat"],
        dock_lon=dock["lon"],
        zones=zones,
        rules=rc,
    )
