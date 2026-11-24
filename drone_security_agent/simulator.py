"""Simulated drone feed: telemetry + 'video frames' as text descriptions.

Two generators:
* `scripted_day()` - a hand-designed 24h scenario that exercises every rule
  (midnight loitering, fence climb, repeat vehicle, unattended package,
  drone low battery, altitude breach ...). Deterministic -> used by tests/demo.
* `random_frames(n)` - procedurally generated frames for scalability tests.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

from .config import AppConfig, load_config
from .models import Frame, Telemetry

SCENARIO_DATE = datetime(2026, 11, 23)


@dataclass
class ScriptedEvent:
    time: str  # "HH:MM" or "HH:MM:SS"
    zone: str
    description: str
    overrides: dict = field(default_factory=dict)  # telemetry overrides (battery, altitude ...)


# --------------------------------------------------------------------------- #
# Scripted day. Each line reads like a VLM caption of what the camera sees.
# --------------------------------------------------------------------------- #
SCRIPTED_EVENTS: list[ScriptedEvent] = [
    ScriptedEvent("00:01", "main_gate", "Person in dark hoodie standing near the main gate, looking around"),
    ScriptedEvent("00:02", "main_gate", "Person in dark hoodie still standing at the main gate, pacing slowly"),
    ScriptedEvent("00:03", "main_gate", "Person in dark hoodie loitering at the main gate, checking the gate latch"),
    ScriptedEvent("00:04", "main_gate", "Person in dark hoodie walking away from the main gate toward the road"),
    ScriptedEvent("00:40", "backyard", "Fox crossing the backyard lawn"),
    ScriptedEvent("02:15", "perimeter_fence", "Person climbing the perimeter fence near the north corner"),
    ScriptedEvent("02:16", "perimeter_fence", "Person in black jacket running along the inside of the perimeter fence carrying a bag"),
    ScriptedEvent("06:45", "parking_lot", "White Toyota Camry parked in the parking lot, plate MH12AB1234"),
    ScriptedEvent("07:30", "main_gate", "Blue Ford F150 pickup truck entering through the main gate, plate MH14CD5678"),
    ScriptedEvent("07:32", "driveway", "Blue Ford F150 driving along the driveway toward the garage"),
    ScriptedEvent("08:05", "loading_dock", "White Mercedes Sprinter delivery van parked at the loading dock, two people unloading boxes"),
    ScriptedEvent("08:20", "loading_dock", "White Mercedes Sprinter van exiting the loading dock"),
    ScriptedEvent("09:10", "front_door", "Package left at the front door by a courier in a yellow vest"),
    ScriptedEvent("09:12", "front_door", "Courier in yellow vest walking back toward the main gate"),
    ScriptedEvent("10:40", "front_door", "Package still at the front door, no one around"),
    ScriptedEvent("12:00", "garage", "Blue Ford F150 parked at the garage"),
    ScriptedEvent("13:15", "parking_lot", "Red Honda Civic parked in the parking lot"),
    ScriptedEvent("14:30", "backyard", "Dog running across the backyard"),
    ScriptedEvent("15:10", "driveway", "Clear view of the driveway, no activity", {"altitude_m": 131.0}),
    ScriptedEvent("16:20", "garage", "Blue Ford F150 exiting the garage toward the main gate"),
    ScriptedEvent("16:22", "main_gate", "Blue Ford F150 exiting through the main gate"),
    ScriptedEvent("17:45", "main_gate", "Blue Ford F150 pickup truck entering through the main gate, plate MH14CD5678"),
    ScriptedEvent("19:30", "main_gate", "Main gate left open, no one present"),
    ScriptedEvent("21:05", "parking_lot", "Red Honda Civic exiting the parking lot"),
    ScriptedEvent("22:30", "perimeter_fence", "Grey sedan with headlights off circling slowly on the perimeter road"),
    ScriptedEvent("23:10", "warehouse", "Two people near the warehouse door with a flashlight, testing the door handle"),
    ScriptedEvent("23:40", "driveway", "Clear view of the driveway, no activity", {"battery_pct": 14.0, "flight_mode": "RTL"}),
]

PATROL_TEMPLATES = [
    "Clear view of the {z}, no activity",
    "Routine patrol over the {z}, area quiet",
    "{Z} empty, lighting normal",
    "No people or vehicles visible at the {z}",
]


class TelemetryModel:
    """Very small physical model: battery drains while airborne, recharges at dock
    every `sortie_min` minutes; altitude depends on mode; position from zone."""

    def __init__(self, cfg: AppConfig, seed: int = 7, sortie_min: int = 90):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.sortie_min = sortie_min

    def sample(self, ts: datetime, zone: str, hovering: bool, overrides: Optional[dict] = None) -> Telemetry:
        z = self.cfg.zones.get(zone)
        lat = (z.lat if z else self.cfg.dock_lat) + self.rng.uniform(-0.00004, 0.00004)
        lon = (z.lon if z else self.cfg.dock_lon) + self.rng.uniform(-0.00004, 0.00004)
        minutes_into_sortie = (ts.hour * 60 + ts.minute) % self.sortie_min
        battery = round(100 - minutes_into_sortie * 0.8, 1)
        t = Telemetry(
            ts=ts,
            lat=round(lat, 6),
            lon=round(lon, 6),
            altitude_m=round(25.0 if hovering else 45.0 + self.rng.uniform(-3, 3), 1),
            heading_deg=round(self.rng.uniform(0, 359), 1),
            speed_mps=0.0 if hovering else round(self.rng.uniform(4, 8), 1),
            battery_pct=battery,
            zone=zone,
            flight_mode="HOVER" if hovering else "PATROL",
        )
        for k, v in (overrides or {}).items():
            setattr(t, k, v)
        return t


def _parse_time(day: datetime, s: str) -> datetime:
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.append(0)
    return day.replace(hour=parts[0], minute=parts[1], second=parts[2])


def scripted_day(cfg: Optional[AppConfig] = None, patrol_every_min: int = 20,
                 day: datetime = SCENARIO_DATE, seed: int = 7) -> list[Frame]:
    """Merge scripted events with routine patrol frames into one ordered feed."""
    cfg = cfg or load_config()
    tm = TelemetryModel(cfg, seed=seed)
    rng = random.Random(seed)
    items: list[tuple[datetime, str, str, dict, bool]] = []
    event_minutes = set()
    for ev in SCRIPTED_EVENTS:
        ts = _parse_time(day, ev.time)
        event_minutes.add(ts.hour * 60 + ts.minute)
        items.append((ts, ev.zone, ev.description, ev.overrides, True))

    zone_names = list(cfg.zones)
    for i, minute in enumerate(range(10, 24 * 60, patrol_every_min)):
        if any(abs(minute - m) < 3 for m in event_minutes):
            continue  # drone is busy tracking an event
        zone = zone_names[i % len(zone_names)]
        tmpl = rng.choice(PATROL_TEMPLATES)
        pretty = zone.replace("_", " ")
        desc = tmpl.format(z=pretty, Z=pretty.capitalize())
        items.append((day + timedelta(minutes=minute), zone, desc, {}, False))

    items.sort(key=lambda x: x[0])
    frames = []
    for fid, (ts, zone, desc, overrides, is_event) in enumerate(items, start=1):
        frames.append(Frame(frame_id=fid, ts=ts, description=desc,
                            telemetry=tm.sample(ts, zone, hovering=is_event, overrides=overrides)))
    return frames


# --------------------------------------------------------------------------- #
# Procedural generator for scale tests
# --------------------------------------------------------------------------- #
_COLORS = ["red", "blue", "white", "black", "grey", "silver", "green"]
_VEHICLES = [("Ford", "F150", "pickup truck"), ("Toyota", "Camry", "sedan"), ("Honda", "Civic", "sedan"),
             ("Mercedes", "Sprinter", "van"), ("Tesla", "Model 3", "sedan"), ("Mahindra", "Thar", "SUV")]
_ACTIONS_V = ["entering through the {z}", "parked at the {z}", "exiting the {z}", "driving past the {z}"]
_ACTIONS_P = ["walking near the {z}", "standing at the {z}", "loitering at the {z}", "running across the {z}"]
_ANIMALS = ["Dog", "Cat", "Fox", "Deer"]


def random_frames(n: int, cfg: Optional[AppConfig] = None, seed: int = 42,
                  start: datetime = SCENARIO_DATE, step_s: int = 2) -> Iterator[Frame]:
    """Yield `n` synthetic frames (streaming, O(1) memory)."""
    cfg = cfg or load_config()
    rng = random.Random(seed)
    tm = TelemetryModel(cfg, seed=seed)
    zones = list(cfg.zones)
    for i in range(n):
        ts = start + timedelta(seconds=i * step_s)
        zone = rng.choice(zones)
        z = zone.replace("_", " ")
        roll = rng.random()
        if roll < 0.45:
            desc = rng.choice(PATROL_TEMPLATES).format(z=z, Z=z.capitalize())
        elif roll < 0.75:
            make, model, kind = rng.choice(_VEHICLES)
            desc = f"{rng.choice(_COLORS).capitalize()} {make} {model} {kind} " + rng.choice(_ACTIONS_V).format(z=z)
        elif roll < 0.93:
            desc = f"Person in {rng.choice(_COLORS)} jacket " + rng.choice(_ACTIONS_P).format(z=z)
        else:
            desc = f"{rng.choice(_ANIMALS)} crossing the {z}"
        yield Frame(frame_id=i + 1, ts=ts, description=desc, telemetry=tm.sample(ts, zone, hovering=roll >= 0.45))


def save_frames(frames: list[Frame], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([f.to_dict() for f in frames], indent=2), encoding="utf-8")


def load_frames(path: str | Path) -> list[Frame]:
    return [Frame.from_dict(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]
