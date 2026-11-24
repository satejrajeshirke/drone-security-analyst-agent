"""Core data structures shared by every stage of the pipeline.

Everything is a plain dataclass so it can be serialised to JSON / SQLite
without an ORM and is trivial to construct in tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Optional


class Severity(IntEnum):
    """Alert severity. IntEnum so alerts can be sorted / compared."""

    INFO = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    CRITICAL = 5

    @classmethod
    def parse(cls, value: "str | int | Severity") -> "Severity":
        if isinstance(value, Severity):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[str(value).upper()]


@dataclass
class Telemetry:
    """One drone telemetry sample (simulated, but shaped like a real MAVLink/DJI feed)."""

    ts: datetime
    lat: float
    lon: float
    altitude_m: float
    heading_deg: float
    speed_mps: float
    battery_pct: float
    zone: str  # semantic location derived from the geofence map, e.g. "main_gate"
    flight_mode: str = "PATROL"  # PATROL | HOVER | RTL | DOCKED

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Telemetry":
        d = dict(d)
        d["ts"] = datetime.fromisoformat(d["ts"])
        return cls(**d)


@dataclass
class Frame:
    """A video frame. In the prototype `description` stands in for the pixels:
    it is what a VLM would emit for the frame. `image_path` is set when a real
    image is processed through a VLM backend instead."""

    frame_id: int
    ts: datetime
    description: str
    telemetry: Telemetry
    image_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "ts": self.ts.isoformat(),
            "description": self.description,
            "telemetry": self.telemetry.to_dict(),
            "image_path": self.image_path,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Frame":
        return cls(
            frame_id=int(d["frame_id"]),
            ts=datetime.fromisoformat(d["ts"]),
            description=d["description"],
            telemetry=Telemetry.from_dict(d["telemetry"]),
            image_path=d.get("image_path"),
        )


@dataclass
class Detection:
    """A structured object/event extracted from one frame."""

    object_type: str  # person | vehicle | animal | package | gate | fire | ...
    subtype: str = ""  # truck, van, sedan, dog ...
    color: str = ""
    make: str = ""
    model: str = ""
    plate: str = ""
    count: int = 1
    action: str = ""  # entering, exiting, loitering, parked, climbing ...
    zone: str = ""
    attributes: list[str] = field(default_factory=list)  # free-form cues: hoodie, flashlight, masked ...
    confidence: float = 0.9
    entity_id: str = ""  # filled by the tracker

    @property
    def label(self) -> str:
        """Human-readable label, e.g. 'Blue Ford F150' or 'Person'."""
        if self.object_type == "vehicle":
            parts = [self.color.title(), self.make, self.model]
            name = " ".join(p for p in parts if p)
            if not (self.make or self.model):
                name = " ".join(p for p in [self.color.title(), self.subtype or "vehicle"] if p)
            name = name.strip() or "Vehicle"
            return name[0].upper() + name[1:]
        base = self.subtype or self.object_type
        if self.count <= 1:
            return base.capitalize()
        plural = "people" if base == "person" else (base if base.endswith("s") else base + "s")
        return f"{self.count} {plural}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["label"] = self.label
        return d


@dataclass
class Alert:
    rule_id: str
    severity: Severity
    message: str
    ts: datetime
    frame_id: int
    zone: str = ""
    entity_id: str = ""
    recommendation: str = ""
    related_rules: list[str] = field(default_factory=list)  # other rules merged into this alert

    def __str__(self) -> str:
        return f"[{self.severity.name}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "related_rules": list(self.related_rules),
            "rule_id": self.rule_id,
            "severity": self.severity.name,
            "message": self.message,
            "ts": self.ts.isoformat(),
            "frame_id": self.frame_id,
            "zone": self.zone,
            "entity_id": self.entity_id,
            "recommendation": self.recommendation,
        }


@dataclass
class LogEntry:
    """Human-readable activity log line (the 'Logs' sample output in the brief)."""

    ts: datetime
    frame_id: int
    message: str
    entity_id: str = ""

    def __str__(self) -> str:  # "Blue Ford F150 spotted at garage, 12:00."
        return self.message


def hhmm(ts: datetime) -> str:
    return ts.strftime("%H:%M")


def pretty_zone(zone: str) -> str:
    return zone.replace("_", " ")
