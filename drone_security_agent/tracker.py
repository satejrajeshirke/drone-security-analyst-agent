"""Entity tracking / lightweight re-identification across frames.

Turns per-frame detections into persistent *entities* so the agent can say
"the blue Ford F150 entered twice today" or "person has been at the gate for
3 minutes". Identity keys (prototype-grade re-ID):

* vehicle : licence plate when visible, else colour+make+model+type signature.
            Once a signature has been seen with a plate the two are linked, so
            later plate-less sightings resolve to the same vehicle.
* person  : clothing signature ("dark hoodie") + zone fallback.
* others  : type + zone (package at front door, gate, hazard ...).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .models import Detection, pretty_zone

CLOTHING_SUFFIXES = ("hoodie", "jacket", "vest", "shirt", "coat", "uniform", "helmet", "mask", "cap", "raincoat")


@dataclass
class Visit:
    start: datetime
    end: datetime
    entered: bool = False


@dataclass
class EntityState:
    entity_id: str
    object_type: str
    label: str
    first_seen: datetime
    last_seen: datetime
    plate: str = ""
    sightings: int = 0
    entries: list[datetime] = field(default_factory=list)
    visits: list[Visit] = field(default_factory=list)
    zones: list[str] = field(default_factory=list)
    last_action: str = ""
    last_zone: str = ""

    @property
    def current_visit(self) -> Visit:
        return self.visits[-1]

    @property
    def dwell_s(self) -> float:
        v = self.current_visit
        return (v.end - v.start).total_seconds()

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "object_type": self.object_type,
            "label": self.label,
            "plate": self.plate,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "sightings": self.sightings,
            "entries": [e.isoformat() for e in self.entries],
            "visits": len(self.visits),
            "zones": self.zones,
            "last_action": self.last_action,
        }


class EntityTracker:
    def __init__(self, visit_gap_s: int = 600):
        self.visit_gap_s = visit_gap_s
        self.entities: dict[str, EntityState] = {}
        self._sig_to_id: dict[str, str] = {}

    # ------------------------------------------------------------------ #
    def _key(self, d: Detection) -> str:
        if d.object_type == "vehicle":
            sig = "veh:" + "-".join(x.lower() for x in (d.color, d.make, d.model, d.subtype) if x)
            if d.plate:
                pid = f"veh:{d.plate}"
                self._sig_to_id[sig] = pid
                return pid
            return self._sig_to_id.get(sig, sig)
        if d.object_type == "person":
            clothing = next((a for a in d.attributes if a.endswith(CLOTHING_SUFFIXES)), "")
            if clothing:
                return "per:" + clothing.replace(" ", "-")
            group = f"x{d.count}" if d.count > 1 else ""
            return f"per:unknown{group}@{d.zone}"
        return f"{d.object_type[:3]}:{d.subtype}@{d.zone}"

    def update(self, d: Detection, ts: datetime) -> EntityState:
        eid = self._key(d)
        d.entity_id = eid
        e: Optional[EntityState] = self.entities.get(eid)
        if e is None:
            e = EntityState(entity_id=eid, object_type=d.object_type, label=d.label,
                            first_seen=ts, last_seen=ts, plate=d.plate)
            e.visits.append(Visit(ts, ts))
            self.entities[eid] = e
        else:
            if (ts - e.last_seen).total_seconds() > self.visit_gap_s:
                e.visits.append(Visit(ts, ts))
            else:
                e.current_visit.end = ts
            e.last_seen = ts
            # prefer the most specific label we have seen
            if len(d.label) > len(e.label):
                e.label = d.label
            e.plate = e.plate or d.plate
        e.sightings += 1
        e.last_action = d.action
        e.last_zone = d.zone
        if d.zone and d.zone not in e.zones:
            e.zones.append(d.zone)
        if d.action == "entering" and not e.current_visit.entered:
            e.current_visit.entered = True
            e.entries.append(ts)
        return e

    def get(self, entity_id: str) -> Optional[EntityState]:
        return self.entities.get(entity_id)

    def describe(self, entity_id: str) -> str:
        e = self.entities[entity_id]
        return (f"{e.label} ({e.entity_id}) seen {e.sightings}x, {len(e.visits)} visit(s), "
                f"{len(e.entries)} entr{'y' if len(e.entries) == 1 else 'ies'}, zones: "
                f"{', '.join(pretty_zone(z) for z in e.zones)}")
