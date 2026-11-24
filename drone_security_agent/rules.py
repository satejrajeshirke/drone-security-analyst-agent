"""Real-time security & safety alert rules.

Each rule is a small pure function `(RuleContext) -> list[Alert]`, which makes
rules easy to unit-test, enable/disable and extend (an LLM drafted the
initial rule set from the brief; thresholds live in config/default.yaml).

The engine then
1. runs every rule on the frame,
2. merges alerts that concern the same entity in the same frame (the highest
   severity wins, the others are listed in `related_rules`), so an operator gets
   ONE actionable alert, not three, for "two people with a flashlight testing
   the warehouse door at 23:10",
3. applies a per-(rule, entity) cooldown so a person standing at the gate does
   not raise an alert every frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from .config import AppConfig, haversine_m
from .models import Alert, Detection, Frame, Severity, hhmm, pretty_zone
from .tracker import EntityTracker

ACTIVE_VEHICLE_ACTIONS = {"entering", "parked", "circling", "driving", "standing", ""}


@dataclass
class RuleContext:
    frame: Frame
    detections: list[Detection]
    tracker: EntityTracker
    cfg: AppConfig

    @property
    def ts(self) -> datetime:
        return self.frame.ts

    def zone_of(self, d: Detection) -> str:
        return d.zone or self.frame.telemetry.zone


Rule = Callable[[RuleContext], list[Alert]]
RECOMMENDATIONS = {
    "PERIMETER_BREACH": "Dispatch guard to the fence line, keep drone tracking the intruder, notify police.",
    "TAMPERING": "Dispatch guard immediately; trigger siren/spotlight; preserve footage as evidence.",
    "RESTRICTED_ZONE": "Verify identity via speaker challenge; dispatch guard if not authorised.",
    "NIGHT_LOITERING": "Activate gate floodlight and two-way speaker warning; continue drone overwatch.",
    "SUSPICIOUS_BEHAVIOR": "Increase patrol frequency in this zone and keep the drone on the subject.",
    "AFTER_HOURS_VEHICLE": "Capture plate, cross-check visitor list, notify site manager.",
    "REPEAT_VEHICLE": "Informational: confirm the vehicle is expected; add plate to whitelist if staff.",
    "UNATTENDED_PACKAGE": "Ask reception to collect the package; inspect if no delivery was expected.",
    "GATE_OPEN_AFTER_HOURS": "Close the gate remotely or send staff to secure it.",
    "HAZARD": "Call emergency services and alert everyone on site.",
    "DRONE_LOW_BATTERY": "Return drone to dock; hand over coverage to fixed cameras.",
    "DRONE_ALTITUDE": "Descend below the regulatory ceiling immediately.",
    "DRONE_GEOFENCE": "Return inside the geofence; check navigation / wind conditions.",
}


def _alert(ctx: RuleContext, rule_id: str, sev: Severity, msg: str, d: Detection | None = None,
           zone: str = "") -> Alert:
    return Alert(rule_id=rule_id, severity=sev, message=msg, ts=ctx.ts, frame_id=ctx.frame.frame_id,
                 zone=zone or (ctx.zone_of(d) if d else ctx.frame.telemetry.zone),
                 entity_id=d.entity_id if d else "", recommendation=RECOMMENDATIONS.get(rule_id, ""))


# ----------------------------- security rules ------------------------------ #
def perimeter_breach(ctx: RuleContext) -> list[Alert]:
    out = []
    for d in ctx.detections:
        if d.object_type == "person" and d.action == "climbing":
            out.append(_alert(ctx, "PERIMETER_BREACH", Severity.CRITICAL,
                              f"Perimeter breach: {d.label} climbing the {pretty_zone(ctx.zone_of(d))}, "
                              f"{hhmm(ctx.ts)}.", d))
    return out


def night_loitering(ctx: RuleContext) -> list[Alert]:
    out = []
    rc = ctx.cfg.rules
    for d in ctx.detections:
        if d.object_type != "person" or d.action in ("leaving", "running", "climbing"):
            continue
        e = ctx.tracker.get(d.entity_id)
        if e is None:
            continue
        long_dwell = e.dwell_s >= rc.loitering_min_dwell_s
        if not (long_dwell or d.action == "loitering"):
            continue
        night = rc.is_night(ctx.ts)
        sev = Severity.HIGH if night else Severity.MEDIUM
        start = e.current_visit.start
        out.append(_alert(
            ctx, "NIGHT_LOITERING" if night else "LOITERING", sev,
            f"{d.label} loitering at {pretty_zone(ctx.zone_of(d))}, {hhmm(start)} "
            f"(dwell {int(e.dwell_s // 60)} min, still present {hhmm(ctx.ts)}).", d))
    return out


def tampering(ctx: RuleContext) -> list[Alert]:
    out = []
    for d in ctx.detections:
        if d.object_type == "person" and d.action == "tampering":
            sev = Severity.CRITICAL if not ctx.cfg.rules.is_business_hours(ctx.ts) else Severity.HIGH
            out.append(_alert(ctx, "TAMPERING", sev,
                              f"{d.label} tampering with entry point at {pretty_zone(ctx.zone_of(d))}, "
                              f"{hhmm(ctx.ts)}.", d))
    return out


def restricted_zone(ctx: RuleContext) -> list[Alert]:
    out = []
    restricted = ctx.cfg.restricted_zones()
    for d in ctx.detections:
        zone = ctx.zone_of(d)
        if d.object_type != "person" or zone not in restricted or d.action == "climbing":
            continue
        if ctx.cfg.rules.is_business_hours(ctx.ts):
            continue
        sev = Severity.CRITICAL if ctx.cfg.rules.is_night(ctx.ts) else Severity.HIGH
        out.append(_alert(ctx, "RESTRICTED_ZONE", sev,
                          f"{d.label} in restricted zone ({pretty_zone(zone)}) after hours, {hhmm(ctx.ts)}.", d))
    return out


def suspicious_behavior(ctx: RuleContext) -> list[Alert]:
    out = []
    for d in ctx.detections:
        if d.object_type in ("person", "vehicle") and d.attributes:
            cues = [a for a in d.attributes if a in ctx.cfg.rules.suspicious_keywords]
            if cues:
                out.append(_alert(ctx, "SUSPICIOUS_BEHAVIOR", Severity.HIGH,
                                  f"Suspicious activity: {d.label} at {pretty_zone(ctx.zone_of(d))} "
                                  f"({', '.join(cues)}), {hhmm(ctx.ts)}.", d))
    return out


def after_hours_vehicle(ctx: RuleContext) -> list[Alert]:
    out = []
    rc = ctx.cfg.rules
    if rc.is_business_hours(ctx.ts):
        return out
    for d in ctx.detections:
        if d.object_type != "vehicle" or d.action not in ACTIVE_VEHICLE_ACTIONS:
            continue
        e = ctx.tracker.get(d.entity_id)
        plate = (e.plate if e else d.plate) or ""
        if plate and plate in rc.whitelisted_plates:
            continue
        out.append(_alert(ctx, "AFTER_HOURS_VEHICLE", Severity.MEDIUM,
                          f"Unrecognised {d.label} {d.action or 'present'} at {pretty_zone(ctx.zone_of(d))} "
                          f"after hours, {hhmm(ctx.ts)}.", d))
    return out


def repeat_vehicle(ctx: RuleContext) -> list[Alert]:
    out = []
    n = ctx.cfg.rules.repeat_vehicle_entries
    for d in ctx.detections:
        if d.object_type != "vehicle" or d.action != "entering":
            continue
        e = ctx.tracker.get(d.entity_id)
        if e and len(e.entries) >= n and e.entries[-1] == ctx.ts:
            times = ", ".join(hhmm(t) for t in e.entries)
            word = {2: "twice", 3: "three times"}.get(len(e.entries), f"{len(e.entries)} times")
            out.append(_alert(ctx, "REPEAT_VEHICLE", Severity.INFO,
                              f"{e.label} entered {word} today ({times}).", d))
    return out


def unattended_package(ctx: RuleContext) -> list[Alert]:
    out = []
    for d in ctx.detections:
        if d.object_type != "package":
            continue
        e = ctx.tracker.get(d.entity_id)
        if e and (ctx.ts - e.first_seen).total_seconds() >= ctx.cfg.rules.unattended_package_s:
            mins = int((ctx.ts - e.first_seen).total_seconds() // 60)
            out.append(_alert(ctx, "UNATTENDED_PACKAGE", Severity.LOW,
                              f"Package unattended at {pretty_zone(ctx.zone_of(d))} for {mins} min "
                              f"(since {hhmm(e.first_seen)}), {hhmm(ctx.ts)}.", d))
    return out


def gate_open(ctx: RuleContext) -> list[Alert]:
    if ctx.cfg.rules.is_business_hours(ctx.ts):
        return []
    return [_alert(ctx, "GATE_OPEN_AFTER_HOURS", Severity.MEDIUM,
                   f"Main gate left open after hours, {hhmm(ctx.ts)}.", d)
            for d in ctx.detections if d.object_type == "gate" and d.action == "open"]


def hazard(ctx: RuleContext) -> list[Alert]:
    return [_alert(ctx, "HAZARD", Severity.CRITICAL,
                   f"{d.subtype.capitalize()} detected at {pretty_zone(ctx.zone_of(d))}, {hhmm(ctx.ts)}.", d)
            for d in ctx.detections if d.object_type == "hazard"]


# ------------------------------ drone safety ------------------------------- #
def drone_battery(ctx: RuleContext) -> list[Alert]:
    t, rc = ctx.frame.telemetry, ctx.cfg.rules
    if t.battery_pct <= rc.critical_battery_pct:
        return [_alert(ctx, "DRONE_LOW_BATTERY", Severity.CRITICAL,
                       f"Drone battery critical ({t.battery_pct:.0f}%) - return-to-dock, {hhmm(ctx.ts)}.")]
    if t.battery_pct <= rc.low_battery_pct:
        return [_alert(ctx, "DRONE_LOW_BATTERY", Severity.MEDIUM,
                       f"Drone battery low ({t.battery_pct:.0f}%), {hhmm(ctx.ts)}.")]
    return []


def drone_altitude(ctx: RuleContext) -> list[Alert]:
    t = ctx.frame.telemetry
    if t.altitude_m > ctx.cfg.rules.max_altitude_m:
        return [_alert(ctx, "DRONE_ALTITUDE", Severity.HIGH,
                       f"Drone altitude {t.altitude_m:.0f} m exceeds {ctx.cfg.rules.max_altitude_m:.0f} m "
                       f"ceiling over {pretty_zone(t.zone)}, {hhmm(ctx.ts)}.")]
    return []


def drone_geofence(ctx: RuleContext) -> list[Alert]:
    t = ctx.frame.telemetry
    dist = haversine_m(t.lat, t.lon, ctx.cfg.dock_lat, ctx.cfg.dock_lon)
    if dist > ctx.cfg.rules.geofence_radius_m:
        return [_alert(ctx, "DRONE_GEOFENCE", Severity.HIGH,
                       f"Drone {dist:.0f} m from dock, outside {ctx.cfg.rules.geofence_radius_m:.0f} m geofence, "
                       f"{hhmm(ctx.ts)}.")]
    return []


DEFAULT_RULES: list[Rule] = [
    perimeter_breach, tampering, restricted_zone, night_loitering, suspicious_behavior,
    after_hours_vehicle, repeat_vehicle, unattended_package, gate_open, hazard,
    drone_battery, drone_altitude, drone_geofence,
]


class AlertEngine:
    def __init__(self, cfg: AppConfig, rules: list[Rule] | None = None):
        self.cfg = cfg
        self.rules = list(rules or DEFAULT_RULES)
        self._last_fired: dict[tuple[str, str], datetime] = {}

    def evaluate(self, ctx: RuleContext) -> list[Alert]:
        candidates: list[Alert] = []
        for rule in self.rules:
            candidates.extend(rule(ctx))
        merged = self._merge(candidates)
        return [a for a in merged if self._passes_cooldown(a)]

    @staticmethod
    def _merge(alerts: list[Alert]) -> list[Alert]:
        by_key: dict[str, Alert] = {}
        for a in alerts:
            key = a.entity_id or f"{a.rule_id}@{a.zone}"
            cur = by_key.get(key)
            if cur is None:
                by_key[key] = a
            elif a.severity > cur.severity:
                a.related_rules = [cur.rule_id, *cur.related_rules, *a.related_rules]
                by_key[key] = a
            elif a.rule_id != cur.rule_id and a.rule_id not in cur.related_rules:
                cur.related_rules.append(a.rule_id)
        return sorted(by_key.values(), key=lambda x: -x.severity)

    def _passes_cooldown(self, a: Alert) -> bool:
        if a.rule_id == "REPEAT_VEHICLE":
            return True  # fires once per new entry by construction
        key = (a.rule_id, a.entity_id or a.zone)
        last = self._last_fired.get(key)
        if last is not None and (a.ts - last).total_seconds() < self.cfg.rules.alert_cooldown_s:
            return False
        self._last_fired[key] = a.ts
        return True
