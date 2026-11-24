"""Video summarisation and security recommendations (deterministic baseline).

These work with no LLM at all - the LangChain agent can call them as tools and
then rephrase / extend them, which keeps the core facts grounded in the index.
"""
from __future__ import annotations

from collections import Counter

from .indexer import FrameIndex
from .models import Severity, pretty_zone
from .tracker import EntityTracker


def one_sentence_summary(index: FrameIndex, tracker: EntityTracker) -> str:
    st = index.stats()
    if not st["frames"]:
        return "No frames have been processed yet."
    start, end = st["first_ts"][11:16], st["last_ts"][11:16]
    by_type = Counter({e.object_type: 0 for e in tracker.entities.values()})
    for e in tracker.entities.values():
        by_type[e.object_type] += 1
    people, vehicles = by_type.get("person", 0), by_type.get("vehicle", 0)
    worst = index.alerts(min_severity=Severity.HIGH, limit=1000)
    worst.sort(key=lambda r: (-r["severity"], r["ts_epoch"]))
    repeat = max((e for e in tracker.entities.values() if e.object_type == "vehicle"),
                 key=lambda e: len(e.entries), default=None)
    parts = [f"Between {start} and {end} the drone indexed {st['frames']} frames and tracked "
             f"{vehicles} distinct vehicle{'s' if vehicles != 1 else ''} and {people} "
             f"distinct person{'s' if people != 1 else ''}"]
    if repeat and len(repeat.entries) >= 2:
        parts.append(f" (the {repeat.label} entered {len(repeat.entries)} times)")
    parts.append(f", raising {st['alerts']} alerts")
    if worst:
        w = worst[0]
        parts.append(f" - most serious: {w['message'].rstrip('.')} [{w['severity_name']}]")
    return "".join(parts) + "."


def recommendations(index: FrameIndex, tracker: EntityTracker) -> list[str]:
    """Pattern-level advice derived from the whole day's alerts, not one frame."""
    recs: list[str] = []
    alerts = index.alerts(limit=10_000)
    by_rule = Counter(a["rule_id"] for a in alerts)
    night_zones = Counter(a["zone"] for a in alerts
                          if a["rule_id"] in ("NIGHT_LOITERING", "RESTRICTED_ZONE", "TAMPERING",
                                              "PERIMETER_BREACH", "SUSPICIOUS_BEHAVIOR", "AFTER_HOURS_VEHICLE"))
    if night_zones:
        hot = ", ".join(f"{pretty_zone(z)} ({n})" for z, n in night_zones.most_common(4))
        recs.append(f"After-hours hot-spots: {hot} - add motion-triggered floodlights there and shift extra "
                    f"drone patrols to 22:00-03:00, when most incidents occurred.")
    if by_rule.get("PERIMETER_BREACH"):
        recs.append("Perimeter breach detected - inspect the fence line for damage and consider fence "
                    "vibration sensors to cue the drone automatically.")
    if by_rule.get("TAMPERING"):
        recs.append("Entry-point tampering observed - review lock integrity on the affected doors/gate and "
                    "preserve the indexed frames as evidence.")
    for e in tracker.entities.values():
        if e.object_type == "vehicle" and len(e.entries) >= 2:
            plate = f" (plate {e.plate})" if e.plate else ""
            recs.append(f"{e.label}{plate} entered {len(e.entries)} times - confirm it is a known staff/visitor "
                        f"vehicle and add it to the whitelist to cut noise, or flag it for follow-up.")
    if by_rule.get("UNATTENDED_PACKAGE"):
        recs.append("Packages sat unattended at the front door - set up a reception pickup SLA or a parcel locker.")
    if by_rule.get("GATE_OPEN_AFTER_HOURS"):
        recs.append("Main gate was left open after hours - add an automatic close timer / gate-state sensor.")
    if by_rule.get("DRONE_LOW_BATTERY"):
        recs.append("Drone hit critical battery during a night patrol - shorten late sorties or add a "
                    "battery swap before 23:00 so coverage is not lost at the riskiest hour.")
    if by_rule.get("DRONE_ALTITUDE") or by_rule.get("DRONE_GEOFENCE"):
        recs.append("Flight envelope violation logged - review wind limits / mission altitude for compliance.")
    if not recs:
        recs.append("No actionable patterns - current patrol schedule looks sufficient.")
    return recs
