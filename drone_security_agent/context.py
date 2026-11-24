"""Context management: keeps what the agent knows about a long video small.

A day of 1-fps video is ~86k frames; nobody can put that in an LLM prompt.
The ContextManager keeps three tiers of memory, written to as frames stream in:

1. short-term  - a ring buffer of the last N frame records (verbatim)
2. mid-term    - per-hour digests: counts of labels, zones, alerts (compressed)
3. long-term   - entity memory (from the tracker) + the full SQLite index,
                 queried on demand (retrieval) rather than kept in the prompt

`build_context(question)` assembles a prompt-sized snapshot in priority order
(header > alerts > entities > hourly digests > retrieved frames > recent frames)
and trims to a character budget, so the prompt cost is roughly constant
however long the video is.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from .models import Alert, Detection, Frame, Severity, hhmm, pretty_zone

if TYPE_CHECKING:  # pragma: no cover
    from .indexer import FrameIndex
    from .tracker import EntityTracker


@dataclass
class HourDigest:
    hour: int
    frames: int = 0
    labels: Counter = field(default_factory=Counter)
    zones: Counter = field(default_factory=Counter)
    alerts: list[str] = field(default_factory=list)

    def render(self) -> str:
        if not self.labels and not self.alerts:
            return f"{self.hour:02d}:00 - {self.frames} frames, quiet."
        objs = ", ".join(f"{k} x{v}" if v > 1 else k for k, v in self.labels.most_common(6))
        zones = ", ".join(pretty_zone(z) for z, _ in self.zones.most_common(3))
        s = f"{self.hour:02d}:00 - {self.frames} frames; seen: {objs or 'nothing'}"
        if zones:
            s += f" (at {zones})"
        if self.alerts:
            s += "; alerts: " + "; ".join(self.alerts[:4])
        return s


class ContextManager:
    def __init__(self, property_name: str = "Property", short_term_frames: int = 15):
        self.property_name = property_name
        self.recent: deque[str] = deque(maxlen=short_term_frames)
        self.hours: dict[int, HourDigest] = {}
        self.alerts: list[Alert] = []
        self.frames_seen = 0
        self.first_ts = None
        self.last_ts = None

    def observe(self, frame: Frame, detections: list[Detection], alerts: list[Alert]) -> None:
        self.frames_seen += 1
        self.first_ts = self.first_ts or frame.ts
        self.last_ts = frame.ts
        h = self.hours.setdefault(frame.ts.hour, HourDigest(frame.ts.hour))
        h.frames += 1
        for d in detections:
            h.labels[d.label] += 1
            if d.zone:
                h.zones[d.zone] += 1
        for a in alerts:
            h.alerts.append(f"{a.severity.name} {a.rule_id} {hhmm(a.ts)}")
        self.alerts.extend(alerts)
        line = f"[{hhmm(frame.ts)} #{frame.frame_id} {pretty_zone(frame.telemetry.zone)}] {frame.description}"
        if alerts:
            line += " => " + "; ".join(str(a) for a in alerts)
        self.recent.append(line)

    # ------------------------------------------------------------------ #
    def build_context(self, question: Optional[str] = None, index: "FrameIndex | None" = None,
                      tracker: "EntityTracker | None" = None, budget_chars: int = 8000) -> str:
        sections: list[tuple[str, list[str]]] = []
        span = (f"{hhmm(self.first_ts)}-{hhmm(self.last_ts)} on {self.first_ts:%Y-%m-%d}"
                if self.first_ts else "n/a")
        sections.append(("SITE", [f"{self.property_name}; {self.frames_seen} frames processed; span {span}."]))

        important = [a for a in self.alerts if a.severity >= Severity.MEDIUM]
        minor = len(self.alerts) - len(important)
        alert_lines = [f"{hhmm(a.ts)} {a.severity.name} {a.rule_id}: {a.message} (frame {a.frame_id})"
                       for a in important]
        if minor:
            alert_lines.append(f"+{minor} INFO/LOW alerts: " +
                               "; ".join(a.message for a in self.alerts if a.severity < Severity.MEDIUM)[:400])
        sections.append(("ALERTS", alert_lines or ["none"]))

        if tracker is not None:
            ents = sorted(tracker.entities.values(), key=lambda e: (-len(e.entries), -e.sightings))
            sections.append(("ENTITIES", [
                f"{e.label} [{e.entity_id}] first {hhmm(e.first_seen)} last {hhmm(e.last_seen)}, "
                f"{e.sightings} sightings, entries: {', '.join(hhmm(t) for t in e.entries) or '0'}, "
                f"zones: {', '.join(pretty_zone(z) for z in e.zones)}" for e in ents[:25]]))

        sections.append(("HOURLY DIGEST", [self.hours[h].render() for h in sorted(self.hours)]))

        if question and index is not None:
            hits = index.search_text(question, limit=8)
            sections.append(("RETRIEVED FRAMES (for this question)", [
                f"[{r['ts'][11:16]} #{r['frame_id']} {pretty_zone(r['zone'] or '')}] {r['description']}"
                for r in hits] or ["no direct matches"]))

        sections.append(("MOST RECENT FRAMES", list(self.recent)))
        return self._render(sections, budget_chars)

    @staticmethod
    def _render(sections: list[tuple[str, list[str]]], budget: int) -> str:
        out, used = [], 0
        for title, lines in sections:
            header = f"## {title}"
            if used + len(header) > budget:
                break
            out.append(header)
            used += len(header) + 1
            for i, line in enumerate(lines):
                if used + len(line) + 1 > budget:
                    out.append(f"... ({len(lines) - i} more lines truncated for context budget)")
                    used += 60
                    break
                out.append(line)
                used += len(line) + 1
        return "\n".join(out)
