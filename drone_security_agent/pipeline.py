"""Streaming pipeline: frame -> describe -> detect -> track -> rules -> index.

Every stage is synchronous and in-process for the prototype; the stage
boundaries are exactly where a production system would put a message queue
(see docs/ARCHITECTURE.md).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .config import AppConfig, load_config
from .context import ContextManager
from .detection import Detector
from .indexer import FrameIndex
from .models import Alert, Detection, Frame, LogEntry, hhmm, pretty_zone
from .rules import AlertEngine, RuleContext
from .tracker import EntityTracker
from .vlm import FrameDescriber, PassthroughDescriber

ACTION_TEXT = {
    "entering": "entering", "exiting": "exiting", "leaving": "leaving", "parked": "parked",
    "loitering": "loitering", "standing": "standing", "climbing": "climbing", "running": "running",
    "walking": "walking", "circling": "circling", "tampering": "tampering with entry point",
    "left": "left behind", "unattended": "unattended", "unloading": "unloading", "crossing": "crossing",
    "driving": "driving", "open": "open",
}


@dataclass
class FrameResult:
    frame: Frame
    detections: list[Detection]
    alerts: list[Alert]
    logs: list[LogEntry] = field(default_factory=list)
    latency_ms: float = 0.0


class SecurityPipeline:
    def __init__(self, cfg: Optional[AppConfig] = None, index: Optional[FrameIndex] = None,
                 describer: Optional[FrameDescriber] = None):
        self.cfg = cfg or load_config()
        self.index = index or FrameIndex(":memory:")
        self.describer = describer or PassthroughDescriber()
        self.detector = Detector(self.cfg.rules.suspicious_keywords)
        self.tracker = EntityTracker(visit_gap_s=self.cfg.rules.visit_gap_s)
        self.engine = AlertEngine(self.cfg)
        self.context = ContextManager(self.cfg.property_name)
        self.results: list[FrameResult] = []
        self.keep_results = True
        self.vlm_errors = 0

    def process(self, frame: Frame) -> FrameResult:
        t0 = time.perf_counter()
        if frame.image_path:
            try:
                frame.description = self.describer.describe(frame)
            except Exception as e:  # VLM outage must not stop the patrol: telemetry rules still run
                self.vlm_errors += 1
                frame.description = f"[VLM unavailable: {type(e).__name__}] frame not described"
        detections = self.detector.extract(frame.description, fallback_zone=frame.telemetry.zone)
        logs = []
        for d in detections:
            e = self.tracker.update(d, frame.ts)
            logs.append(self._log(frame, d, e))
        alerts = self.engine.evaluate(RuleContext(frame, detections, self.tracker, self.cfg))
        self.index.add(frame, detections, alerts)
        self.context.observe(frame, detections, alerts)
        res = FrameResult(frame, detections, alerts, logs, (time.perf_counter() - t0) * 1000)
        if self.keep_results:
            self.results.append(res)
        return res

    def run(self, frames: Iterable[Frame], on_result: Optional[Callable[[FrameResult], None]] = None,
            delay_s: float = 0.0) -> list[FrameResult]:
        out = []
        for f in frames:
            r = self.process(f)
            if on_result:
                on_result(r)
            if self.keep_results:
                out.append(r)
            if delay_s:
                time.sleep(delay_s)
        self.index.commit()
        return out

    @staticmethod
    def _log(frame: Frame, d: Detection, e) -> LogEntry:
        zone = pretty_zone(d.zone or frame.telemetry.zone)
        msg = f"{d.label} spotted at {zone}, {hhmm(frame.ts)}."
        extras = []
        if d.action:
            extras.append(ACTION_TEXT.get(d.action, d.action))
        if e.plate:
            extras.append(f"plate {e.plate}")
        if d.object_type == "vehicle" and e.entries:
            extras.append(f"entries today: {len(e.entries)}")
        if e.sightings > 1:
            extras.append(f"sighting #{e.sightings}")
        if extras:
            msg += " [" + " | ".join(extras) + "]"
        return LogEntry(frame.ts, frame.frame_id, msg, d.entity_id)
