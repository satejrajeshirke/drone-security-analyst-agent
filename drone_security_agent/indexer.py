"""Frame-by-frame video index (the cross-domain component).

Storage: a single SQLite file (zero-ops, ships with Python) with
* `frames`      - one row per frame: time, caption, drone telemetry
* `detections`  - one row per object in a frame (normalised, B-tree indexed
                  on time / type / zone / entity for structured queries)
* `alerts`      - every alert raised, linked to its frame
* `frames_fts`  - FTS5 full-text index over captions + an enriched "tags"
                  column (object types, subtypes, labels, actions, zones).
                  The tags column is what lets "show all truck events" find a
                  frame captioned "Blue Ford F150 parked at the garage": the
                  detector tagged it `vehicle truck`.

Three query styles are exposed:
1. structured   - `detections(object_type="vehicle", zone="main_gate", start="07:00")`
2. full-text    - `search_text("blue truck")` ranked with BM25
3. natural      - `natural_query("show all truck events after 12:00")`
                  -> parsed into structured filters, FTS fallback.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, time
from pathlib import Path
from typing import Any, Iterable, Optional

from .detection import COLORS, ZONE_PHRASES
from .models import Alert, Detection, Frame, Severity

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS frames (
    frame_id     INTEGER PRIMARY KEY,
    ts           TEXT NOT NULL,
    ts_epoch     REAL NOT NULL,
    day          TEXT NOT NULL,
    minute_of_day INTEGER NOT NULL,
    description  TEXT NOT NULL,
    tags         TEXT NOT NULL DEFAULT '',
    zone         TEXT,
    lat REAL, lon REAL, altitude_m REAL, battery_pct REAL, flight_mode TEXT,
    image_path   TEXT
);
CREATE TABLE IF NOT EXISTS detections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(frame_id),
    ts           TEXT NOT NULL,
    ts_epoch     REAL NOT NULL,
    minute_of_day INTEGER NOT NULL,
    object_type  TEXT NOT NULL,
    subtype      TEXT, label TEXT, color TEXT, make TEXT, model TEXT, plate TEXT,
    count        INTEGER, action TEXT, zone TEXT, entity_id TEXT, confidence REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(frame_id),
    ts           TEXT NOT NULL,
    ts_epoch     REAL NOT NULL,
    rule_id      TEXT NOT NULL,
    severity     INTEGER NOT NULL,
    severity_name TEXT NOT NULL,
    message      TEXT NOT NULL,
    zone TEXT, entity_id TEXT, recommendation TEXT, related_rules TEXT
);
CREATE INDEX IF NOT EXISTS ix_frames_ts      ON frames(ts_epoch);
CREATE INDEX IF NOT EXISTS ix_det_type_ts    ON detections(object_type, ts_epoch);
CREATE INDEX IF NOT EXISTS ix_det_subtype    ON detections(subtype);
CREATE INDEX IF NOT EXISTS ix_det_zone       ON detections(zone, ts_epoch);
CREATE INDEX IF NOT EXISTS ix_det_entity     ON detections(entity_id, ts_epoch);
CREATE INDEX IF NOT EXISTS ix_alerts_sev_ts  ON alerts(severity, ts_epoch);
CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
    description, tags, content='frames', content_rowid='frame_id', tokenize='porter unicode61'
);
"""

# query word -> structured filter
OBJECT_TERMS: dict[str, dict[str, str]] = {
    "truck": {"subtype": "truck"}, "trucks": {"subtype": "truck"}, "pickup": {"subtype": "truck"},
    "van": {"subtype": "van"}, "vans": {"subtype": "van"},
    "sedan": {"subtype": "sedan"}, "sedans": {"subtype": "sedan"},
    "suv": {"subtype": "suv"}, "motorcycle": {"subtype": "motorcycle"},
    "car": {"object_type": "vehicle"}, "cars": {"object_type": "vehicle"},
    "vehicle": {"object_type": "vehicle"}, "vehicles": {"object_type": "vehicle"},
    "person": {"object_type": "person"}, "people": {"object_type": "person"},
    "persons": {"object_type": "person"}, "man": {"object_type": "person"},
    "intruder": {"object_type": "person"}, "intruders": {"object_type": "person"},
    "courier": {"subtype": "courier"},
    "animal": {"object_type": "animal"}, "animals": {"object_type": "animal"},
    "dog": {"subtype": "dog"}, "fox": {"subtype": "fox"}, "cat": {"subtype": "cat"},
    "package": {"object_type": "package"}, "packages": {"object_type": "package"},
    "parcel": {"object_type": "package"},
}
MAKES_MODELS = {"ford": "make", "toyota": "make", "honda": "make", "mercedes": "make", "tesla": "make",
                "f150": "model", "camry": "model", "civic": "model", "sprinter": "model"}
TIME_RE = r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)"


def _to_minute(value: str | time | datetime | None) -> Optional[int]:
    """'07:30' | '7pm' | time | datetime -> minute of day."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.hour * 60 + value.minute
    if isinstance(value, time):
        return value.hour * 60 + value.minute
    s = value.strip().lower().replace(" ", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", s)
    if not m:
        raise ValueError(f"Unrecognised time: {value!r}")
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return (h % 24) * 60 + mi


def _time_clause(col: str, start: Optional[int], end: Optional[int], params: list) -> str:
    """Minute-of-day window; supports windows that wrap midnight (22:00-06:00)."""
    if start is None and end is None:
        return ""
    if start is not None and end is not None and start > end:
        params += [start, end]
        return f" AND ({col} >= ? OR {col} <= ?)"
    sql = ""
    if start is not None:
        sql += f" AND {col} >= ?"
        params.append(start)
    if end is not None:
        sql += f" AND {col} <= ?"
        params.append(end)
    return sql


class FrameIndex:
    def __init__(self, path: str | Path = ":memory:", commit_every: int = 500):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.commit_every = commit_every
        self._pending = 0

    # ------------------------------------------------------------- writes #
    def reset(self) -> None:
        self.conn.executescript(
            "DELETE FROM alerts; DELETE FROM detections; DELETE FROM frames;"
            "INSERT INTO frames_fts(frames_fts) VALUES('delete-all');")
        self.conn.commit()

    def add(self, frame: Frame, detections: Iterable[Detection], alerts: Iterable[Alert] = ()) -> None:
        detections = list(detections)
        t = frame.telemetry
        minute = frame.ts.hour * 60 + frame.ts.minute
        tags = " ".join(sorted({tok for d in detections for tok in
                                (d.object_type, d.subtype, d.label.lower(), d.action, d.color,
                                 d.zone.replace("_", " "), d.plate.lower()) if tok}))
        epoch = frame.ts.timestamp()
        self.conn.execute(
            "INSERT OR REPLACE INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (frame.frame_id, frame.ts.isoformat(), epoch, frame.ts.date().isoformat(), minute,
             frame.description, tags, t.zone, t.lat, t.lon, t.altitude_m, t.battery_pct,
             t.flight_mode, frame.image_path))
        self.conn.execute("INSERT INTO frames_fts(rowid, description, tags) VALUES (?,?,?)",
                          (frame.frame_id, frame.description, tags))
        if detections:
            self.conn.executemany(
                "INSERT INTO detections(frame_id, ts, ts_epoch, minute_of_day, object_type, subtype, label,"
                " color, make, model, plate, count, action, zone, entity_id, confidence)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(frame.frame_id, frame.ts.isoformat(), epoch, minute, d.object_type, d.subtype, d.label,
                  d.color, d.make, d.model, d.plate, d.count, d.action, d.zone, d.entity_id, d.confidence)
                 for d in detections])
        for a in alerts:
            self.add_alert(a)
        self._pending += 1
        if self._pending >= self.commit_every:
            self.commit()

    def add_alert(self, a: Alert) -> None:
        self.conn.execute(
            "INSERT INTO alerts(frame_id, ts, ts_epoch, rule_id, severity, severity_name, message, zone,"
            " entity_id, recommendation, related_rules) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (a.frame_id, a.ts.isoformat(), a.ts.timestamp(), a.rule_id, int(a.severity), a.severity.name,
             a.message, a.zone, a.entity_id, a.recommendation, ",".join(a.related_rules)))

    def commit(self) -> None:
        self.conn.commit()
        self._pending = 0

    def close(self) -> None:
        self.commit()
        self.conn.close()

    # -------------------------------------------------------------- reads #
    def _rows(self, sql: str, params: list | tuple = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.conn.execute(sql, params).fetchall()]

    def frame(self, frame_id: int) -> Optional[dict[str, Any]]:
        rows = self._rows("SELECT * FROM frames WHERE frame_id=?", (frame_id,))
        return rows[0] if rows else None

    def frames_between(self, start: str | None = None, end: str | None = None,
                       limit: int = 500) -> list[dict[str, Any]]:
        params: list = []
        sql = "SELECT * FROM frames WHERE 1=1" + _time_clause("minute_of_day", _to_minute(start),
                                                                _to_minute(end), params)
        return self._rows(sql + " ORDER BY ts_epoch LIMIT ?", params + [limit])

    def detections(self, object_type: str | None = None, subtype: str | None = None,
                   color: str | None = None, make: str | None = None, model: str | None = None,
                   zone: str | None = None, action: str | None = None, entity_id: str | None = None,
                   label_like: str | None = None, start: str | None = None, end: str | None = None,
                   limit: int = 500) -> list[dict[str, Any]]:
        params: list = []
        sql = ("SELECT d.*, f.description FROM detections d JOIN frames f USING(frame_id) WHERE 1=1")
        for col, val in (("object_type", object_type), ("subtype", subtype), ("color", color),
                         ("zone", zone), ("action", action), ("entity_id", entity_id)):
            if val:
                sql += f" AND d.{col} = ?"
                params.append(val.lower() if col in ("object_type", "subtype", "color", "action") else val)
        for col, val in (("make", make), ("model", model)):
            if val:
                sql += f" AND lower(d.{col}) = ?"
                params.append(val.lower())
        if label_like:
            sql += " AND lower(d.label) LIKE ?"
            params.append(f"%{label_like.lower()}%")
        sql += _time_clause("d.minute_of_day", _to_minute(start), _to_minute(end), params)
        return self._rows(sql + " ORDER BY d.ts_epoch LIMIT ?", params + [limit])

    def search_text(self, query: str, limit: int = 50) -> list[dict[str, Any]]:
        """BM25-ranked full-text search over captions + detector tags."""
        tokens = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in _STOPWORDS]
        if not tokens:
            return []
        match = " OR ".join(f'"{t}"*' for t in tokens)
        sql = ("SELECT f.*, bm25(frames_fts) AS score FROM frames_fts JOIN frames f "
               "ON f.frame_id = frames_fts.rowid WHERE frames_fts MATCH ? ORDER BY score LIMIT ?")
        return self._rows(sql, (match, limit))

    def alerts(self, min_severity: str | int | Severity = Severity.INFO, start: str | None = None,
               end: str | None = None, zone: str | None = None, rule_id: str | None = None,
               limit: int = 500) -> list[dict[str, Any]]:
        params: list = [int(Severity.parse(min_severity))]
        sql = ("SELECT *, CAST(strftime('%H', ts) AS INTEGER)*60 + CAST(strftime('%M', ts) AS INTEGER) AS m"
               " FROM alerts WHERE severity >= ?")
        if zone:
            sql += " AND zone = ?"
            params.append(zone)
        if rule_id:
            sql += " AND rule_id = ?"
            params.append(rule_id.upper())
        sql += _time_clause("m", _to_minute(start), _to_minute(end), params)
        return self._rows(sql + " ORDER BY ts_epoch LIMIT ?", params + [limit])

    def entity_timeline(self, entity_id: str) -> list[dict[str, Any]]:
        return self.detections(entity_id=entity_id)

    def stats(self) -> dict[str, Any]:
        q = self.conn.execute
        return {
            "frames": q("SELECT COUNT(*) FROM frames").fetchone()[0],
            "detections": q("SELECT COUNT(*) FROM detections").fetchone()[0],
            "alerts": q("SELECT COUNT(*) FROM alerts").fetchone()[0],
            "first_ts": q("SELECT MIN(ts) FROM frames").fetchone()[0],
            "last_ts": q("SELECT MAX(ts) FROM frames").fetchone()[0],
            "by_object_type": dict(q("SELECT object_type, COUNT(*) FROM detections GROUP BY 1 ORDER BY 2 DESC")
                                   .fetchall()),
            "by_label": dict(q("SELECT label, COUNT(*) FROM detections GROUP BY 1 ORDER BY 2 DESC LIMIT 15")
                             .fetchall()),
            "by_zone": dict(q("SELECT zone, COUNT(*) FROM detections GROUP BY 1 ORDER BY 2 DESC").fetchall()),
            "alerts_by_severity": dict(q("SELECT severity_name, COUNT(*) FROM alerts GROUP BY 1 "
                                         "ORDER BY MAX(severity) DESC").fetchall()),
            "distinct_entities": q("SELECT COUNT(DISTINCT entity_id) FROM detections").fetchone()[0],
        }

    # ------------------------------------------------- natural language #
    def parse_query(self, text: str) -> dict[str, Any]:
        """Turn 'show all blue truck events at the main gate after 5pm' into filters."""
        q = text.lower()
        f: dict[str, Any] = {"table": "alerts" if re.search(r"\balerts?\b", q) else "detections"}
        if m := re.search(rf"between {TIME_RE} and {TIME_RE}", q):
            f["start"], f["end"] = m.group(1), m.group(2)
        elif m := re.search(rf"from {TIME_RE} to {TIME_RE}", q):
            f["start"], f["end"] = m.group(1), m.group(2)
        else:
            if m := re.search(rf"\b(?:after|since) {TIME_RE}", q):
                f["start"] = m.group(1)
            if m := re.search(rf"\bbefore {TIME_RE}", q):
                f["end"] = m.group(1)
            if m := re.search(rf"\bat {TIME_RE}(?!\s*\w*\s*(?:gate|door))", q):
                if re.fullmatch(r"\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm)", m.group(1).strip()):
                    mins = _to_minute(m.group(1))
                    f["start"] = f"{(mins - 5) // 60 % 24:02d}:{(mins - 5) % 60:02d}"
                    f["end"] = f"{(mins + 5) // 60 % 24:02d}:{(mins + 5) % 60:02d}"
        if re.search(r"\b(at night|overnight|night ?time)\b", q):
            f["start"], f["end"] = "22:00", "06:00"
        if re.search(r"\b(morning)\b", q):
            f["start"], f["end"] = "06:00", "12:00"
        if re.search(r"\b(afternoon)\b", q):
            f["start"], f["end"] = "12:00", "18:00"
        if re.search(r"\b(evening)\b", q):
            f["start"], f["end"] = "18:00", "22:00"
        for word in re.findall(r"[a-z0-9]+", q):
            if word in OBJECT_TERMS:
                f.update(OBJECT_TERMS[word])
            if word in MAKES_MODELS:
                f[MAKES_MODELS[word]] = word
            if word in COLORS and word not in ("dark",):
                f["color"] = "grey" if word == "gray" else word
        zone_text = re.sub(rf"\bat {TIME_RE}", "", q)
        for phrase, zone in ZONE_PHRASES:
            if re.search(rf"\b{phrase}\b", zone_text):
                f["zone"] = zone
                break
        for sev in ("critical", "high", "medium", "low"):
            if re.search(rf"\b{sev}\b", q):
                f["min_severity"] = sev
                break
        return f

    def natural_query(self, text: str, limit: int = 100) -> dict[str, Any]:
        f = self.parse_query(text)
        table = f.pop("table")
        if table == "alerts":
            rows = self.alerts(min_severity=f.get("min_severity", "INFO"), start=f.get("start"),
                               end=f.get("end"), zone=f.get("zone"), limit=limit)
            return {"mode": "alerts", "filters": f, "results": rows}
        f.pop("min_severity", None)
        structured = {k: v for k, v in f.items() if k not in ("start", "end")}
        if structured or "start" in f or "end" in f:
            rows = self.detections(**f, limit=limit)
            return {"mode": "structured", "filters": f, "results": rows}
        return {"mode": "full_text", "filters": {"match": text}, "results": self.search_text(text, limit)}


_STOPWORDS = {"show", "all", "the", "a", "an", "of", "events", "event", "me", "find", "list", "any",
              "were", "was", "there", "what", "which", "when", "did", "in", "on", "at", "frames", "frame",
              "with", "and", "or", "is", "are", "today", "video"}
