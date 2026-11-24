"""Object / event extraction: free-text frame description -> structured Detections.

This is the "object detection logic" component. In production the text comes
from a VLM caption (see `vlm.py`); here it can also come straight from the
simulator. The parser is deliberately deterministic (lexicon + regex) so that
it is fast (>50k frames/s), explainable and unit-testable; an LLM extractor
can be swapped in through the same interface (`Detector.extract`).

Design notes
* The description is split into clauses (",", ";", " while ").
  Each clause yields at most one vehicle / person-group / animal / package.
* Actions are looked up in the entity's own clause, falling back to the first
  action anywhere in the description ("two people near the door, testing the
  handle" -> people: tampering).
* Directional phrases ("toward the main gate") are removed before zone
  resolution so that the zone is where the object *is*, not where it heads.
* Negated mentions ("no one present", "no activity") produce no detections.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import Detection

COLORS = ["black", "white", "grey", "gray", "silver", "red", "blue", "green", "yellow",
          "orange", "brown", "dark", "beige", "maroon", "purple"]

# model keyword -> (make, model, subtype)
VEHICLE_MODELS: dict[str, tuple[str, str, str]] = {
    "f150": ("Ford", "F150", "truck"),
    "f-150": ("Ford", "F150", "truck"),
    "camry": ("Toyota", "Camry", "sedan"),
    "civic": ("Honda", "Civic", "sedan"),
    "sprinter": ("Mercedes", "Sprinter", "van"),
    "model 3": ("Tesla", "Model 3", "sedan"),
    "thar": ("Mahindra", "Thar", "suv"),
    "swift": ("Suzuki", "Swift", "hatchback"),
    "ram": ("Dodge", "Ram", "truck"),
    "silverado": ("Chevrolet", "Silverado", "truck"),
    "hilux": ("Toyota", "Hilux", "truck"),
}
VEHICLE_MAKES = ["ford", "toyota", "honda", "mercedes", "tesla", "mahindra", "suzuki", "bmw", "hyundai", "tata",
                 "dodge", "chevrolet", "nissan", "kia", "volkswagen", "audi", "renault"]
# phrase -> canonical subtype (longest first matters)
VEHICLE_TYPES: list[tuple[str, str]] = [
    ("pickup truck", "truck"), ("pickup", "truck"), ("trucks", "truck"), ("truck", "truck"), ("lorry", "truck"),
    ("vans", "van"), ("van", "van"), ("sedans", "sedan"), ("sedan", "sedan"), ("suvs", "suv"), ("suv", "suv"),
    ("hatchback", "hatchback"), ("motorcycles", "motorcycle"), ("motorcycle", "motorcycle"),
    ("motorbike", "motorcycle"), ("scooter", "motorcycle"), ("bicycle", "bicycle"), ("bus", "bus"),
    ("cars", "car"), ("car", "car"), ("vehicles", "vehicle"), ("vehicle", "vehicle"),
]
PERSON_WORDS = ["person", "people", "man", "men", "woman", "women", "individual", "individuals",
                "figure", "intruder", "intruders", "courier", "guard", "worker", "workers",
                "someone", "child", "children", "pedestrian"]
ROLE_WORDS = {"courier": "courier", "guard": "guard", "worker": "worker", "workers": "worker"}
ANIMALS = ["dog", "cat", "fox", "deer", "cow", "monkey", "bird", "snake"]
NUMBER_WORDS = {"two": 2, "three": 3, "four": 4, "five": 5, "several": 3, "group of": 4, "couple of": 2}

# phrase -> canonical action. Checked in order (multi-word before single word).
ACTIONS: list[tuple[str, str]] = [
    ("jumping over", "climbing"), ("climbing", "climbing"), ("climbs", "climbing"), ("scaling", "climbing"),
    ("testing the door", "tampering"), ("checking the gate latch", "tampering"), ("forcing", "tampering"),
    ("prying", "tampering"), ("cutting", "tampering"),
    ("walking away", "leaving"), ("walking back", "leaving"), ("leaving", "leaving"),
    ("exiting", "exiting"), ("exits", "exiting"), ("departing", "exiting"),
    ("entering", "entering"), ("enters", "entering"), ("arriving", "entering"), ("pulling in", "entering"),
    ("loitering", "loitering"), ("lingering", "loitering"), ("pacing", "loitering"),
    ("standing", "standing"), ("stands", "standing"), ("waiting", "standing"), ("sitting", "standing"),
    ("sits", "standing"),
    ("left open", "open"), ("left at", "left"), ("dropped", "left"), ("still at", "unattended"),
    ("parked", "parked"), ("stationary", "parked"),
    ("circling", "circling"), ("prowling", "circling"),
    ("running", "running"), ("fleeing", "running"), ("sprinting", "running"),
    ("unloading", "unloading"), ("loading", "loading"), ("carrying", "carrying"),
    ("driving", "driving"), ("crossing", "crossing"), ("walking", "walking"),
]
ZONE_PHRASES: list[tuple[str, str]] = [
    ("main gate", "main_gate"), ("front gate", "main_gate"), ("gate", "main_gate"),
    ("driveway", "driveway"), ("garage", "garage"),
    ("front door", "front_door"), ("porch", "front_door"),
    ("parking lot", "parking_lot"), ("parking", "parking_lot"),
    ("loading dock", "loading_dock"), ("warehouse", "warehouse"),
    ("backyard", "backyard"), ("back yard", "backyard"), ("lawn", "backyard"),
    ("perimeter fence", "perimeter_fence"), ("perimeter road", "perimeter_fence"),
    ("perimeter", "perimeter_fence"), ("fence", "perimeter_fence"),
]
CLOTHING_RE = re.compile(
    r"\bin (?:an? )?((?:[a-z]+ )?(?:hoodie|jacket|vest|shirt|coat|uniform|helmet|mask|cap|raincoat))\b")
# The keyword is case-insensitive; the plate itself must be upper-case letters/digits,
# so "license plate WDS-947 is parked" yields WDS-947 (not "WDS-947 IS").
PLATE_RE = re.compile(
    r"(?i:\b(?:licen[cs]e\s+)?plate\s*(?:number\s*)?[:#]?\s*)([A-Z0-9]{2,4}(?:[ -]?[A-Z0-9]{1,4}){1,3})\b")
DIRECTIONAL_RE = re.compile(r"\b(?:toward|towards|into the direction of|heading to)\s+(?:the\s+)?[a-z ]+?(?=,|;|$)")
NEGATION_RE = re.compile(
    r"\bno (?:one|people(?: or vehicles)?|persons?|vehicles?|activity|movement|motion)\b[^,;]*")
CLAUSE_SPLIT_RE = re.compile(r"[,;]| while | and then ")


def _word_in(text: str, word: str) -> Optional[re.Match]:
    return re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text)


# When a clause mentions several actions, the most security-relevant one wins
# ("two men, one carrying a briefcase, are climbing the fence" -> climbing); ties go to the earliest.
ACTION_PRIORITY = {"climbing": 10, "tampering": 9, "running": 7, "loitering": 6, "circling": 6,
                   "entering": 5, "exiting": 5, "leaving": 5, "open": 4, "left": 4, "unattended": 4,
                   "standing": 3, "parked": 3, "carrying": 2}


def _first_action(text: str) -> str:
    best: tuple[int, int, str] | None = None  # (-priority, position, action)
    for phrase, canon in ACTIONS:
        m = _word_in(text, phrase)
        if m:
            key = (-ACTION_PRIORITY.get(canon, 3), m.start(), canon)
            if best is None or key < best:
                best = key
    return best[2] if best else ""


LOCATIVE = r"(?:at|in|on|near|by|inside|outside|through|along|across|into|from|beside|behind|around|within)"


def _zone(text: str, locative_only: bool = False) -> str:
    """Most specific mention wins: 'At the perimeter fence ... climbing over a gate' -> perimeter_fence.
    With locative_only, a zone word only counts as a *location* ("at the gate", "near the warehouse door"),
    not as an object being acted on ("climbing over a gate")."""
    text = DIRECTIONAL_RE.sub("", text)
    for phrase, zone in sorted(ZONE_PHRASES, key=lambda pz: -len(pz[0])):
        if locative_only:
            if re.search(rf"\b{LOCATIVE}\s+(?:the\s+|a\s+|an\s+)?(?:[a-z]+\s+){{0,2}}{re.escape(phrase)}\b", text):
                return zone
        elif _word_in(text, phrase):
            return zone
    return ""


def _color_before(text: str, pos: int) -> str:
    window = text[:pos]
    best, best_pos = "", -1
    for c in COLORS:
        for m in re.finditer(rf"\b{c}\b", window):
            if m.start() > best_pos:
                best, best_pos = c, m.start()
    return "grey" if best == "gray" else best


class Detector:
    """Text -> Detections. Stateless and thread-safe."""

    def __init__(self, suspicious_keywords: Optional[list[str]] = None):
        self.suspicious_keywords = [k.lower() for k in (suspicious_keywords or [])]

    def extract(self, description: str, fallback_zone: str = "") -> list[Detection]:
        raw = description.strip()
        text = NEGATION_RE.sub("", raw.lower())
        plate_m = PLATE_RE.search(raw)
        plate = plate_m.group(1).upper().replace(" ", "").replace("-", "") if plate_m else ""
        if plate_m:
            text = text.replace(plate_m.group(0).lower(), "")

        global_action = _first_action(text)
        global_zone = _zone(text) or fallback_zone
        cues = [k for k in self.suspicious_keywords if k in raw.lower()]

        detections: list[Detection] = []
        for clause in (c.strip() for c in CLAUSE_SPLIT_RE.split(text)):
            if not clause:
                continue
            action = _first_action(clause) or global_action
            zone = _zone(clause, locative_only=True) or global_zone
            detections.extend(self._clause_detections(clause, action, zone, plate, cues))
            plate = "" if any(d.object_type == "vehicle" for d in detections) else plate

        detections = self._merge_people(detections)

        # gate state & hazards are scene-level facts
        if re.search(r"\bgate\b[^,;]*\b(?:left )?open\b", text):
            detections.append(Detection("gate", subtype="gate", action="open", zone="main_gate", attributes=cues))
        for hazard in ("fire", "smoke"):
            if _word_in(text, hazard):
                detections.append(Detection("hazard", subtype=hazard, action="detected", zone=global_zone,
                                            attributes=cues, confidence=0.8))
        return detections

    # ------------------------------------------------------------------ #
    @staticmethod
    def _merge_people(detections: list[Detection]) -> list[Detection]:
        """'Two men climbing the fence, one of them carrying a bag' is ONE group, not two detections.
        People in the same zone within one frame are merged (max count, first action, union of cues)."""
        merged: list[Detection] = []
        first_by_zone: dict[str, Detection] = {}
        for d in detections:
            if d.object_type != "person":
                merged.append(d)
                continue
            first = first_by_zone.get(d.zone)
            if first is None:
                first_by_zone[d.zone] = d
                merged.append(d)
                continue
            first.count = max(first.count, d.count)
            if ACTION_PRIORITY.get(d.action, 3) > ACTION_PRIORITY.get(first.action, 3 if first.action else 0):
                first.action = d.action
            first.attributes += [a for a in d.attributes if a not in first.attributes]
        return merged

    def _clause_detections(self, clause: str, action: str, zone: str, plate: str,
                           cues: list[str]) -> list[Detection]:
        out: list[Detection] = []

        # ---- vehicle ----
        make = model = subtype = ""
        v_pos = None
        for kw, (mk, md, st) in VEHICLE_MODELS.items():
            m = _word_in(clause, kw)
            if m:
                make, model, subtype, v_pos = mk, md, st, m.start()
                break
        if not make:
            for mk in VEHICLE_MAKES:
                m = _word_in(clause, mk)
                if m:
                    make, v_pos = mk.capitalize(), m.start()
                    break
        for phrase, st in VEHICLE_TYPES:
            m = _word_in(clause, phrase)
            if m:
                if not subtype or subtype == "vehicle":
                    subtype = st
                v_pos = m.start() if v_pos is None else min(v_pos, m.start())
                break
        if v_pos is not None:
            out.append(Detection(
                "vehicle", subtype=subtype or "vehicle", color=_color_before(clause, v_pos + 1),
                make=make, model=model, plate=plate, action=action, zone=zone,
                attributes=list(cues), confidence=0.9))

        # ---- people ----
        for word in PERSON_WORDS:
            m = _word_in(clause, word)
            if not m:
                continue
            count = 1
            if word in ("people", "men", "women", "individuals", "intruders", "workers", "children"):
                count = 2
            for nw, n in NUMBER_WORDS.items():
                if _word_in(clause[: m.start()], nw):
                    count = n
            cm = CLOTHING_RE.search(clause)
            attrs = list(cues)
            color = ""
            if cm:
                attrs.insert(0, cm.group(1))
                color = cm.group(1).split()[0] if cm.group(1).split()[0] in COLORS else ""
            out.append(Detection(
                "person", subtype=ROLE_WORDS.get(word, "person"), color=color, count=count,
                action=action, zone=zone, attributes=attrs, confidence=0.88))
            break

        # ---- animals ----
        for a in ANIMALS:
            if _word_in(clause, a):
                out.append(Detection("animal", subtype=a, action=action, zone=zone, confidence=0.85))
                break

        # ---- packages ----
        if _word_in(clause, "package") or _word_in(clause, "parcel"):
            out.append(Detection("package", subtype="package", action=action or "present", zone=zone,
                                 confidence=0.85))
        return out
