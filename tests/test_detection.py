"""Object/event extraction from frame descriptions."""
import pytest

from drone_security_agent.detection import Detector

det = Detector(["flashlight", "headlights off", "circling", "crowbar", "masked", "fire", "smoke"])


def one(desc, kind, zone=""):
    ds = [d for d in det.extract(desc, zone) if d.object_type == kind]
    assert len(ds) == 1, f"expected one {kind} in {desc!r}, got {ds}"
    return ds[0]


def test_truck_logged_with_make_model_color_plate():
    d = one("Blue Ford F150 pickup truck entering through the main gate, plate MH14CD5678", "vehicle")
    assert (d.color, d.make, d.model, d.subtype) == ("blue", "Ford", "F150", "truck")
    assert d.plate == "MH14CD5678"
    assert d.action == "entering"
    assert d.zone == "main_gate"
    assert d.label == "Blue Ford F150"


def test_brief_example_blue_truck_at_gate():
    d = one("Blue truck at gate", "vehicle")
    assert d.subtype == "truck" and d.color == "blue" and d.zone == "main_gate"


def test_person_clothing_and_action():
    d = one("Person in dark hoodie loitering at the main gate", "person")
    assert d.action == "loitering"
    assert "dark hoodie" in d.attributes


def test_group_count_and_action_from_other_clause():
    d = one("Two people near the warehouse door with a flashlight, testing the door handle", "person")
    assert d.count == 2
    assert d.action == "tampering"
    assert d.zone == "warehouse"
    assert "flashlight" in d.attributes


def test_multiple_objects_in_one_frame():
    ds = det.extract("White Mercedes Sprinter delivery van parked at the loading dock, two people unloading boxes")
    kinds = sorted(d.object_type for d in ds)
    assert kinds == ["person", "vehicle"]
    van = next(d for d in ds if d.object_type == "vehicle")
    assert van.subtype == "van" and van.action == "parked"


@pytest.mark.parametrize("desc", [
    "Clear view of the driveway, no activity",
    "No people or vehicles visible at the garage",
    "Routine patrol over the backyard, area quiet",
])
def test_negated_or_empty_scenes_produce_no_detections(desc):
    assert det.extract(desc) == []


def test_directional_phrase_does_not_move_zone():
    d = one("Blue Ford F150 exiting the garage toward the main gate", "vehicle")
    assert d.zone == "garage"


def test_fallback_zone_from_telemetry():
    d = one("Courier in yellow vest walking back", "person", zone="front_door")
    assert d.zone == "front_door"


def test_suspicious_cues_and_hazards():
    car = one("Grey sedan with headlights off circling slowly on the perimeter road", "vehicle")
    assert {"headlights off", "circling"} <= set(car.attributes)
    assert car.zone == "perimeter_fence"
    assert one("Smoke rising from the warehouse roof", "hazard").subtype == "smoke"


def test_gate_open_state():
    g = one("Main gate left open, no one present", "gate")
    assert g.action == "open"
    assert not [d for d in det.extract("Main gate left open, no one present") if d.object_type == "person"]


def test_animals_are_not_people():
    ds = det.extract("Fox crossing the backyard lawn")
    assert [d.object_type for d in ds] == ["animal"]
