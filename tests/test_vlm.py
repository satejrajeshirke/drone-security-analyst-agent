"""VLM layer: real-image frames, Gemini failover (mocked, no network), and caption parsing of real VLM output."""
import json
import os
from types import SimpleNamespace

import pytest

from drone_security_agent.detection import Detector
from drone_security_agent.indexer import FrameIndex
from drone_security_agent.models import Severity
from drone_security_agent.pipeline import SecurityPipeline
from drone_security_agent.vlm import GeminiVisionDescriber, frames_from_images

genai_errors = pytest.importorskip("google.genai.errors")


class FakeModels:
    """Stands in for client.models: fails for the listed models, captions otherwise."""

    def __init__(self, failing: dict[str, Exception], caption: str):
        self.failing, self.caption, self.calls = failing, caption, []

    def generate_content(self, model, contents, config=None):
        self.calls.append(model)
        if model in self.failing:
            raise self.failing[model]
        return SimpleNamespace(text=self.caption)


def _server_error():
    return genai_errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}})


def _not_found():
    return genai_errors.ClientError(404, {"error": {"code": 404, "message": "retired", "status": "NOT_FOUND"}})


@pytest.fixture()
def image_dir(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (tmp_path / "b.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (tmp_path / "manifest.json").write_text(json.dumps([
        {"file": "b.jpg", "zone": "perimeter_fence", "time": "02:20"},
        {"file": "a.jpg", "zone": "driveway", "time": "07:31"},
    ]))
    return tmp_path


def test_frames_from_images_uses_manifest_zone_and_time(image_dir):
    frames = list(frames_from_images(image_dir))
    assert [(f.ts.strftime("%H:%M"), f.telemetry.zone) for f in frames] == [("02:20", "perimeter_fence"),
                                                                            ("07:31", "driveway")]
    assert all(f.image_path and f.description == "" for f in frames)


def test_gemini_fails_over_to_next_model_and_sticks(image_dir, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeModels({"m1": _server_error(), "m2": _not_found()}, "Two men are climbing over the perimeter fence.")
    g = GeminiVisionDescriber(models=("m1", "m2", "m3"), client=SimpleNamespace(models=fake))
    frames = list(frames_from_images(image_dir))
    assert g.describe(frames[0]) == "Two men are climbing over the perimeter fence."
    assert g.last_model == "m3"
    assert fake.calls == ["m1", "m1", "m2", "m3"]  # 503 retried once, 404 skipped immediately
    fake.calls.clear()
    g.describe(frames[1])
    assert fake.calls == ["m3"]  # sticky: the model that worked is tried first


def test_pipeline_survives_vlm_outage(image_dir, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeModels({"m1": _server_error()}, "")
    g = GeminiVisionDescriber(models=("m1",), client=SimpleNamespace(models=fake))
    p = SecurityPipeline(index=FrameIndex(":memory:"), describer=g)
    results = p.run(frames_from_images(image_dir))
    assert len(results) == 2 and p.vlm_errors == 2
    assert results[0].frame.description.startswith("[VLM unavailable")
    assert results[0].detections == []


def test_real_image_pipeline_with_mocked_caption(image_dir):
    fake = FakeModels({}, "At the perimeter fence, two men wearing dark jackets are climbing over a metal fence, "
                          "and one of the men is carrying a black briefcase.")
    g = GeminiVisionDescriber(models=("m",), client=SimpleNamespace(models=fake))
    p = SecurityPipeline(index=FrameIndex(":memory:"), describer=g)
    r = p.process(next(iter(frames_from_images(image_dir))))
    assert [(d.label, d.count, d.action) for d in r.detections] == [("2 people", 2, "climbing")]
    assert r.alerts[0].rule_id == "PERIMETER_BREACH" and r.alerts[0].severity == Severity.CRITICAL


@pytest.mark.parametrize("caption,plate,label", [
    ("One white Dodge Ram 3500 pickup truck with license plate WDS-947 is parked in the driveway.",
     "WDS947", "White Dodge Ram"),
    ("Blue Ford F150 entering through the main gate, plate MH14CD5678", "MH14CD5678", "Blue Ford F150"),
    ("Grey sedan with licence plate KA 01 AB 1234 parked", "KA01AB1234", "Grey sedan"),
])
def test_plates_from_real_vlm_phrasing(caption, plate, label):
    v = next(d for d in Detector().extract(caption) if d.object_type == "vehicle")
    assert (v.plate, v.label) == (plate, label)


def test_people_in_one_scene_are_one_group():
    ds = Detector().extract("two men are climbing over a metal fence, and one of the men is carrying a briefcase",
                            "perimeter_fence")
    people = [d for d in ds if d.object_type == "person"]
    assert len(people) == 1 and people[0].count == 2


@pytest.mark.skipif(not os.environ.get("GEMINI_API_KEY"), reason="needs GEMINI_API_KEY (live Gemini call)")
def test_gemini_live_fence_image():
    from pathlib import Path

    folder = Path(__file__).resolve().parent.parent / "data" / "real_frames"
    if not (folder / "04_fence_climb.jpg").exists():
        pytest.skip("run scripts/fetch_sample_images.py first")
    frame = next(f for f in frames_from_images(folder) if f.image_path.endswith("04_fence_climb.jpg"))
    caption = GeminiVisionDescriber().describe(frame).lower()
    assert "fence" in caption and ("climb" in caption or "over" in caption)


def test_most_relevant_action_wins():
    ds = Detector().extract("two men wearing dark jackets—one carrying a black briefcase—are climbing over a "
                            "blue metal fence", "perimeter_fence")
    assert [(d.object_type, d.action) for d in ds] == [("person", "climbing")]
    g = next(d for d in Detector().extract("a man stands beside a guard booth, a white sedan parked behind")
             if d.object_type == "person")
    assert g.action == "standing"


def test_negative_caption_is_verified_and_corrected(image_dir):
    class Seq(FakeModels):
        def __init__(self):
            super().__init__({}, "")
            self.answers = iter(["Clear view of the backyard, no activity",
                                 "1 small white dog",
                                 "A small white dog is running across the backyard."])

        def generate_content(self, model, contents, config=None):
            self.calls.append(model)
            return SimpleNamespace(text=next(self.answers))

    g = GeminiVisionDescriber(models=("m",), client=SimpleNamespace(models=Seq()))
    cap = g.describe(next(iter(frames_from_images(image_dir))))
    assert cap == "A small white dog is running across the backyard."
    assert (g.verifications, g.corrections) == (1, 1)


def test_true_negative_is_kept(image_dir):
    class Seq(FakeModels):
        def __init__(self):
            super().__init__({}, "")
            self.answers = iter(["Clear view of the driveway, no activity", "NONE"])

        def generate_content(self, model, contents, config=None):
            return SimpleNamespace(text=next(self.answers))

    g = GeminiVisionDescriber(models=("m",), client=SimpleNamespace(models=Seq()))
    assert g.describe(next(iter(frames_from_images(image_dir)))) == "Clear view of the driveway, no activity"
    assert (g.verifications, g.corrections) == (1, 0)


def test_object_being_climbed_is_not_a_location():
    cap = ("At the perimeter fence, one man wearing a black jacket is climbing over while carrying a briefcase, "
           "and another man wearing a dark jacket and jeans is climbing over a gate next to a green train.")
    people = [d for d in Detector().extract(cap) if d.object_type == "person"]
    assert len(people) == 1 and people[0].zone == "perimeter_fence" and people[0].action == "climbing"
