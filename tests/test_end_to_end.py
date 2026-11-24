"""Whole-system scenarios: the scripted day, the agent, context, scale, robustness."""
import os
import random
import time

import pytest

from drone_security_agent.agent import AnalystToolkit, OfflineAnalyst, make_analyst
from drone_security_agent.indexer import FrameIndex
from drone_security_agent.insights import one_sentence_summary, recommendations
from drone_security_agent.models import Severity
from drone_security_agent.pipeline import SecurityPipeline
from drone_security_agent.simulator import random_frames, scripted_day

from .conftest import make_frame

EXPECTED_ALERTS = [
    ("00:03", "NIGHT_LOITERING"), ("02:15", "PERIMETER_BREACH"), ("02:16", "RESTRICTED_ZONE"),
    ("10:40", "UNATTENDED_PACKAGE"), ("15:10", "DRONE_ALTITUDE"), ("17:45", "REPEAT_VEHICLE"),
    ("19:30", "GATE_OPEN_AFTER_HOURS"), ("22:30", "SUSPICIOUS_BEHAVIOR"), ("23:10", "TAMPERING"),
    ("23:40", "DRONE_LOW_BATTERY"),
]


def test_scripted_day_raises_exactly_the_expected_alerts(day_pipeline):
    got = [(a.ts.strftime("%H:%M"), a.rule_id) for r in day_pipeline.results for a in r.alerts]
    assert got == EXPECTED_ALERTS


def test_no_false_alarms_on_quiet_patrol_or_animals(day_pipeline):
    for r in day_pipeline.results:
        if any(d.object_type == "animal" for d in r.detections) or not r.detections:
            assert all(a.rule_id.startswith("DRONE_") for a in r.alerts), r.frame.description


def test_sample_log_line_matches_brief(day_pipeline):
    logs = [l.message for r in day_pipeline.results for l in r.logs]
    assert any(m.startswith("Blue Ford F150 spotted at garage, 12:00.") for m in logs)


def test_sample_alert_matches_brief(day_pipeline):
    msgs = [a.message for r in day_pipeline.results for a in r.alerts]
    assert any(m.startswith("Person loitering at main gate, 00:01") for m in msgs)


def test_whitelisted_staff_car_creates_no_alert(day_pipeline):
    r = next(r for r in day_pipeline.results if "MH12AB1234" in r.frame.description)
    assert r.detections and r.alerts == []


def test_entity_reid_links_plate_less_sightings(day_pipeline):
    e = day_pipeline.tracker.get("veh:MH14CD5678")
    assert e.sightings == 6 and len(e.entries) == 2


def test_summary_is_one_sentence(day_pipeline):
    s = one_sentence_summary(day_pipeline.index, day_pipeline.tracker)
    assert s.endswith(".") and s.count(". ") == 0
    assert "92 frames" in s and "entered 2 times" in s and "CRITICAL" in s


def test_recommendations_reflect_incidents(day_pipeline):
    recs = " ".join(recommendations(day_pipeline.index, day_pipeline.tracker))
    assert "fence" in recs and "MH14CD5678" in recs and "battery" in recs


@pytest.mark.parametrize("question,expected", [
    ("What objects were in the video?", ["Blue Ford F150", "Fox", "Package"]),
    ("How many times did the blue Ford F150 enter?", ["entered 2 time(s) at 07:30, 17:45"]),
    ("Show critical alerts", ["Perimeter breach", "tampering"]),
    ("show all truck events", ["6 match(es)"]),
    ("Summarize the video", ["92 frames"]),
    ("What do you recommend?", ["Recommendations"]),
    ("any dogs?", ["Dog"]),
])
def test_offline_agent_follow_up_questions(day_pipeline, question, expected):
    answer = OfflineAnalyst(day_pipeline).ask(question)
    for e in expected:
        assert e in answer, answer


def test_make_analyst_falls_back_offline_without_key(day_pipeline, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert make_analyst(day_pipeline).name == "offline"


def test_make_analyst_prefers_groq_when_key_present(day_pipeline, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_dummy")
    a = make_analyst(day_pipeline)
    assert a.name.startswith("llm:groq")


def test_toolkit_outputs_are_json_serialisable(day_pipeline):
    import json

    tk = AnalystToolkit(day_pipeline)
    for out in (tk.search_frames("truck"), tk.get_alerts("HIGH"), tk.object_inventory(),
                tk.entity_history("MH14CD5678"), tk.frames_in_window("23:00", "23:59"), tk.video_summary()):
        json.dumps(out, default=str)


def test_context_is_bounded_and_prioritised(day_pipeline):
    ctx = day_pipeline.context.build_context("truck", day_pipeline.index, day_pipeline.tracker, budget_chars=1500)
    assert len(ctx) <= 1700
    assert ctx.index("## ALERTS") < ctx.index("## ENTITIES")  # alerts survive truncation first


try:  # let the live test pick up keys from .env
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@pytest.mark.skipif(not (os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
                    reason="needs GROQ_API_KEY or ANTHROPIC_API_KEY (live LLM call)")
def test_llm_agent_live(day_pipeline):
    from drone_security_agent.agent import LLMAnalyst, available_provider

    a = LLMAnalyst(day_pipeline, provider=available_provider())
    ans = a.ask("How many times did the blue Ford F150 enter today, and when?")
    assert "07:30" in ans and "17:45" in ans
    follow = a.ask("And when did it leave?")  # uses conversation memory
    assert "16:2" in follow


# ------------------------- dynamic inputs / robustness ------------------------- #
def test_fuzzed_descriptions_never_crash(pipeline):
    rng = random.Random(0)
    words = ("person truck van red blue gate fence warehouse climbing loitering parked entering at the near "
             "with flashlight no one , ; 12 plate MH01AA0001 fire dog").split()
    for i in range(500):
        desc = " ".join(rng.choice(words) for _ in range(rng.randint(0, 14)))
        pipeline.process(make_frame(i + 1, f"{(i // 60) % 24:02d}:{i % 60:02d}", desc))
    assert pipeline.index.stats()["frames"] == 500


def test_out_of_order_and_unknown_zone_frames(pipeline):
    pipeline.process(make_frame(1, "10:00", "Person walking near the barn"))
    r = pipeline.process(make_frame(2, "09:00", "Person climbing the perimeter fence", "perimeter_fence"))
    assert r.alerts[0].severity == Severity.CRITICAL


def test_emergency_alert_is_emitted_in_the_same_frame(pipeline):
    """Latency check: critical alerts are produced synchronously with the frame."""
    t = time.perf_counter()
    r = pipeline.process(make_frame(1, "03:00", "Fire spreading at the warehouse", "warehouse"))
    assert (time.perf_counter() - t) < 0.05
    assert r.alerts and r.alerts[0].severity == Severity.CRITICAL


# --------------------------------- scalability --------------------------------- #
def test_scalability_10k_frames():
    p = SecurityPipeline(index=FrameIndex(":memory:", commit_every=2000))
    p.keep_results = False
    t = time.perf_counter()
    for f in random_frames(10_000):
        p.process(f)
    elapsed = time.perf_counter() - t
    assert p.index.stats()["frames"] == 10_000
    assert elapsed < 60, f"ingest too slow: {elapsed:.1f}s"
    t = time.perf_counter()
    rows = p.index.natural_query("show all truck events", limit=100_000)["results"]
    assert rows and (time.perf_counter() - t) < 1.0
    assert len(p.context.build_context("truck", p.index, p.tracker)) < 9000
