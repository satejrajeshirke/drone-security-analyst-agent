"""Frame-by-frame index: storage and queries by time / object / zone / text."""
from drone_security_agent.indexer import FrameIndex, _to_minute


def test_every_frame_is_indexed(day_pipeline):
    st = day_pipeline.index.stats()
    assert st["frames"] == len(day_pipeline.results)
    assert st["detections"] == sum(len(r.detections) for r in day_pipeline.results)


def test_show_all_truck_events(day_pipeline):
    res = day_pipeline.index.natural_query("show all truck events")
    assert res["mode"] == "structured"
    times = [r["ts"][11:16] for r in res["results"]]
    assert times == ["07:30", "07:32", "12:00", "16:20", "16:22", "17:45"]
    # 12:00 frame never says "truck" - found via detector tags
    assert any(r["description"] == "Blue Ford F150 parked at the garage" for r in res["results"])


def test_query_by_time_window_wrapping_midnight(day_pipeline):
    rows = day_pipeline.index.detections(object_type="person", start="22:00", end="06:00")
    times = sorted({r["ts"][11:16] for r in rows})
    assert times[0] == "00:01" and "23:10" in times
    assert all(t >= "22:00" or t <= "06:00" for t in times)


def test_query_by_zone_and_type(day_pipeline):
    rows = day_pipeline.index.detections(object_type="person", zone="main_gate")
    assert {r["ts"][11:16] for r in rows} == {"00:01", "00:02", "00:03", "00:04"}


def test_full_text_search_ranks_matches(day_pipeline):
    rows = day_pipeline.index.search_text("flashlight warehouse")
    assert rows and "flashlight" in rows[0]["description"]


def test_natural_query_alerts_after_time(day_pipeline):
    res = day_pipeline.index.natural_query("alerts after 22:00")
    assert res["mode"] == "alerts"
    assert all(r["ts"][11:16] >= "22:00" for r in res["results"])
    assert len(res["results"]) == 3


def test_natural_query_parsing():
    idx = FrameIndex()
    f = idx.parse_query("blue truck at the main gate between 7am and 9:30")
    assert f["subtype"] == "truck" and f["color"] == "blue" and f["zone"] == "main_gate"
    assert _to_minute(f["start"]) == 7 * 60 and _to_minute(f["end"]) == 9 * 60 + 30


def test_frames_between(day_pipeline):
    rows = day_pipeline.index.frames_between("02:00", "02:20")
    assert [r["ts"][11:16] for r in rows] == ["02:10", "02:15", "02:16"]


def test_entity_timeline(day_pipeline):
    rows = day_pipeline.index.entity_timeline("veh:MH14CD5678")
    assert len(rows) == 6


def test_persistence_on_disk(tmp_path, day_pipeline):
    from drone_security_agent.pipeline import SecurityPipeline
    from drone_security_agent.simulator import scripted_day

    db = tmp_path / "idx.db"
    p = SecurityPipeline(index=FrameIndex(db))
    p.run(scripted_day())
    p.index.close()
    reopened = FrameIndex(db)
    assert reopened.stats()["frames"] == day_pipeline.index.stats()["frames"]
    assert reopened.natural_query("show all truck events")["results"]
