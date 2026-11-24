"""Alert rules - one test per rule plus engine behaviour (merge, cooldown, config)."""
import copy

from drone_security_agent.models import Severity
from drone_security_agent.pipeline import SecurityPipeline
from drone_security_agent.indexer import FrameIndex

from .conftest import make_frame


def run(pipeline, frames):
    alerts = []
    for f in frames:
        alerts.extend(pipeline.process(f).alerts)
    return alerts


def rules_of(alerts):
    return [a.rule_id for a in alerts]


def test_alert_triggered_at_midnight_for_loitering(pipeline):
    alerts = run(pipeline, [
        make_frame(1, "00:01", "Person in dark hoodie standing near the main gate"),
        make_frame(2, "00:02", "Person in dark hoodie still standing at the main gate"),
        make_frame(3, "00:03", "Person in dark hoodie standing at the main gate"),
    ])
    assert rules_of(alerts) == ["NIGHT_LOITERING"]
    a = alerts[0]
    assert a.severity == Severity.HIGH
    assert a.message.startswith("Person loitering at main gate, 00:01")


def test_brief_dwell_does_not_alert(pipeline):
    alerts = run(pipeline, [make_frame(1, "00:01", "Person walking near the main gate")])
    assert alerts == []


def test_daytime_loitering_is_lower_severity(pipeline):
    alerts = run(pipeline, [make_frame(1, "14:00", "Person loitering at the main gate")])
    assert rules_of(alerts) == ["LOITERING"] and alerts[0].severity == Severity.MEDIUM


def test_perimeter_breach_is_critical(pipeline):
    alerts = run(pipeline, [make_frame(1, "02:15", "Person climbing the perimeter fence", "perimeter_fence")])
    assert alerts[0].rule_id == "PERIMETER_BREACH" and alerts[0].severity == Severity.CRITICAL


def test_repeat_vehicle_entered_twice(pipeline):
    alerts = run(pipeline, [
        make_frame(1, "07:30", "Blue Ford F150 entering through the main gate, plate MH14CD5678"),
        make_frame(2, "12:00", "Blue Ford F150 parked at the garage", "garage"),
        make_frame(3, "16:22", "Blue Ford F150 exiting through the main gate"),
        make_frame(4, "17:45", "Blue Ford F150 entering through the main gate"),
    ])
    assert rules_of(alerts) == ["REPEAT_VEHICLE"]
    assert alerts[0].message == "Blue Ford F150 entered twice today (07:30, 17:45)."


def test_same_visit_entry_not_double_counted(pipeline):
    alerts = run(pipeline, [
        make_frame(1, "07:30", "Blue Ford F150 entering through the main gate"),
        make_frame(2, "07:31", "Blue Ford F150 entering through the main gate"),
    ])
    assert "REPEAT_VEHICLE" not in rules_of(alerts)


def test_after_hours_vehicle_respects_whitelist(pipeline):
    alerts = run(pipeline, [
        make_frame(1, "05:30", "White Toyota Camry parked in the parking lot, plate MH12AB1234", "parking_lot"),
        make_frame(2, "05:31", "Black Hyundai sedan parked in the parking lot, plate KA01ZZ9999", "parking_lot"),
    ])
    assert rules_of(alerts) == ["AFTER_HOURS_VEHICLE"]
    assert "Black" in alerts[0].message


def test_restricted_zone_after_hours_only(pipeline):
    day = run(pipeline, [make_frame(1, "11:00", "Worker walking near the warehouse", "warehouse")])
    night = run(pipeline, [make_frame(2, "23:30", "Person walking near the warehouse", "warehouse")])
    assert day == []
    assert rules_of(night) == ["RESTRICTED_ZONE"] and night[0].severity == Severity.CRITICAL


def test_alerts_for_same_entity_are_merged(pipeline):
    alerts = run(pipeline, [make_frame(1, "23:10",
                 "Two people near the warehouse door with a flashlight, testing the door handle", "warehouse")])
    assert len(alerts) == 1
    a = alerts[0]
    assert a.rule_id in ("TAMPERING", "RESTRICTED_ZONE") and a.severity == Severity.CRITICAL
    assert {"SUSPICIOUS_BEHAVIOR"} <= set(a.related_rules)


def test_cooldown_suppresses_duplicates(pipeline):
    frames = [make_frame(i, f"00:{i:02d}", "Person in red jacket loitering at the main gate") for i in range(1, 6)]
    alerts = run(pipeline, frames)
    assert rules_of(alerts) == ["NIGHT_LOITERING"]


def test_unattended_package(pipeline):
    alerts = run(pipeline, [
        make_frame(1, "09:10", "Package left at the front door", "front_door"),
        make_frame(2, "09:20", "Package still at the front door", "front_door"),
        make_frame(3, "09:45", "Package still at the front door", "front_door"),
    ])
    assert rules_of(alerts) == ["UNATTENDED_PACKAGE"] and "35 min" in alerts[0].message


def test_gate_open_after_hours(pipeline):
    assert run(pipeline, [make_frame(1, "12:00", "Main gate left open")]) == []
    assert rules_of(run(pipeline, [make_frame(2, "20:00", "Main gate left open")])) == ["GATE_OPEN_AFTER_HOURS"]


def test_fire_is_emergency(pipeline):
    alerts = run(pipeline, [make_frame(1, "13:00", "Smoke and fire visible at the loading dock", "loading_dock")])
    assert alerts[0].rule_id == "HAZARD" and alerts[0].severity == Severity.CRITICAL
    assert "emergency" in alerts[0].recommendation.lower()


def test_drone_safety_rules(pipeline):
    alerts = run(pipeline, [
        make_frame(1, "10:00", "Clear view of the driveway, no activity", "driveway", battery_pct=22),
        make_frame(2, "10:30", "Clear view of the driveway, no activity", "driveway", battery_pct=12),
        make_frame(3, "11:00", "Clear view of the driveway, no activity", "driveway", altitude_m=140),
    ])
    sev = {(a.rule_id, a.severity) for a in alerts}
    assert ("DRONE_LOW_BATTERY", Severity.MEDIUM) in sev
    assert ("DRONE_LOW_BATTERY", Severity.CRITICAL) in sev
    assert ("DRONE_ALTITUDE", Severity.HIGH) in sev


def test_geofence(pipeline):
    f = make_frame(1, "10:00", "Clear view", "driveway")
    f.telemetry.lat += 0.01  # ~1.1 km north
    assert rules_of(run(pipeline, [f])) == ["DRONE_GEOFENCE"]


def test_config_change_changes_behaviour(cfg):
    """Rules are data: lowering the dwell threshold makes a 1-minute stay alert."""
    c = copy.deepcopy(cfg)
    c.rules.loitering_min_dwell_s = 60
    p = SecurityPipeline(c, FrameIndex(":memory:"))
    alerts = run(p, [make_frame(1, "01:00", "Person standing at the main gate"),
                     make_frame(2, "01:01", "Person standing at the main gate")])
    assert rules_of(alerts) == ["NIGHT_LOITERING"]
