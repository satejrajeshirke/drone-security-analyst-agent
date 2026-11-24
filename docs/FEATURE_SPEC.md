# Feature Spec: Drone Security Analyst Agent

## Problem
A docked drone patrols a fixed property (a logistics yard) every day. It produces telemetry and hours of
video that nobody has time to watch. Property owners care about two things:
1. **"Tell me now"**: anything that needs action, such as an intruder at the fence or someone loitering at the gate at midnight.
2. **"Tell me later"**: a searchable memory of what happened, such as "a blue Ford F150 entered twice today" or "show all truck events".

## Value to property owners
- **Automated monitoring.** Every frame is described, parsed into objects and checked against security rules in about 1 ms, so nothing is missed at 3 a.m.
- **Fewer, better alerts.** Several rule hits on the same subject are merged into one alert with a recommended action. A per-subject cooldown stops repeat alerts, and known staff vehicles are whitelisted.
- **Searchable footage.** Every frame is stored by time, object, zone and text, so "what happened at the gate last night?" takes seconds instead of an hour of scrubbing video.
- **Context, not just detections.** The agent remembers each subject: it knows the F150 at 17:45 is the same truck that entered at 07:30, and that the hoodie person has been at the gate for 2 minutes.
- **Drone safety included.** The same rule engine watches battery level, the altitude ceiling and the geofence.

## Key requirements
| # | Requirement | Acceptance criteria | Where |
|---|---|---|---|
| R1 | **Real-time detection and alerting.** Every frame is analysed as it arrives. Rule-based alerts have a severity, a time, a zone and a recommendation. | Alert is emitted in the same call as its frame (<50 ms); "Person loitering at main gate, 00:01" is raised; harmless animals and quiet patrol frames raise nothing | `rules.py`, `pipeline.py`, `tests/test_rules.py` |
| R2 | **Contextual event log and entity memory.** Detections are logged with time, place, action and identity, and are tracked across frames and visits. | "Blue Ford F150 spotted at garage, 12:00."; "Blue Ford F150 entered twice today (07:30, 17:45)." | `detection.py`, `tracker.py` |
| R3 | **Queryable frame index plus analyst Q&A.** Every frame is indexed and can be queried by time, object, zone or free text. An agent answers follow-up questions, summarises the video and recommends actions. | "show all truck events" returns all 6 F150 frames, including ones whose caption never says "truck"; a 50k-frame index answers queries in under 40 ms | `indexer.py`, `agent.py`, `insights.py` |

## Out of scope (prototype)
Live RTSP ingestion, real drone SDK control, multi-camera fusion, face recognition, and identity verification
beyond licence plates. The [architecture doc](ARCHITECTURE.md) shows where each of these would plug in.
