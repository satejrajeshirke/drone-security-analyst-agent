# Testing & Validation

```bash
python -m pytest -q          # 76 passed (live Groq + Gemini tests run when the keys are in .env, else skipped)
python -m drone_security_agent images                 # real photos through Gemini vision -> docs/vlm_results.md
python -m drone_security_agent bench --frames 50000   # scalability test
```

## Strategy
1. **Unit tests per stage** cover the detector, each alert rule and the index queries.
2. **Golden-scenario test.** The scripted 24-hour day (`simulator.SCRIPTED_EVENTS`) was designed so that each event
   exercises one rule. The test asserts the exact ordered list of 10 alerts, so any regression
   (missed alert or false alarm) fails.
3. **Negative tests** check that quiet patrol frames, animals, a whitelisted staff car and daytime warehouse
   workers raise no alert.
4. **Dynamic inputs and robustness** cover 500 fuzzed random-word captions, out-of-order timestamps, unknown zones and config changes at runtime.
5. **Emergency response** checks that fire, smoke, perimeter breach and critical battery produce a CRITICAL alert in the same call as
   the frame, with an emergency recommendation.
6. **Scale** ingests 10k frames in the test suite and 50k in the benchmark, measures query latency, and checks that the LLM context stays bounded.

## Test matrix
| Area | Test | What it proves |
|---|---|---|
| Detection | `test_truck_logged_with_make_model_color_plate` | "truck logged correctly": colour, make, model, plate, action, zone |
| | `test_brief_example_blue_truck_at_gate` | The brief's own example "Blue truck at gate" |
| | `test_group_count_and_action_from_other_clause` | "Two people ... testing the door" gives count=2, action=tampering |
| | `test_negated_or_empty_scenes_produce_no_detections` | "no one present" does not create a person |
| | `test_directional_phrase_does_not_move_zone` | "exiting the garage toward the gate" places the object at the garage |
| Rules | `test_alert_triggered_at_midnight_for_loitering` | "alert triggered at midnight": HIGH, "Person loitering at main gate, 00:01" |
| | `test_brief_dwell_does_not_alert` | A single sighting is not loitering |
| | `test_repeat_vehicle_entered_twice` | "Blue Ford F150 entered twice today (07:30, 17:45)." |
| | `test_same_visit_entry_not_double_counted` | Re-ID visit logic |
| | `test_after_hours_vehicle_respects_whitelist` | Staff plate suppressed, unknown car alerted |
| | `test_alerts_for_same_entity_are_merged` | Tampering + restricted zone + flashlight become one CRITICAL alert |
| | `test_cooldown_suppresses_duplicates` | 5 loitering frames raise 1 alert |
| | `test_fire_is_emergency`, `test_drone_safety_rules`, `test_geofence` | Emergency and drone safety alerts |
| | `test_config_change_changes_behaviour` | Thresholds are data |
| Index | `test_show_all_truck_events` | The brief's "show all truck events" query, including captions without the word "truck" |
| | `test_query_by_time_window_wrapping_midnight` | 22:00-06:00 windows |
| | `test_full_text_search_ranks_matches` | FTS5 BM25 |
| | `test_persistence_on_disk` | The index survives a process restart |
| Agent | `test_offline_agent_follow_up_questions` (7 cases) | "What objects were in the video?", entity history, alerts, summary, recommendations |
| | `test_llm_agent_live` | Claude agent plus conversation memory (needs key) |
| | `test_context_is_bounded_and_prioritised` | Context budget is respected; alerts are kept first |
| E2E | `test_scripted_day_raises_exactly_the_expected_alerts` | Golden 24-hour run |
| | `test_sample_log_line_matches_brief`, `test_sample_alert_matches_brief` | The brief's expected sample outputs |
| Scale | `test_scalability_10k_frames` | 10k frames in under 60 s, query under 1 s, bounded context |
| VLM (real images) | `test_gemini_fails_over_to_next_model_and_sticks` | 503 is retried, then the next model is tried; a 404 (retired model) is skipped; the working model is tried first next time (mocked) |
| | `test_pipeline_survives_vlm_outage` | A VLM outage flags the frame and the patrol keeps running |
| | `test_negative_caption_is_verified_and_corrected`, `test_true_negative_is_kept` | Self-check pass: a wrong "no activity" is corrected, a true empty scene is kept |
| | `test_plates_from_real_vlm_phrasing` | "license plate WDS-947 is parked" gives WDS947 (was WDS947IS) |
| | `test_most_relevant_action_wins`, `test_object_being_climbed_is_not_a_location`, `test_people_in_one_scene_are_one_group` | Parser fixes found on real Gemini captions |
| | `test_gemini_live_fence_image` | Live: the real fence-climb photo is captioned as climbing a fence |

## Golden scenario: expected alerts
| Time | Frame caption (abridged) | Alert |
|---|---|---|
| 00:03 | Person in dark hoodie loitering at the main gate | HIGH NIGHT_LOITERING: "Person loitering at main gate, 00:01 (dwell 2 min...)" |
| 02:15 | Person climbing the perimeter fence | CRITICAL PERIMETER_BREACH |
| 02:16 | Person in black jacket running inside the fence | CRITICAL RESTRICTED_ZONE |
| 06:45 | White Toyota Camry, plate MH12AB1234 (staff) | none (whitelisted) |
| 10:40 | Package still at the front door | LOW UNATTENDED_PACKAGE (90 min) |
| 15:10 | (telemetry) altitude 131 m | HIGH DRONE_ALTITUDE |
| 17:45 | Blue Ford F150 entering, plate MH14CD5678 | INFO REPEAT_VEHICLE: "entered twice today (07:30, 17:45)" |
| 19:30 | Main gate left open | MEDIUM GATE_OPEN_AFTER_HOURS |
| 22:30 | Grey sedan, headlights off, circling | HIGH SUSPICIOUS_BEHAVIOR (+AFTER_HOURS_VEHICLE) |
| 23:10 | Two people, flashlight, testing warehouse door | CRITICAL TAMPERING (+RESTRICTED_ZONE, SUSPICIOUS_BEHAVIOR) |
| 23:40 | (telemetry) battery 14% | CRITICAL DRONE_LOW_BATTERY (return to dock) |

## Scalability results (laptop, Python 3.12, single process)
From [benchmark_output.txt](benchmark_output.txt):

| Metric | Value |
|---|---|
| Frames ingested | 50,000 (about 28 h of footage at one frame per 2 s) |
| Throughput | about 1,270 frames/s end-to-end (describe, detect, track, rules, index) |
| Per-frame latency | p50 0.64 ms, p99 2.0 ms |
| "show all truck events" over 50k frames | 36 ms (2,475 rows) |
| "people at the main gate at night" | 13 ms |
| Alerts >= HIGH | 11 ms |
| LLM context size | 8,029 chars, constant regardless of video length |

Even a real VLM at 0.5 to 2 s per keyframe would be the bottleneck, not the analytics or storage layer.
