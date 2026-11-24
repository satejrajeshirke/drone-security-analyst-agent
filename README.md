# Drone Security Analyst Agent

A working prototype of an AI security analyst for a **docked drone that patrols a fixed property**. It processes
drone telemetry and video frames in real time, identifies and logs objects and events with context, raises
rule-based security and safety alerts, indexes every frame for later search, and answers follow-up questions with
a LangChain agent on Groq (gpt-oss-120b) or Claude.

```
Time: 00:03, Location: Main Gate, Alt 25 m, Battery 98%, Mode HOVER
  Frame 3: Person in dark hoodie loitering at the main gate, checking the gate latch
  LOG   Person spotted at main gate, 00:03. [loitering | sighting #3]
  ALERT HIGH Person loitering at main gate, 00:01 (dwell 2 min, still present 00:03).
        -> Activate gate floodlight and two-way speaker warning; continue drone overwatch.
...
  LOG   Blue Ford F150 spotted at garage, 12:00. [parked | plate MH14CD5678 | entries today: 1 | sighting #3]
...
  ALERT INFO Blue Ford F150 entered twice today (07:30, 17:45).

$ python -m drone_security_agent query "show all truck events"
6 result(s) - mode=structured filters={'subtype': 'truck'}   07:30 07:32 12:00 16:20 16:22 17:45
```

| Brief requirement | Where |
|---|---|
| Feature spec (value + 3 key requirements) | [docs/FEATURE_SPEC.md](docs/FEATURE_SPEC.md) |
| Architecture / design artifacts (flowchart, sequence, agent flow, ER diagram) | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Simulated telemetry + frames as text | [simulator.py](drone_security_agent/simulator.py), [data/scenario_day.json](data/scenario_day.json) |
| Object/event detection and logging with context | [detection.py](drone_security_agent/detection.py), [tracker.py](drone_security_agent/tracker.py) |
| Real-time alerts from predefined rules | [rules.py](drone_security_agent/rules.py), [config/default.yaml](config/default.yaml) |
| **Frame-by-frame indexing** (cross-domain) | [indexer.py](drone_security_agent/indexer.py): SQLite + FTS5 |
| VLM scripts | [vlm.py](drone_security_agent/vlm.py): Gemini vision (tested on 8 real photos) / Claude / BLIP / OpenCV; [results](docs/vlm_results.md) |
| LangChain agent | [agent.py](drone_security_agent/agent.py) |
| Context management | [context.py](drone_security_agent/context.py) |
| Bonus: video summary + follow-up Q&A | [insights.py](drone_security_agent/insights.py), `ask` / `chat` commands |
| QA / tests (76 cases) | [tests/](tests/), [docs/TESTING.md](docs/TESTING.md) |
| AI tools used | [docs/AI_TOOLS.md](docs/AI_TOOLS.md) |
| Report (PDF) | [docs/REPORT.pdf](docs/REPORT.pdf) |
| **Demo video (voiceover)** | [Google Drive](https://drive.google.com/file/d/1kBkghCLg-n2GAupJo2UftY_kq_alLgo8/view?usp=drive_link) |
| Demo video script | [docs/VIDEO_SCRIPT.md](docs/VIDEO_SCRIPT.md) |

---

## 1. Setup

Requires **Python 3.10+** (developed and tested on 3.12.10, Windows 11). No GPU is needed.

```bash
git clone <this repo> && cd FlytBase
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q                      # 76 passed (live Groq/Gemini tests are skipped without keys)
```

**Optional: LLM-powered agent.** Without a key, the agent runs in a deterministic offline mode. Keys are read
from a `.env` file in the project root (gitignored, never committed):
```bash
# .env
GROQ_API_KEY=gsk_...            # Groq, default model openai/gpt-oss-120b (override with GROQ_MODEL=...)
# ANTHROPIC_API_KEY=sk-ant-...  # alternative: Claude (claude-opus-5-5)
```
`--mode auto` (the default) picks Groq if `GROQ_API_KEY` is set, then Claude, then offline. You can force a mode with
`--mode groq|claude|offline`:
```bash
python -m drone_security_agent ask "What happened at night and what should I do about it?"
python -m drone_security_agent chat --mode groq
```

**Real images through a VLM (Gemini).** Add a free Gemini key (aistudio.google.com/apikey) to `.env`:
```bash
# .env
GEMINI_API_KEY=...
python scripts/fetch_sample_images.py      # 8 openly licensed real photos -> data/real_frames (already included)
python -m drone_security_agent images      # real photo -> Gemini caption -> detect -> track -> rules -> index
```
Captions are cached in `out/vlm_cache.json`, so re-runs are instant and repeatable; add `--fresh` to call Gemini
again. Results: [docs/vlm_results.md](docs/vlm_results.md).

**Optional: process a real video file** (OpenCV frame sampling, then the same VLM step):
```bash
pip install -r requirements-vlm.txt
python -m drone_security_agent video clip.mp4 --describer gemini --every 2   # or claude / blip (local)
```

## 2. Running

| Command | What it does |
|---|---|
| `python -m drone_security_agent run` | Streams the simulated 24-hour day. It prints telemetry, captions, logs and alerts, then an alert table, a 1-sentence summary and recommendations. It writes `out/index.db` and `out/report.json`. Add `--delay 0.2` for demo pacing. |
| `python -m drone_security_agent query "show all truck events"` | Plain-English query of the frame index. Other examples: `"people at the main gate at night"`, `"blue vehicles between 7am and 13:00"`, `"alerts after 22:00"`, `"flashlight"` |
| `python -m drone_security_agent ask "What objects were in the video?"` | One question to the analyst agent (`--mode auto\|llm\|offline`) |
| `python -m drone_security_agent chat` | Interactive follow-up Q&A |
| `python -m drone_security_agent summary [-v]` | 1-sentence video summary; `-v` also prints the agent's context snapshot |
| `python -m drone_security_agent bench --frames 50000` | Scalability test: ingest throughput, latency and query timings |
| `python -m drone_security_agent images [--fresh]` | Real photos from `data/real_frames` through Gemini vision and the full pipeline; writes [docs/vlm_results.md](docs/vlm_results.md) |
| `python -m drone_security_agent simulate --out data/scenario_day.json` | Exports the simulated feed; replay it with `--input <file>` |

Captured outputs: [docs/sample_output.txt](docs/sample_output.txt), [docs/benchmark_output.txt](docs/benchmark_output.txt).

## 3. How it works

```
frame + telemetry ─► Describe (VLM) ─► Detect ─► Track / re-ID ─► Rules ─► alerts + log
                                          │            │            │
                                          └────────────┴────────────┴─► SQLite index (+FTS5) ◄─► Analyst agent
                                                                    └─► Context manager ──────►  (LangChain + Claude)
```

1. **Describe** ([vlm.py](drone_security_agent/vlm.py)). Each frame becomes one security-focused caption. In the
   simulation the caption is given. For real images, **Gemini vision** generates it (temperature 0). If a model is
   overloaded, the next Flash model takes over. A **self-check** re-examines any "no activity" caption with a narrow
   "list every person, vehicle, animal or package" question and regenerates the caption when something was missed.
   Claude vision and local BLIP are alternative backends.
2. **Detect** ([detection.py](drone_security_agent/detection.py)). Captions are parsed into typed detections:
   `vehicle / truck / blue / Ford F150 / plate MH14CD5678 / entering / main_gate`. The parser handles multi-object frames,
   counts ("two people"), clothing, suspicious cues, negation ("no one present") and direction words ("toward the gate").
3. **Track** ([tracker.py](drone_security_agent/tracker.py)). Detections are linked into entities across frames: by plate,
   by visual signature, or by a signature previously linked to a plate. The tracker keeps visits (10-minute gap), entries and dwell time.
4. **Rules** ([rules.py](drone_security_agent/rules.py)). 13 rules: perimeter breach, tampering,
   restricted zone, night loitering, suspicious behaviour, after-hours vehicle (with whitelist), repeat vehicle,
   unattended package, gate open, fire/smoke, drone low battery, altitude ceiling and geofence. Alerts for the same entity
   in one frame are **merged** (highest severity wins) and a **cooldown** prevents alert storms. Every alert carries a
   recommended action.
5. **Index** ([indexer.py](drone_security_agent/indexer.py)). SQLite tables `frames`, `detections` and `alerts` have B-tree
   indexes on time, type, zone and entity, plus an FTS5 table over captions and detector tags. Queries can be
   structured, BM25 full-text, or natural language (parsed into filters, with FTS as fallback). Time windows that wrap midnight are supported.
6. **Context** ([context.py](drone_security_agent/context.py)). Memory is kept in three tiers: a recent-frames ring buffer, hourly digests,
   and entity memory plus the index. `build_context()` assembles a prioritised snapshot under a character budget, so the
   LLM prompt is about 8k characters whether the video has 92 frames or 50,000.
7. **Agent** ([agent.py](drone_security_agent/agent.py)). A LangChain `create_agent` on Groq (`openai/gpt-oss-120b`) or Claude, with 6 tools (search_frames,
   get_alerts, object_inventory, entity_history, frames_in_window, video_summary), the context snapshot in its system prompt,
   a 12-message sliding-window memory and server-side refusal fallback. `OfflineAnalyst` answers the same
   question types with no LLM.

## 4. Design decisions (summary)
Full table in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#5-design-decisions).
- **Caption first, then parse.** One text representation works for search, rules, the LLM and humans, and the same
  downstream code handles simulated and real frames.
- **Deterministic per-frame analytics, LLM at the edges.** Per-frame work takes about 1 ms, is free and is testable. The LLM is used where it
  adds the most: describing pixels and answering open questions.
- **SQLite + FTS5** is zero-ops and supports both time/object queries and text search; Postgres/pgvector is the scale-up path.
- **Rules as data** in YAML: owners tune hours, thresholds, whitelists and suspicious keywords.
- **Merged alerts with recommendations.** Operators get one actionable alert per subject, not three.

## 5. AI tools used
Built with **Claude Code** (Claude Opus 5.5) as an agentic pair-programmer. It handled environment setup, architecture proposals, code
for every module, the test suite, the benchmark and doc drafts. The work was then checked by running it, which caught several bugs.
Details and examples are in [docs/AI_TOOLS.md](docs/AI_TOOLS.md).

## 6. Project layout
```
drone_security_agent/   models, config, simulator, vlm, detection, tracker, rules,
                        indexer, context, insights, agent, pipeline, cli
config/default.yaml     property zones + rule thresholds
data/scenario_day.json  the simulated 24 h feed (92 frames)
tests/                  76 pytest cases
docs/                   spec, architecture, testing, AI tools, report, video script, sample outputs
```

## 7. Limitations and next steps
- The text parser covers the vocabulary of the scenario and common phrasings. Unusual captions fall back to full-text search.
  Next step: let Claude return structured JSON detections directly (structured outputs), keeping the parser as a fast path.
- Person re-ID uses clothing text only. Next step: appearance embeddings.
- Add CLIP/SigLIP embeddings for image similarity search, plus YOLO + ByteTrack for boxes and tracks on real video.
- Alert delivery (push, webhook, VMS) and guard acknowledgement workflow.
