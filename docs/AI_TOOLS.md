# How AI tools were used

**Primary tool:** Claude Code (Claude Opus 5.5) running as an agent inside VS Code. It read the assignment
PDF, set up the environment, wrote and ran the code and tests, and drafted the documentation.

| Stage | What the AI did | What was decided or checked by hand |
|---|---|---|
| Environment | Found that Python was missing, installed Python 3.12 with `winget`, created the venv, installed LangChain, langchain-anthropic and pytest, and confirmed that SQLite had FTS5 | Pinned versions in `requirements.txt` |
| Spec and architecture | Turned the brief into 3 requirements with acceptance criteria and proposed the caption-then-parse pipeline and the SQLite + FTS5 index. It also proposed the trade-off table in ARCHITECTURE.md | Chose deterministic rules over LLM-per-frame for cost and latency, and SQLite over a vector DB for the prototype |
| Scenario design | Wrote a 24-hour scripted day in which each event targets one rule (midnight loitering, fence climb, repeat F150, whitelisted staff car, unattended package, altitude and battery faults) | Checked that the brief's sample outputs appear verbatim |
| **AI-generated components** | `detection.py` (lexicon/regex object and action extraction), `rules.py` (13 alert rules + merge + cooldown), `indexer.py` natural-query parser, `context.py` tiered memory | Reviewed the outputs; fixed issues found by running the code (below) |
| Agent | Generated the LangChain `create_agent` wiring, tool wrappers, system prompt, sliding-window memory and server-side refusal fallback; looked up the current Claude API shape (adaptive thinking defaults, `fallbacks`, `output_config.effort`) instead of relying on memory | Added the offline analyst so the demo runs without an API key |
| QA | Wrote 62 pytest cases (golden scenario, negatives, fuzzing, emergency, scale) and the benchmark command | All tests run locally: 61 pass, 1 skipped without a key |
| Docs | Drafted README, spec, architecture (Mermaid), testing doc, report and video script | Edited for accuracy against real outputs |

## Bugs found by running the AI's code
- **Rich markup swallowed log details.** Log suffixes such as `[parked | plate ...]` disappeared because Rich read them as
  markup. Found on the first verbose run and fixed with `rich.markup.escape`.
- **Repetitive recommendations.** The first version produced one near-identical recommendation per zone. It was merged
  into a single "after-hours hot-spots" line.
- **Alert storms by design.** Early rule drafts raised 3 separate alerts for one event ("two people with a flashlight
  testing the warehouse door"). This led to per-entity alert merging with `related_rules`.

- **Night-time answer gap (Groq agent).** "What happened at night?" first covered only 00:00–06:00 and missed
  22:30–23:40. The system prompt now states the night window explicitly.
- **Real-image VLM issues.** Running Gemini on 8 real photos and comparing the captions with the images found:
  - "lazy negative" captions ("no activity" for a truck, a dog and 1,000 cars), fixed with a stricter prompt and a self-check pass
  - zone-hint hallucination ("warehouse roof")
  - four parser bugs on real phrasing: plate `WDS947IS`, unknown "Dodge Ram", *carrying* chosen over *climbing*, and "climbing over a gate" read as the main gate

  Each fix has a regression test.
- **Model availability.** `gemini-2.5-flash` was retired for new keys and the newer Flash models returned 503 under load.
  This was found by calling the API, and led to the failover chain and a caption cache for repeatable demos.

## Impact
- A working, tested end-to-end prototype in a single session instead of several days.
- Researching unfamiliar areas (SQLite FTS5 external-content tables, BM25 ranking, LangChain 1.x
  `create_agent`, the current Claude API parameters) was done inside the coding loop, and each finding was checked by running code.
- **Limitations:** the AI's first drafts were not always right, as the bugs above show. Running the code and tests,
  not reading the code, is what caught them.
