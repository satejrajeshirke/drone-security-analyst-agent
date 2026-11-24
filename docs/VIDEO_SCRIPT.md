# Demo video plan (record with voiceover)

The brief asks for screen recordings **with voiceover** covering video processing, context summaries, agent
recommendations, the scalability test and innovative features. Suggested clips are 1–3 minutes each.
OBS Studio or the Windows Game Bar (Win+Alt+R) work, with your microphone on.

Before recording: `.venv\Scripts\activate`, then widen the terminal (about 120 columns).

### Clip 1: Overview and architecture (2 min)
Show `docs/ARCHITECTURE.md` rendered on GitHub.
> "The agent processes each frame in five stages: describe, detect, track, apply rules, index. Captions come from a
> VLM; for this prototype they're simulated text, as the brief allows, but the same pipeline accepts
> real video through Claude vision or BLIP..."

### Clip 2: Live processing, logs and alerts (3 min)
```
python -m drone_security_agent run --delay 0.15
```
Point out:
- The telemetry line ("Time: 00:01, Location: Main Gate").
- The 00:03 HIGH alert "Person loitering at main gate, 00:01".
- The 02:15 perimeter breach.
- The 06:45 staff car raising no alert (whitelist).
- The 17:45 "Blue Ford F150 entered twice today".
- The 23:10 merged CRITICAL alert.
- The 23:40 drone battery return-to-dock.

### Clip 3: Frame index queries (2 min)
```
python -m drone_security_agent query "show all truck events"
python -m drone_security_agent query "people at the main gate at night"
python -m drone_security_agent query "alerts after 22:00"
python -m drone_security_agent query "flashlight"
```
> "The 12:00 frame says 'Blue Ford F150 parked at the garage'. It never says 'truck', but the detector's tags
> are indexed in FTS5, so it's still found."

### Clip 4: Agent follow-up questions, summary and recommendations (3 min)
```
python -m drone_security_agent chat
  What objects were in the video?
  How many times did the blue Ford F150 enter?
  Show critical alerts
  Summarize the video
  What do you recommend?
```
If you have an API key, run `set ANTHROPIC_API_KEY=...` first and show the Claude-backed agent answering the same questions,
plus a follow-up such as "and when did it leave?" (conversation memory).
Also show `python -m drone_security_agent summary -v` to explain the bounded context snapshot.

### Clip 4b: Real images through the VLM (2–3 min)
```
python -m drone_security_agent images
```
Captions come from the cache, so this run is instant and matches [docs/vlm_results.md](vlm_results.md). Use `--fresh` to call
Gemini live, which is slower and may give different captions. Open a few photos from `data/real_frames` next to the output and point out:
- The fence-climb photo produces a CRITICAL perimeter breach.
- The guard at the night gate is logged as a guard and gets no alert.
- The plate is read from the pickup.
- Be honest about the limitation: the cars on the street outside raise an after-hours vehicle alert.

> "My first prompt let the small model say 'no activity' for a truck and a dog, so I added a self-check pass:
> any 'no activity' caption is re-checked with a narrow question and regenerated if something was missed."

### Clip 5: Scalability (1–2 min)
```
python -m drone_security_agent bench --frames 50000
```
> "50,000 frames at about 1,200 frames per second, queries in milliseconds, and the context fed to the LLM stays at about 8k
> characters regardless of video length."

### Clip 6: Tests (1 min)
```
python -m pytest -v
```
Scroll through the names: midnight alert, truck logged correctly, emergency, fuzzing, scale.
