# Architecture

## 1. System overview

```mermaid
flowchart LR
    subgraph Drone["Docked drone (simulated)"]
        CAM[Camera frames]
        TEL[Telemetry<br/>GPS, alt, battery, mode]
    end

    subgraph Pipeline["Streaming pipeline (pipeline.py)"]
        VLM["1. Describe<br/>vlm.py<br/>simulated caption | Gemini vision | Claude | BLIP"]
        DET["2. Detect<br/>detection.py<br/>caption to typed objects"]
        TRK["3. Track / re-ID<br/>tracker.py<br/>entities, visits, entries"]
        RUL["4. Rules<br/>rules.py<br/>13 rules, merge, cooldown"]
    end

    subgraph Storage["Memory"]
        IDX[("SQLite index<br/>frames, detections, alerts<br/>+ FTS5 full-text")]
        CTX["Context manager<br/>recent ring buffer,<br/>hourly digests, entity memory"]
    end

    subgraph Outputs
        LOG[Activity log]
        ALR[Real-time alerts<br/>+ recommendations]
        AGT["Analyst agent (agent.py)<br/>LangChain + Groq or Claude<br/>or offline router"]
        USR((Owner / operator))
    end

    CAM --> VLM
    TEL --> VLM
    VLM --> DET --> TRK --> RUL
    TRK --> LOG
    RUL --> ALR
    DET --> IDX
    RUL --> IDX
    RUL --> CTX
    IDX <--> AGT
    CTX --> AGT
    ALR --> USR
    LOG --> USR
    USR <-- "questions / answers, summary" --> AGT
```

## 2. Per-frame sequence

```mermaid
sequenceDiagram
    participant F as Frame + telemetry
    participant V as Describer (VLM)
    participant D as Detector
    participant T as Tracker
    participant R as AlertEngine
    participant I as FrameIndex (SQLite)
    participant C as ContextManager
    F->>V: image or caption
    V-->>D: "Blue Ford F150 entering through the main gate, plate MH14CD5678"
    D-->>T: Detection(vehicle, truck, blue, Ford, F150, plate, entering, main_gate)
    T-->>T: resolve entity veh:MH14CD5678, visit #2, entries=[07:30, 17:45]
    T-->>R: detections + entity state
    R-->>R: run 13 rules, merge per entity, apply cooldown
    R-->>I: INSERT frame, detections, alerts, FTS row
    R-->>C: update hourly digest, recent buffer
    R-->>F: ALERT INFO "Blue Ford F150 entered twice today (07:30, 17:45)."
```

## 3. Agent question flow (context management)

```mermaid
flowchart TD
    Q[User question] --> M{GROQ_API_KEY or ANTHROPIC_API_KEY set?}
    M -- no --> OFF["OfflineAnalyst<br/>regex intent router"] --> TK
    M -- yes --> SNAP["Build context snapshot (bounded ~8k chars)<br/>1 site header, 2 alerts, 3 entities,<br/>4 hourly digests, 5 FTS-retrieved frames, 6 recent frames"]
    SNAP --> SYS[System prompt]
    HIST["Sliding-window memory<br/>(last 12 messages)"] --> LLM
    SYS --> LLM["LangChain create_agent<br/>Groq gpt-oss-120b or Claude"]
    LLM -- tool calls --> TK["AnalystToolkit<br/>search_frames, get_alerts, object_inventory,<br/>entity_history, frames_in_window, video_summary"]
    TK --> IDX[(SQLite index)]
    TK --> LLM
    LLM --> A[Grounded answer with HH:MM + frame ids]
```

## 4. Data model (SQLite)

```mermaid
erDiagram
    frames ||--o{ detections : contains
    frames ||--o{ alerts : triggers
    frames ||--|| frames_fts : "FTS5 (description, tags)"
    frames {
        int frame_id PK
        text ts
        real ts_epoch
        int minute_of_day
        text description
        text tags "detector labels for search"
        text zone
        real lat
        real lon
        real altitude_m
        real battery_pct
    }
    detections {
        int id PK
        int frame_id FK
        text object_type
        text subtype
        text label
        text color
        text make
        text model
        text plate
        text action
        text zone
        text entity_id
    }
    alerts {
        int id PK
        int frame_id FK
        text rule_id
        int severity
        text message
        text recommendation
        text related_rules
    }
```

## 5. Design decisions

| Decision | Why | Trade-off |
|---|---|---|
| **One function per stage, in-process** | Simple to test and debug; every stage boundary is where a queue would go in production | Single process; no horizontal scaling yet |
| **Caption first, then parse** (VLM → text → structured) | A caption works for indexing, search, the LLM and humans at once. The same parser serves simulated and real frames | Parsing depends on caption quality; mitigated by a security-specific VLM prompt |
| **Deterministic detector and rules** (lexicon + regex, pure functions) | About 1 ms/frame, explainable, unit-testable, zero cost. An LLM is used where it adds value: captioning and Q&A | Lower recall on unusual phrasing; the LLM agent can still use free-text search |
| **SQLite + FTS5** for the index | Zero-ops, ships with Python, B-tree and BM25 search in one file, handles millions of rows | Single writer; Postgres/pgvector or OpenSearch at fleet scale |
| **`tags` column in FTS** | "show all truck events" finds "Blue Ford F150 parked at the garage", even though the caption never says "truck" | Small storage overhead |
| **Entity re-ID** by plate, then visual signature, with plate-to-signature linking | Enables "entered twice today" and dwell-time loitering | Naive for people; production would use appearance embeddings |
| **Alert merge + cooldown** | One actionable alert per subject per frame, with no alert storms | Related rules are listed rather than raised separately |
| **Rules as data** (`config/default.yaml`) | Owners can tune hours, thresholds and whitelists without code | Complex rules still need Python |
| **Tiered context manager** | Prompt cost stays roughly constant (~8k chars) whether the video has 90 or 50,000 frames | Summaries lose detail, which the tools recover on demand |
| **Two agent brains, one toolkit** | Works offline for demos and CI; Claude adds reasoning, phrasing and multi-step tool use | Offline answers are templated |

## 6. Production path
1. Ingest RTSP or a drone SDK stream through a message queue (Kafka/NATS), with one topic per stage.
2. Add a real detector (YOLOv8/RT-DETR) for boxes and tracking (ByteTrack), plus a VLM for captions of keyframes only.
3. Add CLIP/SigLIP image embeddings in a vector store for "find frames that look like this".
4. Deliver alerts through webhooks, push notifications or a VMS integration, with guard acknowledgement and an escalation timer.
5. Move the index to Postgres (Timescale + pgvector) with per-site partitioning and a retention policy.
