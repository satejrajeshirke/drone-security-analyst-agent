"""Security Analyst Agent - follow-up Q&A, summaries and recommendations.

Two interchangeable brains over the SAME toolkit:

* `LLMAnalyst`     - a LangChain agent (`langchain.agents.create_agent`) on
  Claude (claude-opus-5-5) that decides which tools to call. The system prompt
  carries a compact, budgeted context snapshot (see context.py) so simple
  questions need no tool call; detailed questions trigger retrieval tools.
  Conversation memory is a sliding window of the last N messages.
* `OfflineAnalyst` - deterministic intent router used when no API key is set
  (and in tests). Same tools, template answers.

Every fact the agent states comes from a tool backed by the SQLite index, so
answers stay grounded and cite times / frame ids.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

from .indexer import FrameIndex
from .insights import one_sentence_summary, recommendations
from .models import pretty_zone
from .pipeline import SecurityPipeline

SYSTEM_PROMPT = """You are the Drone Security Analyst for {site}. A docked drone patrols the property \
and every frame it captured has been captioned, parsed into objects and indexed. You answer the \
property owner's questions about today's footage, explain alerts and recommend actions.

How to work:
- Ground every factual claim in the context below or in a tool result. Cite the time (HH:MM) and \
frame id, e.g. "02:15 (frame 12)". If the data does not show something, say so - never invent sightings.
- The context snapshot is a compressed summary. Use the tools for exact lists, counts, time windows \
or any object/zone not covered there.
- Be concise: lead with the answer, then the supporting evidence, then (when useful) a recommendation.
- Severity order: CRITICAL > HIGH > MEDIUM > LOW > INFO.
- Site hours: night is {night}; business hours are {business}. The footage covers one calendar day, so \
"night" means BOTH the early morning (00:00 to the end of night) AND the late evening (from the start of \
night to 23:59). Time-window tools accept wrapping windows such as start="22:00", end="06:00".

=== CONTEXT SNAPSHOT ===
{context}
"""


class AnalystToolkit:
    """Plain-Python tools over the index; wrapped as LangChain tools below."""

    def __init__(self, pipeline: SecurityPipeline):
        self.p = pipeline
        self.index: FrameIndex = pipeline.index

    @staticmethod
    def _slim(rows: list[dict[str, Any]], keys: tuple[str, ...], limit: int = 25) -> list[dict[str, Any]]:
        out = []
        for r in rows[:limit]:
            out.append({k: (r[k][11:16] if k == "ts" and r.get(k) else r.get(k)) for k in keys if k in r})
        return out

    def search_frames(self, query: str) -> dict[str, Any]:
        """Search indexed frames/detections with a natural-language query, e.g. 'truck events after 12:00'."""
        res = self.index.natural_query(query, limit=50)
        keys = ("ts", "frame_id", "label", "action", "zone", "plate", "description", "message", "severity_name")
        return {"mode": res["mode"], "filters": res["filters"], "count": len(res["results"]),
                "results": self._slim(res["results"], keys)}

    def get_alerts(self, min_severity: str = "INFO", start: Optional[str] = None,
                   end: Optional[str] = None) -> list[dict[str, Any]]:
        """List alerts at or above a severity, optionally within a HH:MM time window."""
        rows = self.index.alerts(min_severity=min_severity, start=start, end=end)
        return self._slim(rows, ("ts", "frame_id", "severity_name", "rule_id", "message", "zone",
                                 "recommendation", "related_rules"), limit=50)

    def object_inventory(self) -> dict[str, Any]:
        """Distinct objects/entities seen in the video, with counts and first/last sighting."""
        ents = sorted(self.p.tracker.entities.values(), key=lambda e: e.first_seen)
        return {
            "detections_by_type": self.index.stats()["by_object_type"],
            "entities": [{"label": e.label, "type": e.object_type, "plate": e.plate,
                          "first_seen": e.first_seen.strftime("%H:%M"), "last_seen": e.last_seen.strftime("%H:%M"),
                          "sightings": e.sightings, "entries": [t.strftime("%H:%M") for t in e.entries],
                          "zones": [pretty_zone(z) for z in e.zones]} for e in ents],
        }

    def entity_history(self, name: str) -> list[dict[str, Any]]:
        """Timeline for an object by label or plate, e.g. 'blue ford f150' or 'MH14CD5678'."""
        q = name.lower()
        tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t not in ("the", "a", "an")]
        matches = [e for e in self.p.tracker.entities.values()
                   if q.replace(" ", "") == e.plate.lower() or all(t in e.label.lower() or t in e.entity_id.lower()
                                                                 for t in tokens)]
        out = []
        for e in matches:
            rows = self.index.entity_timeline(e.entity_id)
            out.append({"label": e.label, "entity_id": e.entity_id, "plate": e.plate,
                        "entries": [t.strftime("%H:%M") for t in e.entries],
                        "timeline": self._slim(rows, ("ts", "frame_id", "action", "zone", "description"))})
        return out

    def frames_in_window(self, start: str, end: str) -> list[dict[str, Any]]:
        """All frames between two HH:MM times (window may wrap midnight)."""
        return self._slim(self.index.frames_between(start, end),
                          ("ts", "frame_id", "zone", "description", "battery_pct", "altitude_m"), limit=60)

    def video_summary(self) -> dict[str, Any]:
        """One-sentence summary of the whole video plus recommended actions."""
        return {"summary": one_sentence_summary(self.index, self.p.tracker),
                "recommendations": recommendations(self.index, self.p.tracker),
                "stats": self.index.stats()}


# --------------------------------------------------------------------------- #
class OfflineAnalyst:
    """Rule-based intent router - zero cost, deterministic, used when no LLM is configured."""

    name = "offline"

    def __init__(self, pipeline: SecurityPipeline):
        self.tools = AnalystToolkit(pipeline)

    def ask(self, question: str) -> str:
        q = question.lower().strip()
        t = self.tools
        if re.search(r"summar|overview|what happened|recap", q):
            s = t.video_summary()
            return s["summary"]
        if re.search(r"recommend|should (i|we)|advice|suggest|improve", q):
            return "Recommendations:\n" + "\n".join(f"- {r}" for r in t.video_summary()["recommendations"])
        if re.search(r"what (objects|things)|inventory|what did (you|the drone) see|objects (were|in)", q):
            inv = t.object_inventory()
            lines = [f"- {e['label']}" + (f" [{e['plate']}]" if e["plate"] else "") +
                     f": {e['sightings']} sighting(s), {e['first_seen']}-{e['last_seen']}, at "
                     f"{', '.join(e['zones'])}" for e in inv["entities"]]
            types = ", ".join(f"{k}: {v}" for k, v in inv["detections_by_type"].items())
            return f"Objects detected ({types}):\n" + "\n".join(lines)
        if m := re.search(r"(?:how many times|when did|history of|timeline (?:of|for)|track) (?:did |the |a )?"
                          r"(.+?)(?: enter| come| visit| appear| arrive)?(?:\?|$)", q):
            hist = t.entity_history(m.group(1))
            if hist:
                parts = []
                for h in hist:
                    tl = "; ".join(f"{r['ts']} {r['action'] or 'seen'} at {pretty_zone(r['zone'] or '')}"
                                   f" (frame {r['frame_id']})" for r in h["timeline"])
                    parts.append(f"{h['label']}: entered {len(h['entries'])} time(s)"
                                 f"{' at ' + ', '.join(h['entries']) if h['entries'] else ''}. Timeline: {tl}")
                return "\n".join(parts)
        if re.search(r"\balerts?\b|incident|threat|danger|emergenc", q):
            sev = next((s for s in ("critical", "high", "medium", "low") if s in q), "info")
            res = t.search_frames(question) if re.search(r"\d{1,2}:\d{2}|night|zone|gate|door", q) else None
            rows = res["results"] if res and res["mode"] == "alerts" else t.get_alerts(sev)
            if not rows:
                return "No alerts matched."
            return f"{len(rows)} alert(s):\n" + "\n".join(
                f"- {r['ts']} [{r['severity_name']}] {r['message']} (frame {r['frame_id']})" for r in rows)
        res = t.search_frames(question)
        if not res["results"]:
            return f"No matching frames found (filters: {res['filters']})."
        lines = [f"- {r['ts']} frame {r['frame_id']}: {r.get('label') or ''} "
                 f"{r.get('action') or ''} @ {pretty_zone(r.get('zone') or '')} - {r.get('description', '')}"
                 for r in res["results"]]
        return f"{res['count']} match(es) [{res['mode']} search, filters={res['filters']}]:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
class LLMAnalyst:
    """LangChain + Claude tool-using agent with budgeted context and sliding-window memory."""

    name = "llm"

    def __init__(self, pipeline: SecurityPipeline, provider: str = "groq", model: Optional[str] = None,
                 memory_messages: int = 12, context_chars: int = 8000):
        from langchain.agents import create_agent
        from langchain_core.tools import StructuredTool

        self.pipeline = pipeline
        self.toolkit = AnalystToolkit(pipeline)
        self.memory_messages = memory_messages
        self.context_chars = context_chars
        self.history: list[Any] = []
        self.provider = provider
        self.name = f"llm:{provider}"
        if provider == "groq":
            from langchain_groq import ChatGroq

            self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
            llm = ChatGroq(model=self.model, max_tokens=4096, temperature=0)
        elif provider == "claude":
            from langchain_anthropic import ChatAnthropic

            self.model = model or "claude-opus-5-5"
            llm = ChatAnthropic(
                model=self.model,
                max_tokens=4096,
                betas=["server-side-fallback-2026-07-01"],
                model_kwargs={"fallbacks": "default", "output_config": {"effort": "medium"}},
            )
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        self.name = f"llm:{provider} ({self.model})"
        tk = self.toolkit
        tools = [StructuredTool.from_function(fn) for fn in (
            tk.search_frames, tk.get_alerts, tk.object_inventory, tk.entity_history,
            tk.frames_in_window, tk.video_summary)]
        self._create_agent = lambda system: create_agent(llm, tools=tools, system_prompt=system)

    def ask(self, question: str) -> str:
        ctx = self.pipeline.context.build_context(question, self.pipeline.index, self.pipeline.tracker,
                                                  budget_chars=self.context_chars)
        rc = self.pipeline.cfg.rules
        agent = self._create_agent(SYSTEM_PROMPT.format(
            site=self.pipeline.cfg.property_name, context=ctx,
            night=f"{rc.night_start:%H:%M}-{rc.night_end:%H:%M}",
            business=f"{rc.business_start:%H:%M}-{rc.business_end:%H:%M}"))
        messages = self.history[-self.memory_messages:] + [{"role": "user", "content": question}]
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1]
        text = answer.content if isinstance(answer.content, str) else "".join(
            b.get("text", "") for b in answer.content if isinstance(b, dict))
        self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": text}]
        return text


def available_provider() -> Optional[str]:
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    return None


def make_analyst(pipeline: SecurityPipeline, mode: str = "auto"):
    """mode: auto | groq | claude | llm | offline.
    auto -> Groq if GROQ_API_KEY is set, else Claude if ANTHROPIC_API_KEY is set, else offline."""
    if mode == "offline":
        return OfflineAnalyst(pipeline)
    if mode in ("groq", "claude"):
        return LLMAnalyst(pipeline, provider=mode)
    provider = available_provider()
    if provider is None:
        if mode == "llm":
            raise RuntimeError("No LLM key found: set GROQ_API_KEY or ANTHROPIC_API_KEY (e.g. in .env)")
        return OfflineAnalyst(pipeline)
    return LLMAnalyst(pipeline, provider=provider)


def to_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str)
