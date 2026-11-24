"""Command-line interface.

    python -m drone_security_agent run                  # stream the simulated day, print logs + alerts
    python -m drone_security_agent query "show all truck events"
    python -m drone_security_agent ask "What objects were in the video?"
    python -m drone_security_agent chat                 # interactive follow-up questions
    python -m drone_security_agent summary
    python -m drone_security_agent bench --frames 50000 # scalability test
    python -m drone_security_agent video clip.mp4 --describer claude
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .agent import make_analyst, to_json
from .config import load_config
from .indexer import FrameIndex
from .insights import one_sentence_summary, recommendations
from .models import Severity, hhmm, pretty_zone
from .pipeline import FrameResult, SecurityPipeline
from .simulator import load_frames, random_frames, save_frames, scripted_day

console = Console(highlight=False)
SEV_STYLE = {"INFO": "cyan", "LOW": "green", "MEDIUM": "yellow", "HIGH": "bold red", "CRITICAL": "bold white on red"}
DEFAULT_DB = "out/index.db"


def _frames(args):
    if getattr(args, "input", None):
        return load_frames(args.input)
    return scripted_day(load_config(args.config))


def _build(args, db: str = ":memory:", verbose: bool = False, delay: float = 0.0) -> SecurityPipeline:
    cfg = load_config(args.config)
    index = FrameIndex(db)
    index.reset()
    p = SecurityPipeline(cfg, index)
    p.run(_frames(args), on_result=_print_result if verbose else None, delay_s=delay)
    return p


def _print_result(r: FrameResult) -> None:
    t = r.frame.telemetry
    console.print(f"[dim]Time: {hhmm(r.frame.ts)}, Location: {pretty_zone(t.zone).title()}, "
                  f"Alt {t.altitude_m:.0f} m, Battery {t.battery_pct:.0f}%, Mode {t.flight_mode}[/dim]")
    console.print(f"  Frame {r.frame.frame_id}: {escape(r.frame.description)}")
    for log in r.logs:
        console.print(f"  [green]LOG[/green]   {escape(log.message)}")
    for a in r.alerts:
        style = SEV_STYLE[a.severity.name]
        extra = f" (+{', '.join(a.related_rules)})" if a.related_rules else ""
        console.print(f"  [{style}]ALERT {a.severity.name}[/{style}] {escape(a.message + extra)}")
        if a.recommendation:
            console.print(f"        [italic]-> {a.recommendation}[/italic]")


# --------------------------------------------------------------------------- #
def cmd_simulate(args):
    frames = scripted_day(load_config(args.config))
    save_frames(frames, args.out)
    console.print(f"Wrote {len(frames)} frames to {args.out}")


def cmd_run(args):
    console.rule("[bold]Drone Security Analyst Agent - live feed")
    p = _build(args, db=args.db, verbose=not args.quiet, delay=args.delay)
    console.rule("[bold]End of feed")
    _print_summary(p)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({
            "summary": one_sentence_summary(p.index, p.tracker),
            "recommendations": recommendations(p.index, p.tracker),
            "stats": p.index.stats(),
            "alerts": [a.to_dict() for r in p.results for a in r.alerts],
            "logs": [{"ts": l.ts.isoformat(), "frame_id": l.frame_id, "message": l.message}
                     for r in p.results for l in r.logs],
            "entities": [e.to_dict() for e in p.tracker.entities.values()],
        }, indent=2), encoding="utf-8")
        console.print(f"\nJSON report written to {args.report}; index at {args.db}")


def _print_summary(p: SecurityPipeline) -> None:
    st = p.index.stats()
    tbl = Table(title="Alerts raised", show_lines=False)
    for col in ("Time", "Severity", "Rule", "Message"):
        tbl.add_column(col)
    for r in p.index.alerts():
        tbl.add_row(r["ts"][11:16], f"[{SEV_STYLE[r['severity_name']]}]{r['severity_name']}[/]",
                    r["rule_id"], r["message"])
    console.print(tbl)
    console.print(Panel(one_sentence_summary(p.index, p.tracker), title="Video summary (1 sentence)"))
    console.print(Panel("\n".join(f"- {r}" for r in recommendations(p.index, p.tracker)),
                        title="Agent recommendations"))
    console.print(f"[dim]Indexed {st['frames']} frames, {st['detections']} detections, "
                  f"{st['distinct_entities']} entities, {st['alerts']} alerts.[/dim]")


def cmd_query(args):
    db = args.db if Path(args.db).exists() and not args.rebuild else None
    if db:
        index = FrameIndex(db)
    else:
        index = _build(args).index
    res = index.natural_query(args.text, limit=args.limit)
    console.print(f"[bold]{len(res['results'])} result(s)[/bold] - mode={res['mode']} filters=" + escape(str(res['filters'])))
    tbl = Table()
    if res["mode"] == "alerts":
        cols = ("ts", "frame_id", "severity_name", "message")
    elif res["mode"] == "structured":
        cols = ("ts", "frame_id", "label", "action", "zone", "description")
    else:
        cols = ("ts", "frame_id", "zone", "description")
    for c in cols:
        tbl.add_column(c)
    for r in res["results"]:
        tbl.add_row(*[escape(str(r.get(c))[11:16] if c == "ts" else str(r.get(c) or "")) for c in cols])
    console.print(tbl)


def cmd_ask(args):
    p = _build(args)
    analyst = make_analyst(p, args.mode)
    console.print(f"[dim]analyst={analyst.name}[/dim]")
    console.print(f"[bold]Q:[/bold] {args.question}")
    console.print("[bold]A:[/bold] " + escape(analyst.ask(args.question)))


def cmd_chat(args):
    p = _build(args)
    analyst = make_analyst(p, args.mode)
    console.print(Panel(f"Analyst ready ({analyst.name}). {p.index.stats()['frames']} frames indexed. "
                        "Ask anything - 'exit' to quit.", title="Drone Security Analyst"))
    while True:
        try:
            q = console.input("[bold cyan]you> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if q:
            console.print(f"[bold green]agent>[/bold green] {analyst.ask(q)}\n")


def cmd_summary(args):
    p = _build(args)
    console.print(one_sentence_summary(p.index, p.tracker))
    if args.verbose:
        console.print(p.context.build_context(index=p.index, tracker=p.tracker))


def cmd_bench(args):
    """Scalability test: ingest N synthetic frames, then time representative queries."""
    db = args.db if args.db != ":memory:" else ":memory:"
    if db != ":memory:" and Path(db).exists():
        Path(db).unlink()
    index = FrameIndex(db, commit_every=2000)
    p = SecurityPipeline(load_config(args.config), index)
    p.keep_results = False
    lat = []
    t0 = time.perf_counter()
    for f in random_frames(args.frames):
        lat.append(p.process(f).latency_ms)
    index.commit()
    ingest_s = time.perf_counter() - t0
    st = index.stats()
    console.print(Panel(
        f"Frames: {args.frames:,}  |  detections: {st['detections']:,}  |  alerts: {st['alerts']:,}  |  "
        f"entities: {st['distinct_entities']:,}\n"
        f"Ingest: {ingest_s:.2f}s -> {args.frames / ingest_s:,.0f} frames/s  |  per-frame latency "
        f"p50 {statistics.median(lat):.3f} ms, p99 {sorted(lat)[int(len(lat) * .99) - 1]:.3f} ms",
        title="Ingest (describe->detect->track->rules->index)"))
    tbl = Table(title="Query latency")
    tbl.add_column("Query")
    tbl.add_column("Rows", justify="right")
    tbl.add_column("ms", justify="right")
    queries = [
        ("natural: show all truck events", lambda: index.natural_query("show all truck events", limit=10_000)["results"]),
        ("natural: people at the main gate at night",
         lambda: index.natural_query("people at the main gate at night", limit=10_000)["results"]),
        ("structured: vehicles 07:00-09:00", lambda: index.detections(object_type="vehicle", start="07:00",
                                                                        end="09:00", limit=100_000)),
        ("full-text: 'blue ford loitering'", lambda: index.search_text("blue ford loitering", limit=100)),
        ("alerts >= HIGH", lambda: index.alerts(min_severity="HIGH", limit=100_000)),
        ("stats()", lambda: [index.stats()]),
    ]
    for name, fn in queries:
        reps = []
        for _ in range(5):
            t = time.perf_counter()
            rows = fn()
            reps.append((time.perf_counter() - t) * 1000)
        tbl.add_row(name, f"{len(rows):,}", f"{statistics.median(reps):.2f}")
    console.print(tbl)
    ctx = p.context.build_context("truck at gate", index, p.tracker)
    console.print(f"[dim]LLM context snapshot stays bounded: {len(ctx):,} chars for {args.frames:,} frames.[/dim]")


def cmd_video(args):
    from .vlm import frames_from_video, get_describer

    cfg = load_config(args.config)
    index = FrameIndex(args.db)
    index.reset()
    p = SecurityPipeline(cfg, index, describer=get_describer(args.describer))
    start = datetime.fromisoformat(args.start) if args.start else datetime.now().replace(microsecond=0)
    p.run(frames_from_video(args.path, args.frames_dir, start, zone=args.zone, every_s=args.every),
          on_result=_print_result)
    _print_summary(p)


def cmd_images(args):
    """Real images -> VLM caption -> detection -> tracking -> rules -> index, plus a Markdown results table."""
    from .vlm import CachedDescriber, frames_from_images, get_describer

    cfg = load_config(args.config)
    index = FrameIndex(args.db)
    index.reset()
    cache_path = Path("out/vlm_cache.json")
    if args.fresh and cache_path.exists():
        cache_path.unlink()
    describer = CachedDescriber(get_describer(args.describer), cache_path)
    p = SecurityPipeline(cfg, index, describer=describer)
    rows = []

    def on_result(r: FrameResult) -> None:
        _print_result(r)
        model = getattr(describer, "last_model", None) or getattr(describer, "name", args.describer)
        console.print(f"        [dim]VLM: {model}, {r.latency_ms / 1000:.1f}s[/dim]")
        rows.append((r, model))

    p.run(frames_from_images(args.folder), on_result=on_result)
    _print_summary(p)
    checks = getattr(describer, "verifications", None)
    stats = (f"VLM self-check: {checks} 'no activity' caption(s) re-checked, "
             f"{getattr(describer, 'corrections', 0)} corrected; VLM errors: {p.vlm_errors}") if checks is not None \
        else f"VLM errors: {p.vlm_errors}"
    if getattr(describer, "hits", 0):
        stats += f"; {describer.hits} caption(s) replayed from cache (out/vlm_cache.json; use --fresh to re-caption)"
    console.print(f"[dim]{stats}[/dim]")
    if args.out:
        folder = Path(args.folder)
        out = Path(args.out)
        rel = Path("..") / folder if not folder.is_absolute() else folder
        lines = [f"# Real-image VLM results ({args.describer})", "",
                 f"Command: `python -m drone_security_agent images {args.folder} --describer {args.describer}`", "",
                 "| Image | Time / zone | VLM caption | Detections | Alerts |", "|---|---|---|---|---|"]
        for r, model in rows:
            img = Path(r.frame.image_path).name
            dets = "; ".join(f"{d.label} ({d.action or 'seen'})" for d in r.detections) or "none"
            alerts = "; ".join(f"**{a.severity.name}** {a.rule_id}" for a in r.alerts) or "none"
            cap = r.frame.description.replace("|", "/").replace("\n", " ")
            lines.append(f"| [{img}]({(rel / img).as_posix()}) | {hhmm(r.frame.ts)} {pretty_zone(r.frame.telemetry.zone)}"
                         f" | {cap} <br><sub>{model}</sub> | {dets} | {alerts} |")
        lines += ["", stats + ".", "", "Image sources and licences: "
                  f"[ATTRIBUTION.md]({(rel / 'ATTRIBUTION.md').as_posix()})"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"\nResults table written to {out}")


def main(argv=None):
    try:
        from dotenv import load_dotenv

        load_dotenv()  # picks up GROQ_API_KEY / ANTHROPIC_API_KEY from .env
    except ImportError:
        pass
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="drone_security_agent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: config/default.yaml)")
    ap.add_argument("--input", default=None, help="frames JSON (default: built-in scripted day)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="write the scripted day to JSON")
    s.add_argument("--out", default="data/scenario_day.json")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("run", help="process the feed, print logs/alerts, build the index")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--delay", type=float, default=0.0, help="seconds between frames (demo pacing)")
    s.add_argument("--quiet", action="store_true")
    s.add_argument("--report", default="out/report.json")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("query", help="query the frame index in plain English")
    s.add_argument("text")
    s.add_argument("--db", default=DEFAULT_DB)
    s.add_argument("--limit", type=int, default=50)
    s.add_argument("--rebuild", action="store_true")
    s.set_defaults(fn=cmd_query)

    for name, fn in (("ask", cmd_ask), ("chat", cmd_chat)):
        s = sub.add_parser(name, help="ask the analyst agent" if name == "ask" else "interactive Q&A")
        if name == "ask":
            s.add_argument("question")
        s.add_argument("--mode", choices=["auto", "groq", "claude", "llm", "offline"], default="auto")
        s.set_defaults(fn=fn)

    s = sub.add_parser("summary", help="one-sentence video summary")
    s.add_argument("-v", "--verbose", action="store_true", help="also print the agent context snapshot")
    s.set_defaults(fn=cmd_summary)

    s = sub.add_parser("bench", help="scalability benchmark")
    s.add_argument("--frames", type=int, default=20_000)
    s.add_argument("--db", default="out/bench.db")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("video", help="process a real video file through a VLM")
    s.add_argument("path")
    s.add_argument("--describer", default="gemini", choices=["gemini", "claude", "blip"])
    s.add_argument("--every", type=float, default=2.0, help="sample one frame every N seconds")
    s.add_argument("--zone", default="main_gate")
    s.add_argument("--start", default=None, help="ISO timestamp of the first frame")
    s.add_argument("--frames-dir", default="out/video_frames")
    s.add_argument("--db", default="out/video_index.db")
    s.set_defaults(fn=cmd_video)

    s = sub.add_parser("images", help="process a folder of real images through a VLM + the full pipeline")
    s.add_argument("folder", nargs="?", default="data/real_frames")
    s.add_argument("--describer", default="gemini", choices=["gemini", "claude", "blip"])
    s.add_argument("--db", default="out/images_index.db")
    s.add_argument("--out", default="docs/vlm_results.md", help="Markdown results table ('' to skip)")
    s.add_argument("--fresh", action="store_true", help="ignore cached captions and call the VLM again")
    s.set_defaults(fn=cmd_images)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
