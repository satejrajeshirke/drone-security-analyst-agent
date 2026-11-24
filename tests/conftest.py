from datetime import datetime
from pathlib import Path

import pytest

try:  # live-API tests pick up GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY from .env
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

from drone_security_agent.config import load_config
from drone_security_agent.indexer import FrameIndex
from drone_security_agent.models import Frame, Telemetry
from drone_security_agent.pipeline import SecurityPipeline
from drone_security_agent.simulator import scripted_day

DAY = datetime(2026, 11, 23)


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture()
def pipeline(cfg):
    return SecurityPipeline(cfg, FrameIndex(":memory:"))


@pytest.fixture(scope="session")
def day_pipeline():
    """The full scripted day, processed once and shared (read-only) by tests."""
    p = SecurityPipeline(load_config(), FrameIndex(":memory:"))
    p.run(scripted_day())
    return p


def make_frame(fid: int, hhmm: str, description: str, zone: str = "main_gate", **telemetry) -> Frame:
    h, m = map(int, hhmm.split(":")[:2])
    s = int(hhmm.split(":")[2]) if hhmm.count(":") == 2 else 0
    ts = DAY.replace(hour=h, minute=m, second=s)
    cfg = load_config()
    z = cfg.zones[zone]
    t = Telemetry(ts=ts, lat=z.lat, lon=z.lon, altitude_m=telemetry.pop("altitude_m", 30.0), heading_deg=0,
                  speed_mps=0, battery_pct=telemetry.pop("battery_pct", 80.0), zone=zone, **telemetry)
    return Frame(frame_id=fid, ts=ts, description=description, telemetry=t)
