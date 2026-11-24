"""Frame describers (VLM layer).

The pipeline only needs `describe(frame) -> str`. Three interchangeable backends:

* `PassthroughDescriber` - simulated frames already carry a caption (default,
  required by the brief: "simulate video frames with text descriptions").
* `ClaudeVisionDescriber` - sends the real image to Claude (claude-opus-5-5)
  with a security-analyst prompt; best caption quality, understands context
  such as "climbing", "headlights off", plates. Needs ANTHROPIC_API_KEY.
* `BlipDescriber` - local Hugging Face BLIP captioning model; offline and
  free, but captions are generic ("a man standing next to a gate").

Plus `extract_video_frames()` (OpenCV) to turn a real .mp4 into frames.
Heavy dependencies (anthropic, transformers, torch, cv2) are imported lazily
so the core prototype installs in seconds.
"""
from __future__ import annotations

import base64
import mimetypes
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional, Protocol

from .models import Frame

SECURITY_CAPTION_PROMPT = (
    "You are the camera analyst on a security drone patrolling a private property. "
    "Describe this frame in ONE sentence of plain text for a security log.\n"
    "Rules:\n"
    "1. Mention EVERY visible person, vehicle, animal and package/box - including parked or stationary "
    "ones. For vehicles give colour, type, make/model and licence plate if readable; for people give "
    "count, clothing and what they are doing (entering, exiting, parked, standing, loitering, climbing, "
    "running, carrying ...).\n"
    "2. Mention anything suspicious: masks, tools, flashlights, headlights off, open gates, fire, smoke.\n"
    "3. Only if there are NO people, vehicles, animals or packages at all, write exactly "
    "'Clear view of the <area>, no activity'.\n"
    "4. The telemetry hint below names the area the drone is over - use it as the location, but describe "
    "only what is actually visible; never invent objects or places.\n"
    "No preamble, no markdown."
)


VERIFY_PROMPT = (
    "Look carefully at the whole image, including the background and small or distant objects. "
    "List every person, vehicle (including parked ones), animal and package/box that is visible, as a short "
    "comma-separated list with counts (e.g. '2 people, 1 white van, about 200 parked cars'). "
    "If there are none at all, answer exactly NONE."
)
NEGATIVE_CAPTION_RE = re.compile(r"\bno activity\b|\bclear view\b|\bnothing (?:is )?(?:visible|happening)\b", re.I)


class FrameDescriber(Protocol):
    def describe(self, frame: Frame) -> str: ...


class PassthroughDescriber:
    """Simulated feed: the description *is* the VLM output."""

    name = "simulated"

    def describe(self, frame: Frame) -> str:
        return frame.description


class ClaudeVisionDescriber:
    name = "claude-vision"

    def __init__(self, model: str = "claude-opus-5-5", max_tokens: int = 300):
        import anthropic  # lazy

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def describe(self, frame: Frame) -> str:
        if not frame.image_path:
            return frame.description
        path = Path(frame.image_path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        data = base64.standard_b64encode(path.read_bytes()).decode()
        hint = f"Drone telemetry: area={frame.telemetry.zone.replace('_', ' ')}, time={frame.ts:%H:%M}."
        resp = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # Server-side refusal fallback: if the primary model declines, the API re-runs on a fallback.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "low"},  # captioning is a simple task
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                {"type": "text", "text": f"{SECURITY_CAPTION_PROMPT}\n{hint}"},
            ]}],
        )
        if resp.stop_reason == "refusal":
            return frame.description or "Frame could not be described."
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return text or frame.description


class GeminiVisionDescriber:
    """Google Gemini vision captioner. Tries a list of Flash models in order, so a temporarily
    overloaded (503) or retired (404) model falls back to the next one instead of failing."""

    name = "gemini-vision"
    DEFAULT_MODELS = ("gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.5-flash",
                      "gemini-3.5-flash-lite", "gemini-3.1-flash-lite")

    def __init__(self, models: Optional[tuple[str, ...]] = None, retries_per_model: int = 2,
                 client=None, timeout_s: int = 30, verify_negatives: bool = True):
        import os

        if client is None:
            from google import genai  # lazy
            from google.genai import types

            # SDK-internal retries disabled: we fail over to the next model instead of waiting minutes.
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=types.HttpOptions(
                timeout=timeout_s * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.client = client
        env_model = os.environ.get("GEMINI_MODEL")
        # GEMINI_MODEL is tried first, but the defaults stay behind it as fallbacks
        self.models = models or (((env_model,) if env_model else ()) +
                                 tuple(m for m in self.DEFAULT_MODELS if m != env_model))
        self.retries_per_model = retries_per_model
        self.verify_negatives = verify_negatives
        self.last_model: Optional[str] = None
        self.verifications = 0  # how many "no activity" captions were re-checked
        self.corrections = 0  # ... and how many of those turned out to be wrong

    def describe(self, frame: Frame) -> str:
        if not frame.image_path:
            return frame.description
        from google.genai import types

        path = Path(frame.image_path)
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image = types.Part.from_bytes(data=path.read_bytes(), mime_type=media_type)
        hint = f"Drone telemetry: area={frame.telemetry.zone.replace('_', ' ')}, time={frame.ts:%H:%M}."
        caption = self._generate([image, f"{SECURITY_CAPTION_PROMPT}\n{hint}"])
        if self.verify_negatives and NEGATIVE_CAPTION_RE.search(caption):
            # Small/fast VLMs often take the "nothing here" shortcut. Re-check with a narrow
            # enumeration question; only if it finds objects, regenerate the caption with them.
            self.verifications += 1
            found = self._generate([image, VERIFY_PROMPT]).strip()
            if found and not found.upper().startswith("NONE"):
                self.corrections += 1
                caption = self._generate([image, f"{SECURITY_CAPTION_PROMPT}\n{hint}\n"
                                                  f"A careful check found these in the frame: {found}. "
                                                  f"Your sentence must include them."])
        return caption or frame.description

    def _generate(self, contents: list) -> str:
        import time

        from google.genai import errors, types

        config = types.GenerateContentConfig(temperature=0)
        last_err: Exception | None = None
        # try the model that worked last time first (sticky), then the rest in order
        order = ([self.last_model] if self.last_model else []) + [m for m in self.models if m != self.last_model]
        for model in order:
            for attempt in range(self.retries_per_model):
                try:
                    resp = self.client.models.generate_content(model=model, contents=contents, config=config)
                    self.last_model = model
                    return (resp.text or "").strip()
                except errors.ServerError as e:  # 5xx: overloaded -> retry, then next model
                    last_err = e
                    if attempt + 1 < self.retries_per_model:
                        time.sleep(2 * (attempt + 1))
                except errors.ClientError as e:  # 404 retired model / 429 quota -> next model
                    last_err = e
                    break
        raise RuntimeError(f"All Gemini models failed: {last_err}")


class CachedDescriber:
    """Caches captions on disk, keyed by image content + describer + prompt version.

    Makes demos and re-runs instant and repeatable (VLM output varies between calls), and in
    production avoids paying twice for the same keyframe. Delete the file or use --fresh to re-caption."""

    def __init__(self, inner: FrameDescriber, path: str | Path = "out/vlm_cache.json"):
        import json

        self.inner = inner
        self.name = getattr(inner, "name", type(inner).__name__)
        self.path = Path(path)
        self.cache: dict[str, dict] = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.hits = 0
        self.last_model: Optional[str] = None

    def _key(self, frame: Frame) -> str:
        import hashlib

        h = hashlib.sha256(Path(frame.image_path).read_bytes()).hexdigest()[:16]
        p = hashlib.sha256((SECURITY_CAPTION_PROMPT + VERIFY_PROMPT).encode()).hexdigest()[:8]
        return f"{self.name}:{p}:{h}:{frame.telemetry.zone}:{frame.ts:%H:%M}"

    def describe(self, frame: Frame) -> str:
        import json

        if not frame.image_path:
            return frame.description
        key = self._key(frame)
        if key in self.cache:
            self.hits += 1
            self.last_model = self.cache[key]["model"] + " (cached)"
            return self.cache[key]["caption"]
        caption = self.inner.describe(frame)
        self.last_model = getattr(self.inner, "last_model", None) or self.name
        self.cache[key] = {"caption": caption, "model": self.last_model, "image": Path(frame.image_path).name}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False), encoding="utf-8")
        return caption

    def __getattr__(self, item):  # expose inner stats (verifications, corrections ...)
        return getattr(self.inner, item)


class BlipDescriber:
    name = "blip"

    def __init__(self, model_id: str = "Salesforce/blip-image-captioning-base"):
        from transformers import BlipForConditionalGeneration, BlipProcessor  # lazy

        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForConditionalGeneration.from_pretrained(model_id)

    def describe(self, frame: Frame) -> str:
        if not frame.image_path:
            return frame.description
        from PIL import Image

        image = Image.open(frame.image_path).convert("RGB")
        inputs = self.processor(image, "an aerial security camera photo of", return_tensors="pt")
        out = self.model.generate(**inputs, max_new_tokens=40)
        return self.processor.decode(out[0], skip_special_tokens=True)


def get_describer(name: str = "simulated") -> FrameDescriber:
    name = name.lower()
    if name in ("simulated", "passthrough", "text"):
        return PassthroughDescriber()
    if name in ("claude", "claude-vision"):
        return ClaudeVisionDescriber()
    if name in ("gemini", "gemini-vision"):
        return GeminiVisionDescriber()
    if name == "blip":
        return BlipDescriber()
    raise ValueError(f"Unknown describer: {name}")


def extract_video_frames(video_path: str | Path, out_dir: str | Path, every_s: float = 1.0
                         ) -> Iterator[tuple[float, Path]]:
    """Sample a real video every `every_s` seconds -> (offset_seconds, jpg_path)."""
    import cv2  # lazy (pip install opencv-python-headless)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * every_s)))
    idx = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if idx % step == 0:
            p = out_dir / f"frame_{idx:07d}.jpg"
            cv2.imwrite(str(p), img)
            yield idx / fps, p
        idx += 1
    cap.release()


def frames_from_video(video_path: str | Path, out_dir: str | Path, start: datetime,
                      zone: str = "main_gate", every_s: float = 1.0) -> Iterator[Frame]:
    """Wrap sampled video frames as `Frame`s with placeholder telemetry (real
    deployments would join the drone's telemetry log on timestamp)."""
    from .config import load_config
    from .simulator import TelemetryModel

    tm = TelemetryModel(load_config())
    for i, (offset, path) in enumerate(extract_video_frames(video_path, out_dir, every_s), start=1):
        ts = start + timedelta(seconds=offset)
        yield Frame(frame_id=i, ts=ts, description="", telemetry=tm.sample(ts, zone, hovering=True),
                    image_path=str(path))


def frames_from_images(folder: str | Path, day: Optional[datetime] = None) -> Iterator[Frame]:
    """Real still images as frames. `manifest.json` in the folder (optional) gives each image a zone
    and time-of-day, so time-dependent rules (night, after hours) apply as they would on patrol."""
    import json

    from .config import load_config
    from .simulator import SCENARIO_DATE, TelemetryModel

    folder = Path(folder)
    day = day or SCENARIO_DATE
    mpath = folder / "manifest.json"
    if mpath.exists():
        entries = json.loads(mpath.read_text(encoding="utf-8"))
    else:
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
        entries = [{"file": p.name, "zone": "main_gate", "time": f"{12 + i // 60:02d}:{i % 60:02d}"}
                   for i, p in enumerate(files)]
    tm = TelemetryModel(load_config())
    for i, e in enumerate(sorted(entries, key=lambda e: e["time"]), start=1):
        h, m = map(int, e["time"].split(":"))
        ts = day.replace(hour=h, minute=m)
        yield Frame(frame_id=i, ts=ts, description="", telemetry=tm.sample(ts, e["zone"], hovering=True),
                    image_path=str(folder / e["file"]))


def describer_name(d: Optional[FrameDescriber]) -> str:
    return getattr(d, "name", type(d).__name__) if d else "simulated"
