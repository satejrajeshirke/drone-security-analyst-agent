"""Download a small set of openly licensed real-world images from Wikimedia Commons
to test the VLM stage on real pixels (not simulated captions).

Each scenario has a search query plus the zone/time the image is "replayed" at,
so the downstream rules (night, restricted zone ...) apply realistically.

    python scripts/fetch_sample_images.py          -> data/real_frames/*.jpg + manifest.json + ATTRIBUTION.md
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "real_frames"
UA = "drone-security-agent-prototype/0.1 (https://github.com/; educational research)"
OK_LICENSES = re.compile(r"^(CC0|Public domain|No restrictions|CC BY(-SA)? [0-9.]+)$", re.I)

SCENARIOS = [
    # (file stem, [search queries, tried in order], zone, HH:MM)
    ("01_parking_lot_aerial", ["aerial view parking lot cars"], "parking_lot", "13:20"),
    ("02_pickup_truck", ["pickup truck driveway"], "driveway", "07:31"),
    ("03_person_at_gate", ["security guard standing at gate", "man standing in front of gate",
                           "person waiting at gate"], "main_gate", "00:05"),
    ("04_fence_climb", ["person climbing fence"], "perimeter_fence", "02:20"),
    ("05_loading_dock", ["truck at loading dock"], "loading_dock", "08:10"),
    ("06_night_people", ["man with flashlight at night", "people with flashlight at night",
                         "person walking at night parking lot"], "warehouse", "23:15"),
    ("07_package_doorstep", ["package on porch", "parcel at front door", "delivery box doorstep"],
     "front_door", "09:15"),
    ("08_dog_in_yard", ["dog running in backyard"], "backyard", "14:35"),
]


def _get(url: str, params: dict | None = None, retries: int = 5) -> bytes:
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            wait = int(e.headers.get("Retry-After") or 0) or 5 * 2 ** attempt
            print(f"  rate limited, waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def search(query: str, limit: int = 10) -> list[dict]:
    data = json.loads(_get("https://commons.wikimedia.org/w/api.php", {
        "action": "query", "format": "json", "generator": "search", "gsrnamespace": 6,
        "gsrsearch": f"{query} filetype:bitmap", "gsrlimit": limit,
        "prop": "imageinfo", "iiprop": "url|extmetadata|mime", "iiurlwidth": 1024,
    }))
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda p: p.get("index", 0))
    out = []
    for p in pages:
        ii = p["imageinfo"][0]
        meta = ii.get("extmetadata", {})
        lic = meta.get("LicenseShortName", {}).get("value", "")
        if ii.get("mime") not in ("image/jpeg", "image/png") or not OK_LICENSES.match(lic.strip()):
            continue
        artist = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", "unknown")).strip()
        out.append({"title": p["title"], "license": lic, "artist": artist,
                    "page": ii.get("descriptionurl"), "url": ii.get("thumburl") or ii["url"]})
    return out


def main() -> None:
    """Idempotent: images already on disk (and in the manifest) are kept; delete a file to re-fetch it."""
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    old = {m["file"]: m for m in json.loads(mpath.read_text(encoding="utf-8"))} if mpath.exists() else {}
    manifest = []
    for stem, queries, zone, hhmm in SCENARIOS:
        path = OUT / f"{stem}.jpg"
        if path.exists() and path.name in old:
            manifest.append(old[path.name])
            continue
        hit, used = None, None
        for q in queries:
            hits = search(q)
            time.sleep(2)
            if hits:
                hit, used = hits[0], q
                break
        if hit is None:
            print(f"  no licensed result for {stem} ({queries})", file=sys.stderr)
            continue
        path.write_bytes(_get(hit["url"]))
        manifest.append({"file": path.name, "zone": zone, "time": hhmm, "query": used, "source": hit["page"],
                         "title": hit["title"], "artist": hit["artist"], "license": hit["license"]})
        print(f"  {path.name}  <- {hit['title']} ({hit['license']})")
        time.sleep(4)
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Image attribution", "", "Real-world test images from Wikimedia Commons, used under their licences.", ""]
    for m in manifest:
        lines.append(f"- **{m['file']}**: [{m.get('title', m['source'])}]({m['source']}), "
                     f"{m.get('artist', 'see source page')}, {m.get('license', 'see source page')}")
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(manifest)} images in {OUT}")


if __name__ == "__main__":
    main()
