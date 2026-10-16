"""Zapisnik zamrzavanja (korak 1 protokola): python3 -m kfb.tools.freeze OUT.json

Beleži MODEL_ID mašine, SHA-256 svih enkodera i porodica i git commit. Poziva se
POSLE commit-a enkodera, a PRE pravljenja master seed-a; seed se zatim pravi
tek kada je ovaj zapis napisan i objavljen.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from kfb import record  # noqa: E402


def tree_hash(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(ROOT)).encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def main() -> None:
    out = Path(sys.argv[1])
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    dirty = git("status", "--porcelain", "--", "kfb", "igc")
    rec = {
        "model_id": record.model_id(),
        "encoders_sha256": tree_hash(list((ROOT / "kfb" / "encoders").glob("*.py"))),
        "families_sha256": tree_hash(list((ROOT / "kfb" / "families").glob("*.py"))),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty_kfb_igc": bool(dirty),
        "ffmpeg": record.ffmpeg_version().splitlines()[0],
    }
    if rec["git_dirty_kfb_igc"]:
        raise SystemExit("kfb/ ili igc/ imaju nekomitovane izmene — prvo commit (zamrzavanje)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
