"""Pravi izolovan "sandbox" za spoljne (LLM) enkodere.

    python3 -m kfb.tools.make_sandbox --suite DIR --iids a,b,c --out SANDBOX

Sandbox sadrži SAMO: KFL VM (kfl.py), format filma (igc/film.py), specifikaciju,
kanonske filmove izabranih instanci i check.py. NE sadrži generatore porodica,
enkodere, ground truth ni rezultate. Agent radi isključivo u sandbox-u.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECK = '''"""Provera kandidata: python3 check.py films/<iid>.film kandidat.rec

Ispisuje veličinu zapisa i da li se dekoduje TAČNO u film (i samoograničavanje).
Zapis = bytes([0x10]) + meta.to_bytes() + program.to_bytes()  (vidi KFL_SPEC.md).
"""
import hashlib, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from igc.film import Meta, parse_film
from kfb import kfl

def decode(rec):
    if rec[0] != 0x10:
        raise ValueError("sandbox prihvata samo KFL zapise (metoda 0x10)")
    meta, pos = Meta.from_bytes(rec, 1)
    prog, pos = kfl.Program.from_bytes(rec, pos, meta)
    return meta, kfl.run(prog, meta), pos

if __name__ == "__main__":
    film = open(sys.argv[1], "rb").read()
    rec = open(sys.argv[2], "rb").read()
    meta0, payload0 = parse_film(film)
    meta, payload, used = decode(rec + os.urandom(16))
    exact = meta == meta0 and payload == payload0
    diff = None
    if meta == meta0 and len(payload) == len(payload0) and not exact:
        diff = sum(a != b for a, b in zip(payload, payload0))
    print(f"size={len(rec)} exact={exact} self_delimiting={used == len(rec)} differing_bytes={diff}")
'''


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--iids", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "kfb").mkdir(parents=True, exist_ok=True)
    (out / "igc").mkdir(parents=True, exist_ok=True)
    (out / "films").mkdir(exist_ok=True)
    (out / "out").mkdir(exist_ok=True)
    shutil.copy(ROOT / "kfb" / "kfl.py", out / "kfb" / "kfl.py")
    (out / "kfb" / "__init__.py").write_text("")
    shutil.copy(ROOT / "igc" / "film.py", out / "igc" / "film.py")
    (out / "igc" / "__init__.py").write_text("")
    shutil.copy(ROOT / "kfb" / "KFL_SPEC.md", out / "KFL_SPEC.md")
    (out / "check.py").write_text(CHECK)
    for iid in a.iids.split(","):
        shutil.copy(Path(a.suite) / "films" / f"{iid}.film", out / "films" / f"{iid}.film")
    print(out)


if __name__ == "__main__":
    main()
