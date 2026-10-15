"""Pravljenje skupa instanci KFB.

    python3 -m kfb.suite --master-seed-file SEED --out DIR [--reps 2] [--tiers mini,std]

DIR/films/<iid>.film     kanonski film (igc.film); iid je anonimni heš (ne otkriva porodicu)
DIR/manifest.json        lista instanci: iid, tier, N (bez porodice i parametara)
DIR/truth/<iid>.json     GROUND TRUTH: porodica, nivo, K̄, ideal_bits, parametri
DIR/truth/<iid>.rec      oracle zapis (dokaz da je K̄ dostižno)
Vremensko razdvajanje: master seed se pravi i objavljuje TEK POSLE zamrzavanja
enkodera; enkoderi i LLM agenti nikada ne smeju čitati DIR/truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igc.film import Meta, film_bytes, video_meta  # noqa: E402
from igc import codec as igc_codec  # noqa: E402
from kfb import families, kfl, record  # noqa: E402

TIERS = {
    "mini": video_meta(32, 32, 24, 24, 8000, 1),   # 1 s, ~90 KB
    "std": video_meta(96, 96, 96, 24, 8000, 1),    # 4 s, ~2,7 MB
}


def instance_rng(master: bytes, family: str, level: str, tier: str, rep: int) -> np.random.Generator:
    h = hashlib.sha256(master + f"|{family}|{level}|{tier}|{rep}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(h[:16], "little")))


def make(master: bytes, out: Path, reps: int = 2, tiers=("mini", "std"), only: set | None = None) -> list:
    fams = families.load_all()
    (out / "films").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    manifest = []
    mid = record.model_id()
    for fname, fam in fams.items():
        if only and fname not in only:
            continue
        for lname, level in fam.LEVELS.items():
            for tier in tiers:
                meta = TIERS[tier]
                for rep in range(reps):
                    rng = instance_rng(master, fname, lname, tier, rep)
                    obj, info = fam.generate(rng, meta, level)
                    if isinstance(obj, kfl.Program):
                        payload = kfl.run(obj, meta)
                        oracle = record.encode_kfl(meta, obj)
                    else:
                        payload = bytes(obj)
                        oracle = igc_codec.encode("raw", meta, payload)
                    m2, chk, used = record.decode(oracle)
                    if chk != payload or used != len(oracle):
                        raise RuntimeError(f"oracle se ne dekoduje: {fname}/{lname}")
                    film = film_bytes(meta, payload)
                    iid = hashlib.sha256(master + b"iid" + f"{fname}{lname}{tier}{rep}".encode()).hexdigest()[:12]
                    (out / "films" / f"{iid}.film").write_bytes(film)
                    (out / "truth" / f"{iid}.rec").write_bytes(oracle)
                    truth = {"iid": iid, "family": fname, "level": lname, "tier": tier, "rep": rep,
                             "N_bytes": len(film), "payload_bytes": len(payload), "K_bar_bytes": len(oracle),
                             "oracle_kind": "kfl" if isinstance(obj, kfl.Program) else "raw",
                             "film_sha256": hashlib.sha256(film).hexdigest(), "model_id": mid, **info}
                    (out / "truth" / f"{iid}.json").write_text(json.dumps(truth, indent=1, default=float))
                    manifest.append({"iid": iid, "tier": tier, "N_bytes": len(film)})
                    print(f"{iid} {fname:10s} {lname:20s} {tier:4s} N={len(film):8d} K̄={len(oracle)}", flush=True)
    manifest.sort(key=lambda r: r["iid"])
    (out / "manifest.json").write_text(json.dumps({"model_id": mid, "instances": manifest}, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master-seed-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tiers", default="mini,std")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    master = Path(a.master_seed_file).read_bytes()
    make(master, Path(a.out), a.reps, tuple(a.tiers.split(",")), set(filter(None, a.only.split(","))))


if __name__ == "__main__":
    main()
