"""Inverzija HASH generatora porodice PRNG grubom silom (do 2^24 seed-ova).

Family-aware: pretpostavlja konvenciju porodice prng (ključ = seed << 32), ali
ne zna seed. Kandidati se filtriraju po prvim bajtovima videa (vektorski).
"""
from __future__ import annotations

import time

import numpy as np

from kfb import record
from kfb.families.prng import program_for

from . import Encoder

MAX_BITS = 24


def _hash(z: np.ndarray) -> np.ndarray:
    z = z.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(1)).astype(np.int64)


def encode(meta, payload, budget_s):
    if meta.video_bytes < 8:
        return None
    t0 = time.perf_counter()
    target = np.frombuffer(payload[:4], dtype=np.uint8).astype(np.int64)
    chunk = 1 << 20
    with np.errstate(all="ignore"):
        for start in range(0, 1 << MAX_BITS, chunk):
            if time.perf_counter() - t0 > budget_s:
                return None
            s = np.arange(start, start + chunk, dtype=np.int64)
            key = s << 32
            ok = (_hash(0 ^ key) & 255) == target[0]
            cand = s[ok]
            for i in range(1, 4):
                vidx = (i // 3) * 4 + i % 3          # bajt i -> idx piksela (4 kanala po pikselu)
                cand = cand[(_hash(np.int64(vidx) ^ (cand << 32)) & 255) == target[i]]
            for seed in cand.tolist():
                prog = program_for(int(seed))
                if prog.run(meta) == payload:
                    return record.encode_kfl(meta, prog)
    return None


ENCODERS = [Encoder("prng_infer", encode, family_aware=True,
                    note="gruba sila nad seed-om HASH porodice, do 2^24")]
