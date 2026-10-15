"""Porodica PRNG: "nasumičan" film iz HASH generatora VM-a sa seed-om od k bita.

VIDEO = HASH(idx XOR (seed << 32)) & 255,  idx = ((T*HEI + Y)*WID + X)*4 + C
AUDIO = HASH(N XOR (seed << 32) XOR 2^62) >> 8
Statistički izgleda kao šum; K̄ ≈ 25 B. Pronalaženje seed-a je pretraga 2^k.
"""
from __future__ import annotations

from kfb import kfl
from kfb.kfl import C, HEI, N, T, WID, X, Y, op

NAME = "prng"
LEVELS = {"seed16": {"bits": 16}, "seed24": {"bits": 24}, "seed64": {"bits": 63}}


def program_for(seed: int) -> kfl.Program:
    key = seed << 32 if seed < (1 << 31) else seed
    idx = ((T * HEI + Y) * WID + X) * 4 + C
    video = op("HASH", op("XOR", idx, key))
    audio = op("SHR", op("HASH", op("XOR", N, key ^ (1 << 62))), 8)
    return kfl.Program(video=video, audio=audio)


def generate(rng, meta, level):
    b = level["bits"]
    seed = int(rng.integers(0, 1 << b)) if b < 63 else int(rng.integers(1 << 31, 1 << 62))
    return program_for(seed), {"ideal_bits": float(b), "params": {"seed_bits": b}}
