"""Porodica CA: Life-like ćelijski automat sa šumom procesa (izuzeci stanja).

INIT  = nasumična "supa" s x s (BITS literal) na nasumičnom mestu
STEP  = LUT[18](S*9 + NB8(S))          (pravilo B/S kao tabela)
VIDEO = LUT[6](S*3 + C)                (dve boje)
AUDIO = ISIN((N * (SUMS % 61 + 20)) >> 3) * 100   (ton zavisi od populacije)
Šum: svaka ćelija u svakom prelazu se invertuje sa verovatnoćom p.
"""
from __future__ import annotations

import math

import numpy as np

from kfb import kfl
from kfb.kfl import C, N, S, SUMS, X, Y, bits, lut, op

NAME = "ca"
CONWAY = (1 << 3) | (1 << (9 + 2)) | (1 << (9 + 3))
LEVELS = {
    "conway_p0": {"rule": "conway", "p": 0.0},
    "conway_p1e-3": {"rule": "conway", "p": 1e-3},
    "random_rule_p1e-4": {"rule": "random", "p": 1e-4},
}


def program_for(rule, colors, soup, y0, x0, s, exc_positions):
    region = op("AND", op("AND", op("LE", y0, Y), op("LT", Y, y0 + s)),
                op("AND", op("LE", x0, X), op("LT", X, x0 + s)))
    init = op("WHERE", region, bits(soup, (Y - y0) * s + (X - x0)), 0)
    table = [(rule >> k) & 1 for k in range(18)]
    step = lut(table, S * 9 + op("NB8", S))
    video = lut(colors, S * 3 + C)
    audio = op("ISIN", (N * (op("MOD", SUMS, 61) + 20)) >> 3) * 100
    exc = kfl.Exceptions(np.asarray(exc_positions, np.int64)) if len(exc_positions) else None
    return kfl.Program(video=video, init=init, step=step, audio=audio, exc_state=exc)


def generate(rng, meta, level):
    H, W, T = meta.height, meta.width, meta.frames
    s = max(4, min(H, W) // 3)
    y0 = int(rng.integers(0, H - s + 1))
    x0 = int(rng.integers(0, W - s + 1))
    soup = rng.integers(0, 2, s * s).astype(np.uint8)
    if level["rule"] == "conway":
        rule = CONWAY
    else:
        rule = int(rng.integers(0, 1 << 18)) & ~1      # bez B0 (inače treperi)
    colors = [int(v) for v in rng.integers(0, 256, 6)]
    p = float(level["p"])
    Ncells = max(0, T - 1) * H * W
    K = int(rng.binomial(Ncells, p)) if p > 0 else 0
    pos = np.sort(rng.choice(Ncells, size=K, replace=False)) if K else np.zeros(0, np.int64)
    prog = program_for(rule, colors, soup, y0, x0, s, pos)
    ideal = s * s + (K * -math.log2(p) + (Ncells - K) * -math.log2(1 - p) if p > 0 else 0.0)
    return prog, {"ideal_bits": ideal, "params": {"rule": rule, "s": s, "y0": y0, "x0": x0,
                                                   "exceptions": K, "p": p}}
