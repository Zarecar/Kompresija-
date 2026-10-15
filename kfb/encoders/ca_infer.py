"""MDL indukcija pravila + izuzeci za CA porodicu (port igc.ca u KFL).

Family-aware: pretpostavlja dvobojni Life-like CA i audio konvenciju porodice;
NE zna pravilo, boje, supu, šum ni p. Zvuk: probaju se hipoteze (formula
porodice, tišina, sirovi WORDS literal) i bira najkraća.
"""
from __future__ import annotations

import numpy as np

from kfb import kfl, record
from kfb.kfl import C, N, S, SUMS, X, Y, bits, lut, op, words_lit

from . import Encoder


def _neighbors(F):
    n = np.zeros(F.shape, dtype=np.int64)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                n += np.roll(np.roll(F, dy, axis=-2), dx, axis=-1)
    return n


def encode(meta, payload, budget_s):
    T, H, W = meta.frames, meta.height, meta.width
    if T < 2 or H < 1 or W < 1:
        return None
    video = np.frombuffer(payload[:meta.video_bytes], dtype=np.uint8).reshape(T, H, W, 3)
    audio = np.frombuffer(payload[meta.video_bytes:], dtype="<i2")
    packed = (video[..., 0].astype(np.int64) << 16) | (video[..., 1].astype(np.int64) << 8) | video[..., 2]
    cols = np.unique(packed)
    if len(cols) > 2:
        return None
    cols = [int(c) for c in cols] + ([int(cols[0]) ^ 1] if len(cols) == 1 else [])
    best = None
    for alive, dead in ((cols[0], cols[1]), (cols[1], cols[0])):
        F = (packed == alive).astype(np.int64)
        idx = F[:-1] * 9 + _neighbors(F[:-1])
        nxt = F[1:]
        ones = np.bincount(idx[nxt == 1], minlength=18)
        tot = np.bincount(idx.ravel(), minlength=18)
        table = [1 if 2 * ones[k] > tot[k] else 0 for k in range(18)]
        E = nxt ^ np.array(table, dtype=np.int64)[idx]
        ys, xs = np.nonzero(F[0])
        if len(ys):
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
            h, w = y1 - y0, x1 - x0
            region = op("AND", op("AND", op("LE", y0, Y), op("LT", Y, y1)),
                        op("AND", op("LE", x0, X), op("LT", X, x1)))
            init = op("WHERE", region, bits(F[0, y0:y1, x0:x1].reshape(-1), (Y - y0) * w + (X - x0)), 0)
        else:
            init = kfl.const(0)
        step = lut(table, S * 9 + op("NB8", S))
        a_rgb = [(alive >> 16) & 255, (alive >> 8) & 255, alive & 255]
        d_rgb = [(dead >> 16) & 255, (dead >> 8) & 255, dead & 255]
        video_e = lut(d_rgb + a_rgb, S * 3 + C)
        pos = np.flatnonzero(E.reshape(-1))
        exc = kfl.Exceptions(pos.astype(np.int64)) if len(pos) else None
        hyps = [None] if not len(audio) or not audio.any() else []
        hyps.append(op("ISIN", (N * (op("MOD", SUMS, 61) + 20)) >> 3) * 100)
        if len(audio):
            hyps.append(op("SUB", words_lit(audio.view("<u2"), N), 0))
        for a_expr in hyps:
            prog = kfl.Program(video=video_e, init=init, step=step, audio=a_expr, exc_state=exc)
            rec = record.encode_kfl(meta, prog)
            if best is not None and len(rec) >= len(best):
                continue
            if prog.run(meta) == payload:
                best = rec
    return best


ENCODERS = [Encoder("ca_infer", encode, family_aware=True,
                    note="MDL: pravilo (većinsko glasanje) + supa (BITS) + izuzeci (Rice)")]
