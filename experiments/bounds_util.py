"""Analitičke donje granice za kontrolne izvore (van modela M, da ne menja MODEL_ID)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from igc import sources  # noqa: E402
from igc.film import Meta  # noqa: E402


def _clipped_entropy_bits(base: np.ndarray, sigma: float, lo: int, hi: int, step: int = 1) -> float:
    ks = np.arange(-int(12 * sigma) - 2, int(12 * sigma) + 3)
    cdf = np.vectorize(lambda v: 0.5 * (1 + math.erf(v / (sigma * math.sqrt(2)))))
    pk = cdf(ks + 0.5) - cdf(ks - 0.5)
    total = 0.0
    vals, counts = np.unique(base, return_counts=True)
    for v, c in zip(vals.tolist(), counts.tolist()):
        _, inv = np.unique(np.clip(v + step * ks, lo, hi), return_inverse=True)
        q = np.bincount(inv, weights=pk)
        q = q[q > 0]
        total += c * float(-(q * np.log2(q)).sum())
    return total


def structured_noisy_entropy_bits(meta: Meta, phase: int, sigma: float = 2.0) -> float:
    """Tačna entropija šuma u `structured_noisy` (sa odsecanjem na opseg).

    To je donja granica očekivane dužine SVAKOG tačnog koda (uz poznatu fazu),
    osim ako koder ne invertuje CSPRNG koji je proizveo šum.
    """
    pl = sources._structured(meta, phase)
    v = np.frombuffer(pl[:meta.video_bytes], dtype=np.uint8).astype(np.int64)
    a = np.frombuffer(pl[meta.video_bytes:], dtype="<i2").astype(np.int64)
    return (_clipped_entropy_bits(v, sigma, 0, 255)
            + _clipped_entropy_bits(a, sigma, -32768, 32767, step=64))
