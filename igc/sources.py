"""Izvori testnih filmova.

Ključno razlikovanje (zahtev 7): "nasumičan" nije jedno svojstvo.
  * csprng          - os.urandom/secrets: fizička + kriptografska entropija.
  * shake_keyed     - SHAKE-256 sa tajnim 256-bitnim ključem (CSPRNG, ponovljiv
                      samo uz ključ; ključ se odbacuje i enkoder ga ne zna).
  * np_pcg64        - NumPy default_rng (PCG64): nije kriptografski, ali ga naš
                      enkoder ne invertuje (poznati napad košta ~1e4 CPU-sati).
  * np_legacy       - NumPy RandomState(seed32) -> MT19937, 32-bitni seed.
  * py_random       - Python random.Random(seed32) -> MT19937.
  * py_random_os    - Python random.Random() bez seed-a (seed = 2496 B iz OS-a).
  * shake_weak20    - SHAKE-256 iz poznate porodice sa 20-bitnim seed-om.
  * structured      - proceduralni video + sinusni zvuk (pozitivna kontrola).
  * structured_noisy- isto + Gausov šum sigma=2 iz CSPRNG-a ("šum senzora");
                      ima izračunljivu donju granicu entropije.
Seed/ključ se NIKAD ne prosleđuje enkoderu; čuva se samo radi izveštaja.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import secrets

import numpy as np

from .film import Meta

WEAK_LABEL = b"IGC-weak-shake-v1"
WEAK_BITS = 20


def shake_stream(key: bytes, n: int) -> bytes:
    return hashlib.shake_256(key).digest(n)


def weak_shake(seed: int, n: int) -> bytes:
    return hashlib.shake_256(WEAK_LABEL + seed.to_bytes(4, "little")).digest(n)


def _structured(meta: Meta, phase: int, noise_sigma: float = 0.0) -> bytes:
    W, H, F = meta.width, meta.height, meta.frames
    t = np.arange(F).reshape(F, 1, 1)
    y = np.arange(H).reshape(1, H, 1)
    x = np.arange(W).reshape(1, 1, W)
    r = (x * 4 + t * 3 + phase) & 255
    g = (y * 4 + t + 2 * phase) & 255
    b = ((x * y) // 4 + t * 2) & 255
    s = max(2, min(W, H) // 4)
    cx = (t * 2 + phase) % max(1, W - s)
    cy = (t + phase) % max(1, H - s)
    inside = (x >= cx) & (x < cx + s) & (y >= cy) & (y < cy + s)
    r = np.where(inside, 250, r)
    g = np.where(inside, 40, g)
    b = np.where(inside, 40, b)
    video = np.stack(np.broadcast_arrays(r, g, b), axis=-1).astype(np.int32)
    n_a = meta.audio_samples * meta.audio_channels
    k = np.arange(meta.audio_samples)
    sr = max(1, meta.audio_rate)
    tone = 6000 * np.sin(2 * math.pi * 440 * k / sr + phase) + 3000 * np.sin(2 * math.pi * 660 * k / sr)
    audio = np.repeat(np.round(tone).astype(np.int32)[:, None], meta.audio_channels, axis=1).reshape(-1)
    if noise_sigma > 0:
        key = os.urandom(32)
        u = np.frombuffer(shake_stream(key, 8 * (video.size + n_a)), dtype="<u8")
        # Box-Muller iz CSPRNG bitova (deterministično za dati ključ)
        u1 = (u[0::2] >> 11).astype(np.float64) / float(1 << 53) + 1e-300
        u2 = (u[1::2] >> 11).astype(np.float64) / float(1 << 53)
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * math.pi * u2)
        z = np.concatenate([z, np.sqrt(-2 * np.log(u1)) * np.sin(2 * math.pi * u2)])
        nz = np.round(z * noise_sigma).astype(np.int32)
        video = video + nz[:video.size].reshape(video.shape)
        audio = audio + nz[video.size:video.size + n_a] * 64
    video = np.clip(video, 0, 255).astype(np.uint8)
    audio = np.clip(audio, -32768, 32767).astype("<i2")
    if meta.sample_bytes != 2:
        raise ValueError("structured izvor očekuje 16-bitni zvuk")
    return video.tobytes() + audio.tobytes()


def make(source: str, meta: Meta) -> tuple[bytes, dict]:
    n = meta.payload_bytes
    if source == "csprng":
        return secrets.token_bytes(n), {}
    if source == "shake_keyed":
        return shake_stream(os.urandom(32) + b"keyed", n), {"key": "odbačen"}
    if source == "np_pcg64":
        seed = int.from_bytes(os.urandom(16), "little")
        return np.random.default_rng(seed).bytes(n), {"seed_bits": 128}
    if source == "np_legacy":
        seed = int.from_bytes(os.urandom(4), "little")
        return np.random.RandomState(seed).bytes(n), {"seed": seed}
    if source == "py_random":
        seed = int.from_bytes(os.urandom(4), "little")
        return random.Random(seed).randbytes(n), {"seed": seed}
    if source == "py_random_os":
        return random.Random().randbytes(n), {"seed": "OS (19968 bita)"}
    if source == "shake_weak20":
        seed = int.from_bytes(os.urandom(4), "little") & ((1 << WEAK_BITS) - 1)
        return weak_shake(seed, n), {"seed": seed}
    if source == "structured":
        phase = os.urandom(1)[0]
        return _structured(meta, phase), {"phase": phase}
    if source == "structured_noisy":
        phase = os.urandom(1)[0]
        return _structured(meta, phase, noise_sigma=2.0), {"phase": phase, "sigma": 2.0}
    raise ValueError(source)


def gaussian_noise_entropy_bits(sigma: float) -> float:
    """Entropija zaokruženog Gausovog šuma (bita po uzorku), bez odsecanja."""
    ks = np.arange(-int(12 * sigma) - 2, int(12 * sigma) + 3)
    cdf = lambda v: 0.5 * (1 + np.vectorize(math.erf)(v / (sigma * math.sqrt(2))))
    p = cdf(ks + 0.5) - cdf(ks - 0.5)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


SOURCES = ["csprng", "shake_keyed", "np_pcg64", "np_legacy", "py_random", "py_random_os",
           "shake_weak20", "structured", "structured_noisy"]
