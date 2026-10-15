"""Adaptivni kontekstni prediktor + logistički mikser + aritmetičko kodovanje.

Ovo je naš "alternativni matematički prediktor" (i minimalni neuronski model:
mikser je jedan logistički neuron koji uči online, kao u PAQ/cmix porodici).
Nema unapred treniranih težina, pa se ništa specifično za film ne može
"sakriti" u modelu. Sve operacije su celobrojne -> bit-identično dekodovanje
na svakoj platformi.

Konteksti (po bajtu, za svaki bit sa delimičnim bajtom c0):
  video: kanal, prethodni bajt, MED/LOCO-I prediktor (levo/gore/gore-levo),
         isti piksel u prethodnom frejmu, kvantizovan par (levo, gore),
         vremensko-prostorni prediktor T + L - TL.
  audio: pozicija bajta u uzorku, prethodni bajt, linearni prediktor 2L-U,
         prethodni uzorak istog kanala, kvantizovan par, uzorak pre dva koraka.

Pure-Python implementacija je spora (~10 KB/s); namenjena je malim testovima.
"""
from __future__ import annotations

from .film import Meta
from .rangecoder import COUNTER_INIT, DT, LIMIT, MASK32

NM = 6            # broj modela
TB = 18           # bitova indeksa po modelu
S = 1 << TB
LR_SHIFT = 10


def _squash_point(d: int) -> int:
    pts = [1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047,
           2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089,
           4092, 4093, 4094]
    if d > 2047:
        return 4095
    if d < -2047:
        return 1
    w = d & 127
    i = (d >> 7) + 16
    return (pts[i] * (128 - w) + pts[i + 1] * w + 64) >> 7


SQUASH = [_squash_point(d) for d in range(-2047, 2048)]
STRETCH = [0] * 4096
_pi = 0
for _x in range(-2047, 2048):
    _v = _squash_point(_x)
    for _j in range(_pi, _v + 1):
        STRETCH[_j] = _x
    _pi = _v + 1
for _j in range(_pi, 4096):
    STRETCH[_j] = 2047


def _clamp8(v: int) -> int:
    return 0 if v < 0 else (255 if v > 255 else v)


def _contexts(meta: Meta, buf, i: int) -> tuple[int, int, int, int, int, int, int]:
    """Vraća 6 konteksta (svaki < 2^10) i selektor skupa težina."""
    vb = meta.video_bytes
    p1 = buf[i - 1] if i > 0 else 0
    if i < vb:
        fb = meta.frame_bytes
        rb = meta.width * 3
        f, r = divmod(i, fb)
        y, r2 = divmod(r, rb)
        x, ch = divmod(r2, 3)
        L = buf[i - 3] if x else 0
        U = buf[i - rb] if y else 0
        UL = buf[i - rb - 3] if (x and y) else 0
        T = buf[i - fb] if f else 0
        TL = buf[i - fb - 3] if (f and x) else 0
        mx, mn = (L, U) if L > U else (U, L)
        if UL >= mx:
            med = mn
        elif UL <= mn:
            med = mx
        else:
            med = L + U - UL
        return (ch, p1, med, T, ((L >> 3) << 5) | (U >> 3), _clamp8(T + L - TL), ch)
    j = i - vb
    sb = meta.sample_bytes
    fs = sb * meta.audio_channels
    bpos = j % sb
    L = buf[i - fs] if j >= fs else 0
    U = buf[i - 2 * fs] if j >= 2 * fs else 0
    return (4 + bpos, 256 | p1, 512 | (bpos << 8) | _clamp8(2 * L - U), 256 | L,
            512 | ((L >> 3) << 5) | (U >> 3), 256 | U, 3 + min(bpos, 1))


def _run(meta: Meta, data: bytes | None, rec: bytes | None, pos: int):
    """Zajednički kod za enkoder (data != None) i dekoder (rec != None)."""
    n = meta.payload_bytes
    T = [COUNTER_INIT] * (NM * S)
    W = [[65536 // NM] * (NM + 1) for _ in range(5 * 256)]
    ST, SQ, dt, lim = STRETCH, SQUASH, DT, LIMIT
    offs = [m * S for m in range(NM)]
    enc = data is not None
    buf = data if enc else bytearray(n)
    out = bytearray()
    x1, x2 = 0, MASK32
    if not enc:
        x = int.from_bytes(rec[pos:pos + 4], "big")
        rp = pos + 4
        rlen = len(rec)
        if rp > rlen:
            raise ValueError("skraćen CM tok")
    for i in range(n):
        c = _contexts(meta, buf, i)
        bases = [offs[m] + (c[m] << 8) for m in range(NM)]
        wbase = c[6] * 256
        byte = buf[i] if enc else 0
        c0 = 1
        for k in range(7, -1, -1):
            idx = [b + c0 for b in bases]
            vals = [T[j] for j in idx]
            st = [ST[v >> 12] for v in vals]
            w = W[wbase + c0]
            dot = 256 * w[NM]
            for m in range(NM):
                dot += w[m] * st[m]
            d = dot >> 16
            if d > 2047:
                d = 2047
            elif d < -2047:
                d = -2047
            p = SQ[d + 2047]
            xmid = x1 + ((x2 - x1) >> 12) * p
            if enc:
                bit = (byte >> k) & 1
                if bit:
                    x2 = xmid
                else:
                    x1 = xmid + 1
                while not ((x1 ^ x2) & 0xFF000000):
                    out.append(x2 >> 24)
                    x1 = (x1 << 8) & MASK32
                    x2 = ((x2 << 8) & MASK32) | 0xFF
            else:
                if x <= xmid:
                    bit = 1
                    x2 = xmid
                else:
                    bit = 0
                    x1 = xmid + 1
                while not ((x1 ^ x2) & 0xFF000000):
                    x1 = (x1 << 8) & MASK32
                    x2 = ((x2 << 8) & MASK32) | 0xFF
                    if rp >= rlen:
                        raise ValueError("skraćen CM tok")
                    x = ((x << 8) & MASK32) | rec[rp]
                    rp += 1
            # učenje: mikser (gradijent log-gubitka) i brojači
            err = (bit << 12) - p
            for m in range(NM):
                w[m] += (st[m] * err) >> LR_SHIFT
            w[NM] += (256 * err) >> LR_SHIFT
            target = bit << 16
            for m in range(NM):
                v = vals[m]
                pp = v >> 8
                cnt = v & 0xFF
                pp += ((target - pp) * dt[cnt]) >> 16
                if cnt < lim:
                    cnt += 1
                T[idx[m]] = (pp << 8) | cnt
            c0 = (c0 << 1) | bit
        if not enc:
            buf[i] = c0 & 0xFF
    if enc:
        out += x1.to_bytes(4, "big")
        return bytes(out)
    return bytes(buf), rp


def encode(meta: Meta, payload: bytes) -> bytes:
    return _run(meta, payload, None, 0)


def decode(meta: Meta, rec: bytes, pos: int) -> tuple[bytes, int]:
    return _run(meta, None, rec, pos)
