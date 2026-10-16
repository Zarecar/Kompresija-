"""Porodica SPRITES: k objekata (krugovi i pravougaonici) koji se odbijaju od zidova.

Gornji levi ugao objekta širine w:  px(T) = TRI(x0 + vx*T, W - w)   (odbijanje u [0, W-w])
Pravougaonik w x h:  van  <=>  OR(DIV(X - px, w), DIV(Y - py, h)) != 0
Krug (w = h = 2r+1): van  <=>  (X - px - r)^2 + (Y - py - r)^2 > R2,  R2 = r*r + r
                     (u zapisu najmanji R2 sa istim diskom)
Jednobojna pozadina: VIDEO = BYTES[3(k+1)](id + C),  id = WHERE(van_k, WHERE(.., 3(k-1)), 3k)
Gradijent:           VIDEO = WHERE(van_k, WHERE(.., pozadina, boja_1), boja_k),
                     pozadina = baza_c + ax_c*X + ay_c*Y (po kanalu)
Kasniji objekat je iznad ranijeg (fiksan z-red). Graditelj bira najkraći ekvivalentan oblik
(i oracle i enkoder ga koriste, pa je K̄ dužina kanonskog opisa instance).
AUDIO = ISIN((N*f) >> 3) * amp (ton) ili tišina.
Šum (k3_noise): svaki bajt videa se XOR-uje nasumičnim nenultim bajtom sa verovatnoćom p
(izuzeci videa sa vrednostima).
"""
from __future__ import annotations

import math

import numpy as np

from kfb import kfl
from kfb.kfl import C, N, T, X, Y, bytes_lit, const, lut, op

NAME = "sprites"
LEVELS = {
    "k2": {"k": 2, "p": 0.0, "audio": "tone"},
    "k6": {"k": 6, "p": 0.0, "audio": "tone"},
    "k3_noise": {"k": 3, "p": 1e-3, "audio": "silence"},
}
TONE_SHIFT = 3


# ------------------------------------------------------------ graditelj izraza
def blen(node: kfl.Node) -> int:
    """Dužina izraza u bajtovima."""
    buf = bytearray()
    kfl.emit(node, buf)
    return len(buf)


def shortest(nodes):
    """Najkraći izraz; pri istoj dužini prvi po redu (kandidati su poređani po čitljivosti)."""
    return min(nodes, key=blen)


def chan(vals) -> kfl.Node:
    """Vrednost po kanalu (R, G, B) kao najkraći izraz (izlaz se ionako seče na & 255)."""
    v = [int(x) for x in vals]
    if v[0] == v[1] == v[2]:
        return shortest([const(v[0]), const(v[0] % 256), const(v[0] % 256 - 256)])
    sg = [((x + 128) % 256) - 128 for x in v]
    return shortest([lut(v, C), lut(sg, C), bytes_lit(bytes(x % 256 for x in v), C)])


def add_c(node: kfl.Node, c: int) -> kfl.Node:
    if not c:
        return node
    if node.op == "CONST":
        return const(node.value + c)
    return node + c


def tri_value(a: int, L: int) -> int:
    return L - abs(a % (2 * L) - L) if L > 0 else 0


def tri_node(p0: int, v: int, L: int) -> kfl.Node:
    """TRI(p0 + v*T, L) u najkraćem obliku (p0, v mod 2L; simetrija (p0, v) ~ (-p0, -v))."""
    if L <= 0:
        return const(0)
    P = 2 * L
    if v % P == 0:
        return const(tri_value(p0, L))
    pairs = []
    for s in (1, -1):
        a0, b0 = (s * p0) % P, (s * v) % P
        pairs += [(a, b) for a in (a0, a0 - P) for b in (b0, b0 - P)]
    pairs.sort(key=lambda ab: (abs(ab[1]), ab[1] < 0, abs(ab[0]), ab[0] < 0))
    cands = []
    for a, b in pairs:
        if b == 1:
            inner = T if a == 0 else const(a) + T
        elif b == -1:
            inner = op("NEG", T) if a == 0 else const(a) - T
        else:
            inner = const(b) * T if a == 0 else const(a) + const(b) * T
        cands.append(op("TRI", inner, L))
    return shortest(cands)


def outside(o: dict, W: int, H: int) -> kfl.Node:
    """Uslov "piksel nije u objektu" (nenula van objekta)."""
    px = tri_node(o["x0"], o["vx"], W - o["w"])
    py = tri_node(o["y0"], o["vy"], H - o["h"])
    if o["shape"] == "rect":
        return op("OR", op("DIV", X - px, o["w"]), op("DIV", Y - py, o["h"]))
    r = (o["w"] - 1) // 2
    return op("LT", canon_r2(r, o["R2"]), op("SQ", X - add_c(px, r)) + op("SQ", Y - add_c(py, r)))


def canon_r2(r: int, R2: int) -> int:
    """Najmanji R2 sa istim diskom: najveći zbir dx^2 + dy^2 <= R2 (|dx|, |dy| <= r)."""
    d = np.arange(r + 1)
    s = (d[:, None] ** 2 + d[None, :] ** 2).ravel()
    s = s[s <= R2]
    return int(s.max()) if len(s) else int(R2)


def bg_node(bg: dict) -> kfl.Node:
    base, ax, ay = bg["base"], bg["ax"], bg["ay"]
    e = chan(base)
    if any(int(v) % 256 for v in ax) and [int(v) % 256 for v in ax] == [int(v) % 256 for v in ay]:
        return e + chan(ax) * (X + Y)
    if any(int(v) % 256 for v in ax):
        e = e + chan(ax) * X
    if any(int(v) % 256 for v in ay):
        e = e + chan(ay) * Y
    return e


def is_solid(bg: dict) -> bool:
    return not any(int(v) % 256 for v in list(bg["ax"]) + list(bg["ay"]))


def video_node(bg: dict, objs: list, W: int, H: int) -> kfl.Node:
    """Najkraći od dva oblika: lanac WHERE sa bojom po objektu, ili zajednička tabela boja."""
    outs = [outside(o, W, H) for o in objs]
    v = bg_node(bg)
    for o, out in zip(objs, outs):
        v = op("WHERE", out, v, chan(o["color"]))
    cands = [v]
    if is_solid(bg):
        idn = const(0)
        for i, out in enumerate(outs, 1):
            idn = op("WHERE", out, idn, 3 * i)
        tab = bytes(int(c) % 256 for c in bg["base"]) + b"".join(
            bytes(int(c) % 256 for c in o["color"]) for o in objs)
        cands.append(bytes_lit(tab, idn + C))
    return shortest(cands)


def tone_node(f: int, s: int, amp: int) -> kfl.Node:
    """ISIN((N*f) >> s) * amp; (2g, s) je isto što i (g, s-1), pa se skraćuje."""
    while s > 0 and f % 2 == 0 and f:
        f, s = f // 2, s - 1
    arg = N if f == 1 else N * f
    if s:
        arg = arg >> s
    e = op("ISIN", arg)
    return e if amp == 1 else e * amp


def program_for(bg, objs, W, H, tone=None, exc_video=None, audio=None, exc_audio=None) -> kfl.Program:
    """Oracle/enkoderski program. audio (gotov izraz) ima prednost nad tone=(f, s, amp)."""
    if audio is None and tone is not None:
        audio = tone_node(*tone)
    return kfl.Program(video=video_node(bg, objs, W, H), audio=audio,
                       exc_video=exc_video, exc_audio=exc_audio)


# ------------------------------------------------------------ generator
def _color(rng, avoid, min_d=110):
    c = [int(v) for v in rng.integers(0, 256, 3)]
    for _ in range(200):
        if all(sum(abs(a - b) for a, b in zip(c, q)) >= min_d for q in avoid):
            break
        c = [int(v) for v in rng.integers(0, 256, 3)]
    return c


def _background(rng, W, H):
    if rng.random() < 0.5:
        base = [int(v) for v in rng.integers(0, 256, 3)]
        return {"base": base, "ax": [0, 0, 0], "ay": [0, 0, 0]}, 1 + 24.0, base
    m = max(1, 96 // max(W, H))
    while True:
        ax = [int(v) for v in rng.integers(-m, m + 1, 3)]
        ay = [int(v) for v in rng.integers(-m, m + 1, 3)]
        if any(ax) or any(ay):
            break
    base, bits = [], 1 + 6 * math.log2(2 * m + 1)
    for c in range(3):
        lo = -min(0, ax[c] * (W - 1)) - min(0, ay[c] * (H - 1))
        hi = 255 - max(0, ax[c] * (W - 1)) - max(0, ay[c] * (H - 1))
        base.append(int(rng.integers(lo, hi + 1)))
        bits += math.log2(hi - lo + 1)
    mid = [base[c] + (ax[c] * (W - 1) + ay[c] * (H - 1)) // 2 for c in range(3)]
    return {"base": base, "ax": ax, "ay": ay}, bits, mid


def generate(rng, meta, level):
    H, W = meta.height, meta.width
    s = min(H, W)
    vmax = max(2, s // 24)
    bg, ideal, bg_mid = _background(rng, W, H)
    avoid = [bg_mid]
    objs = []
    for _ in range(int(level["k"])):
        o = {}
        if rng.random() < 0.5:
            hi = min(max(4, s // 3), s - 1)
            lo = min(max(3, s // 6), hi)
            o.update(shape="rect", w=int(rng.integers(lo, hi + 1)), h=int(rng.integers(lo, hi + 1)))
            ideal += 1 + 2 * math.log2(hi - lo + 1)
        else:
            hi = min(max(3, s // 6), (s - 2) // 2)
            lo = min(max(2, s // 12), hi)
            r = int(rng.integers(lo, hi + 1))
            o.update(shape="circ", w=2 * r + 1, h=2 * r + 1, R2=r * r + r)
            ideal += 1 + math.log2(hi - lo + 1)
        for ax_, n in (("x", W), ("y", H)):
            L = n - o["w" if ax_ == "x" else "h"]
            o[ax_ + "0"] = int(rng.integers(0, 2 * L))
            o["v" + ax_] = int(rng.integers(1, vmax + 1)) * int(rng.choice([-1, 1]))
            ideal += math.log2(2 * L * vmax)          # (p0, v) ~ (-p0, -v): pola od 2L * 2vmax
        o["color"] = _color(rng, avoid)
        avoid.append(o["color"])
        ideal += 24
        objs.append(o)
    tone = None
    if level["audio"] == "tone" and meta.audio_samples:
        tone = (int(rng.integers(20, 201)), TONE_SHIFT, int(rng.integers(40, 251)))
        ideal += math.log2(181) + math.log2(211)
    p = float(level["p"])
    Nv = meta.video_bytes
    K = int(rng.binomial(Nv, p)) if p > 0 else 0
    exc = None
    if K:
        pos = np.sort(rng.choice(Nv, size=K, replace=False)).astype(np.int64)
        vals = rng.integers(1, 256, K).astype(np.int64)
        exc = kfl.Exceptions(pos, vals)
        ideal += Nv * -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) + K * math.log2(255)
    prog = program_for(bg, objs, W, H, tone=tone, exc_video=exc)
    params = {"bg": bg, "objects": objs, "tone": list(tone) if tone else None,
              "exceptions": K, "p": p}
    return prog, {"ideal_bits": float(ideal), "params": params}
