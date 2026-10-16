"""Porodica SHADER: proceduralni "demoscene" uzorci po pikselu (bez stanja).

Šablon "sum" (plazma):
    VIDEO = bias_c + sum_j a_j * ISIN(fx_j*X + fy_j*Y + ft_j*T + k_j*C + ph_j)   (& 255)
    bias_c je po kanalu (konstanta, linearno u C ili BYTES[3](C), šta je kraće);
    kanali dele članove, a razlikuju se faznim pomakom k_j*C.
Šablon "rings" (prstenovi XOR rampa):
    R     = ISQRT(SQ(X - cx) + SQ(Y - cy))
    VIDEO = (R * (f + s*C) + v*T) XOR (gx*X + gy*Y + u*T)                      (& 255)
    s je neparno, pa bar jedan kanal ima neparnu frekvenciju prstenova.
AUDIO = sum_i A_i * ISIN((N * p_i) >> 2)     ton ili akord od dva tona
        (paran p se skraćuje: (N*2p) >> 2 == (N*p) >> 1).
Šum (nivo *_noise): svaki bajt videa se XOR-uje nasumičnim nenultim bajtom sa
verovatnoćom p (izuzeci videa sa vrednostima).

Svi koeficijenti unutar ISIN i XOR deluju po modulu 256, pa se emituju u
najkraćem obliku (lin()); oracle je zato i kanonski najkraći zapis porodice.
"""
from __future__ import annotations

import math

import numpy as np

from kfb import kfl
from kfb.kfl import C, N, T, X, Y, Node, bytes_lit, const, op

NAME = "shader"
LEVELS = {
    "one_term": {"template": "sum", "terms": 1, "p": 0.0},
    "three_terms": {"template": "sum", "terms": 3, "p": 0.0},
    "rings_xor": {"template": "rings", "p": 0.0},
    "three_terms_noise": {"template": "sum", "terms": 3, "p": 1e-3},
}
AUDIO_SHIFT = 2
K_MAX = 32          # opseg faznog pomaka po kanalu: k u [-K_MAX, K_MAX]


# ------------------------------------------------------------ opsezi
def ranges(meta) -> dict:
    """Opsezi parametara zavise samo od dimenzija: do ≈ 2 ciklusa preko kadra / filma,
    ali najviše 12 po koraku (inače se zbir prelama skoro svuda i liči na šum)."""
    W, H, Tn = meta.width, meta.height, meta.frames
    return {"fx": max(3, min(12, 512 // max(W, 1))), "fy": max(3, min(12, 512 // max(H, 1))),
            "ft": max(2, min(12, 384 // max(Tn, 1))), "fr": max(2, 384 // max(W, H, 1))}


# ------------------------------------------------- najkraće emitovanje
def nbytes(n: Node) -> int:
    buf = bytearray()
    kfl.emit(n, buf)
    return len(buf)


def ccost(v: int) -> int:
    """Broj bajtova konstante v u KFL-u."""
    if -16 <= v < 64:
        return 1
    z, n = kfl.zigzag(v), 1
    while z >= 128:
        z >>= 7
        n += 1
    return 1 + n


def rep(v: int, mod: int | None) -> int:
    """Najjeftiniji predstavnik v po modulu mod (None = tačna vrednost)."""
    if not mod:
        return int(v)
    v = int(v) % mod
    return min((v, v - mod), key=lambda c: (ccost(c), abs(c)))


def _prod(c: int, node: Node) -> Node:
    if c == 1:
        return node
    if c == -1:
        return Node("NEG", (node,))
    return Node("MUL", (node, const(c)))


def lin(pairs, c0: int = 0, mod: int | None = 256) -> Node:
    """Najkraći izraz za c0 + sum c*node (po modulu mod), lanac ADD/SUB."""
    elems = []
    for c, node in pairs:
        c = int(c) % mod if mod else int(c)
        if c == 0:
            continue
        elems.append((_prod(rep(c, mod), node), _prod(rep(-c, mod), node)))
    c0 = int(c0) % mod if mod else int(c0)
    if c0:
        elems.append((const(rep(c0, mod)), const(rep(-c0, mod))))
    if not elems:
        return const(0)
    sizes = [(nbytes(p), nbytes(m)) for p, m in elems]
    first = min(range(len(elems)), key=lambda i: (sizes[i][0] - min(sizes[i]), i))
    expr = elems[first][0]
    for i, (p, m) in enumerate(elems):
        if i != first:
            expr = Node("ADD", (expr, p)) if sizes[i][0] <= sizes[i][1] else Node("SUB", (expr, m))
    return expr


# --------------------------------------------------------- šablon sum
def term_forms(t: dict) -> list[dict]:
    """Četiri ekvivalentna oblika člana (ISIN je neparan i simetričan oko 64)."""
    fx, fy, ft, k, ph, a = t["fx"], t["fy"], t["ft"], t["k"], t["ph"], t["a"]
    return [dict(fx=fx, fy=fy, ft=ft, k=k, ph=ph, a=a),
            dict(fx=-fx, fy=-fy, ft=-ft, k=-k, ph=128 - ph, a=a),
            dict(fx=fx, fy=fy, ft=ft, k=k, ph=ph + 128, a=-a),
            dict(fx=-fx, fy=-fy, ft=-ft, k=-k, ph=-ph, a=-a)]


def term_node(t: dict) -> Node:
    """ISIN(arg) člana (bez amplitude) u najkraćem obliku."""
    return op("ISIN", lin([(t["fx"], X), (t["fy"], Y), (t["ft"], T), (t["k"], C)], t["ph"], 256))


def _best_form(t: dict) -> tuple[Node, int]:
    best = None
    for f in term_forms(t):
        n = term_node(f)
        cost = nbytes(n) + min(nbytes(_prod(rep(f["a"], 256), n)), nbytes(_prod(rep(-f["a"], 256), n)))
        if best is None or cost < best[0]:
            best = (cost, n, f["a"])
    return best[1], best[2]


def bias_options(bias) -> list[tuple[list, int]]:
    """Načini zapisa bias_c: (dodatni parovi, konstanta)."""
    b = [int(v) % 256 for v in bias]
    opts = [([], b[0])] if b[0] == b[1] == b[2] else []
    d = (b[1] - b[0]) % 256
    if (b[2] - b[1]) % 256 == d:
        opts.append(([(d, C)], b[0]))
    opts.append(([(1, bytes_lit(bytes(b), C))], 0))
    return opts


def sum_video(terms: list[dict], bias) -> Node:
    pairs = []
    for t in terms:
        n, a = _best_form(t)
        pairs.append((a, n))
    best = None
    for extra, c0 in bias_options(bias):
        e = lin(pairs + extra, c0, 256)
        if best is None or nbytes(e) < nbytes(best):
            best = e
    return best


# ------------------------------------------------------- šablon rings
def rings_video(p: dict) -> Node:
    R = op("ISQRT", op("ADD", op("SQ", lin([(1, X)], -p["cx"], None)),
                       op("SQ", lin([(1, Y)], -p["cy"], None))))
    ring = lin([(1, Node("MUL", (R, lin([(p["s"], C)], p["f"], 256)))), (p["v"], T)], 0, 256)
    ramp = lin([(p["gx"], X), (p["gy"], Y), (p["u"], T)], 0, 256)
    return op("XOR", ring, ramp)


# -------------------------------------------------------------- zvuk
def audio_expr(tones: list[dict]) -> Node | None:
    """tones: [{"A", "p", "q"}]; vrednost & 0xFFFF, pa su amplitude po modulu 2^16."""
    if not tones:
        return None
    pairs = []
    for tn in tones:
        p, q = int(tn["p"]), int(tn["q"])
        while q and not p & 1:           # (N*2p) >> q == (N*p) >> (q-1)
            p, q = p >> 1, q - 1
        arg = Node("MUL", (N, const(p)))
        if q:
            arg = Node("SHR", (arg, const(q)))
        pairs.append((tn["A"], op("ISIN", arg)))
    return lin(pairs, 0, 65536)


def video_exceptions(pos, vals) -> kfl.Exceptions | None:
    if not len(pos):
        return None
    return kfl.Exceptions(np.asarray(pos, np.int64), np.asarray(vals, np.int64))


# ---------------------------------------------------------- generator
def _visible(fx: int, fy: int, meta) -> bool:
    """Talas mora da pređe bar četvrtinu periode preko kadra (inače je skoro ravna boja)."""
    r = ranges(meta)
    need = max(1, min(64, r["fx"] * meta.width, r["fy"] * meta.height))   # mali kadrovi
    return max(abs(fx) * meta.width, abs(fy) * meta.height) >= need


def _n_spatial(meta) -> int:
    r = ranges(meta)
    return sum(_visible(fx, fy, meta) for fx in range(-r["fx"], r["fx"] + 1)
               for fy in range(-r["fy"], r["fy"] + 1))


def _sample_terms(rng, meta, n: int) -> list[dict]:
    r = ranges(meta)
    while True:
        terms, keys = [], set()
        while len(terms) < n:
            fx = int(rng.integers(-r["fx"], r["fx"] + 1))
            fy = int(rng.integers(-r["fy"], r["fy"] + 1))
            ft = int(rng.integers(-r["ft"], r["ft"] + 1))
            if not _visible(fx, fy, meta):
                continue
            if (fx, fy, ft) in keys or (-fx, -fy, -ft) in keys:
                continue
            keys.add((fx, fy, ft))
            terms.append({"fx": fx, "fy": fy, "ft": ft, "k": int(rng.integers(-K_MAX, K_MAX + 1)),
                          "ph": int(rng.integers(0, 256)), "a": int(rng.integers(1, 3))})
        if any(t["ft"] for t in terms):
            return terms


def _sample_tones(rng) -> list[dict]:
    n = int(rng.integers(1, 3))
    ps = rng.choice(np.arange(8, 129), size=n, replace=False)
    amax = 240 if n == 1 else 120
    return [{"A": int(rng.integers(16, amax + 1)), "p": int(p), "q": AUDIO_SHIFT} for p in ps]


def generate(rng, meta, level):
    r = ranges(meta)
    bits = 0.0
    if level["template"] == "sum":
        nt = int(level["terms"])
        terms = _sample_terms(rng, meta, nt)
        bias = [int(v) for v in rng.integers(0, 256, 3)]
        video = sum_video(terms, bias)
        # (w, k, ph, a) i (-w, -k, 128-ph, a) daju isti talas i jednako su verovatni: -1 bit po članu
        per = (math.log2(max(_n_spatial(meta), 1)) + math.log2(2 * r["ft"] + 1)
               + math.log2(2 * K_MAX + 1) + 8 + 1 - 1)
        bits += nt * per - math.log2(math.factorial(nt)) + 24
        params = {"template": "sum", "terms": terms, "bias": bias}
    else:
        W, H = meta.width, meta.height
        while True:
            p = {"cx": int(rng.integers(0, W)), "cy": int(rng.integers(0, H)),
                 "f": int(rng.integers(1, r["fr"] + 1)), "s": int(rng.choice([1, 3])),
                 "v": int(rng.integers(-12, 13)), "u": int(rng.integers(-8, 9)),
                 "gx": int(rng.integers(-4, 5)), "gy": int(rng.integers(-4, 5))}
            if (p["gx"] or p["gy"]) and (p["v"] or p["u"]):
                break
        video = rings_video(p)
        bits += math.log2(W * H * r["fr"] * 2 * 25 * 17 * 80)
        params = {"template": "rings", **p}
    tones = _sample_tones(rng) if meta.audio_samples else []
    if tones:
        na = 225 if len(tones) == 1 else 105
        bits += 1 + math.log2(math.comb(121, len(tones))) + len(tones) * math.log2(na)
    pr = float(level.get("p", 0.0))
    Nv = meta.video_bytes
    K = int(rng.binomial(Nv, pr)) if pr > 0 else 0
    pos = np.sort(rng.choice(Nv, size=K, replace=False)) if K else np.zeros(0, np.int64)
    vals = rng.integers(1, 256, K) if K else np.zeros(0, np.int64)
    if pr > 0:
        bits += K * (-math.log2(pr) + math.log2(255)) + (Nv - K) * -math.log2(1 - pr)
    prog = kfl.Program(video=video, audio=audio_expr(tones), exc_video=video_exceptions(pos, vals))
    params.update({"tones": tones, "p": pr, "exceptions": K})
    return prog, {"ideal_bits": float(bits), "params": params}
