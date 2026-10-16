"""Porodica DIFFUSION: celobrojne dinamičke teksture (lokalna dinamika stanja na torusu).

Nivo "smooth" (difuzija prvog reda sa pokretnim izvorom toplote):
  STEP  = ((a*S + b*NB4(S) + d*NB8(S) + izvor) >> sh) & 255,     a + 4b + 8d = 2^sh
  izvor = LT(SQ(X - px) + SQ(Y - py), R2) * amp   (vrući disk koji se odbija od ivica)
          px = TRI(x0 + vx*T, W-1),  py = TRI(y0 + vy*T, H-1)
  INIT  = HASH((Y*WID + X) ^ seed) >> 55       (šum iz seed-a od 20 bita)
          ili WHERE(van zakrpe, 0, BYTES[h*w](idx))   (mala zakrpa nasumičnih bajtova na nuli)
Nivo "wave" (prigušeni talas drugog reda; dva polja u jednom int64 stanju):
  S = (cur << 20) | (prev & M),  cur = S >> 20,  prev = (S << 44) >> 44,  M = 2^20 - 1
  new  = (A*cur + B*NB4(cur) - P*prev + izvor) >> sh,  A + 4B = 2^(sh+1) - 2δ,  P = 2^sh - δ
  izvor = EQ(X, px) * EQ(Y, py) * (ISIN(T*w) * amp)   (oscilujući izvor koji se kreće)
  STEP = (new << 20) | (cur & M);   INIT = zakrpa << 20   (prev0 = 0)
  (cur je u višem polju jer je tako zapis kraći: cur se koristi češće od prev.)
VIDEO = E * m[C] + o[C]  (afina paleta po kanalu, izlaz & 255), E = S (smooth) ili cur (wave).
        m[0] je neparan, pa se stanje tačno čita iz videa (do pomeraja/znaka palete);
        generator talasa bira amplitude tako da je cur u [-128, 127].
AUDIO = ISIN((N * (MOD(SUMS >> k, R) + base)) >> 3) * amp  (ton vođen zbirom stanja) ili tišina.
Šum (smooth_noise): svaka ćelija u svakom prelazu se sa verovatnoćom p XOR-uje slučajnom
vrednošću 1..15 (izuzeci stanja sa vrednostima).
Graditelj bira najkraći ekvivalentan oblik izraza; isti graditelj koristi i enkoder.
"""
from __future__ import annotations

import math

import numpy as np

from kfb import kfl
from kfb.kfl import C, N, S, SUMS, T, WID, X, Y, bytes_lit, const, lut, op

NAME = "diffusion"
LEVELS = {
    "smooth": {"kind": "smooth", "p": 0.0, "audio": "sums"},
    "wave": {"kind": "wave", "p": 0.0, "audio": "sums"},
    "smooth_noise": {"kind": "smooth", "p": 1e-4, "audio": "silence"},
}
FIELD = 20                       # širina polja prev u pakovanom stanju talasa
FMASK = (1 << FIELD) - 1
HASH_SHIFT = 55                  # gornjih 8 bita HASH-a (HASH je nenegativan, 63 bita)
SEED_BITS = 20
PAL_M0 = (1, -1, 3, -3)          # množilac prvog kanala (neparan -> bijekcija mod 256)
PAL_M = (-3, -2, -1, 1, 2, 3)
DISK_R2 = (2, 3, 5, 6, 9, 10, 11, 14)   # kanonski R2 (diskovi od 5 do 45 ćelija)
ISIN_LUT = kfl.ISIN_LUT


def _smooth_rules():
    """(a, b, d, sh): a + 4b + 8d = 2^sh, a >= 2^sh/4 (bez treperenja šahovnice),
    bar jedan sused, bez svodljivih (sve parno) trojki."""
    out = []
    for sh in (3, 4, 5):
        tot = 1 << sh
        for d in range(tot // 8 + 1):
            for b in range(tot // 4 + 1):
                a = tot - 4 * b - 8 * d
                if a >= tot // 4 and (b or d) and (a | b | d) & 1:
                    out.append((a, b, d, sh))
    return out


SMOOTH_RULES = _smooth_rules()


def _wave_rules():
    """(A, B, P, sh): c^2 = B/2^sh u [1/8, 3/8] (stabilno), prigušenje δ/2^sh, nesvodljiva."""
    out = []
    for sh in (4, 5, 6):
        tot = 1 << sh
        for dl in (1, 2):
            for B in range(-(-tot // 8), 3 * tot // 8 + 1):
                A, P = 2 * tot - 2 * dl - 4 * B, tot - dl
                if (A | B | P) & 1:             # bez svodljivih (sve parno) pravila
                    out.append((A, B, P, sh))
    return out


WAVE_RULES = _wave_rules()


# ------------------------------------------------------------ graditelj izraza
def blen(node: kfl.Node) -> int:
    """Dužina izraza u bajtovima."""
    buf = bytearray()
    kfl.emit(node, buf)
    return len(buf)


def shortest(nodes):
    return min(nodes, key=blen)


def chan(vals) -> kfl.Node:
    """Vrednost po kanalu (R, G, B) mod 256 kao najkraći izraz (izlaz videa se seče na & 255)."""
    v = [int(x) % 256 for x in vals]
    if v[0] == v[1] == v[2]:
        return shortest([const(v[0]), const(v[0] - 256)])
    sg = [((x + 128) % 256) - 128 for x in v]
    return shortest([lut(v, C), lut(sg, C), bytes_lit(bytes(v), C)])


def lin_comb(terms) -> kfl.Node | None:
    """Zbir c_i * e_i u najkraćem obliku (c=1 -> e, negativni članovi se oduzimaju)."""
    pos = [(c, e) for c, e in terms if c > 0]
    neg = [(-c, e) for c, e in terms if c < 0]

    def mul(c, e):
        return e if c == 1 else const(c) * e
    acc = None
    for c, e in pos:
        acc = mul(c, e) if acc is None else acc + mul(c, e)
    for c, e in neg:
        acc = (const(-c) * e if c != 1 else op("NEG", e)) if acc is None else acc - mul(c, e)
    return acc


def add_node(a: kfl.Node | None, b: kfl.Node | None) -> kfl.Node:
    if a is None:
        return b if b is not None else const(0)
    return a if b is None else a + b


def tri_value(a, L: int):
    """TRI(a, L) nad NumPy nizom (isto kao VM)."""
    if L <= 0:
        return np.zeros_like(np.asarray(a))
    return L - np.abs(np.mod(a, 2 * L) - L)


def tri_node(p0: int, v: int, L: int) -> kfl.Node:
    """TRI(p0 + v*T, L) u najkraćem obliku (p0, v mod 2L; simetrija (p0, v) ~ (-p0, -v))."""
    if L <= 0:
        return const(0)
    P = 2 * L
    if v % P == 0:
        return const(int(tri_value(p0, L)))
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


def point_node(src: dict, W: int, H: int) -> kfl.Node:
    """1 u tački izvora (px(T), py(T)), inače 0."""
    px = tri_node(src["x0"], src["vx"], W - 1)
    py = tri_node(src["y0"], src["vy"], H - 1)
    return op("EQ", X, px) * op("EQ", Y, py)


def disk_node(src: dict, W: int, H: int) -> kfl.Node:
    """1 u disku (X-px)^2 + (Y-py)^2 < R2 oko izvora, inače 0."""
    px = tri_node(src["x0"], src["vx"], W - 1)
    py = tri_node(src["y0"], src["vy"], H - 1)
    dx = X if px.op == "CONST" and px.value == 0 else X - px
    dy = Y if py.op == "CONST" and py.value == 0 else Y - py
    return op("LT", op("SQ", dx) + op("SQ", dy), int(src["R2"]))


def canon_r2(R2: int) -> int:
    """Najmanji R2 sa istim diskom {dx^2 + dy^2 < R2}."""
    s = sorted({a * a + b * b for a in range(int(R2) + 1) for b in range(int(R2) + 1) if a * a + b * b < R2})
    return s[-1] + 1 if s else 0


def src_positions(src: dict, W: int, H: int, ts) -> tuple[np.ndarray, np.ndarray]:
    ts = np.asarray(ts, dtype=np.int64)
    return (tri_value(src["x0"] + src["vx"] * ts, W - 1),
            tri_value(src["y0"] + src["vy"] * ts, H - 1))


def cur_node() -> kfl.Node:
    return op("SHR", S, FIELD)


def prev_node() -> kfl.Node:
    return op("SHR", op("SHL", S, 64 - FIELD), 64 - FIELD)


def step_node(kind: str, rule, src: dict | None, W: int, H: int) -> kfl.Node:
    if kind == "smooth":
        a, b, d, sh = rule
        lin = lin_comb([(a, S), (b, op("NB4", S)), (d, op("NB8", S))])
        drv = None if src is None else _scale(disk_node(src, W, H), src["amp"])
        return op("AND", op("SHR", add_node(lin, drv), sh), 255)
    A, B, P, sh = rule
    cur = cur_node()
    lin = lin_comb([(A, cur), (B, op("NB4", cur_node())), (-P, prev_node())])
    drv = None
    if src is not None:
        drv = point_node(src, W, H) * _scale(op("ISIN", _scale(T, src["w"])), src["amp"])
    new = op("SHR", add_node(lin, drv), sh)
    return op("OR", op("SHL", new, FIELD), op("AND", cur_node(), FMASK))


def _scale(e: kfl.Node, c: int) -> kfl.Node:
    return e if c == 1 else e * const(c)


def _rel(axv: kfl.Node, v0: int) -> kfl.Node:
    return axv if v0 == 0 else axv - const(v0)


def patch_node(data, y0: int, x0: int, h: int, w: int, H: int, W: int, off: int = 0,
               bg: int = 0) -> kfl.Node:
    """WHERE(van pravougaonika, bg, BYTES[h*w](idx) + off); BYTES uzima idx mod h*w."""
    L = h * w
    lit = bytes(int(v) for v in np.asarray(data, dtype=np.int64).reshape(-1))
    if h == 1:
        k0, e = x0 % L, X
        idx_cands = [_rel(X, x0)]
    else:
        k0 = (y0 * w + x0) % L
        e = (Y * WID if w == W else Y * const(w)) + X
        idx_cands = [_rel(Y, y0) * const(w) + _rel(X, x0)]
    for kk in {k0, k0 - L}:
        idx_cands.append(e if kk == 0 else e - const(kk))
    lit_e = bytes_lit(lit, shortest(idx_cands))
    if off:
        lit_e = lit_e + const(off)
    conds = []
    for v0, n, NN, axv in ((y0, h, H, Y), (x0, w, W, X)):
        if n >= NN and v0 == 0:
            continue
        conds.append(_rel(axv, v0) if n == 1 else op("DIV", _rel(axv, v0), n))
    if not conds:
        return lit_e
    cond = conds[0] if len(conds) == 1 else op("OR", conds[0], conds[1])
    return op("WHERE", cond, bg, lit_e)


def hash_init_node(seed: int) -> kfl.Node:
    idx = Y * WID + X
    return op("SHR", op("HASH", idx if seed == 0 else op("XOR", idx, seed)), HASH_SHIFT)


def hash_init_values(seed: int, H: int, W: int) -> np.ndarray:
    idx = np.arange(H * W, dtype=np.int64) ^ np.int64(seed)
    return (hash64(idx) >> HASH_SHIFT).reshape(H, W)


def hash64(a: np.ndarray) -> np.ndarray:
    """HASH iz VM-a (splitmix64 završnica, >> 1)."""
    with np.errstate(all="ignore"):
        z = np.asarray(a).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return (z >> np.uint64(1)).astype(np.int64)


def init_node(kind: str, init: dict, H: int, W: int) -> kfl.Node:
    t = init["type"]
    if t == "const":
        e = const(init["v"])
    elif t == "hash":
        e = hash_init_node(init["seed"])
    elif t == "patch":
        e = patch_node(init["data"], init["y0"], init["x0"], init["h"], init["w"], H, W,
                       init.get("off", 0), init.get("bg", 0))
    else:
        raise ValueError(t)
    if kind == "smooth":
        return e
    p0 = init.get("prev0", "zero")          # "zero", "cur" ili konstanta (int)
    if e.op == "CONST":
        v = e.value
        p = v if p0 == "cur" else (0 if p0 == "zero" else int(p0))
        return const((v << FIELD) | (p & FMASK))
    if not isinstance(p0, str):
        return op("OR", op("SHL", e, FIELD), const(int(p0) & FMASK)) if int(p0) & FMASK else op("SHL", e, FIELD)
    if p0 == "cur":
        if init.get("nonneg", False):
            return e * const((1 << FIELD) + 1)
        return op("OR", op("SHL", e, FIELD), op("AND", init_node("smooth", init, H, W), FMASK))
    return op("SHL", e, FIELD)


def video_node(kind: str, pal: dict) -> kfl.Node:
    E = S if kind == "smooth" else cur_node()
    if "lut" in pal:
        tab = bytes(int(v) % 256 for v in pal["lut"])
        return bytes_lit(tab, E * 3 + C)
    m = [int(v) % 256 for v in pal["m"]]
    o = [int(v) % 256 for v in pal["o"]]
    e = E if m == [1, 1, 1] else E * chan(m)
    if any(o):
        e = e + chan(o)
    return e


def audio_node(aud: dict | None) -> kfl.Node | None:
    if aud is None:
        return None
    if "words" in aud:
        return kfl.words_lit(np.asarray(aud["words"], dtype=np.int64) & 0xFFFF, N)
    if "f" in aud:                       # stalni ton
        f, s = int(aud["f"]), 3
        while s > 0 and f % 2 == 0 and f:
            f, s = f // 2, s - 1
        arg = N if f == 1 else N * f
        e = op("ISIN", arg >> s if s else arg)
    else:
        k, R, base = int(aud["k"]), int(aud["R"]), int(aud["base"])
        sm = SUMS if k == 0 else op("SHR", SUMS, k)
        f = op("MOD", sm, R)
        if base:
            f = f + const(base)
        e = op("ISIN", (N * f) >> 3)
    amp = int(aud["amp"])
    return e if amp == 1 else e * const(amp)


def program_for(kind: str, rule, src, init: dict, pal: dict, aud, meta, exc_state=None,
                exc_audio=None) -> kfl.Program:
    H, W = meta.height, meta.width
    return kfl.Program(video=video_node(kind, pal), init=init_node(kind, init, H, W),
                       step=step_node(kind, rule, src, W, H), audio=audio_node(aud),
                       exc_state=exc_state, exc_audio=exc_audio)


# ------------------------------------------------------------ NumPy replike
def nb4(F: np.ndarray) -> np.ndarray:
    return (np.roll(F, 1, -2) + np.roll(F, -1, -2) + np.roll(F, 1, -1) + np.roll(F, -1, -1))


def nb8(F: np.ndarray) -> np.ndarray:
    s = nb4(F)
    for dy in (1, -1):
        r = np.roll(F, dy, -2)
        s = s + np.roll(r, 1, -1) + np.roll(r, -1, -1)
    return s


def drive_values(kind: str, src: dict | None, ts) -> np.ndarray:
    """Amplituda izvora po prelazu t (bez prostornog dela)."""
    ts = np.asarray(ts, dtype=np.int64)
    if src is None:
        return np.zeros(len(ts), np.int64)
    if kind == "smooth":
        return np.full(len(ts), int(src["amp"]), np.int64)
    return ISIN_LUT[(ts * int(src["w"])) & 255] * int(src["amp"])


def drive_mask(kind: str, src: dict, H: int, W: int, ts) -> np.ndarray:
    """Prostorni deo izvora (nT, H, W): disk (smooth) ili tačka (wave)."""
    ts = np.asarray(ts, dtype=np.int64)
    px, py = src_positions(src, W, H, ts)
    if kind == "smooth":
        dy = np.arange(H, dtype=np.int64)[None, :, None] - py[:, None, None]
        dx = np.arange(W, dtype=np.int64)[None, None, :] - px[:, None, None]
        return (dx * dx + dy * dy < int(src["R2"])).astype(np.int64)
    m = np.zeros((len(ts), H, W), np.int64)
    m[np.arange(len(ts)), py, px] = 1
    return m


def add_drive(lin: np.ndarray, kind: str, src: dict | None, ts) -> np.ndarray:
    """lin (nT, H, W) += izvor u prelazima ts."""
    if src is None:
        return lin
    H, W = lin.shape[-2:]
    amp = drive_values(kind, src, ts)
    return lin + drive_mask(kind, src, H, W, ts) * amp[:, None, None]


def smooth_next(Sp: np.ndarray, rule, src, ts) -> np.ndarray:
    """Sledeće stanje za niz stanja Sp (nT, H, W) u prelazima ts."""
    a, b, d, sh = rule
    with np.errstate(all="ignore"):
        lin = a * Sp + b * nb4(Sp) + (d * nb8(Sp) if d else 0)
        lin = add_drive(np.array(lin, dtype=np.int64), "smooth", src, ts)
        return (lin >> sh) & 255


def wave_new(cur: np.ndarray, prev: np.ndarray, rule, src, ts) -> np.ndarray:
    A, B, P, sh = rule
    with np.errstate(all="ignore"):
        lin = np.array(A * cur + B * nb4(cur) - P * prev, dtype=np.int64)
        lin = add_drive(lin, "wave", src, ts)
        return lin >> sh


def simulate(kind, rule, src, S0, T_, prev0=None) -> np.ndarray:
    """Polje stanja (smooth: S, wave: cur) za T_ frejmova, bez izuzetaka."""
    out = np.empty((T_,) + S0.shape, np.int64)
    out[0] = S0
    prev = np.zeros_like(S0) if prev0 is None else prev0
    for t in range(T_ - 1):
        if kind == "smooth":
            out[t + 1] = smooth_next(out[t][None], rule, src, [t])[0]
        else:
            out[t + 1] = wave_new(out[t][None], prev[None], rule, src, [t])[0]
            prev = out[t]
    return out


# ------------------------------------------------------------ generator
def _palette(rng):
    m = [int(rng.choice(PAL_M0))] + [int(rng.choice(PAL_M)) for _ in range(2)]
    o = [int(v) for v in rng.integers(0, 256, 3)]
    return {"m": m, "o": o}, math.log2(len(PAL_M0)) + 2 * math.log2(len(PAL_M)) + 24


def _source(rng, W, H, vmax):
    while True:
        vx, vy = (int(v) for v in rng.integers(-vmax, vmax + 1, 2))
        if vx or vy:
            break
    src = {"x0": int(rng.integers(0, 2 * (W - 1))), "y0": int(rng.integers(0, 2 * (H - 1))),
           "vx": vx, "vy": vy}
    # (x0, vx) ~ (-x0, -vx) za TRI, pa je informacija nešto manja; procena
    bits = math.log2(2 * (W - 1)) + math.log2(2 * (H - 1)) + math.log2((2 * vmax + 1) ** 2 - 1)
    return src, bits


def _patch(rng, H, W, smax, vmax):
    h = int(rng.integers(4, smax + 1))
    w = int(rng.integers(4, smax + 1))
    y0 = int(rng.integers(0, H - h + 1))
    x0 = int(rng.integers(0, W - w + 1))
    data = rng.integers(0, vmax + 1, h * w).astype(np.int64)
    bits = (2 * math.log2(smax - 3) + math.log2(H - h + 1) + math.log2(W - w + 1)
            + h * w * math.log2(vmax + 1))
    return {"type": "patch", "y0": y0, "x0": x0, "h": h, "w": w, "data": data}, bits


def _audio(rng, kind):
    if kind == "smooth":
        k = int(rng.integers(2, 9))
    else:
        k = int(rng.integers(FIELD + 2, FIELD + 9))
    aud = {"k": k, "R": int(rng.integers(16, 65)), "base": int(rng.integers(16, 65)),
           "amp": int(rng.integers(40, 201))}
    return aud, math.log2(7) + 2 * math.log2(49) + math.log2(161)


def generate(rng, meta, level):
    H, W, Tn = meta.height, meta.width, meta.frames
    kind = level["kind"]
    pal, ideal = _palette(rng)
    if kind == "smooth":
        rule = SMOOTH_RULES[int(rng.integers(len(SMOOTH_RULES)))]
        sh = rule[3]
        src, b = _source(rng, W, H, 2)
        src["R2"] = int(rng.choice(DISK_R2))
        src["amp"] = int(rng.integers(16 << sh, (64 << sh) + 1))
        ideal += math.log2(len(SMOOTH_RULES)) + b + math.log2(len(DISK_R2)) + math.log2(48 << sh) + 1
        if rng.random() < 0.5:
            init = {"type": "hash", "seed": int(rng.integers(0, 1 << SEED_BITS))}
            ideal += SEED_BITS
        else:
            init, b = _patch(rng, H, W, 8, 255)
            ideal += b
    else:
        rule = WAVE_RULES[int(rng.integers(len(WAVE_RULES)))]
        sh = rule[3]
        src, b = _source(rng, W, H, 1)
        src["w"] = int(rng.integers(8, 41))
        src["amp"] = int(rng.integers(max(2, (1 << sh) // 4), (1 << sh) + 1))
        init, b2 = _patch(rng, H, W, 6, int(rng.integers(24, 49)))
        ideal += math.log2(len(WAVE_RULES)) + b + b2 + math.log2(33) + math.log2(1 << sh)
        # amplitude se smanjuju dok cur ne ostane u [-128, 127] (stanje vidljivo iz videa)
        for _ in range(40):
            S0 = _patch_field(init, H, W)
            cur = simulate("wave", rule, src, S0, Tn)
            if cur.min() >= -128 and cur.max() <= 127:
                break
            if src["amp"] > 1:
                src["amp"] = src["amp"] * 3 // 4
            else:
                init["data"] = init["data"] * 3 // 4
        else:
            raise RuntimeError("diffusion/wave: amplituda se nije smirila")
    aud = None
    if level["audio"] == "sums" and meta.audio_samples:
        aud, b = _audio(rng, kind)
        ideal += b
    p = float(level["p"])
    exc = None
    K = 0
    if p > 0:
        Ncells = max(0, Tn - 1) * H * W
        K = int(rng.binomial(Ncells, p))
        if K:
            pos = np.sort(rng.choice(Ncells, size=K, replace=False)).astype(np.int64)
            vals = rng.integers(1, 16, K).astype(np.int64)
            exc = kfl.Exceptions(pos, vals)
        ideal += Ncells * -(p * math.log2(p) + (1 - p) * math.log2(1 - p)) + K * math.log2(15)
    prog = program_for(kind, rule, src, init, pal, aud, meta, exc_state=exc)
    init_p = {k: (bytes(int(x) for x in v).hex() if k == "data" else v) for k, v in init.items()}
    params = {"kind": kind, "rule": list(rule), "src": src, "init": init_p, "palette": pal,
              "audio": aud, "exceptions": K, "p": p}
    return prog, {"ideal_bits": float(ideal), "params": params}


def _patch_field(init: dict, H: int, W: int) -> np.ndarray:
    F = np.full((H, W), init.get("bg", 0), np.int64)
    F[init["y0"]:init["y0"] + init["h"], init["x0"]:init["x0"] + init["w"]] = \
        np.asarray(init["data"], np.int64).reshape(init["h"], init["w"]) + init.get("off", 0)
    return F
