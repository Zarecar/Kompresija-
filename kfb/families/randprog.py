"""Porodica RANDPROG: nasumični KFL VIDEO izrazi iz gramatike (bez stanja).

Uloga: "neviđena" porodica za ručno pisane inferencere. Ne postoji šablon;
film je izlaz slučajno izvučenog programa od približno n čvorova.

Gramatika (težine u zagradama):
    list    := T(3) | Y(2.5) | X(3) | C(1.5) | k(3), k u 1..63 | K(0.4), K u [64, 4096) log-uniformno
    unarni  := ISIN(8) | SQ(3) | HASH(0.7, retko)
    binarni := ADD(10) SUB(6) MUL(9) XOR(10) AND(6) OR(5) SHR(7) SHL(3) MOD(5)
               MIN(4) MAX(4) TRI(5) LT(3)
    ternarni:= WHERE(4)   (uslov je LT(..) sa verovatnoćom 0.6)
Desni argument SHR/SHL/MOD/TRI je najčešće mala konstanta (pomeraj 1..7 / 1..5,
modul i period 2..63). Broj čvorova je n-1, n ili n+1 (n = 5, 9, 15); podela
čvorova među decom je uniformna.

Odbacivanje (ponovno izvlačenje istim rng-om), na domenu D (mini: ceo film;
std: svi frejmovi, svaki 3. red i kolona; prolaz na D povlači prolaz na celom filmu za
sve provere osim udela najčešće vrednosti, koja se zato ponavlja na celom filmu):
    - manje od 16 različitih bajtova ili jedna vrednost > 50 % bajtova;
    - svi frejmovi isti (period 1 po T) ili je svaki frejm ravan (period 1 po Y i X);
    - podizraz bez promenljive, podizraz konstantan na D ili jednak svom detetu
      (npr. MIN(X, 50) na 32 kolone) -> takav program nije zbijen, pa K̄ ne bi bilo pošteno;
    - podizraz (ili niži bajt korena) jednak nekom strogo kraćem izrazu od <= 3 bajta
      (npr. WHERE(LT(X, T), T, X) == MAX(X, T)); provera na 128 fiksnih tačaka iz D;
    - zamena nekog čvora njegovim detetom (npr. WHERE(ISIN(T), a, b) == WHERE(T, a, b) na
      24 frejma), ili velike konstante jeftinijim konstantnim izrazom (-64..63 ili U(c), npr.
      HASH(53) umesto 1211 kad je bitan samo niži bajt), ne menja film na D; prvo 128 tačaka, pa ceo D.
AUDIO (nivoi n9, n15): tišina | ton ISIN((N*k) >> s) * a | testera
    (((N*k) >> s) & 63 - 32) * a | mali nasumični izraz nad N (bez čvora zamenljivog detetom).
K̄ = dužina zapisa izvučenog programa; ideal_bits = -log2 verovatnoće svih izbora
(gornja procena: odbacivanje samo povećava verovatnoću prihvaćenog programa).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from kfb import kfl
from kfb.kfl import N, Node, const, op

NAME = "randprog"
LEVELS = {
    "n5": {"nodes": 5, "audio": False},
    "n9": {"nodes": 9, "audio": True},
    "n15": {"nodes": 15, "audio": True},
}

LEAF_W = {"T": 3.0, "Y": 2.5, "X": 3.0, "C": 1.5, "k": 3.0, "K": 0.4}
UNARY_W = {"ISIN": 8.0, "SQ": 3.0, "HASH": 0.7}
BINARY_W = {"ADD": 10.0, "SUB": 6.0, "MUL": 9.0, "XOR": 10.0, "AND": 6.0, "OR": 5.0, "SHR": 7.0,
            "SHL": 3.0, "MOD": 5.0, "MIN": 4.0, "MAX": 4.0, "TRI": 5.0, "LT": 3.0}
TERNARY_W = {"WHERE": 4.0}
# desni argument je mala konstanta: (verovatnoća, najmanja, najveća)
CONST_RIGHT = {"SHR": (0.75, 1, 7), "SHL": (0.75, 1, 5), "MOD": (0.6, 2, 63), "TRI": (0.75, 2, 63)}
VARS = ("T", "Y", "X", "C")
AUDIO_W = {"silence": 0.25, "tone": 0.45, "saw": 0.15, "expr": 0.15}
MAX_ATTEMPTS = 20000


class _Sampler:
    """Izvori slučajnosti sa brojanjem informacije (-log2 p) svakog izbora."""

    def __init__(self, rng):
        self.rng = rng
        self.bits = 0.0

    def choice(self, weights: dict):
        keys = list(weights)
        w = np.array([weights[k] for k in keys], dtype=np.float64)
        p = w / w.sum()
        i = int(self.rng.choice(len(keys), p=p))
        self.bits -= math.log2(p[i])
        return keys[i]

    def integer(self, lo: int, hi: int) -> int:
        """Uniformno u [lo, hi]."""
        self.bits += math.log2(hi - lo + 1)
        return int(self.rng.integers(lo, hi + 1))

    def coin(self, p: float) -> bool:
        r = bool(self.rng.random() < p)
        self.bits -= math.log2(p if r else 1.0 - p)
        return r


def _has_var(n: Node, names=VARS) -> bool:
    return n.op in names or any(_has_var(k, names) for k in n.kids)


def _leaf(sm: _Sampler) -> Node:
    kind = sm.choice(LEAF_W)
    if kind == "k":
        return const(sm.integer(1, 63))
    if kind == "K":
        v = int(2.0 ** sm.rng.uniform(6.0, 12.0))        # log-uniformno u [64, 4096)
        # P(v) = (log2(v + 1) - log2(v)) / 6  ->  8..14 bita
        sm.bits -= math.log2((math.log2(v + 1) - math.log2(v)) / 6.0)
        return const(v)
    return Node(kind)


def _split(sm: _Sampler, total: int, parts: int) -> list[int]:
    """Uniformna kompozicija total = s_1 + ... + s_parts, s_i >= 1."""
    if parts == 1:
        return [total]
    cuts = sorted(sm.rng.choice(total - 1, size=parts - 1, replace=False).tolist())
    sm.bits += math.log2(math.comb(total - 1, parts - 1))
    b = [0] + [c + 1 for c in cuts] + [total]
    return [b[i + 1] - b[i] for i in range(parts)]


def _gen(sm: _Sampler, n: int, leaf_fn=_leaf, varnames=VARS, unary=UNARY_W, binary=BINARY_W,
         ternary=TERNARY_W) -> Node:
    """Nasumično stablo sa tačno n čvorova; svaki podizraz (n > 1) sadrži promenljivu.
    Bitovi odbačenih pokušaja se ne broje (-log2 P(prihvaćene putanje) ostaje gornja procena)."""
    b0 = sm.bits
    for _ in range(1000):
        sm.bits = b0
        node = _gen_once(sm, n, leaf_fn, varnames, unary, binary, ternary)
        if n == 1 or _has_var(node, varnames):
            return node
    raise RuntimeError("randprog: ne mogu da izvučem podizraz")


def _gen_once(sm, n, leaf_fn, varnames, unary, binary, ternary) -> Node:
    if n == 1:
        return leaf_fn(sm)
    ops = dict(unary)
    if n >= 3:
        ops.update(binary)
    if n >= 4:
        ops.update(ternary)
    name = sm.choice(ops)
    gen = lambda m: _gen(sm, m, leaf_fn, varnames, unary, binary, ternary)  # noqa: E731
    if name in unary:
        return Node(name, (gen(n - 1),))
    if name in ternary:
        if n >= 6 and sm.coin(0.6):
            s_l, s_a, s_b = _split(sm, n - 3, 3)          # uslov LT(l, r) ima >= 3 čvora
            s_r = 1
            if s_l > 1 and sm.coin(0.5):
                s_l, s_r = _split(sm, s_l + 1, 2)
            cond = Node("LT", (gen(s_l), gen(s_r)))
            return Node(name, (cond, gen(s_a), gen(s_b)))
        s = _split(sm, n - 1, 3)
        return Node(name, tuple(gen(m) for m in s))
    if name in CONST_RIGHT:
        p, lo, hi = CONST_RIGHT[name]
        if sm.coin(p):
            return Node(name, (gen(n - 2), const(sm.integer(lo, hi))))
    s = _split(sm, n - 1, 2)
    return Node(name, (gen(s[0]), gen(s[1])))


# ------------------------------------------------------- domen i provere
def _domain(meta):
    """Indeksi domena D: (T,1,1,1), (1,h,1,1), (1,1,w,1), (1,1,1,3)."""
    H, W, Tn = meta.height, meta.width, meta.frames
    step = 1 if H * W <= 64 * 64 else 3
    ys = np.arange(0, H, step, dtype=np.int64)
    xs = np.arange(0, W, step, dtype=np.int64)
    return {"T": np.arange(Tn, dtype=np.int64)[:, None, None, None], "Y": ys[None, :, None, None],
            "X": xs[None, None, :, None], "C": np.arange(3, dtype=np.int64)[None, None, None, :],
            "WID": np.int64(W), "HEI": np.int64(H), "NFR": np.int64(Tn)}


def _nodes(n: Node, acc: list) -> list:
    acc.append(n)
    for k in n.kids:
        _nodes(k, acc)
    return acc


def evaluate(n: Node, env: dict, shape) -> tuple[np.ndarray, list]:
    """Vrednost izraza nad env (broadcasting) normativnim evaluatorom VM-a (kfl.evaluate_all);
    vraća (vrednost, [(čvor, vrednost, vrednosti dece)] za unutrašnje čvorove)."""
    all_nodes = _nodes(n, [])
    r, vals = kfl.evaluate_all(n, env, shape, (1, 2))
    out = [(m, vals[id(m)], [vals[id(k)] for k in m.kids]) for m in all_nodes if m.kids]
    return r, out


_SHORT: dict = {}
_UN = ("NEG", "ABS", "NOT", "ISIN", "HASH", "ISQRT", "BOOL", "SQ")
_BIN = ("ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "MIN", "MAX",
        "LT", "LE", "EQ", "NE", "TRI")
_NS = 128
_R = (np.arange(1, _NS + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) | np.uint64(1)


def _hash(V: np.ndarray) -> np.ndarray:
    x = np.ascontiguousarray(V).view(np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(31)
    return (x * _R).sum(axis=-1, dtype=np.uint64)


def _short_bank(env: dict, shape) -> dict:
    """Heševi svih izraza od <= 3 bajta (bez prostornih i literala) na _NS fiksnih tačaka domena,
    sa najmanjom cenom; keš po obliku domena. Vrednosti računa normativni evaluator VM-a."""
    key = (shape, int(env["WID"]), int(env["HEI"]))
    if key in _SHORT:
        return _SHORT[key]
    size = int(np.prod(shape))
    idx = (np.arange(_NS, dtype=np.int64) * 2654435761) % size
    pts = {nm: np.broadcast_to(env[nm], shape).reshape(-1)[idx] for nm in VARS}
    lv1 = [pts[nm] for nm in VARS] + [np.full(_NS, v, np.int64) for v in
                                      (int(env["WID"]), int(env["HEI"]), int(env["NFR"]))]
    lv1 += [np.full(_NS, c, np.int64) for c in range(-16, 64)]
    A1 = np.stack(lv1)
    ev = lambda node, e: kfl.evaluate(node, e, (), (0, 0))  # noqa: E731
    A, B = Node("A"), Node("B")
    blocks = [(A1, 1)]
    with np.errstate(all="ignore"):
        lv2 = [np.broadcast_to(ev(Node(u, (A,)), {"A": A1}), A1.shape) for u in _UN]
        lv2.append(np.repeat(np.arange(-64, -16, dtype=np.int64)[:, None], _NS, 1))
        A2 = np.concatenate(lv2)
        blocks.append((A2, 2))
        blocks += [(np.broadcast_to(ev(Node(u, (A,)), {"A": A2}), A2.shape), 3) for u in _UN]
        for o in _BIN:
            r = ev(Node(o, (A, B)), {"A": A1[:, None, :], "B": A1[None, :, :]})
            blocks.append((np.broadcast_to(r, (len(A1), len(A1), _NS)).reshape(-1, _NS), 3))
    res = {"idx": idx}
    for name, f in (("full", lambda V: V), ("low", lambda V: V & 255)):
        h = np.concatenate([_hash(f(V)) for V, _ in blocks])
        c = np.concatenate([np.full(len(V), k, np.int64) for V, k in blocks])
        order = np.lexsort((c, h))
        h, c = h[order], c[order]
        first = np.r_[True, h[1:] != h[:-1]]
        res[name] = (h[first], c[first])
    _SHORT[key] = res
    return res


def _cost_of(n: Node) -> int:
    buf = bytearray()
    kfl.emit(n, buf)
    return len(buf)


def _reducible(out: list, env: dict, shape, root_low: np.ndarray) -> bool:
    """Da li je neki podizraz (ili niži bajt korena) jednak strogo kraćem izrazu od <= 3 bajta."""
    bank = _short_bank(env, shape)
    idx = bank["idx"]

    def short(V, table):
        h = _hash(V[None, :])
        hs, cs = bank[table]
        i = int(np.searchsorted(hs, h[0]))
        return int(cs[i]) if i < len(hs) and hs[i] == h[0] else None

    for node, r, _ in out:
        c = _cost_of(node)
        if c <= 1:
            continue
        cb = short(np.broadcast_to(r, shape).reshape(-1)[idx], "full")
        if cb is not None and cb < c:
            return True
    cb = short(np.broadcast_to(root_low, shape).reshape(-1)[idx].astype(np.int64), "low")
    return cb is not None and cb < _cost_of(out[0][0])


def _replace(n: Node, path: tuple, new: Node) -> Node:
    """Kopija stabla n u kojoj je čvor na putanji path zamenjen sa new."""
    if not path:
        return new
    kids = list(n.kids)
    kids[path[0]] = _replace(kids[path[0]], path[1:], new)
    return Node(n.op, kids)


_CONST2: list = []


def _cheaper_consts(v: int, cost: int) -> list:
    """Konstantni izrazi jeftiniji od konstante v: c u -64..63 i U(c) (2 B, npr. HASH(53)),
    ovi drugi samo ako se slažu sa v u nižih 6 bita (inače skoro nikad ne daju isti film)."""
    if not _CONST2:
        with np.errstate(all="ignore"):
            for u in _UN:
                for c in range(-16, 64):
                    nd = Node(u, (const(c),))
                    _CONST2.append((int(kfl.evaluate(nd, {}, (), (0, 0))), nd))
    out = [const(c) for c in range(-64, 64) if _cost_of(const(c)) < cost]
    if cost > 2:
        out += [nd for val, nd in _CONST2 if (val ^ v) & 63 == 0]
    return out


def _child_reducible(expr: Node, env: dict, shape, v: np.ndarray) -> bool:
    """Da li zamena nekog unutrašnjeg čvora njegovim detetom, ili velike konstante jeftinijim
    konstantnim izrazom (_cheaper_consts), daje isti film na D. Primeri: WHERE(ISIN(T), a, b)
    == WHERE(T, a, b) na 24 frejma; AND(MUL(202, e), 37) == AND(MUL(10, e), 37).
    Prvo _NS tačaka, pa ceo D."""
    idx = _short_bank(env, shape)["idx"]
    senv = {nm: np.broadcast_to(env[nm], shape).reshape(-1)[idx] for nm in VARS}
    senv.update({nm: env[nm] for nm in ("WID", "HEI", "NFR")})
    vs = v.reshape(-1)[idx]
    stack = [((), expr)]
    while stack:
        path, n = stack.pop()
        alts = []
        for i, k in enumerate(n.kids):
            stack.append((path + (i,), k))
            if path:                                      # koren: proverava video_ok
                alts.append(k)
        if n.op == "CONST" and _cost_of(n) > 1:
            alts += _cheaper_consts(n.value, _cost_of(n))
        for alt in alts:
            e2 = _replace(expr, path, alt)
            with np.errstate(all="ignore"):
                s2 = np.broadcast_to(kfl.evaluate(e2, senv, (_NS,), (0, 0)), (_NS,)) & 255
            if not np.array_equal(s2, vs):
                continue
            r2, _ = evaluate(e2, env, shape)
            if np.array_equal(np.broadcast_to(np.bitwise_and(r2, 255), shape), v):
                return True
    return False


def _tight(out: list, shape) -> bool:
    """Nijedan unutrašnji čvor nije konstantan niti jednak svom detetu na domenu."""
    for node, r, kids in out:
        rf = np.broadcast_to(r, shape)
        if rf.min() == rf.max():
            return False
        for kv in kids:
            if np.array_equal(rf, np.broadcast_to(kv, shape)):
                return False
    return True


def video_ok(expr: Node, env: dict) -> bool:
    """Provere nedegenerisanosti i zbijenosti (vidi docstring modula)."""
    shape = np.broadcast_shapes(env["T"].shape, env["Y"].shape, env["X"].shape, env["C"].shape)
    v, out = evaluate(expr, env, shape)
    if not out or not _tight(out, shape):          # sam list nikad nije dovoljan film
        return False
    v = np.broadcast_to(np.bitwise_and(v, 255), shape).astype(np.uint8)
    # koren: niži bajt ne sme biti isti kao kod deteta (npr. AND(e, 255))
    for kv in out[0][2]:
        if np.array_equal(v, np.broadcast_to(np.bitwise_and(kv, 255), shape)):
            return False
    if not _histogram_ok(v):
        return False
    if v.shape[0] > 1 and np.array_equal(v[1:], v[:-1]):
        return False
    if np.array_equal(v, np.broadcast_to(v[:, :1, :1, :], v.shape)):
        return False
    return not _reducible(out, env, shape, v) and not _child_reducible(expr, env, shape, v)


def _histogram_ok(v: np.ndarray) -> bool:
    cnt = np.bincount(v.reshape(-1), minlength=256)
    return np.count_nonzero(cnt) >= 16 and cnt.max() <= 0.5 * v.size


# -------------------------------------------------------------- zvuk
def _audio_leaf(sm: _Sampler) -> Node:
    if sm.coin(0.45):
        return N
    return const(sm.integer(1, 63))


AUDIO_UNARY = {"ISIN": 4.0}
AUDIO_BINARY = {"ADD": 4.0, "SUB": 2.0, "MUL": 6.0, "XOR": 4.0, "AND": 3.0, "OR": 2.0, "SHR": 5.0,
                "SHL": 3.0, "TRI": 2.0, "MOD": 2.0}


def _audio(sm: _Sampler, meta) -> tuple[Node | None, str]:
    kind = sm.choice(AUDIO_W)
    if kind == "silence" or meta.audio_samples == 0:
        return None, "silence"
    if kind in ("tone", "saw"):
        k, s = sm.integer(2, 63), sm.integer(0, 3)
        while s and not k & 1:                  # (N*2k) >> s == (N*k) >> (s-1): kanonski oblik
            k, s = k >> 1, s - 1
        arg = N if k == 1 else op("MUL", N, k)
        if s:
            arg = op("SHR", arg, s)
        if kind == "tone":
            return op("MUL", op("ISIN", arg), sm.integer(16, 250)), "tone"
        return op("MUL", op("SUB", op("AND", arg, 63), 32), sm.integer(64, 1000)), "saw"
    n = np.arange(meta.audio_samples, dtype=np.int64)
    b0 = sm.bits
    for _ in range(1000):
        sm.bits = b0                            # bitovi odbačenih izraza se ne broje
        e = _gen(sm, sm.integer(3, 7), _audio_leaf, ("N",), AUDIO_UNARY, AUDIO_BINARY, {})
        a, out = evaluate(e, {"N": n}, n.shape)
        a = np.broadcast_to(np.bitwise_and(a, 0xFFFF), n.shape)
        if len(np.unique(a)) >= 16 and _tight(out, n.shape) and not _audio_reducible(e, n, a):
            return e, "expr"
    raise RuntimeError("randprog: ne mogu da izvučem zvuk")


def _audio_reducible(e: Node, n: np.ndarray, a: np.ndarray) -> bool:
    """Da li zamena nekog čvora njegovim detetom daje isti zvuk (npr. SHR(XOR(12, ADD(N, N)), 5)
    == SHR(ADD(N, N), 5): XOR menja samo bitove koje SHR odbacuje)."""
    stack = [((), e)]
    while stack:
        path, nd = stack.pop()
        for i, k in enumerate(nd.kids):
            stack.append((path + (i,), k))
            r, _ = evaluate(_replace(e, path, k), {"N": n}, n.shape)
            if np.array_equal(np.broadcast_to(np.bitwise_and(r, 0xFFFF), n.shape), a):
                return True
    return False


# --------------------------------------------------------- generator
def generate(rng, meta, level):
    sm = _Sampler(rng)
    target = int(level["nodes"])
    env = _domain(meta)
    sub = env["Y"].size < meta.height or env["X"].size < meta.width
    for attempt in range(1, MAX_ATTEMPTS + 1):
        sm.bits = 0.0
        n = target + sm.integer(-1, 1)
        expr = _gen(sm, n)
        if not video_ok(expr, env):
            continue
        if not sub:
            break
        # D je poduzorak: udeo najčešće vrednosti proveri i na celom filmu (VM)
        full = np.frombuffer(kfl.run(kfl.Program(video=expr), replace(meta, audio_samples=0)), np.uint8)
        if _histogram_ok(full):
            break
    else:
        raise RuntimeError("randprog: previše odbačenih uzoraka")
    vbits = sm.bits
    audio, akind = (None, "silence")
    if level.get("audio"):
        sm.bits = 0.0
        audio, akind = _audio(sm, meta)
    prog = kfl.Program(video=expr, audio=audio)
    buf = bytearray()
    kfl.emit(expr, buf)
    params = {"nodes": expr.size(), "video_expr": kfl.to_text(expr), "video_expr_bytes": len(buf),
              "audio_kind": akind, "audio_expr": kfl.to_text(audio) if audio is not None else None,
              "attempts": attempt}
    return prog, {"ideal_bits": float(vbits + (sm.bits if level.get("audio") else 0.0)), "params": params}
