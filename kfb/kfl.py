"""KFL v1 — Kolmogorov Film Language: referentna mašina M benchmarka KFB.

Program u KFL-u je kratak bajt-kod koji deterministički proizvodi ceo film
(RGB24 frejmovi + 16-bitni mono PCM). Dužina programa u bajtovima je mera
opisa; dekoder (ovaj interpreter) je fiksiran i hešovan (MODEL_ID).

Specifikacija za ljude i LLM enkodere: kfb/KFL_SPEC.md. Ukratko:

  program = [flags]
            [INIT expr][STEP expr]          ako flags & 1 (stanje H x W)
            [VIDEO expr]                    uvek; vrednost & 255 po (t, y, x, c)
            [AUDIO expr]                    ako flags & 2; vrednost & 0xFFFF -> int16
            [izuzeci stanja][izuzeci videa][izuzeci zvuka]   ako flags & 4/8/16

  expr   = prefiksni bajt-kod nad int64 nizovima (broadcasting, wrap-around).

Sve operacije su celobrojne i definisane za sve ulaze (deljenje nulom = 0,
pomeraji se seku na [0, 63]); nema plutajućeg zareza u dekoderu.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from igc.film import Meta, get_varint, put_varint

# ------------------------------------------------------------- opkodovi
LEAVES = {"CONST": 0x00, "T": 0x01, "Y": 0x02, "X": 0x03, "C": 0x04, "N": 0x05,
          "S": 0x06, "SUMS": 0x07, "WID": 0x08, "HEI": 0x09, "NFR": 0x0A}
UNARY = {"NEG": 0x10, "ABS": 0x11, "NOT": 0x12, "ISIN": 0x13, "HASH": 0x14,
         "ISQRT": 0x15, "BOOL": 0x16, "SQ": 0x17, "NB8": 0x18, "NB4": 0x19}
BINARY = {"ADD": 0x20, "SUB": 0x21, "MUL": 0x22, "DIV": 0x23, "MOD": 0x24,
          "AND": 0x25, "OR": 0x26, "XOR": 0x27, "SHL": 0x28, "SHR": 0x29,
          "MIN": 0x2A, "MAX": 0x2B, "LT": 0x2C, "LE": 0x2D, "EQ": 0x2E, "NE": 0x2F,
          "ROLLY": 0x30, "ROLLX": 0x31, "TRI": 0x32}
TERNARY = {"WHERE": 0x38, "CLIP": 0x39}
LUT_OP = 0x3A
# Literali (arnost 1, podaci su u kodu): BITS -> bit[i] (0 van opsega),
# BYTES -> bajt[i mod L], WORDS -> uint16[i mod L] (LE). Daju efikasan zapis
# nekompresibilnih delova (npr. nasumična početna supa) bez gubitka kroz varint.
LITERALS = {"BITS": 0x3B, "BYTES": 0x3C, "WORDS": 0x3D}
MAX_LITERAL = 1 << 24
SMALL_POS = 0x80     # 0x80..0xBF -> konstante 0..63
SMALL_NEG = 0xC0     # 0xC0..0xCF -> konstante -1..-16

ARITY = {}
for _d, _a in ((LEAVES, 0), (UNARY, 1), (BINARY, 2), (TERNARY, 3)):
    for _k, _v in _d.items():
        ARITY[_v] = (_k, _a)
ARITY[LUT_OP] = ("LUT", 1)
for _k, _v in LITERALS.items():
    ARITY[_v] = (_k, 1)
NAME_TO_CODE = {**LEAVES, **UNARY, **BINARY, **TERNARY, "LUT": LUT_OP, **LITERALS}
INLINE = {"LUT", "BITS", "BYTES", "WORDS"}

SECTION_LEAVES = {
    "INIT": {"CONST", "Y", "X", "WID", "HEI", "NFR"},
    "STEP": {"CONST", "S", "T", "Y", "X", "WID", "HEI", "NFR"},
    "VIDEO": {"CONST", "T", "Y", "X", "C", "S", "WID", "HEI", "NFR"},
    "AUDIO": {"CONST", "N", "T", "SUMS", "WID", "HEI", "NFR"},
}
SPATIAL = {"NB8", "NB4", "ROLLY", "ROLLX"}

# Ograničenja resursa (deo specifikacije mašine)
MAX_NODES = 8192
MAX_DEPTH = 256
MAX_LUT = 65536
MAX_PIXELS = 512 * 512
MAX_FRAMES = 2048
MAX_SAMPLES = 1 << 24

ISIN_LUT = np.array([
    0, 3, 6, 9, 12, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 51, 54, 57, 60, 63, 65, 68,
    71, 73, 76, 78, 81, 83, 85, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 107, 109, 111, 112,
    113, 115, 116, 117, 118, 120, 121, 122, 122, 123, 124, 125, 125, 126, 126, 126, 127, 127, 127,
    127, 127, 127, 127, 126, 126, 126, 125, 125, 124, 123, 122, 122, 121, 120, 118, 117, 116, 115,
    113, 112, 111, 109, 107, 106, 104, 102, 100, 98, 96, 94, 92, 90, 88, 85, 83, 81, 78, 76, 73,
    71, 68, 65, 63, 60, 57, 54, 51, 49, 46, 43, 40, 37, 34, 31, 28, 25, 22, 19, 16, 12, 9, 6, 3,
    0, -3, -6, -9, -12, -16, -19, -22, -25, -28, -31, -34, -37, -40, -43, -46, -49, -51, -54, -57,
    -60, -63, -65, -68, -71, -73, -76, -78, -81, -83, -85, -88, -90, -92, -94, -96, -98, -100,
    -102, -104, -106, -107, -109, -111, -112, -113, -115, -116, -117, -118, -120, -121, -122,
    -122, -123, -124, -125, -125, -126, -126, -126, -127, -127, -127, -127, -127, -127, -127,
    -126, -126, -126, -125, -125, -124, -123, -122, -122, -121, -120, -118, -117, -116, -115,
    -113, -112, -111, -109, -107, -106, -104, -102, -100, -98, -96, -94, -92, -90, -88, -85, -83,
    -81, -78, -76, -73, -71, -68, -65, -63, -60, -57, -54, -51, -49, -46, -43, -40, -37, -34,
    -31, -28, -25, -22, -19, -16, -12, -9, -6, -3], dtype=np.int64)

F_STATE, F_AUDIO, F_EXC_STATE, F_EXC_VIDEO, F_EXC_AUDIO = 1, 2, 4, 8, 16


def zigzag(v: int) -> int:
    return (v << 1) if v >= 0 else ((-v << 1) - 1)


def unzigzag(u: int) -> int:
    return (u >> 1) if not u & 1 else -((u + 1) >> 1)


# ------------------------------------------------------------- AST
class Node:
    """Čvor izraza. Podržava Python operatore radi lakog pisanja programa."""
    __slots__ = ("op", "kids", "value", "table", "data", "nbits")

    def __init__(self, op: str, kids=(), value: int = 0, table=None, data: bytes | None = None,
                 nbits: int = 0):
        self.op = op
        self.kids = tuple(_wrap(k) for k in kids)
        self.value = int(value)
        self.table = None if table is None else tuple(int(v) for v in table)
        self.data = data          # BITS/BYTES/WORDS: sirovi bajtovi literala
        self.nbits = int(nbits)   # BITS: broj bitova

    # aritmetika
    def __add__(self, o): return Node("ADD", (self, o))
    def __radd__(self, o): return Node("ADD", (o, self))
    def __sub__(self, o): return Node("SUB", (self, o))
    def __rsub__(self, o): return Node("SUB", (o, self))
    def __mul__(self, o): return Node("MUL", (self, o))
    def __rmul__(self, o): return Node("MUL", (o, self))
    def __floordiv__(self, o): return Node("DIV", (self, o))
    def __mod__(self, o): return Node("MOD", (self, o))
    def __and__(self, o): return Node("AND", (self, o))
    def __or__(self, o): return Node("OR", (self, o))
    def __xor__(self, o): return Node("XOR", (self, o))
    def __lshift__(self, o): return Node("SHL", (self, o))
    def __rshift__(self, o): return Node("SHR", (self, o))
    def __neg__(self): return Node("NEG", (self,))
    def __invert__(self): return Node("NOT", (self,))
    def __lt__(self, o): return Node("LT", (self, o))
    def __le__(self, o): return Node("LE", (self, o))

    def __repr__(self) -> str:
        return to_text(self)

    def size(self) -> int:
        return 1 + sum(k.size() for k in self.kids)


def _wrap(v) -> Node:
    if isinstance(v, Node):
        return v
    if isinstance(v, (int, np.integer)):
        return Node("CONST", value=int(v))
    raise TypeError(f"ne mogu da pretvorim {v!r} u KFL izraz")


def const(v: int) -> Node:
    return Node("CONST", value=v)


def op(name: str, *kids) -> Node:
    return Node(name, kids)


def lut(table, idx) -> Node:
    return Node("LUT", (idx,), table=table)


def bits(bitarray, idx) -> Node:
    """Literal bitova (MSB-prvi u bajtu); vrednost je bit[idx] ili 0 van opsega."""
    b = np.asarray(bitarray, dtype=np.uint8)
    return Node("BITS", (idx,), data=np.packbits(b).tobytes(), nbits=len(b))


def bytes_lit(data: bytes, idx) -> Node:
    return Node("BYTES", (idx,), data=bytes(data))


def words_lit(values, idx) -> Node:
    return Node("WORDS", (idx,), data=np.asarray(values).astype("<u2").tobytes())


T, Y, X, C, N, S, SUMS = (Node(n) for n in ("T", "Y", "X", "C", "N", "S", "SUMS"))
WID, HEI, NFR = Node("WID"), Node("HEI"), Node("NFR")


def to_text(n: Node) -> str:
    if n.op == "CONST":
        return str(n.value)
    if not n.kids:
        return n.op
    if n.op == "LUT":
        return f"LUT[{len(n.table)}]({to_text(n.kids[0])})"
    if n.op == "BITS":
        return f"BITS[{n.nbits}b]({to_text(n.kids[0])})"
    if n.op in ("BYTES", "WORDS"):
        return f"{n.op}[{len(n.data)}B]({to_text(n.kids[0])})"
    return f"{n.op}(" + ", ".join(to_text(k) for k in n.kids) + ")"


# ------------------------------------------------------- serijalizacija
def emit(n: Node, buf: bytearray) -> None:
    if n.op == "CONST":
        v = n.value
        if 0 <= v < 64:
            buf.append(SMALL_POS + v)
        elif -16 <= v < 0:
            buf.append(SMALL_NEG + (-v - 1))
        else:
            buf.append(LEAVES["CONST"])
            put_varint(buf, zigzag(v))
        return
    code = NAME_TO_CODE[n.op]
    buf.append(code)
    if n.op == "LUT":
        put_varint(buf, len(n.table))
        for v in n.table:
            put_varint(buf, zigzag(v))
    elif n.op == "BITS":
        put_varint(buf, n.nbits)
        buf += n.data
    elif n.op == "BYTES":
        put_varint(buf, len(n.data))
        buf += n.data
    elif n.op == "WORDS":
        put_varint(buf, len(n.data) // 2)
        buf += n.data
    for k in n.kids:
        emit(k, buf)


def parse(data: bytes, pos: int, budget: list[int] | None = None, depth: int = 0) -> tuple[Node, int]:
    if budget is None:
        budget = [MAX_NODES]
    budget[0] -= 1
    if budget[0] < 0 or depth > MAX_DEPTH:
        raise ValueError("KFL: program prevelik ili preduboko ugnježden")
    if pos >= len(data):
        raise ValueError("KFL: skraćen program")
    b = data[pos]
    pos += 1
    if SMALL_POS <= b < SMALL_POS + 64:
        return Node("CONST", value=b - SMALL_POS), pos
    if SMALL_NEG <= b < SMALL_NEG + 16:
        return Node("CONST", value=-(b - SMALL_NEG) - 1), pos
    if b not in ARITY:
        raise ValueError(f"KFL: nepoznat opkod 0x{b:02x}")
    name, ar = ARITY[b]
    if name == "CONST":
        u, pos = get_varint(data, pos)
        v = unzigzag(u)
        if not -(1 << 63) <= v < (1 << 63):
            raise ValueError("KFL: konstanta van int64 opsega")
        return Node("CONST", value=v), pos
    table = None
    lit, nb = None, 0
    if name in LITERALS:
        L, pos = get_varint(data, pos)
        if not 1 <= L <= MAX_LITERAL:
            raise ValueError("KFL: neispravan literal")
        nbytes = (L + 7) // 8 if name == "BITS" else (L if name == "BYTES" else 2 * L)
        lit = bytes(data[pos:pos + nbytes])
        if len(lit) != nbytes:
            raise ValueError("KFL: skraćen literal")
        pos += nbytes
        nb = L
    if name == "LUT":
        L, pos = get_varint(data, pos)
        if not 1 <= L <= MAX_LUT:
            raise ValueError("KFL: neispravna LUT tabela")
        table = []
        for _ in range(L):
            u, pos = get_varint(data, pos)
            table.append(unzigzag(u))
    kids = []
    for _ in range(ar):
        k, pos = parse(data, pos, budget, depth + 1)
        kids.append(k)
    n = Node(name, kids)
    n.table = None if table is None else tuple(table)
    n.data, n.nbits = lit, (nb if name == "BITS" else 0)
    return n, pos


# ------------------------------------------------------------ izuzeci
@dataclass
class Exceptions:
    """Retke korekcije: pozicije (rastuće) i XOR vrednosti."""
    positions: np.ndarray
    values: np.ndarray | None = None   # None -> sve vrednosti su 1

    def to_bytes(self, vmode_hint: int | None = None) -> bytes:
        out = bytearray()
        pos = np.asarray(self.positions, dtype=np.int64)
        K = len(pos)
        put_varint(out, K)
        if K == 0:
            return bytes(out)
        if np.any(np.diff(pos) <= 0) or pos[0] < 0:
            raise ValueError("pozicije izuzetaka moraju biti strogo rastuće i nenegativne")
        gaps = np.diff(np.concatenate([[-1], pos])) - 1
        k = min(range(0, 48), key=lambda kk: int(((gaps >> kk) + 1 + kk).sum()))
        out.append(k)
        out += _rice_bits(gaps, k)
        vals = self.values
        if vals is None or np.all(np.asarray(vals) == 1):
            out.append(0)
        else:
            vals = np.asarray(vals, dtype=np.int64)
            if np.any(vals < 0):
                raise ValueError("XOR vrednosti moraju biti nenegativne")
            if vals.max() < 256:
                out.append(1)
                out += vals.astype(np.uint8).tobytes()
            elif vals.max() < 65536:
                out.append(2)
                out += vals.astype("<u2").tobytes()
            else:
                out.append(3)
                for v in vals.tolist():
                    put_varint(out, v)
        return bytes(out)

    @staticmethod
    def from_bytes(data: bytes, pos: int, limit: int) -> tuple["Exceptions", int]:
        K, pos = get_varint(data, pos)
        if K == 0:
            return Exceptions(np.zeros(0, np.int64)), pos
        if K > limit:
            raise ValueError("KFL: previše izuzetaka")
        k = data[pos]
        gaps, pos = _rice_read(data, pos + 1, K, k)
        positions = np.cumsum(gaps + 1) - 1
        if positions[-1] >= limit:
            raise ValueError("KFL: pozicija izuzetka van opsega")
        vmode = data[pos]
        pos += 1
        if vmode == 0:
            vals = None
        elif vmode == 1:
            vals = np.frombuffer(data[pos:pos + K], dtype=np.uint8).astype(np.int64)
            pos += K
        elif vmode == 2:
            vals = np.frombuffer(data[pos:pos + 2 * K], dtype="<u2").astype(np.int64)
            pos += 2 * K
        elif vmode == 3:
            vl = []
            for _ in range(K):
                v, pos = get_varint(data, pos)
                vl.append(v)
            vals = np.array(vl, dtype=np.int64)
        else:
            raise ValueError("KFL: nepoznat način vrednosti izuzetaka")
        if vals is not None and len(vals) != K:
            raise ValueError("KFL: skraćene vrednosti izuzetaka")
        return Exceptions(positions.astype(np.int64), vals), pos


def _rice_bits(gaps: np.ndarray, k: int) -> bytes:
    acc, nb, out = 0, 0, bytearray()
    for g in gaps.tolist():
        q = g >> k
        while q >= 32:
            acc = (acc << 32) | 0xFFFFFFFF
            nb += 32
            q -= 32
            while nb >= 8:
                nb -= 8
                out.append((acc >> nb) & 0xFF)
            acc &= (1 << nb) - 1
        acc = (acc << (q + 1)) | (((1 << q) - 1) << 1)
        nb += q + 1
        if k:
            acc = (acc << k) | (g & ((1 << k) - 1))
            nb += k
        while nb >= 8:
            nb -= 8
            out.append((acc >> nb) & 0xFF)
        acc &= (1 << nb) - 1
    if nb:
        out.append((acc << (8 - nb)) & 0xFF)
    return bytes(out)


def _rice_read(data: bytes, pos: int, K: int, k: int) -> tuple[np.ndarray, int]:
    bitpos = pos * 8
    end = len(data) * 8
    res = np.empty(K, dtype=np.int64)
    for i in range(K):
        q = 0
        while True:
            if bitpos >= end:
                raise ValueError("KFL: skraćen Rice tok")
            bit = (data[bitpos >> 3] >> (7 - (bitpos & 7))) & 1
            bitpos += 1
            if not bit:
                break
            q += 1
        r = 0
        for _ in range(k):
            if bitpos >= end:
                raise ValueError("KFL: skraćen Rice tok")
            r = (r << 1) | ((data[bitpos >> 3] >> (7 - (bitpos & 7))) & 1)
            bitpos += 1
        res[i] = (q << k) | r
    return res, (bitpos + 7) >> 3


# ------------------------------------------------------------ program
@dataclass
class Program:
    video: Node
    init: Node | None = None
    step: Node | None = None
    audio: Node | None = None
    exc_state: Exceptions | None = None
    exc_video: Exceptions | None = None
    exc_audio: Exceptions | None = None

    def to_bytes(self) -> bytes:
        flags = 0
        if self.init is not None or self.step is not None:
            if self.init is None or self.step is None:
                raise ValueError("INIT i STEP idu zajedno")
            flags |= F_STATE
        if self.audio is not None:
            flags |= F_AUDIO
        for f, e in ((F_EXC_STATE, self.exc_state), (F_EXC_VIDEO, self.exc_video),
                     (F_EXC_AUDIO, self.exc_audio)):
            if e is not None and len(e.positions):
                flags |= f
        if flags & F_EXC_STATE and not flags & F_STATE:
            raise ValueError("izuzeci stanja bez stanja")
        buf = bytearray([flags])
        if flags & F_STATE:
            emit(self.init, buf)
            emit(self.step, buf)
        emit(self.video, buf)
        if flags & F_AUDIO:
            emit(self.audio, buf)
        if flags & F_EXC_STATE:
            buf += self.exc_state.to_bytes()
        if flags & F_EXC_VIDEO:
            buf += self.exc_video.to_bytes()
        if flags & F_EXC_AUDIO:
            buf += self.exc_audio.to_bytes()
        return bytes(buf)

    @staticmethod
    def from_bytes(data: bytes, pos: int, meta: Meta) -> tuple["Program", int]:
        if pos >= len(data):
            raise ValueError("KFL: prazan program")
        flags = data[pos]
        pos += 1
        if flags & ~0x1F:
            raise ValueError("KFL: nepoznate zastavice")
        budget = [MAX_NODES]
        p = Program(video=None)
        if flags & F_STATE:
            p.init, pos = parse(data, pos, budget)
            p.step, pos = parse(data, pos, budget)
        p.video, pos = parse(data, pos, budget)
        if flags & F_AUDIO:
            p.audio, pos = parse(data, pos, budget)
        HW = meta.width * meta.height
        if flags & F_EXC_STATE:
            p.exc_state, pos = Exceptions.from_bytes(data, pos, max(0, meta.frames - 1) * HW)
        if flags & F_EXC_VIDEO:
            p.exc_video, pos = Exceptions.from_bytes(data, pos, meta.video_bytes)
        if flags & F_EXC_AUDIO:
            p.exc_audio, pos = Exceptions.from_bytes(data, pos, meta.audio_samples)
        if flags & F_EXC_STATE and not flags & F_STATE:
            raise ValueError("KFL: izuzeci stanja bez stanja")
        return p, pos

    def run(self, meta: Meta) -> bytes:
        return run(self, meta)


# ----------------------------------------------------------- evaluacija
def _check(node: Node, section: str, has_state: bool) -> None:
    allowed = SECTION_LEAVES[section]
    stack = [node]
    while stack:
        n = stack.pop()
        if not n.kids and n.op not in INLINE:
            if n.op not in allowed:
                raise ValueError(f"KFL: list {n.op} nije dozvoljen u sekciji {section}")
            if n.op in ("S", "SUMS") and not has_state:
                raise ValueError("KFL: S/SUMS bez stanja")
        if n.op in SPATIAL and section == "AUDIO":
            raise ValueError("KFL: prostorni operator u AUDIO sekciji")
        stack.extend(n.kids)


def _dynamic(n: Node, memo: dict) -> bool:
    k = id(n)
    if k not in memo:
        memo[k] = n.op in ("T", "S") or any(_dynamic(c, memo) for c in n.kids)
    return memo[k]


class _Eval:
    def __init__(self, env: dict, shape: tuple, spatial_axes: tuple, cache: dict | None, dyn: dict):
        self.env, self.shape, self.ax, self.cache, self.dyn = env, shape, spatial_axes, cache, dyn

    def full(self, a: np.ndarray) -> np.ndarray:
        return np.broadcast_to(a, np.broadcast_shapes(np.shape(a), self.shape))

    def ev(self, n: Node) -> np.ndarray:
        cache = self.cache
        if cache is not None and not self.dyn.get(id(n), True):
            r = cache.get(id(n))
            if r is None:
                r = self._ev(n)
                cache[id(n)] = r
            return r
        return self._ev(n)

    def _ev(self, n: Node) -> np.ndarray:
        o = n.op
        if o == "CONST":
            return np.int64(n.value)
        if not n.kids and o not in INLINE:
            return self.env[o]
        a = self.ev(n.kids[0])
        if o in UNARY:
            if o == "NEG":
                return np.negative(a)
            if o == "ABS":
                return np.abs(a)
            if o == "NOT":
                return np.invert(a)
            if o == "ISIN":
                return ISIN_LUT[np.bitwise_and(a, 255)]
            if o == "HASH":
                z = np.asarray(a).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
                z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                z = z ^ (z >> np.uint64(31))
                return (z >> np.uint64(1)).astype(np.int64)
            if o == "ISQRT":
                v = np.clip(a, 0, 1 << 52)
                r = np.floor(np.sqrt(v.astype(np.float64))).astype(np.int64)
                r = np.where(r * r > v, r - 1, r)
                r = np.where((r + 1) * (r + 1) <= v, r + 1, r)
                return r
            if o == "BOOL":
                return (np.asarray(a) != 0).astype(np.int64)
            if o == "SQ":
                return np.multiply(a, a)
            if o in ("NB8", "NB4"):
                f = self.full(a)
                ay, ax = self.ax
                s = (np.roll(f, 1, ay) + np.roll(f, -1, ay) + np.roll(f, 1, ax) + np.roll(f, -1, ax))
                if o == "NB8":
                    for dy in (1, -1):
                        for dx in (1, -1):
                            s = s + np.roll(np.roll(f, dy, ay), dx, ax)
                return s
        if o == "LUT":
            tab = np.array(n.table, dtype=np.int64)
            return tab[np.mod(a, len(tab))]
        if o == "BITS":
            bitsarr = np.unpackbits(np.frombuffer(n.data, dtype=np.uint8))[:n.nbits].astype(np.int64)
            ai = np.asarray(a)
            inside = (ai >= 0) & (ai < n.nbits)
            return np.where(inside, bitsarr[np.where(inside, ai, 0)], 0)
        if o == "BYTES":
            tab = np.frombuffer(n.data, dtype=np.uint8).astype(np.int64)
            return tab[np.mod(a, len(tab))]
        if o == "WORDS":
            tab = np.frombuffer(n.data, dtype="<u2").astype(np.int64)
            return tab[np.mod(a, len(tab))]
        b = self.ev(n.kids[1])
        if o in BINARY:
            if o == "ADD":
                return np.add(a, b)
            if o == "SUB":
                return np.subtract(a, b)
            if o == "MUL":
                return np.multiply(a, b)
            if o in ("DIV", "MOD"):
                z = np.asarray(b) == 0
                bb = np.where(z, 1, b)
                r = np.floor_divide(a, bb) if o == "DIV" else np.mod(a, bb)
                return np.where(z, 0, r)
            if o == "AND":
                return np.bitwise_and(a, b)
            if o == "OR":
                return np.bitwise_or(a, b)
            if o == "XOR":
                return np.bitwise_xor(a, b)
            if o in ("SHL", "SHR"):
                sh = np.clip(b, 0, 63)
                return np.left_shift(a, sh) if o == "SHL" else np.right_shift(a, sh)
            if o == "MIN":
                return np.minimum(a, b)
            if o == "MAX":
                return np.maximum(a, b)
            if o == "LT":
                return (np.asarray(a) < b).astype(np.int64)
            if o == "LE":
                return (np.asarray(a) <= b).astype(np.int64)
            if o == "EQ":
                return (np.asarray(a) == b).astype(np.int64)
            if o == "NE":
                return (np.asarray(a) != b).astype(np.int64)
            if o in ("ROLLY", "ROLLX"):
                bv = np.asarray(b)
                if bv.size != 1:
                    raise ValueError("KFL: pomeraj ROLL mora biti skalar")
                axis = self.ax[0] if o == "ROLLY" else self.ax[1]
                return np.roll(self.full(a), int(bv.reshape(-1)[0]), axis)
            if o == "TRI":
                L = np.asarray(b)
                pos = L > 0
                LL = np.where(pos, L, 1)
                m = np.mod(a, 2 * LL)
                return np.where(pos, LL - np.abs(m - LL), 0)
        c = self.ev(n.kids[2])
        if o == "WHERE":
            return np.where(np.asarray(a) != 0, b, c)
        if o == "CLIP":
            return np.minimum(np.maximum(a, b), c)
        raise ValueError(f"KFL: nepoznat operator {o}")


def check_meta(meta: Meta) -> None:
    if meta.width * meta.height > MAX_PIXELS or meta.frames > MAX_FRAMES:
        raise ValueError("KFL: film prevelik za mašinu")
    if meta.audio_samples > MAX_SAMPLES:
        raise ValueError("KFL: previše uzoraka zvuka")
    if meta.audio_samples and (meta.audio_channels != 1 or meta.sample_bytes != 2):
        raise ValueError("KFL: zvuk mora biti mono 16-bit")


def run(p: Program, meta: Meta) -> bytes:
    """Izvršava program i vraća payload (video bajtovi + PCM)."""
    check_meta(meta)
    H, W, Tn = meta.height, meta.width, meta.frames
    has_state = p.init is not None
    with np.errstate(all="ignore"):
        dyn: dict = {}
        if has_state:
            _check(p.init, "INIT", True)
            _check(p.step, "STEP", True)
        _check(p.video, "VIDEO", has_state)
        if p.audio is not None:
            _check(p.audio, "AUDIO", has_state)
        yy = np.arange(H, dtype=np.int64)
        xx = np.arange(W, dtype=np.int64)
        consts = {"WID": np.int64(W), "HEI": np.int64(H), "NFR": np.int64(Tn)}
        states = None
        if has_state and Tn > 0:
            states = np.empty((Tn, H, W), dtype=np.int64)
            env = {**consts, "Y": yy[:, None], "X": xx[None, :]}
            states[0] = np.broadcast_to(_Eval(env, (H, W), (0, 1), None, dyn).ev(p.init), (H, W))
            se = p.exc_state
            if se is not None and len(se.positions):
                per = H * W
                bounds = np.searchsorted(se.positions, np.arange(Tn) * per)
            for t in range(Tn - 1):
                env = {**consts, "S": states[t], "T": np.int64(t), "Y": yy[:, None], "X": xx[None, :]}
                nxt = np.array(np.broadcast_to(_Eval(env, (H, W), (0, 1), None, dyn).ev(p.step), (H, W)),
                               dtype=np.int64)
                if se is not None and len(se.positions):
                    lo, hi = bounds[t], bounds[t + 1]
                    if hi > lo:
                        idx = se.positions[lo:hi] - t * per
                        v = 1 if se.values is None else se.values[lo:hi]
                        flat = nxt.reshape(-1)
                        flat[idx] ^= v
                states[t + 1] = nxt
        video = np.empty((Tn, H, W, 3), dtype=np.uint8)
        memo: dict = {}
        _dynamic(p.video, memo)
        cache: dict = {}
        base = {**consts, "Y": yy[:, None, None], "X": xx[None, :, None],
                "C": np.arange(3, dtype=np.int64)[None, None, :]}
        for t in range(Tn):
            env = dict(base)
            env["T"] = np.int64(t)
            if has_state:
                env["S"] = states[t][:, :, None]
            v = _Eval(env, (H, W, 3), (0, 1), cache, memo).ev(p.video)
            video[t] = np.broadcast_to(np.bitwise_and(v, 255), (H, W, 3)).astype(np.uint8)
        ve = p.exc_video
        if ve is not None and len(ve.positions):
            flat = video.reshape(-1)
            vals = 1 if ve.values is None else (ve.values & 255).astype(np.uint8)
            flat[ve.positions] ^= vals
        S_ = meta.audio_samples
        audio = np.zeros(S_, dtype=np.int64)
        if p.audio is not None and S_:
            n = np.arange(S_, dtype=np.int64)
            tt = (n * meta.fps) // meta.audio_rate if meta.audio_rate else np.zeros(S_, np.int64)
            tt = np.clip(tt, 0, max(Tn - 1, 0))
            env = {**consts, "N": n, "T": tt}
            if has_state:
                env["SUMS"] = states.reshape(Tn, -1).sum(axis=1)[tt]
            audio = np.broadcast_to(_Eval(env, (S_,), (0, 0), None, dyn).ev(p.audio), (S_,))
        a16 = np.bitwise_and(audio, 0xFFFF).astype(np.uint16)
        ae = p.exc_audio
        if ae is not None and len(ae.positions):
            vals = 1 if ae.values is None else (ae.values & 0xFFFF).astype(np.uint16)
            a16 = a16.copy()
            a16[ae.positions] ^= vals
        return video.tobytes() + a16.astype("<u2").tobytes()
