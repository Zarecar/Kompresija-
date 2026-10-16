"""Generička enumerativna pretraga KFL izraza (bottom-up po dužini, posmatračka ekvivalencija).

family_aware=False: ne pretpostavlja ništa o porodici. Traži najkraći VIDEO izraz (bez
stanja) koji tačno daje film i, nezavisno, AUDIO izraz; ako tačnog nema, bira po MDL-u
između (približan izraz + izuzeci) i poštene rezerve (literal nad periodičnom kutijom).

PROSTOR PRETRAGE (cena = dužina serijalizovanog izraza u bajtovima)
  VIDEO listovi: T, Y, X, C -- samo promenljive od kojih film zavisi (provera na celom filmu:
                 ako su svi frejmovi isti, T se izostavlja; isto za redove Y, kolone X, kanale C);
                 WID, HEI, NFR; konstante -16..63 (1 B) i -64..-17 (2 B: CONST + varint).
  AUDIO listovi: N, T (= N*fps // audio_rate), WID, HEI, NFR i iste konstante.
  unarni (8):    NEG ABS NOT ISIN HASH ISQRT BOOL SQ
  binarni (17):  ADD SUB MUL DIV MOD AND OR XOR SHL SHR MIN MAX LT LE EQ NE TRI
                 (komutativni ADD MUL AND OR XOR MIN MAX EQ NE samo u jednom redosledu)
  ternarni (2):  WHERE (uslov: izraz sa mešovitim nula/nenula obrascem na uzorku, po jedan
                 predstavnik obrasca), CLIP (prvi argument nije konstanta; bez lo >= hi konstanti)
  Isključeno: NB8/NB4/ROLLY/ROLLX (prostorni), S/SUMS (stanje), LUT/BITS/BYTES/WORDS (samo u
  rezervi). Veće konstante nastaju kompozicijom (npr. MUL(8, 25)) ili rešavanjem korena.
UZORAK I EKVIVALENCIJA
  Video: P = 192 (mini) / 256 (std) tačaka (t, y, x, c); svaka osa pokrivena ravnomerno
  (permutovani ponovljeni indeksi, fiksan seed). Zvuk: 128 različitih nasumičnih N.
  Posmatračka ekvivalencija (OE): od izraza sa istim int64 vektorom na uzorku (64-bitni heš)
  čuva se samo prvi, tj. najkraći. Kandidat koji se slaže na celom uzorku (niži bajt; za zvuk
  16 bita) proverava se na 4096 dodatnih tačaka, pa na celom filmu (VM). Netačan zapis se nikad
  ne vraća: i konačni zapis se dekoduje (record.decode) i poredi sa payload-om.
FAZE (redom, dok traje budžet; uvek se čuva najbolji tačan kandidat)
  1. banka: svi izrazi cene 1..B, B <= 4 (najviše MAX_BANK izraza), sa punim vrednostima;
  2. posle svakog nivoa: susret u sredini ADD/SUB/XOR(e1, e2), e1 i e2 iz banke (cena do 2B+1),
     preko heš indeksa nižeg bajta (16 bita za zvuk) na prvih 32 tačke, i koren sa rešenom
     konstantom ADD/SUB/XOR/MUL/AND/OR(e, c), e iz banke, c bilo koja konstanta iz uzorka
     (MUL: inverz neparne vrednosti mod 2^nb);
  3. razlaganje WHERE(c, a, b): a se slaže gde je c != 0, b gde je c == 0; jedna strana pokriva
     >= P/2 tačaka (skupovi pogodaka kao bit-maske), pa je trošak linearan u veličini banke;
  4. slojevi cene B+1 .. 2B+1 bez čuvanja, redom po ceni: oblici U(e), B(x, y), U(B(x, y)) i
     CLIP nad bankom (U unarni, B binarni; ADD/SUB/XOR kao koren pokriva faza 2); kaskadni filtar
     tačku po tačku na Q = 8 proba, pa ceo uzorak; na ceni B+1 i koren sa rešenom konstantom
     (+, -, ^, *) nad svakim oblikom;
  5. redukcije korena (pokušaji pre obične pretrage, ako cilj ima potpis): video SHL(e, z) kad svi
     bajtovi imaju z nižih nula (traži se f >> z na 8 - z bita); zvuk MUL(e, d) kad svi označeni
     uzorci imaju NZD d > 1 (traži se f / d);
  6. MDL: do TOPK približnih izraza (najviše pogodaka na uzorku) + izuzeci (XOR vrednosti),
     naspram BYTES[..](indeks) nad najmanjom periodičnom kutijom po (T, Y, X, C), odnosno
     WORDS[p](N) nad najmanjim periodom zvuka; tišina ako je zvuk sav nula. Bira se najkraći
     kompletan zapis.
  7. generička rezerva: bz2 i lzma (igc) ako programski zapis nije kratak (> GENERIC_MIN B);
     bira se najkraći tačan zapis (svaki se dekoduje i poredi).
Budžet (od poziva): rezerva za lzma (~1,5 s po MB, najviše 15 %) se oduzima; od ostatka video do
60 % (90 % ako je zvuk tišina; kad ima zvuka, posle prvog tačnog videa ostaje mu samo FOUND_SHARE
preostalog roka, da zvuk ne ostane bez vremena), zatim zvuk do 90 %; ostatak za MDL, rezervu i
završnu proveru zapisa.
"""
from __future__ import annotations

import time
from bisect import bisect_right
from dataclasses import replace

import numpy as np

from igc import codec as igc_codec
from kfb import kfl, record
from kfb.kfl import Node, bytes_lit, const, op, words_lit

from . import Encoder

UNARY = ("NEG", "ABS", "NOT", "ISIN", "HASH", "ISQRT", "BOOL", "SQ")
BINARY = ("ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "MIN", "MAX",
          "LT", "LE", "EQ", "NE", "TRI")
TERNARY = ("WHERE", "CLIP")
COMM = frozenset({"ADD", "MUL", "AND", "OR", "XOR", "MIN", "MAX", "EQ", "NE"})
BOOLEAN = frozenset({"LT", "LE", "EQ", "NE", "BOOL"})
MITM = frozenset({"ADD", "SUB", "XOR"})
CONST_CODE = -1
MAX_B = 4
MAX_BANK = 900_000
Q_PROBE = 8
Q_INDEX = 32
N_VALID = 4096
CHUNK = 1 << 19               # elemenata po bloku evaluacije (~4 MB int64, staje u keš)
TOPK = 6
SEED = 0x6B6662               # fiksan seed uzorka i heša (enkoder je deterministički)
FOUND_SHARE = 0.3             # posle prvog tačnog videa: udeo preostalog vremena videa (ostalo za zvuk)
LZMA_BPS = 1.0e6              # gruba procena brzine lzma rezerve (bajtova u sekundi)
GENERIC_MIN = 64              # programski zapis kraći od ovoga: generička rezerva se ne pokušava
_HK = np.uint64(0x9E3779B97F4A7C15)


# ------------------------------------------------------------ operacije
def un(o: str, a):
    """Unarna operacija VM-a, elementwise nad int64 (ista semantika kao kfl._Eval)."""
    if o == "NEG":
        return np.negative(a)
    if o == "ABS":
        return np.abs(a)
    if o == "NOT":
        return np.invert(a)
    if o == "ISIN":
        return kfl.ISIN_LUT[np.bitwise_and(a, 255)]
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
        return np.where((r + 1) * (r + 1) <= v, r + 1, r)
    if o == "BOOL":
        return (np.asarray(a) != 0).astype(np.int64)
    if o == "SQ":
        return np.multiply(a, a)
    raise ValueError(o)


def bi(o: str, a, b):
    """Binarna operacija VM-a (bez ROLLY/ROLLX)."""
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
    if o == "TRI":
        L = np.asarray(b)
        pos = L > 0
        LL = np.where(pos, L, 1)
        m = np.mod(a, 2 * LL)
        return np.where(pos, LL - np.abs(m - LL), 0)
    raise ValueError(o)


def ter(o: str, a, b, c):
    if o == "WHERE":
        return np.where(np.asarray(a) != 0, b, c)
    if o == "CLIP":
        return np.minimum(np.maximum(a, b), c)
    raise ValueError(o)


def evaluate(n: Node, env: dict):
    """Vrednost izraza (bez prostornih operatora) nad tačkama iz env."""
    o = n.op
    if o == "CONST":
        return np.int64(n.value)
    if not n.kids:
        return env[o]
    k = [evaluate(c, env) for c in n.kids]
    if len(k) == 1:
        return un(o, k[0])
    if len(k) == 2:
        return bi(o, k[0], k[1])
    return ter(o, k[0], k[1], k[2])


# ------------------------------------------------------- pomoćne funkcije
def ccost(v: int) -> int:
    """Broj bajtova konstante v u KFL-u."""
    if -16 <= v < 64:
        return 1
    z, n = kfl.zigzag(int(v)), 1
    while z >= 128:
        z >>= 7
        n += 1
    return 1 + n


def best_rep(r: int, nb: int) -> tuple[int, int]:
    """Najjeftinija konstanta sa istih nb nižih bitova kao r (r u [0, 2^nb))."""
    c = min((r, r - (1 << nb)), key=lambda v: (ccost(v), abs(v)))
    return c, ccost(c)


def nbytes(n: Node) -> int:
    buf = bytearray()
    kfl.emit(n, buf)
    return len(buf)


def _rowhash(V: np.ndarray, R: np.ndarray) -> np.ndarray:
    """64-bitni heš svakog reda (n, P) int64."""
    x = np.ascontiguousarray(V).view(np.uint64) * _HK
    x ^= x >> np.uint64(29)
    x *= R
    return x.sum(axis=-1, dtype=np.uint64)


def _inv_odd(x: np.ndarray, mask: int) -> np.ndarray:
    """Inverz neparnih x po modulu mask+1 (Njutnova iteracija)."""
    y = x.copy()
    for _ in range(5):
        y = (y * (2 - x * y)) & mask
    return y


def exc_size(K: int, gaps: np.ndarray | None, vbytes: int) -> int:
    """Tačna dužina bloka izuzetaka (kfl.Exceptions.to_bytes) bez serijalizacije."""
    out = bytearray()
    kfl.put_varint(out, K)
    if K == 0:
        return len(out)
    best = None
    for k in range(24):                 # pozicije < 2^24, pa veći k nikad nije bolji
        bits = int(((gaps >> k) + 1 + k).sum())
        if best is None or bits < best:
            best = bits
    return len(out) + 1 + (best + 7) // 8 + 1 + vbytes * K


# ------------------------------------------------------------ banka
class _Bank:
    """Banka izraza po ceni (nivo k = cena k) sa posmatračkom ekvivalencijom na uzorku."""

    def __init__(self, leaves, tgt, nb, deadline, check, rng):
        self.P = len(tgt)
        self.tgt = tgt
        self.nb = nb
        self.mask = (1 << nb) - 1
        self.deadline = deadline
        self.check = check              # check(node) -> bool (validacija + VM)
        r = rng.integers(0, 1 << 62, self.P, dtype=np.int64).astype(np.uint64)
        self.R = (r << np.uint64(1)) | np.uint64(1)                      # neparni množioci heša
        self.R2 = self.R[:Q_INDEX]
        self.V = [np.zeros((0, self.P), np.int64)]
        self.code = [np.zeros(0, np.int16)]
        self.kids = [np.zeros((0, 3), np.int64)]
        self.cv = [np.zeros(0, np.int64)]
        self.off = [0, 0]
        self.seen = np.zeros(0, np.uint64)
        self.best_cost = 1 << 30
        self.best = None
        self.approx: list = []          # (pogoci, -cena, gid)
        self.near: list = []            # izrazi tačni na uzorku koji nisu prošli proveru (za MDL)
        self.leaves = leaves
        self.timed_out = False
        self.found_share = None         # skraćuje rok posle prvog tačnog kandidata (vidi offer)
        self.found_at = None
        self.tvals = set(np.unique(tgt & self.mask).tolist())
        half = 1 << (nb - 1)
        sgn = ((tgt & self.mask) ^ half) - half              # cilj kao označen nb-bitni broj
        self.isin_ok = nb < 8 or bool(np.all(np.abs(sgn) <= 127))
        self.sq_ok = set(((np.arange(1 << min(nb, 16), dtype=np.int64) ** 2) & self.mask).tolist()) \
            if nb <= 16 else None

    # --------------------------------------------------------- osnovno
    def timeout(self) -> bool:
        if time.perf_counter() > self.deadline:
            self.timed_out = True
        return self.timed_out

    @property
    def total(self) -> int:
        return self.off[-1]

    def level_of(self, g: int) -> int:
        return bisect_right(self.off, g) - 1

    def node(self, g: int) -> Node:
        k = self.level_of(g)
        i = g - self.off[k]
        return self._mk(int(self.code[k][i]), self.kids[k][i], int(self.cv[k][i]))

    def _mk(self, code: int, kids, cv: int) -> Node:
        if code == CONST_CODE:
            return const(cv)
        name, ar = kfl.ARITY[code]
        if ar == 0:
            return Node(name)
        return Node(name, tuple(self.node(int(kids[j])) for j in range(ar)))

    def offer(self, cost: int, mk) -> bool:
        """Kandidat koji se slaže na uzorku: proveri i zapamti ako je jeftiniji."""
        if cost >= self.best_cost:
            return False
        nd = mk()
        if self.check(nd):
            self.best_cost, self.best = cost, nd
            if self.found_share is not None and self.found_at is None:
                now = time.perf_counter()
                self.found_at = now
                self.deadline = min(self.deadline, now + self.found_share * max(0.0, self.deadline - now))
            return True
        if len(self.near) < TOPK:
            self.near.append(nd)
        return False

    # ----------------------------------------------------- dodavanje
    def _begin(self, k: int) -> None:
        self.cur = k
        self.pend = {"V": [], "code": [], "kids": [], "cv": [], "h": []}
        self.cnt = 0
        self.lvl_set: set = set()

    def _add(self, R, code, kids, cv=None) -> None:
        """R (m, P) vrednosti; code skalar ili niz; kids (m, 3) globalni id-jevi dece."""
        m = len(R)
        if not m or self.total + self.cnt >= MAX_BANK:
            return
        h = _rowhash(R, self.R)
        if self.seen.size:
            pos = np.searchsorted(self.seen, h)
            pos[pos == self.seen.size] = 0
            fresh = np.flatnonzero(self.seen[pos] != h)
        else:
            fresh = np.arange(m)
        if not fresh.size:
            return
        _, first = np.unique(h[fresh], return_index=True)
        idx = fresh[np.sort(first)]
        S = self.lvl_set
        keep = [i for i, hv in zip(idx.tolist(), h[idx].tolist()) if hv not in S]
        if not keep:
            return
        keep = np.asarray(keep, dtype=np.int64)
        hk = h[keep]
        S.update(hk.tolist())
        Rk = np.ascontiguousarray(R[keep])
        ck = np.full(len(keep), code, np.int16) if np.isscalar(code) else np.asarray(code)[keep]
        kk = np.asarray(kids)[keep]
        vk = np.zeros(len(keep), np.int64) if cv is None else np.asarray(cv, np.int64)[keep]
        base = self.off[self.cur] + self.cnt
        P = self.pend
        P["V"].append(Rk)
        P["code"].append(ck)
        P["kids"].append(kk)
        P["cv"].append(vk)
        P["h"].append(hk)
        self.cnt += len(keep)
        eq = (Rk & self.mask) == self.tgt
        hits = eq.sum(axis=1)
        full = np.flatnonzero(hits == self.P)
        for i in full[:64].tolist():
            self.offer(self.cur, lambda i=i: self._mk(int(ck[i]), kk[i], int(vk[i])))
        # približni kandidati za MDL
        top = np.argsort(-hits, kind="stable")[:TOPK]
        for i in top.tolist():
            self.approx.append((int(hits[i]), -self.cur, base + i))
        self.approx.sort(reverse=True)
        del self.approx[TOPK:]

    def _end(self) -> None:
        P = self.pend
        if P["V"]:
            V = np.concatenate(P["V"])
            code = np.concatenate(P["code"])
            kids = np.concatenate(P["kids"])
            cv = np.concatenate(P["cv"])
            self.seen = np.union1d(self.seen, np.concatenate(P["h"]))
        else:
            V = np.zeros((0, self.P), np.int64)
            code, kids, cv = np.zeros(0, np.int16), np.zeros((0, 3), np.int64), np.zeros(0, np.int64)
        self.V.append(V)
        self.code.append(code)
        self.kids.append(kids)
        self.cv.append(cv)
        self.off.append(self.off[-1] + len(V))
        self.pend = None

    # --------------------------------------------------------- nivoi
    def build_level(self, k: int) -> bool:
        """Gradi nivo k; vraća True ako je potpun (nije prekinut vremenom ni veličinom)."""
        self._begin(k)
        try:
            if k == 1:
                t0 = int(self.tgt[0]) & self.mask
                if np.all((self.tgt & self.mask) == t0):          # konstantan cilj
                    c, cc = best_rep(t0, self.nb)
                    self.offer(cc, lambda c=c: const(c))
                for nd, vals in self.leaves:
                    v = np.broadcast_to(np.asarray(vals, np.int64), (self.P,))[None, :]
                    if nd.op == "CONST":
                        self._add(v, CONST_CODE, np.full((1, 3), -1), [nd.value])
                    else:
                        self._add(v, kfl.NAME_TO_CODE[nd.op], np.full((1, 3), -1))
                for c in range(-16, 64):
                    self._add(np.full((1, self.P), c, np.int64), CONST_CODE, np.full((1, 3), -1), [c])
                return True
            if k == 2:
                cs = np.arange(-64, -16, dtype=np.int64)
                self._add(np.repeat(cs[:, None], self.P, axis=1), CONST_CODE, np.full((len(cs), 3), -1), cs)
            self._unary(k)
            for a in range(1, k - 1):
                if not self._binary(a, k - 1 - a):
                    return False
            if k >= 4 and not self._ternary(k):
                return False
            return not self.timeout() and self.total + self.cnt < MAX_BANK
        finally:
            self._end()

    def _unary(self, k: int) -> None:
        Vp = self.V[k - 1]
        if not len(Vp):
            return
        ids = np.arange(len(Vp)) + self.off[k - 1]
        kids = np.stack([ids, np.full_like(ids, -1), np.full_like(ids, -1)], 1)
        step = max(1, CHUNK // self.P)
        with np.errstate(all="ignore"):
            for name in UNARY:
                for i0 in range(0, len(Vp), step):
                    self._add(un(name, Vp[i0:i0 + step]), kfl.NAME_TO_CODE[name], kids[i0:i0 + step])
                    if self.timeout():
                        return

    def _binary(self, a: int, b: int) -> bool:
        Va, Vb = self.V[a], self.V[b]
        na, nb_ = len(Va), len(Vb)
        if not na or not nb_:
            return True
        P = self.P
        jb = min(nb_, max(1, CHUNK // P))
        ib = max(1, CHUNK // (P * jb))
        with np.errstate(all="ignore"):
            for name in BINARY:
                comm = name in COMM
                if comm and a > b:
                    continue
                code = kfl.NAME_TO_CODE[name]
                for i0 in range(0, na, ib):
                    i1 = min(na, i0 + ib)
                    for j0 in range(0, nb_, jb):
                        j1 = min(nb_, j0 + jb)
                        if comm and a == b and j1 <= i0:
                            continue
                        R = bi(name, Va[i0:i1, None, :], Vb[None, j0:j1, :]).reshape(-1, P)
                        ii = np.repeat(np.arange(i0, i1), j1 - j0)
                        jj = np.tile(np.arange(j0, j1), i1 - i0)
                        if comm and a == b:
                            s = jj >= ii
                            R, ii, jj = R[s], ii[s], jj[s]
                        kids = np.stack([ii + self.off[a], jj + self.off[b], np.full_like(ii, -1)], 1)
                        self._add(R, code, kids)
                        if self.timeout() or self.total + self.cnt >= MAX_BANK:
                            return False
        return True

    def conds(self, k: int) -> np.ndarray:
        """Lokalni indeksi nivoa k pogodni za uslov WHERE (mešovit obrazac, jedan po obrascu)."""
        V = self.V[k]
        if not len(V):
            return np.zeros(0, np.int64)
        nz = V != 0
        mixed = np.flatnonzero(nz.any(1) & ~nz.all(1))
        if not mixed.size:
            return mixed
        h = _rowhash(nz[mixed].astype(np.int64), self.R)
        _, first = np.unique(h, return_index=True)
        return mixed[np.sort(first)]

    def is_const(self, k: int) -> np.ndarray:
        V = self.V[k]
        return (V == V[:, :1]).all(1) if len(V) else np.zeros(0, bool)

    def _ternary(self, k: int) -> bool:
        P = self.P
        with np.errstate(all="ignore"):
            for sc in range(1, k - 2):
                for sa in range(1, k - 1 - sc):
                    sb = k - 1 - sc - sa
                    # WHERE(c, a, b)
                    ci = self.conds(sc)
                    Va, Vb = self.V[sa], self.V[sb]
                    if ci.size and len(Va) and len(Vb):
                        jb = min(len(Vb), max(1, CHUNK // P))
                        ib = max(1, CHUNK // (P * jb))
                        for c in ci.tolist():
                            cnd = self.V[sc][c] != 0
                            for i0 in range(0, len(Va), ib):
                                i1 = min(len(Va), i0 + ib)
                                for j0 in range(0, len(Vb), jb):
                                    j1 = min(len(Vb), j0 + jb)
                                    R = np.where(cnd, Va[i0:i1, None, :], Vb[None, j0:j1, :]).reshape(-1, P)
                                    ii = np.repeat(np.arange(i0, i1), j1 - j0)
                                    jj = np.tile(np.arange(j0, j1), i1 - i0)
                                    kids = np.stack([np.full_like(ii, c + self.off[sc]), ii + self.off[sa],
                                                     jj + self.off[sb]], 1)
                                    self._add(R, kfl.NAME_TO_CODE["WHERE"], kids)
                                    if self.timeout() or self.total + self.cnt >= MAX_BANK:
                                        return False
                    # CLIP(a, lo, hi): a = nivo sc, lo = sa, hi = sb
                    if not self._clip(sc, sa, sb):
                        return False
        return True

    def _clip(self, s0: int, s1: int, s2: int) -> bool:
        Va, Vl, Vh = self.V[s0], self.V[s1], self.V[s2]
        if not (len(Va) and len(Vl) and len(Vh)):
            return True
        P = self.P
        ca, cl, ch = self.is_const(s0), self.is_const(s1), self.is_const(s2)
        lo0, hi0 = Vl[:, 0], Vh[:, 0]
        jb = min(len(Vh), max(1, CHUNK // P))
        ib = max(1, CHUNK // (P * jb))
        code = kfl.NAME_TO_CODE["CLIP"]
        for a in np.flatnonzero(~ca).tolist():              # CLIP(konstanta, ., .) se ne gradi
            for i0 in range(0, len(Vl), ib):
                i1 = min(len(Vl), i0 + ib)
                for j0 in range(0, len(Vh), jb):
                    j1 = min(len(Vh), j0 + jb)
                    ii = np.repeat(np.arange(i0, i1), j1 - j0)
                    jj = np.tile(np.arange(j0, j1), i1 - i0)
                    both = cl[ii] & ch[jj]
                    # konstantno po konstrukciji: lo >= hi (obe konstante)
                    s = ~(both & (lo0[ii] >= hi0[jj]))
                    if not s.any():
                        continue
                    ii, jj = ii[s], jj[s]
                    R = np.minimum(np.maximum(Va[a][None, :], Vl[ii]), Vh[jj])
                    kids = np.stack([np.full_like(ii, a + self.off[s0]), ii + self.off[s1], jj + self.off[s2]], 1)
                    self._add(R, code, kids)
                    if self.timeout() or self.total + self.cnt >= MAX_BANK:
                        return False
        return True

    # ------------------------------------------------- slojevi > B (probe)
    def _prep_probe(self) -> None:
        Q = min(Q_PROBE, self.P)
        self.Qt = [np.ascontiguousarray(V[:, :Q].T) for V in self.V]       # (Q, n) po nivou
        self.tq = self.tgt[:Q] & self.mask
        self.only01 = self.tvals <= {0, 1}
        self.sqok = self.sq_ok is not None and self.tvals <= self.sq_ok
        x = np.arange(1 << self.nb, dtype=np.int64)
        self.inv = np.where(x & 1, _inv_odd(x | 1, self.mask), 0)

    def _root_ok(self, name: str) -> bool:
        """Koren čiji izlaz ne može da da cilj (npr. LT kad cilj nije 0/1) se preskače."""
        return not ((name in BOOLEAN and not self.only01) or (name == "SQ" and not self.sqok)
                    or (name == "ISIN" and not self.isin_ok))

    def _cascade(self, ii, jj, at, full, mk, cost) -> None:
        """Pogoci na probi 0 -> ostale probe -> ceo uzorak -> provera (offer)."""
        m, tq = self.mask, self.tq
        for p in range(1, len(tq)):
            if not ii.size:
                return
            s = (at(p, ii, jj) & m) == tq[p]
            ii, jj = ii[s], jj[s]
        if not ii.size or cost >= self.best_cost:
            return
        ii, jj = ii[:20000], jj[:20000]
        ok = np.flatnonzero(((full(ii, jj) & m) == self.tgt).all(1))
        for t in ok[:64].tolist():
            self.offer(cost, lambda t=t: mk(int(ii[t]), int(jj[t])))

    def _rootc(self, R0, R1, i0, j0, tri, at, full, mk, cost) -> None:
        """Koren op(e, c) sa rešenom konstantom nad oblikom e; R0, R1 = vrednosti na probama 0 i 1.
        Oblici: e + c / e - c, c - e, e ^ c, e * c (c iz inverza neparne vrednosti na probi 0)."""
        if cost + 2 >= self.best_cost:
            return
        m, tq = self.mask, self.tq
        f0, f1 = tq[0], tq[1]
        c = (f0 * self.inv[R0 & m]) & m
        hit = ((((f0 - R0) ^ (f1 - R1)) & m) == 0) | ((((f0 + R0) ^ (f1 + R1)) & m) == 0) \
            | ((((f0 ^ R0) ^ (f1 ^ R1)) & m) == 0) | (((R0 & 1) == 1) & (((R1 * c - f1) & m) == 0))
        ii, jj = np.nonzero(hit)
        if tri:
            s = jj + j0 >= ii + i0
            ii, jj = ii[s], jj[s]
        if not ii.size:
            return
        r0, cm = R0[ii, jj], c[ii, jj]
        ii, jj = ii + i0, jj + j0
        alive = np.ones((4, ii.size), bool)
        alive[3] = (r0 & 1) == 1
        for p in range(1, len(tq)):
            rp = at(p, ii, jj)
            fp = tq[p]
            alive[0] &= (((f0 - r0) ^ (fp - rp)) & m) == 0
            alive[1] &= (((f0 + r0) ^ (fp + rp)) & m) == 0
            alive[2] &= (((f0 ^ r0) ^ (fp ^ rp)) & m) == 0
            alive[3] &= ((rp * cm - fp) & m) == 0
            s = alive.any(0)
            if not s.all():
                ii, jj, r0, cm, alive = ii[s], jj[s], r0[s], cm[s], alive[:, s]
            if not ii.size:
                return
        ii, jj = ii[:20000], jj[:20000]
        X = full(ii, jj) & m
        for form, row, cv, cc in self._solve_const(X, self.tgt & m):
            self.offer(cost + 1 + cc, lambda form=form, row=row, cv=cv:
                       _const_root(form, mk(int(ii[row]), int(jj[row])), cv))

    def probe(self, k: int, rootc: bool) -> bool:
        """Koreni cene k nad bankom (bez čuvanja): U(e), B(x, y), U(B(x, y)), CLIP. False = isteklo vreme."""
        B = len(self.V) - 1
        with np.errstate(all="ignore"):
            if 1 <= k - 1 <= B and len(self.V[k - 1]):
                for u in UNARY:
                    if not self._shape_un(u, k - 1, k, rootc):
                        return False
            for a in range(1, k - 1):
                b = k - 1 - a
                if a > B or b > B:
                    continue
                for name in BINARY:
                    if (name in COMM and a > b) or (name in MITM and not rootc):
                        continue
                    if not self._shape_bin(name, a, b, None, k, rootc):
                        return False
            for a in range(1, k - 2):
                b = k - 2 - a
                if a > B or b > B:
                    continue
                for name in BINARY:
                    if name in COMM and a > b:
                        continue
                    if not self._shape_bin(name, a, b, UNARY, k, False):
                        return False
            for s0 in range(1, k - 2):
                for s1 in range(1, k - 1 - s0):
                    s2 = k - 1 - s0 - s1
                    if max(s0, s1, s2) <= B and not self._shape_clip(s0, s1, s2, k):
                        return False
        return not self.timeout()

    def _shape_un(self, u, a, k, rootc) -> bool:
        Qa, Va, oa = self.Qt[a], self.V[a], self.off[a]
        n = Qa.shape[1]
        at = lambda p, ii, jj: un(u, Qa[p, ii])                       # noqa: E731
        full = lambda ii, jj: un(u, Va[ii])                          # noqa: E731
        mk = lambda i, j: Node(u, (self.node(i + oa),))               # noqa: E731
        for i0 in range(0, n, CHUNK):
            R0 = un(u, Qa[0, i0:i0 + CHUNK])
            if self._root_ok(u):
                ii = np.flatnonzero((R0 & self.mask) == self.tq[0]) + i0
                self._cascade(ii, np.zeros_like(ii), at, full, mk, k)
            if rootc:
                R1 = un(u, Qa[1, i0:i0 + CHUNK])
                self._rootc(R0[:, None], R1[:, None], i0, 0, False, at, full, mk, k)
            if self.timeout():
                return False
        return True

    def _shape_bin(self, name, a, b, unaries, k, rootc) -> bool:
        Qa, Qb, Va, Vb = self.Qt[a], self.Qt[b], self.V[a], self.V[b]
        na, nb_ = Qa.shape[1], Qb.shape[1]
        if not na or not nb_:
            return True
        oa, ob = self.off[a], self.off[b]
        tri = name in COMM and a == b
        roots = [None] if unaries is None else list(unaries)
        roots = [u for u in roots if self._root_ok(u or name)]
        if not roots and not rootc:
            return True
        jb = min(nb_, CHUNK)
        ib = max(1, CHUNK // jb)
        fb = lambda ii, jj: bi(name, Va[ii], Vb[jj])                                  # noqa: E731
        mb = lambda i, j: Node(name, (self.node(i + oa), self.node(j + ob)))           # noqa: E731
        for i0 in range(0, na, ib):
            i1 = min(na, i0 + ib)
            for j0 in range(0, nb_, jb):
                j1 = min(nb_, j0 + jb)
                if tri and j1 <= i0:
                    continue
                R = bi(name, Qa[0, i0:i1, None], Qb[0, None, j0:j1])
                for u in roots:
                    V0 = R if u is None else un(u, R)
                    ii, jj = np.nonzero((V0 & self.mask) == self.tq[0])
                    ii, jj = ii + i0, jj + j0
                    if tri:
                        s = jj >= ii
                        ii, jj = ii[s], jj[s]
                    if u is None:
                        self._cascade(ii, jj, lambda p, ii, jj: bi(name, Qa[p, ii], Qb[p, jj]), fb, mb, k)
                    else:
                        self._cascade(ii, jj, lambda p, ii, jj, u=u: un(u, bi(name, Qa[p, ii], Qb[p, jj])),
                                      lambda ii, jj, u=u: un(u, fb(ii, jj)),
                                      lambda i, j, u=u: Node(u, (mb(i, j),)), k)
                if rootc:
                    R1 = bi(name, Qa[1, i0:i1, None], Qb[1, None, j0:j1])
                    self._rootc(R, R1, i0, j0, tri, lambda p, ii, jj: bi(name, Qa[p, ii], Qb[p, jj]), fb, mb, k)
                if self.timeout():
                    return False
        return True

    def _shape_clip(self, s0, s1, s2, k) -> bool:
        Qa, Ql, Qh = self.Qt[s0], self.Qt[s1], self.Qt[s2]
        if not (Qa.shape[1] and Ql.shape[1] and Qh.shape[1]):
            return True
        Va, Vl, Vh = self.V[s0], self.V[s1], self.V[s2]
        jb = min(Qh.shape[1], CHUNK)
        ib = max(1, CHUNK // jb)
        for a in np.flatnonzero(~self.is_const(s0)).tolist():
            g = a + self.off[s0]
            at = lambda p, ii, jj, a=a: np.minimum(np.maximum(Qa[p, a], Ql[p, ii]), Qh[p, jj])   # noqa: E731
            full = lambda ii, jj, a=a: np.minimum(np.maximum(Va[a], Vl[ii]), Vh[jj])            # noqa: E731
            mk = lambda i, j, g=g: Node("CLIP", (self.node(g), self.node(i + self.off[s1]),     # noqa: E731
                                                 self.node(j + self.off[s2])))
            for i0 in range(0, Ql.shape[1], ib):
                i1 = min(Ql.shape[1], i0 + ib)
                for j0 in range(0, Qh.shape[1], jb):
                    j1 = min(Qh.shape[1], j0 + jb)
                    R = np.minimum(np.maximum(Qa[0, a], Ql[0, i0:i1, None]), Qh[0, None, j0:j1])
                    ii, jj = np.nonzero((R & self.mask) == self.tq[0])
                    self._cascade(ii + i0, jj + j0, at, full, mk, k)
            if self.timeout():
                return False
        return True

    # ------------------------------------------- susret u sredini + konstante
    def _lowhash(self, L: np.ndarray) -> np.ndarray:
        """Heš nižih nb bitova na Q_INDEX tačaka (L je već maskiran)."""
        dt = np.uint8 if self.nb <= 8 else np.dtype("<u2")
        w = np.ascontiguousarray(L.astype(dt)).view(np.uint64)
        x = w * _HK
        x ^= x >> np.uint64(29)
        x *= self.R2[:w.shape[1]]
        return x.sum(axis=-1, dtype=np.uint64)

    def _solve_const(self, X, F, andor=False) -> list:
        """Za svaki red X (maskiran) rešenja op(x, c) == F: lista (oblik, red, c, cena c)."""
        m, nb = self.mask, self.nb
        res = []
        for form, Rr in (("ADDC", F - X), ("CSUB", F + X), ("XORC", F ^ X)):
            r = Rr & m
            for i in np.flatnonzero((r == r[:, :1]).all(1))[:2000].tolist():
                rv = int(r[i, 0])
                c, cc = best_rep(rv, nb)
                if form == "ADDC":
                    c2, cc2 = best_rep((-rv) & m, nb)
                    if cc2 < cc:
                        res.append(("SUBC", i, c2, cc2))
                        continue
                res.append((form, i, c, cc))
        odd = (X & 1) == 1
        rows = np.flatnonzero(odd.any(1))
        if rows.size:
            p = np.argmax(odd[rows], axis=1)
            c = (F[p] * _inv_odd(X[rows, p], m)) & m
            ok = (((X[rows] * c[:, None] - F) & m) == 0).all(1)
            for i, cv in zip(rows[ok][:2000].tolist(), c[ok][:2000].tolist()):
                c_, cc = best_rep(int(cv), nb)
                res.append(("MULC", i, c_, cc))
        if andor:
            f_or = int(np.bitwise_or.reduce(F))
            f_and = int(np.bitwise_and.reduce(F))
            okA = np.flatnonzero(((X & f_or) == F).all(1))
            if okA.size:
                free = (~np.bitwise_or.reduce(X[okA], axis=1)) & m & ~f_or
                for i, fr in zip(okA[:2000].tolist(), free[:2000].tolist()):
                    c, cc = min((best_rep(v, nb) for v in (f_or, f_or | fr)), key=lambda z: z[1])
                    res.append(("ANDC", i, c, cc))
            cmin = np.bitwise_or.reduce(F & ~X, axis=1) & m
            for i in np.flatnonzero(((X | cmin[:, None]) == F).all(1))[:2000].tolist():
                cm = int(cmin[i])
                c, cc = min((best_rep(v, nb) for v in (cm, cm | (f_and & ~cm & m))), key=lambda z: z[1])
                res.append(("ORC", i, c, cc))
        return res

    def combine(self) -> None:
        """Faze 3 i 4 nad celom bankom: ADD/SUB/XOR(e1, e2), op(e, c) sa rešenim c, WHERE razlaganje."""
        if not self.total:
            return
        mask = self.mask
        Vall = np.concatenate([V[:, :Q_INDEX] for V in self.V if len(V)]) & mask
        gids = np.arange(self.total)
        cost = np.concatenate([np.full(len(V), k, np.int64) for k, V in enumerate(self.V) if len(V)])
        H = self._lowhash(Vall)
        order = np.lexsort((gids, H))
        Hs = H[order]
        first = np.r_[True, Hs[1:] != Hs[:-1]]
        ih, ig = Hs[first], gids[order][first]
        F = self.tgt[:Q_INDEX] & mask
        cands = []                      # (cena, oblik, gid_x, gid_y ili konstanta)
        step = max(1, CHUNK // (8 * Q_INDEX))
        with np.errstate(all="ignore"):
            for s0 in range(0, self.total, step):
                if self.timeout() or len(cands) > 200_000:
                    break
                X = Vall[s0:s0 + step]
                cx = cost[s0:s0 + step]
                live = self.best_cost - 1 - cx >= 1
                if not live.any():
                    continue
                for form, Rr in (("ADD", F - X), ("SUBL", X - F), ("SUBR", F + X), ("XOR", F ^ X)):
                    h = self._lowhash(Rr & mask)
                    pos = np.searchsorted(ih, h)
                    pos[pos == len(ih)] = 0
                    hit = np.flatnonzero((ih[pos] == h) & live)
                    if hit.size:
                        oth = ig[pos[hit]]
                        tot = 1 + cx[hit] + cost[oth]
                        s = tot < self.best_cost
                        for t, a_, b_ in zip(tot[s].tolist(), (hit[s] + s0).tolist(), oth[s].tolist()):
                            cands.append((t, form, a_, b_))
                for form, i, c, cc in self._solve_const(X, F, andor=True):
                    t = 1 + int(cx[i]) + cc
                    if t < self.best_cost:
                        cands.append((t, form, s0 + i, c))
        cands.sort(key=lambda z: (z[0], z[2]))
        seen = set()
        for r_, (t, form, a_, b_) in enumerate(cands[:5000]):
            if t >= self.best_cost or (r_ >= 32 and self.timeout()):
                break
            if (form, a_, b_) in seen:
                continue
            seen.add((form, a_, b_))
            xa = self.value(a_)
            if form in ("ADD", "SUBL", "SUBR", "XOR"):
                xb = self.value(b_)
                val = {"ADD": xa + xb, "SUBL": xa - xb, "SUBR": xb - xa, "XOR": xa ^ xb}[form]
                mk = lambda form=form, a_=a_, b_=b_: _pair_root(form, self.node(a_), self.node(b_))  # noqa: E731
            else:
                val = _const_val(form, xa, b_)
                mk = lambda form=form, a_=a_, b_=b_: _const_root(form, self.node(a_), b_)            # noqa: E731
            if ((val & mask) == self.tgt).all():
                self.offer(t, mk)

    def value(self, g: int) -> np.ndarray:
        k = self.level_of(g)
        return self.V[k][g - self.off[k]]

    def where_split(self) -> None:
        """WHERE(c, a, b) == cilj  <=>  a se slaže gde je c != 0, a b gde je c == 0 (na uzorku).
        Jedna strana mora pokriti >= P/2 tačaka, pa se ona traži među retkim "velikim" izrazima."""
        if self.best_cost <= 4 or self.timeout():
            return
        P = self.P
        W = (P + 63) // 64
        M = np.concatenate([(V & self.mask) == self.tgt for V in self.V if len(V)])
        cost = np.concatenate([np.full(len(V), k, np.int64) for k, V in enumerate(self.V) if len(V)])
        pad = W * 64 - P

        def pack(B):
            if pad:
                B = np.concatenate([B, np.zeros((len(B), pad), bool)], 1)
            return np.packbits(B, axis=1).view(np.uint64)

        S = pack(M)
        cnt = M.sum(1)
        big = np.flatnonzero(cnt * 2 >= P)
        if not big.size:
            return
        Sb = S[big]
        valid = pack(np.ones((1, P), bool))[0]
        conds = []
        for k in range(1, len(self.V)):
            for i in self.conds(k).tolist():
                conds.append((k, i))
        if not conds:
            return
        Z = pack(np.stack([self.V[k][i] != 0 for k, i in conds]))
        Zc = ~Z & valid
        for ci, (kc, ic) in enumerate(conds):
            if 1 + kc + 2 >= self.best_cost or (ci % 256 == 0 and self.timeout()):
                if self.timed_out:
                    return
                continue
            for zt, zf, swap in ((Z[ci], Zc[ci], False), (Zc[ci], Z[ci], True)):
                # velika strana pokriva zt (tačke gde se ona bira), mala pokriva zf
                okb = np.flatnonzero(((Sb & zt) == zt).all(1))
                if not okb.size:
                    continue
                gb = int(big[okb[np.argmin(cost[big[okb]])]])
                if 1 + kc + int(cost[gb]) + 1 >= self.best_cost:
                    continue
                oks = np.flatnonzero(((S & zf) == zf).all(1))
                if not oks.size:
                    continue
                gs = int(oks[np.argmin(cost[oks])])
                t = 1 + kc + int(cost[gb]) + int(cost[gs])
                gc = self.off[kc] + ic
                a_, b_ = (gs, gb) if swap else (gb, gs)
                self.offer(t, lambda gc=gc, a_=a_, b_=b_: Node("WHERE", (self.node(gc), self.node(a_),
                                                                         self.node(b_))))


def _pair_root(form: str, x: Node, y: Node) -> Node:
    if form == "SUBR":
        return Node("SUB", (y, x))
    return Node({"ADD": "ADD", "SUBL": "SUB", "XOR": "XOR"}[form], (x, y))


def _const_root(form: str, x: Node, c: int) -> Node:
    if form == "CSUB":
        return Node("SUB", (const(c), x))
    name = {"ADDC": "ADD", "SUBC": "SUB", "XORC": "XOR", "MULC": "MUL", "ANDC": "AND", "ORC": "OR"}[form]
    return Node(name, (x, const(c)))


def _const_val(form: str, x, c: int):
    c = np.int64(c)
    return {"ADDC": lambda: x + c, "SUBC": lambda: x - c, "CSUB": lambda: c - x, "XORC": lambda: x ^ c,
            "MULC": lambda: x * c, "ANDC": lambda: x & c, "ORC": lambda: x | c}[form]()


def search(leaves, tgt, nb, deadline, check, bound=1 << 30, max_b=MAX_B, max_probe=2 * MAX_B + 1,
           rootc_max=MAX_B + 1, seed=SEED, found_share=None) -> _Bank:
    """Faze 1-4 nad jednim domenom (video ili zvuk); prihvata samo kandidate jeftinije od bound.
    found_share: posle prvog tačnog kandidata ostaje samo taj udeo preostalog vremena."""
    bank = _Bank(leaves, tgt, nb, deadline, check, np.random.default_rng(seed))
    bank.best_cost = bound
    bank.found_share = found_share
    complete = 0
    for k in range(1, max_b + 1):
        ok = bank.build_level(k)
        with np.errstate(all="ignore"):
            bank.combine()
        if ok:
            complete = k
        if bank.best_cost <= k + 1 or not ok or bank.timeout():
            break
    if complete >= 3:
        bank.where_split()
    if complete < 2 or bank.timeout():
        return bank
    bank._prep_probe()
    for k in range(complete + 1, max_probe + 1):
        if bank.best_cost <= k or not bank.probe(k, k <= rootc_max):
            break
        if bank.best_cost <= k + 1:
            break
    return bank


# ------------------------------------------------------------- video
def _sample_points(rng, dims, P):
    """Stratifikovan uzorak: svaka osa ravnomerno pokrivena, kombinacije nasumične."""
    cols = []
    for d in dims:
        v = np.resize(np.arange(d, dtype=np.int64), P)
        rng.shuffle(v)
        cols.append(v)
    return cols


def _period(v: np.ndarray, axis: int) -> int:
    """Najmanji p takav da je v[i] == v[i + p] duž ose (n ako nema perioda)."""
    n = v.shape[axis]
    for p in range(1, n):
        a = [slice(None)] * v.ndim
        b = [slice(None)] * v.ndim
        a[axis], b[axis] = slice(p, None), slice(0, n - p)
        if np.array_equal(v[tuple(a)], v[tuple(b)]):
            return p
    return n


def _box_literal(video: np.ndarray, meta) -> Node:
    """BYTES literal nad najmanjom periodičnom kutijom po (T, Y, X, C)."""
    Tn, H, W = meta.frames, meta.height, meta.width
    pt, py, px, pc = (_period(video, ax) for ax in range(4))
    box = video[:pt, :py, :px, :pc]
    idx = None

    def axis_term(leaf, p, full):
        if p == 1:
            return None
        return leaf if p == full else op("MOD", leaf, p)

    for leaf, p, full, mul in ((Node("T"), pt, Tn, py), (Node("Y"), py, H, px), (Node("X"), px, W, pc),
                               (Node("C"), pc, 3, 1)):
        term = axis_term(leaf, p, full)
        if term is not None:
            idx = term if idx is None else op("ADD", idx, term)
        if idx is not None and mul > 1:
            idx = op("MUL", idx, mul)
    if idx is None:
        idx = const(0)
    return bytes_lit(box.reshape(-1).tobytes(), idx)


def _period1d(a: np.ndarray) -> int:
    n = len(a)
    w = min(16, n)
    if n < 2:
        return max(n, 1)
    win = np.lib.stride_tricks.sliding_window_view(a, w)
    cand = np.flatnonzero((win == a[:w]).all(1))
    for p in cand.tolist():
        if p and np.array_equal(a[p:], a[:n - p]):
            return p
    return n


def _attempts(leaves, attempts, deadline, check_full, found_share=None):
    """Uzastopne pretrage nad (redukovanim) ciljevima. attempts: (tgt, nb, wrap, dodatna cena, udeo
    preostalog vremena). Vraća (cena, izraz | None, [približni izrazi])."""
    best_cost, best, approx = 1 << 30, None, []
    for tgt, nb, wrap, extra, share in attempts:
        now = time.perf_counter()
        if now >= deadline or best_cost - extra <= 1:
            break
        dl = now + share * (deadline - now)
        bank = search(leaves, tgt, nb, dl, lambda nd, wrap=wrap: check_full(wrap(nd)), bound=best_cost - extra,
                      found_share=None if best is not None else found_share)
        if bank.found_at is not None:          # tačan izraz postoji: ostatak roka se skraćuje
            deadline = min(deadline, bank.found_at + found_share * max(0.0, deadline - bank.found_at))
        if bank.best is not None and bank.best_cost + extra < best_cost:
            best_cost, best = bank.best_cost + extra, wrap(bank.best)
        approx += [wrap(nd) for nd in bank.near] + [wrap(bank.node(g)) for _, _, g in bank.approx]
    return best_cost, best, approx


def _mdl(nd: Node, got: np.ndarray, want: np.ndarray, vbytes_max: int):
    """(dužina izraza + izuzeci, izraz, Exceptions) za približan izraz."""
    d = got ^ want
    pos = np.flatnonzero(d)
    gaps = np.diff(np.concatenate([[-1], pos])) - 1
    vals = d[pos].astype(np.int64)
    vb = 0 if not len(vals) or np.all(vals == 1) else (1 if vals.max() < 256 else vbytes_max)
    return nbytes(nd) + exc_size(len(pos), gaps, vb), nd, (kfl.Exceptions(pos.astype(np.int64), vals)
                                                          if len(pos) else None)


def _encode_video(meta, video: np.ndarray, deadline: float, P: int, max_exc: int, found_share=None):
    """Kandidati za video: lista (dužina izraza + izuzeci, izraz, Exceptions | None)."""
    Tn, H, W = meta.frames, meta.height, meta.width
    vflat = video.reshape(-1)
    vbytes = vflat.tobytes()
    meta_v = replace(meta, audio_samples=0)
    rng = np.random.default_rng(SEED)
    t, y, x, c = _sample_points(rng, (Tn, H, W, 3), P)
    tgt = video[t, y, x, c].astype(np.int64)
    vt, vy, vx, vc = (rng.integers(0, d, N_VALID) for d in (Tn, H, W, 3))
    vtgt = video[vt, vy, vx, vc].astype(np.int64)
    venv = {"T": vt, "Y": vy, "X": vx, "C": vc, "WID": np.int64(W), "HEI": np.int64(H), "NFR": np.int64(Tn)}

    def check(nd: Node) -> bool:
        try:
            with np.errstate(all="ignore"):
                if not ((np.broadcast_to(evaluate(nd, venv), (N_VALID,)) & 255) == vtgt).all():
                    return False
            return kfl.run(kfl.Program(video=nd), meta_v) == vbytes
        except (ValueError, KeyError, IndexError):
            return False

    deps = {"T": Tn > 1 and not np.array_equal(video[1:], video[:-1]),
            "Y": H > 1 and not np.array_equal(video[:, 1:], video[:, :-1]),
            "X": W > 1 and not np.array_equal(video[:, :, 1:], video[:, :, :-1]),
            "C": not np.array_equal(video[..., 1:], video[..., :-1])}
    env = {"T": t, "Y": y, "X": x, "C": c}
    leaves = [(Node(nm), env[nm]) for nm in ("T", "Y", "X", "C") if deps[nm]]
    leaves += [(Node("WID"), W), (Node("HEI"), H), (Node("NFR"), Tn)]
    attempts = []
    # redukcija korena SHL(e, z): svi bajtovi imaju z nižih nula -> traži se f >> z na 8 - z bita
    orv = int(np.bitwise_or.reduce(vflat)) if vflat.size else 0
    z = (orv & -orv).bit_length() - 1 if orv else 0
    if 1 <= z <= 7:
        attempts.append((tgt >> z, 8 - z, lambda nd, z=z: Node("SHL", (nd, const(z))), 2, 0.6))
    attempts.append((tgt, 8, lambda nd: nd, 0, 1.0))
    cost, best, approx = _attempts(leaves, attempts, deadline, check, found_share)
    if best is not None:
        return [(nbytes(best), best, None)]
    out = []
    for nd in approx:
        try:
            got = np.frombuffer(kfl.run(kfl.Program(video=nd), meta_v), np.uint8)
        except ValueError:
            continue
        if np.count_nonzero(got != vflat) <= max_exc:
            out.append(_mdl(nd, got, vflat, 1))
    lit = _box_literal(video, meta)
    out.append((nbytes(lit), lit, None))
    return out


def _encode_audio(meta, audio: np.ndarray, deadline: float, P: int):
    """Kandidati za zvuk: lista (dužina, izraz | None, Exceptions | None)."""
    NS = len(audio)
    a16 = audio.view("<u2").astype(np.int64)
    if NS == 0 or not a16.any():
        return [(0, None, None)]
    p = _period1d(a16)
    lit = words_lit(a16[:p], Node("N"))
    out = [(nbytes(lit), lit, None)]
    if NS < 2 * Q_INDEX:
        return out
    Tn = meta.frames
    rng = np.random.default_rng(SEED + 1)
    n = rng.choice(NS, size=min(P, NS), replace=False).astype(np.int64)
    tgt = a16[n]

    def tmap(nn):
        tt = (nn * meta.fps) // meta.audio_rate if meta.audio_rate else np.zeros_like(nn)
        return np.clip(tt, 0, max(Tn - 1, 0))

    vn = rng.integers(0, NS, N_VALID)
    venv = {"N": vn, "T": tmap(vn), "WID": np.int64(meta.width), "HEI": np.int64(meta.height),
            "NFR": np.int64(Tn)}
    vtgt = a16[vn]

    def run_audio(nd):
        pl = kfl.run(kfl.Program(video=const(0), audio=nd), meta)
        return np.frombuffer(pl[meta.video_bytes:], "<u2").astype(np.int64)

    def check(nd: Node) -> bool:
        try:
            with np.errstate(all="ignore"):
                if not ((np.broadcast_to(evaluate(nd, venv), (N_VALID,)) & 0xFFFF) == vtgt).all():
                    return False
            return np.array_equal(run_audio(nd), a16)
        except (ValueError, KeyError, IndexError):
            return False

    leaves = [(Node("N"), n), (Node("T"), tmap(n)), (Node("WID"), meta.width), (Node("HEI"), meta.height),
              (Node("NFR"), Tn)]
    attempts = []
    # redukcija korena MUL(e, d): d = NZD svih označenih uzoraka -> traži se f / d
    sv = audio.astype(np.int64)
    d = int(np.gcd.reduce(np.abs(sv)))
    if 1 < d < 1 << 15:
        attempts.append(((sv[n] // d) & 0xFFFF, 16, lambda nd, d=d: Node("MUL", (nd, const(d))),
                         1 + ccost(d), 0.9))
    attempts.append((tgt, 16, lambda nd: nd, 0, 1.0))
    cost, best, approx = _attempts(leaves, attempts, deadline, check)
    if best is not None:
        out.append((nbytes(best), best, None))
        return out
    for nd in approx:
        try:
            out.append(_mdl(nd, run_audio(nd), a16, 2))
        except ValueError:
            continue
    return out


# ------------------------------------------------------------- enkoder
def _generic(meta, payload, best, t_end):
    """MDL rezerva: bz2, pa lzma (ako stane u rok); vraća kraći tačan zapis od best."""
    if best is not None and len(best) <= GENERIC_MIN:
        return best
    for name in ("bz2", "lzma"):
        if name == "lzma" and time.perf_counter() + len(payload) / LZMA_BPS > t_end:
            continue
        try:
            rec = igc_codec.encode(name, meta, payload)
            if rec is not None and (best is None or len(rec) < len(best)) and record.decode(rec)[1] == payload:
                best = rec
        except Exception:          # rezerva nikad ne sme da obori enkoder
            continue
    return best


def encode(meta, payload, budget_s):
    t0 = time.perf_counter()
    if len(payload) != meta.payload_bytes:
        return None
    try:
        kfl.check_meta(meta)
    except ValueError:
        return None
    if meta.audio_samples and (meta.audio_channels != 1 or meta.sample_bytes != 2):
        return None
    Tn, H, W = meta.frames, meta.height, meta.width
    P = 192 if H * W * Tn <= 64 * 64 * 32 else 256
    video = np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(Tn, H, W, 3)
    audio = np.frombuffer(payload[meta.video_bytes:], "<i2")
    silent = not audio.any()
    # rezerva vremena za generičke kompresore (lzma ~1 MB/s), najviše 15 % budžeta
    search_s = budget_s - min(0.15 * budget_s, 1.5 * len(payload) / LZMA_BPS)
    if meta.video_bytes:
        # izuzeci se serijalizuju i čitaju u Python petlji (~8 us po izuzetku): ograničenje po budžetu
        max_exc = int(max(0.0, 0.08 * budget_s) / 8e-6)
        # kad ima zvuka, video posle prvog tačnog izraza ustupa vreme zvuku (FOUND_SHARE)
        v_cands = _encode_video(meta, video, t0 + (0.9 if silent else 0.6) * search_s, P, max_exc,
                                None if silent else FOUND_SHARE)
    else:
        v_cands = [(1, const(0), None)]
    # zvuk posle videa: do 90 % budžeta pretrage (dobija i vreme koje video nije potrošio)
    a_cands = _encode_audio(meta, audio, t0 + 0.9 * search_s, 128)
    v_cands.sort(key=lambda z: z[0])
    a_cands.sort(key=lambda z: z[0])
    best = None
    for _, vn, ve in v_cands:
        for _, an, ae in a_cands[:2]:
            prog = kfl.Program(video=vn, audio=an, exc_video=ve, exc_audio=ae)
            try:
                rec = record.encode_kfl(meta, prog)
                if record.decode(rec)[1] == payload:
                    best = rec
                    break
            except ValueError:
                continue
        if best is not None:
            break
    return _generic(meta, payload, best, t0 + 0.97 * budget_s)


ENCODERS = [Encoder("enum_search", encode, family_aware=False,
                    note="generička bottom-up pretraga KFL izraza (OE na uzorku, MITM, "
                         "rešavanje konstanti, WHERE razlaganje) + MDL (izuzeci, literal, bz2/lzma)")]
