"""Inverzija porodice SHADER: šablon, parametri i izuzeci iz samih bajtova filma.

Family-aware: pretpostavlja oblike šablona porodice (zbir ISIN talasa sa
faznim pomakom po kanalu; prstenovi XOR rampa; ton/akord ISIN((N*p)>>q)*A),
ali sve parametre (broj članova, frekvencije, faze, amplitude, bias, centar,
koeficijente, tonove) zaključuje iz filma.

Šablon sum (plazma):
  1. Razlike po osama x, y, t uzimaju se po modulu 256 i preslikavaju u
     [-128, 128): gde nema prelamanja, to su TAČNE linearne razlike zbira
     a_j*ISIN(...), bez harmonika koje bi stvorio wrap.
  2. OMP: frekvencija (fx, fy, ft) na mreži 256 bira se separabilnim DFT-om
     ostatka; amplitude/faze po kanalu dobijaju se zajedničkim LS fitom;
     prelamanja se ispravljaju u odnosu na tekući model (ostatak se ponovo
     preslikava u [-128, 128)). Lokalno doterivanje: svaka frekvencija +-1,
     zajednički pomak para i zamena člana (bliski talasi se na kratkom
     prozoru mešaju). Kriterijum je rms bez 1 % najgorih redova (šum).
     Članovi se dodaju dok tačna faza (korak 3) ne uspe.
  3. Tačna faza: a_j = round(|A|/127), ph = arg(A); bias = moda ostatka po
     kanalu; lokalno poliranje faza/amplituda prema broju promašaja.
  4. Promašaji -> izuzeci videa (MDL); bira se najkraći tačan zapis.
Šablon rings: centar iz parnosti ISQRT (bit 0, kanal sa neparnim f_c), pa
koeficijenti pretragom bit po bit (niski bitovi zavise samo od niskih bitova).
Zvuk: vrhovi spektra -> kandidati (p, q), LS za amplitude, tačna provera;
rezerva je sirov WORDS literal.
"""
from __future__ import annotations

import itertools
import time

import numpy as np

from kfb import kfl, record
from kfb.families import shader as fam
from kfb.kfl import ISIN_LUT, C, N, T, X, Y, Node, bytes_lit, op, words_lit
from igc import codec as igc_codec

from . import Encoder

TAU = 2.0 * np.pi
NEIGH = [d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)]
MAX_TERMS = 5
EXC_FRAC = 0.03          # iznad ovog udela promašaja ne gradimo izuzetke (ne isplati se)


def _wrap(a):
    return ((a + 128) & 255) - 128


def _canon(w):
    return tuple(w) if tuple(w) > (0, 0, 0) else (-w[0], -w[1], -w[2])


class _Deadline(Exception):
    pass


# ================================================================ sum
class _Rows:
    """Uzorak razlika (osa, t, y, x, D[3]) za LS fit; kolone po frekvenciji se keširaju."""

    def __init__(self, V, rng, nsamp):
        Tn, H, W, _ = V.shape
        ax_l, t_l, y_l, x_l, d_l = [], [], [], [], []
        for ax, sh in enumerate([(Tn, H, W - 1), (Tn, H - 1, W), (Tn - 1, H, W)]):
            n = int(np.prod(sh))
            if n <= 0:
                continue
            sel = np.sort(rng.choice(n, min(n, nsamp), replace=False))
            t, y, x = np.unravel_index(sel, sh)
            d = [0, 0, 0]
            d[ax] = 1
            ax_l.append(np.full(len(t), ax))
            t_l.append(t), y_l.append(y), x_l.append(x)
            d_l.append(_wrap(V[t + d[2], y + d[1], x + d[0]] - V[t, y, x]))
        self.ax = np.concatenate(ax_l)
        self.t, self.y, self.x = np.concatenate(t_l), np.concatenate(y_l), np.concatenate(x_l)
        self.D = np.concatenate(d_l).astype(np.float64)
        self.cache: dict = {}

    def cols(self, w):
        c = self.cache.get(w)
        if c is None:
            B = np.exp(1j * TAU * np.asarray(w, np.float64) / 256) - 1          # po osi
            e = B[self.ax] * np.exp(1j * TAU * ((w[0] * self.x + w[1] * self.y + w[2] * self.t) & 255) / 256)
            c = np.stack([e.real, -e.imag], 1)
            self.cache[w] = c
        return c

    def fit(self, ws, ref=None, iters=2):
        """LS fit kompleksnih amplituda; prelamanja se ispravljaju u odnosu na ref model.
        U drugom krugu se izbacuju redovi sa ogromnim ostatkom (šum, izuzeci)."""
        D = self.D
        model = np.zeros_like(D) if ref is None else ref
        if not ws:
            return _trimmed_rms(_wrap(np.rint(D).astype(np.int64))), np.zeros((0, 3), complex), model
        Xm = np.concatenate([self.cols(w) for w in ws], 1)
        ridge = 1e-6 * np.eye(Xm.shape[1])
        sol = None
        for it in range(iters):
            R = _wrap(np.rint(D - model).astype(np.int64))
            tgt = model + R
            Xk = Xm
            if it:
                keep = np.abs(R).max(1) <= max(24.0, 6.0 * _trimmed_rms(R))
                Xk, tgt = Xm[keep], tgt[keep]
            try:
                sol = np.linalg.solve(Xk.T @ Xk + ridge, Xk.T @ tgt)
            except np.linalg.LinAlgError:
                sol, *_ = np.linalg.lstsq(Xk, tgt, rcond=None)
            model = Xm @ sol
        rms = _trimmed_rms(_wrap(np.rint(D - model).astype(np.int64)))
        return rms, sol[0::2] + 1j * sol[1::2], model


def _trimmed_rms(R) -> float:
    """rms ostatka bez 1 % najgorih redova (retki šumni bajtovi ne kvare kriterijum)."""
    e = (R.astype(np.float64) ** 2).mean(1)
    k = max(1, int(len(e) * 0.99))
    return float(np.sqrt(np.partition(e, k - 1)[:k].mean()))


class _SumSolver:
    def __init__(self, V, meta, deadline):
        self.V = V.astype(np.int64)
        self.meta = meta
        self.deadline = deadline
        Tn, H, W, _ = V.shape
        r = fam.ranges(meta)
        self.box = (min(r["fx"] + 2, 127), min(r["fy"] + 2, 127), min(r["ft"] + 2, 127))
        self.rows = _Rows(self.V, np.random.default_rng(12345), 5000)
        # razlike na celoj mreži (za izbor frekvencije); kod velikih filmova svaki drugi frejm
        step = 1 if Tn * H * W <= 200_000 else 2
        Vs = self.V[::step]
        V0 = self.V[0:Tn - 1:step]
        self.D = [_wrap(np.diff(Vs, axis=2)), _wrap(np.diff(Vs, axis=1)),
                  _wrap(self.V[1::step][:len(V0)] - V0)]
        self.tstep = step
        Fx, Fy, Ft = self.box
        self.om = [np.arange(-Fx, Fx + 1)[None, None, :], np.arange(-Fy, Fy + 1)[None, :, None],
                   np.arange(-Ft, Ft + 1)[:, None, None]]
        self.Bn = sum(np.abs(np.exp(1j * TAU * o / 256) - 1) ** 2 for o in self.om)
        self.Ex = np.exp(-1j * TAU * np.outer(np.arange(W), np.arange(-Fx, Fx + 1)) / 256)
        self.Ey = np.exp(-1j * TAU * np.outer(np.arange(H), np.arange(-Fy, Fy + 1)) / 256)

    def _tick(self):
        if time.perf_counter() > self.deadline:
            raise _Deadline()

    # ---- izbor frekvencije iz spektra ostatka
    def _spectrum(self, ws, A):
        Fx, Fy, Ft = self.box
        S = 0
        for ax in range(3):
            Da = self.D[ax]
            Tn, H, W = Da.shape[:3]
            t0 = np.arange(Tn) * self.tstep
            model = np.zeros(Da.shape)
            for j, w in enumerate(ws):
                B = np.exp(1j * TAU * w[ax] / 256) - 1
                ph = np.exp(1j * TAU * ((w[0] * np.arange(W)[None, None, :] + w[1] * np.arange(H)[None, :, None]
                                         + w[2] * t0[:, None, None]) & 255) / 256)
                model += np.real(ph[..., None] * (A[j] * B)[None, None, None, :])
            R = _wrap(np.rint(Da - model).astype(np.int64)).astype(np.float64)
            Ex = self.Ex[:W]
            Ey = self.Ey[:H]
            Et = np.exp(-1j * TAU * np.outer(t0, np.arange(-Ft, Ft + 1)) / 256)
            a = np.einsum("thwc,wx->thcx", R, Ex, optimize=True)
            a = np.einsum("thcx,hy->tcyx", a, Ey, optimize=True)
            a = np.einsum("tcyx,tf->cfyx", a, Et, optimize=True)
            B = np.exp(1j * TAU * self.om[ax] / 256) - 1
            S = S + np.conj(B)[None] * a
            self._tick()
        score = (np.abs(S) ** 2).sum(0) / np.maximum(self.Bn, 1e-9)
        score[Ft, Fy, Fx] = 0
        for w in ws:
            for s in (1, -1):
                score[s * w[2] + Ft, s * w[1] + Fy, s * w[0] + Fx] = 0
        return score

    def _peaks(self, score, k):
        Fx, Fy, Ft = self.box
        out = []
        for o in np.argsort(score.ravel())[::-1]:
            f, y, x = np.unravel_index(int(o), score.shape)
            w = _canon((int(x - Fx), int(y - Fy), int(f - Ft)))
            if any(max(abs(w[i] - u[i]) for i in range(3)) <= 1 for u in out):
                continue
            out.append(w)
            if len(out) >= k:
                break
        return out

    def _inbox(self, w):
        return w != (0, 0, 0) and all(abs(w[i]) <= self.box[i] for i in range(3))

    def _refine(self, ws, only=None):
        """Lokalno doterivanje frekvencija (+-1 po osi) uz LS refit; only = samo jedan član."""
        rms, A, model = self.rows.fit(ws)
        js = range(len(ws)) if only is None else [only]
        changed = True
        while changed:
            changed = False
            for j in js:
                for d in NEIGH:
                    self._tick()
                    w2 = _canon((ws[j][0] + d[0], ws[j][1] + d[1], ws[j][2] + d[2]))
                    if not self._inbox(w2) or w2 in ws:
                        continue
                    trial = ws[:j] + [w2] + ws[j + 1:]
                    r2, A2, m2 = self.rows.fit(trial, ref=model)
                    if r2 < rms - 1e-9:
                        rms, ws, A, model, changed = r2, trial, A2, m2, True
        return ws, rms, A

    def _pairs(self, ws, A, rms, force=False):
        """Ponavlja zajednički pomak para članova dok god rms opada."""
        good = 0.0 if force else 1.0 + 0.3 * len(ws)
        while rms > good:
            ws2, rms2, A2 = self._pair_step(ws, A, rms)
            if rms2 >= 0.99 * rms or set(ws2) == set(ws):
                break
            ws, rms, A = ws2, rms2, A2
        return ws, rms, A

    def _pair_step(self, ws, A, rms):
        """Zajednički pomak dva "sumnjiva" člana (amplituda daleko od celog broja)."""
        amp = np.abs(A).mean(1) / 127.0
        off = np.abs(amp - np.maximum(np.rint(amp), 1))
        sus = [int(j) for j in np.argsort(-off)[:2] if off[j] > 0.06]
        # sumnjiv član + bilo koji drugi; najpre spektralno najbliži (oni se mešaju)
        pairs = {tuple(sorted((j, k))) for j in sus for k in range(len(ws)) if k != j}
        dist = lambda jk: max(abs(ws[jk[0]][i] - ws[jk[1]][i]) for i in range(3))  # noqa: E731
        _, _, model = self.rows.fit(ws)
        best = (rms, ws, A)
        for j, k in sorted(pairs, key=dist)[:3]:
            for dj in NEIGH:
                self._tick()
                wj = _canon(tuple(ws[j][i] + dj[i] for i in range(3)))
                if not self._inbox(wj) or wj in ws:
                    continue
                for dk in NEIGH:
                    wk = _canon(tuple(ws[k][i] + dk[i] for i in range(3)))
                    if not self._inbox(wk) or wk in ws or wk == wj:
                        continue
                    trial = list(ws)
                    trial[j], trial[k] = wj, wk
                    r2, A2, _ = self.rows.fit(trial, ref=model)
                    if r2 < best[0] - 1e-9:
                        best = (r2, trial, A2)
        if best[1] is not ws:
            return self._refine(best[1])
        return ws, rms, A

    def _swaps(self, ws, A, rms, force=False):
        """Zamena člana: izbaci j, ponovo izaberi iz spektra ostatka (bliski talasi se mešaju)."""
        good = 0.0 if force else 1.0 + 0.3 * len(ws)
        improved = True
        while improved and rms > good:
            improved = False
            amp = np.abs(A).mean(1) / 127.0
            off = np.abs(amp - np.maximum(np.rint(amp), 1))
            for j in np.argsort(-off):
                rest = ws[:j] + ws[j + 1:]
                _, Ar, _ = self.rows.fit(rest)
                for w in self._peaks(self._spectrum(rest, Ar), 3):
                    if w in ws:
                        continue
                    ws2, r2, A2 = self._refine(rest + [w], only=len(rest))
                    if r2 < 0.99 * rms and set(ws2) != set(ws):
                        ws2, r2, A2 = self._refine(ws2)
                        ws, rms, A, improved = ws2, r2, A2, True
                        break
                if improved:
                    break
        return ws, rms, A

    def omp(self, first=None):
        """Pohlepni izbor članova (generator): daje (frekvencije, amplitude, odnos rms)
        svaki put kad ostatak padne na nivo šuma zaokruživanja, pa nastavlja ako
        tačna faza ne uspe (slabi niskofrekventni članovi se jedva vide u razlikama)."""
        ws, A = [], np.zeros((0, 3), complex)
        rms, _, _ = self.rows.fit(ws)
        rms0 = max(rms, 1e-9)
        good = lambda k: 1.0 + 0.3 * k      # noqa: E731  (šum zaokruživanja ISIN-a)
        while len(ws) < MAX_TERMS:
            if ws and rms <= good(len(ws)):
                yield ws, A, rms / rms0
                # tačna faza nije uspela: niske frekvencije se slabo vide u razlikama,
                # pa parove i zamene probamo i ispod praga šuma
                if len(ws) >= 2:
                    ws2, rms2, A2 = self._pairs(ws, A, rms, force=True)
                    ws2, rms2, A2 = self._swaps(ws2, A2, rms2, force=True)
                    if set(ws2) != set(ws):
                        ws, rms, A = ws2, rms2, A2
                        yield ws, A, rms / rms0
            if first is not None and not ws:
                w = first
            else:
                w = self._peaks(self._spectrum(ws, A), 1)[0]
            ws2, rms2, A2 = self._refine(ws + [w], only=len(ws))
            if rms2 > good(len(ws2)):
                ws2, rms2, A2 = self._refine(ws2)
            if rms2 > good(len(ws2)) and len(ws2) >= 3:     # sa 2 člana je ostatak obično samo nepotpun
                ws2, rms2, A2 = self._pairs(ws2, A2, rms2)
            if rms2 > good(len(ws2)) and len(ws2) >= 3:
                ws2, rms2, A2 = self._swaps(ws2, A2, rms2)
            if ws and rms2 > 0.97 * rms:
                break
            if not ws and rms2 > 0.9 * rms:     # nije zbir ISIN talasa (ili sve se prelama)
                yield [], A, 1.0
                return
            ws, rms, A = ws2, rms2, A2
        yield ws, A, rms / rms0

    # ---- tačni parametri
    def _theta(self, w):
        Tn, H, W, _ = self.V.shape
        return (w[0] * np.arange(W)[None, :] + w[1] * np.arange(H)[:, None])[None] + \
            (w[2] * np.arange(Tn))[:, None, None]

    def exact(self, ws, A):
        """Tačni parametri iz LS amplituda: (terms, bias) ili None; term = dict(w, a, ph[3])."""
        terms = []
        for j, w in enumerate(ws):
            a = int(np.rint(np.mean(np.abs(A[j])) / 127.0))
            if a <= 0:
                continue
            ph = [int(np.rint((np.angle(A[j][c]) + np.pi / 2) * 256 / TAU)) % 256 for c in range(3)]
            terms.append({"w": w, "a": a, "ph": ph, "th": self._theta(w)})
        if not terms:
            return None
        miss, bias = self._polish(terms)
        if miss > 0.002 * self.V.size:          # možda je amplituda pogrešno zaokružena
            for j in range(len(terms)):
                for da in (1, -1):
                    if terms[j]["a"] + da < 1:
                        continue
                    trial = [dict(tm, ph=list(tm["ph"])) for tm in terms]
                    trial[j]["a"] += da
                    m2, b2 = self._polish(trial)
                    if m2 < miss:
                        terms, miss, bias = trial, m2, b2
        if miss > EXC_FRAC * self.V.size:
            return None
        return terms, bias

    def _polish(self, terms):
        """Faze po kanalu +-3 prema broju promašaja; bias = moda ostatka. Menja terms."""
        def contrib(tm, c, ph=None):
            return tm["a"] * ISIN_LUT[(tm["th"] + (tm["ph"][c] if ph is None else ph)) & 255]

        bias, miss = [], 0
        for c in range(3):
            parts = [contrib(tm, c) for tm in terms]
            total = sum(parts)
            Vc = self.V[..., c]

            def cost(tot):
                h = np.bincount(((Vc - tot) & 255).ravel(), minlength=256)
                return int(Vc.size - h.max()), int(h.argmax())

            best, b = cost(total)
            improved = best > 0
            while improved:
                improved = False
                for j, tm in enumerate(terms):
                    self._tick()
                    for dph in (1, -1, 2, -2, 3, -3):
                        p2 = (tm["ph"][c] + dph) % 256
                        new = contrib(tm, c, ph=p2)
                        m, b2 = cost(total - parts[j] + new)
                        if m < best:
                            best, b, total = m, b2, total - parts[j] + new
                            parts[j] = new
                            tm["ph"][c] = p2
                            improved = True
            bias.append(b)
            miss += best
        return miss, bias

    def residual_mask(self, terms, bias):
        out = np.zeros(self.V.shape, np.int64)
        for c in range(3):
            tot = sum(tm["a"] * ISIN_LUT[(tm["th"] + tm["ph"][c]) & 255] for tm in terms)
            out[..., c] = (tot + bias[c]) & 255
        return out


def _sum_term_node(w, a, ph):
    """Najkraći ISIN čvor za (w, a, ph[3]); vraća (amplituda, čvor)."""
    best = None
    for sw, sa, phf in ((1, 1, lambda p: p), (-1, 1, lambda p: 128 - p),
                        (1, -1, lambda p: p + 128), (-1, -1, lambda p: -p)):
        fx, fy, ft = (sw * w[0], sw * w[1], sw * w[2])
        p = [phf(v) % 256 for v in ph]
        pairs = [(fx, X), (fy, Y), (ft, T)]
        opts = []
        k = (p[1] - p[0]) % 256
        if (p[2] - p[1]) % 256 == k:
            opts.append(fam.lin(pairs + [(k, C)], p[0], 256))
        opts.append(fam.lin(pairs + [(1, bytes_lit(bytes(p), C))], 0, 256))
        for arg in opts:
            n = op("ISIN", arg)
            cost = fam.nbytes(n) + fam.ccost(fam.rep(sa * a, 256))
            if best is None or cost < best[0]:
                best = (cost, sa * a, n)
    return best[1], best[2]


def sum_video_node(terms, bias) -> Node:
    pairs = [_sum_term_node(tm["w"], tm["a"], tm["ph"]) for tm in terms]
    best = None
    for extra, c0 in fam.bias_options(bias):
        e = fam.lin(pairs + extra, c0, 256)
        if best is None or fam.nbytes(e) < fam.nbytes(best):
            best = e
    return best


def infer_sum(V, meta, deadline):
    """Lista kandidata (video čvor, predikcija) za šablon sum."""
    try:
        sv = _SumSolver(V, meta, deadline)
        firsts = [None]
        out = []
        tried = 0
        while firsts and tried < 6 and not out:
            first = firsts.pop(0)
            tried += 1
            ratio = 1.0
            for ws, A, ratio in sv.omp(first):
                res = sv.exact(ws, A) if ws else None
                if res is not None:
                    terms, bias = res
                    out.append((sum_video_node(terms, bias), sv.residual_mask(terms, bias)))
                    break
            if not out and first is None and 0 < ratio < 0.3:   # restart sa drugim vrhovima spektra
                firsts = sv._peaks(sv._spectrum([], np.zeros((0, 3), complex)), 6)[1:]
        return out
    except _Deadline:
        return []


# ============================================================== rings
def _isqrt(v):
    r = np.floor(np.sqrt(np.maximum(v, 0).astype(np.float64))).astype(np.int64)
    r = np.where(r * r > v, r - 1, r)
    return np.where((r + 1) * (r + 1) <= v, r + 1, r)


def rings_node(cx, cy, F, v, gx, gy, u) -> Node:
    """Kao fam.rings_video, ali frekvencija po kanalu može biti i BYTES[3](C)."""
    R = op("ISQRT", op("ADD", op("SQ", fam.lin([(1, X)], -cx, None)), op("SQ", fam.lin([(1, Y)], -cy, None))))
    F = [int(f) % 256 for f in F]
    opts = []
    s = (F[1] - F[0]) % 256
    if (F[2] - F[1]) % 256 == s:
        opts.append(fam.lin([(s, C)], F[0], 256))
    opts.append(bytes_lit(bytes(F), C))
    ramp = fam.lin([(gx, X), (gy, Y), (u, T)], 0, 256)
    best = None
    for Fn in opts:
        ring = fam.lin([(1, Node("MUL", (R, Fn))), (v, T)], 0, 256)
        n = op("XOR", ring, ramp)
        if best is None or fam.nbytes(n) < fam.nbytes(best):
            best = n
    return best


def infer_rings(V, meta, deadline):
    Tn, H, W, _ = V.shape
    if H < 4 or W < 4 or Tn < 2:
        return []
    V = V.astype(np.int64)
    rng = np.random.default_rng(777)
    t_ = np.arange(Tn)[:, None, None]
    y_ = np.arange(H)[None, :, None]
    x_ = np.arange(W)[None, None, :]
    # kandidati centra (porodica: u kadru; ovde i malo van njega)
    mx, my = max(2, W // 4), max(2, H // 4)
    cxs = np.arange(-mx, W + mx)
    cys = np.arange(-my, H + my)
    npx = min(H * W, 384)
    pix = rng.choice(H * W, npx, replace=False)
    py, px = pix // W, pix % W
    d2 = (px[None, None, :] - cxs[None, :, None]) ** 2 + (py[None, None, :] - cys[:, None, None]) ** 2
    Rpar = _isqrt(d2) & 1                                  # (cy, cx, piksel)
    centers = set()
    for c in range(3):
        B = V[..., c] & 1
        for e, g0, h0 in itertools.product((0, 1), repeat=3):
            if time.perf_counter() > deadline:
                return []
            tgt = B ^ ((e * t_ + g0 * x_ + h0 * y_) & 1)
            img = tgt[0]
            if np.mean(tgt != img[None]) > 0.01:          # mora biti nezavisno od t
                continue
            want = img.reshape(-1)[pix]
            bad = (Rpar != want[None, None, :]).sum(-1)
            for iy, ix in zip(*np.nonzero(bad <= max(1, npx // 100))):
                centers.add((int(cxs[ix]), int(cys[iy])))
        if len(centers) > 16:
            return []
    out = []
    ns = min(Tn * H * W, 512)
    sel = rng.choice(Tn * H * W, ns, replace=False)
    st, sy, sx = np.unravel_index(sel, (Tn, H, W))
    for cx, cy in sorted(centers):
        if time.perf_counter() > deadline:
            break
        Rimg = _isqrt((x_[0] - cx) ** 2 + (y_[0] - cy) ** 2)    # (H, W)
        Rs = Rimg[sy, sx]
        sols = None
        for c0 in range(3):
            Vs = V[st, sy, sx, c0]
            sols = _bitsearch(Rs, st, sy, sx, Vs, deadline)
            if sols:
                break
        if not sols:
            continue
        for (F0, v, gx, gy, u) in sols[:8]:
            Q = (gx * x_ + gy * y_ + u * t_) & 255
            F = []
            for c in range(3):
                Vs = V[st, sy, sx, c]
                Qs = Q[st, sy, sx]
                fs = np.arange(256)
                pred = ((Rs[None] * fs[:, None] + v * st[None]) & 255) ^ Qs[None]
                good = (pred != Vs[None]).sum(1)
                F.append(int(np.argmin(good)))
            pred = np.empty(V.shape, np.int64)
            for c in range(3):
                pred[..., c] = ((Rimg[None] * F[c] + v * t_) & 255) ^ Q
            if np.mean(pred != V) > EXC_FRAC:
                continue
            out.append((rings_node(cx, cy, F, v, gx, gy, u), pred))
    return out


def _bitsearch(Rs, st, sy, sx, Vs, deadline):
    """Niski bitovi (R*F + v*T) XOR (gx*X + gy*Y + u*T) zavise samo od niskih bitova parametara."""
    tol = max(1, len(Vs) // 100)
    cands = [(0, 0, 0, 0, 0)]
    for k in range(8):
        if time.perf_counter() > deadline:
            return []
        bit = 1 << k
        m = (bit << 1) - 1
        nxt = []
        P = np.array(cands, np.int64)
        ext = np.array(list(itertools.product((0, bit), repeat=5)), np.int64)
        allp = (P[:, None, :] + ext[None, :, :]).reshape(-1, 5)
        F, v, gx, gy, u = (allp[:, i][:, None] for i in range(5))
        pred = ((Rs[None] * F + v * st[None]) ^ (gx * sx[None] + gy * sy[None] + u * st[None])) & m
        bad = (pred != (Vs[None] & m)).sum(1)
        keep = allp[bad <= tol]
        nxt = [tuple(int(z) for z in row) for row in keep]
        if not nxt or len(nxt) > 4096:
            return []
        cands = nxt
    pred_bad = []
    for cnd in cands:
        F, v, gx, gy, u = cnd
        pred = ((Rs * F + v * st) ^ (gx * sx + gy * sy + u * st)) & 255
        pred_bad.append(int((pred != Vs).sum()))
    order = np.argsort(pred_bad)
    return [tuple(fam.rep(z, 256) for z in cands[i]) for i in order if pred_bad[i] <= tol]


# =============================================================== zvuk
def _tone(n, p, q):
    return ISIN_LUT[((n * p) >> q) & 255]


def infer_audio(a16: np.ndarray, deadline) -> list:
    """Kandidati (izraz, izuzeci) za zvuk; uvek sadrži i sirovu rezervu."""
    NS = len(a16)
    if NS == 0:
        return [(None, None)]
    a = a16.astype(np.int64)
    out = []
    if not a.any():
        return [(None, None)]
    out.append((words_lit(a16.view("<u2"), N), None))     # sirova rezerva (& 0xFFFF -> int16)
    n = np.arange(NS, dtype=np.int64)
    spec = np.abs(np.fft.rfft(a * np.hanning(NS), 8 * NS))
    freqs = np.fft.rfftfreq(8 * NS)
    peaks = []
    s2 = spec.copy()
    for _ in range(3):
        i = int(np.argmax(s2))
        if s2[i] <= 0:
            break
        peaks.append(freqs[i])
        lo, hi = max(0, i - 8 * 8), i + 8 * 8
        s2[lo:hi] = 0
    target = a & 0xFFFF
    best = None
    for q in range(0, 5):
        cands = []
        for f in peaks:
            p0 = int(round(f * 256 * (1 << q)))
            cands.append([p for p in range(max(1, p0 - 3), p0 + 4)])
        sets = [(p,) for c in cands[:2] for p in c]
        if len(cands) >= 2:
            sets += [(p1, p2) for p1 in cands[0] for p2 in cands[1]]
        for ps in sets:
            if time.perf_counter() > deadline:
                break
            M = np.stack([_tone(n, p, q) for p in ps], 1).astype(np.float64)
            sol, *_ = np.linalg.lstsq(M, a.astype(np.float64), rcond=None)
            A = [int(round(s)) for s in sol]
            if any(v == 0 for v in A):
                continue
            pred = sum(Ai * _tone(n, p, q) for Ai, p in zip(A, ps)) & 0xFFFF
            miss = np.flatnonzero(pred != target)
            if len(miss) > 0.02 * NS:
                continue
            tones = [{"A": Ai, "p": p, "q": q} for Ai, p in zip(A, ps)]
            exc = kfl.Exceptions(miss.astype(np.int64), (pred[miss] ^ target[miss]).astype(np.int64)) \
                if len(miss) else None
            expr = fam.audio_expr(tones)
            size = fam.nbytes(expr) + (len(exc.to_bytes()) if exc else 0)
            if best is None or size < best[0]:
                best = (size, expr, exc)
    if best is not None:
        out.append((best[1], best[2]))
    return out


# ============================================================ sklapanje
def _video_exc(V, pred):
    diff = (V.astype(np.int64) ^ pred).reshape(-1)
    pos = np.flatnonzero(diff)
    if not len(pos):
        return None
    return kfl.Exceptions(pos.astype(np.int64), diff[pos].astype(np.int64))


def encode(meta, payload, budget_s):
    t0 = time.perf_counter()
    if len(payload) != meta.payload_bytes or meta.frames < 1 or meta.width < 2 or meta.height < 2:
        return None
    if meta.audio_samples and (meta.audio_channels != 1 or meta.sample_bytes != 2):
        return None
    deadline = t0 + 0.8 * budget_s
    try:
        kfl.check_meta(meta)
    except ValueError:
        return None
    V = np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(meta.frames, meta.height, meta.width, 3)
    a16 = np.frombuffer(payload[meta.video_bytes:], dtype="<i2")
    audio_c = infer_audio(a16, deadline)
    vids = []
    with np.errstate(all="ignore"):
        vids += infer_rings(V, meta, deadline)
        near_exact = any(np.mean(pred != V) < 0.005 for _, pred in vids)
        if not near_exact and time.perf_counter() < deadline:
            vids += infer_sum(V, meta, deadline)
    if not vids:
        return None
    best = None
    for vnode, pred in vids:
        exc_v = _video_exc(V, pred)
        for anode, exc_a in audio_c:
            prog = kfl.Program(video=vnode, audio=anode, exc_video=exc_v, exc_audio=exc_a)
            rec = record.encode_kfl(meta, prog)
            if best is not None and len(rec) >= len(best):
                continue
            if time.perf_counter() - t0 > budget_s and best is not None:
                break
            if prog.run(meta) == payload:
                best = rec
    if best is not None and len(best) > 4096 and time.perf_counter() - t0 < 0.5 * budget_s:
        # MDL: kod velikih zapisa (mnogo izuzetaka) proveri i običnu rezervu
        try:
            alt = igc_codec.encode("lzma", meta, payload)
            if len(alt) < len(best) and record.decode(alt)[1] == payload:
                best = alt
        except Exception:
            pass
    return best


ENCODERS = [Encoder("shader_infer", encode, family_aware=True,
                    note="OMP nad razlikama mod 256 + tačne faze; prstenovi bit po bit; MDL izuzeci")]
