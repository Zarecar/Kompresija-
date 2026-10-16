"""MDL indukcija celobrojne dinamike za porodicu DIFFUSION (smooth i wave šabloni).

Family-aware: pretpostavlja šablone porodice (difuzija sa diskom-izvorom ili pakovani talas sa
oscilujućim tačkastim izvorom, afina paleta, ton iz SUMS), ali NE zna nijedan parametar.
  1. Paleta -> stanje: kanal c koji određuje ostale kanale; oznaka U = (k*V_c) & 255, k = inverz
     malog neparnog množioca (oznake iz različitih kanala koje se razlikuju za konstantu se
     spajaju). Ostaje dvosmislenost pomeraja q (i znaka): rešava je slaganje sa pravilom.
     Difuzija je invarijantna na pomeraj osim u okolinama koje seku rez 255|0, pa se q računa
     tačno nad svim prelazima; talas nije (prigušenje), pa se q traži zajedno sa pravilom.
  2. Pravilo: robustni (LTS) najmanji kvadrati nad okolinama bez preloma, pa celobrojna pretraga
     (sh, koeficijenti ±1) po broju TAČNO predviđenih ćelija; pravilo i q naizmenično.
  3. Izvor: ćelije neslaganja -> TRI putanja (x0, vx, y0, vy); R2 i amplituda (smooth) ili
     frekvencija i amplituda (wave) glasanjem po intervalima koje ostavlja >> sh.
  4. Kandidati se rangiraju po broju preostalih neslaganja na celom filmu: talas sa δ = 1 ima
     tačnu simetriju S -> -(2^sh - 1) - S, pa i "obrnuta" oznaka skoro savršeno odgovara.
  5. INIT: konstanta, HASH seed (gruba sila), zakrpa na pozadini ili ceo literal; prev0 talasa
     je 0, konstanta ili cur0.
  6. Paleta po kanalu (afina, inače tabela), zvuk (ton iz SUMS, stalni ton, tišina, WORDS).
  7. Preostala neslaganja -> izuzeci stanja. Bira se najkraći tačan zapis; rezerva je lzma.
"""
from __future__ import annotations

import time

import numpy as np

from igc import codec as igc_codec
from kfb import kfl, record
from kfb.families import diffusion as D

from . import Encoder

DEBUG = False                    # True: greške se ne gutaju (razvoj)
KINV = [pow(m % 256, -1, 256) for m in (1, -1, 3, -3, 5, -5, 7, -7)]
FIT_FRAMES = 10                  # prelazi za uklapanje pravila
MAX_SAMPLES = 6000
VMAX = 8                         # najveća brzina izvora koja se traži
R2_CANDS = sorted({a * a + b * b + 1 for a in range(8) for b in range(8) if a * a + b * b + 1 <= 50})
FMAX = 2048                      # ((N*f) >> 3) & 255 je periodično po f sa periodom 2048


class _Timeout(Exception):
    pass


# ------------------------------------------------------------ oznake stanja
def label(U: np.ndarray, q: int, kind: str) -> np.ndarray:
    if kind == "smooth":
        return (U + q) & 255
    return ((U + q + 128) & 255) - 128


def _gap_q(U: np.ndarray, kind: str) -> int:
    """Početni q: rez oznaka (255|0 ili 127|-128) tamo gde polje ima najmanje velikih skokova.
    Par susednih ćelija (u1, u2) daje skok ako je kratak (|u1-u2| < 128) a seče ga rez c
    (min < c <= max), ili ako je dug, a rez ga ne seče."""
    Tn = U.shape[0]
    Us = U[np.unique(np.linspace(0, Tn - 1, min(12, Tn)).astype(np.int64))]
    a = np.concatenate([Us[:, 1:, :].reshape(-1), Us[:, :, 1:].reshape(-1)])
    b = np.concatenate([Us[:, :-1, :].reshape(-1), Us[:, :, :-1].reshape(-1)])
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    short = hi - lo < 128
    diff = np.zeros(258, np.int64)
    np.add.at(diff, lo + 1, np.where(short, 1, -1))
    np.add.at(diff, hi + 1, np.where(short, -1, 1))
    jumps = int((~short).sum()) + np.cumsum(diff)[:256]       # jumps[c] za rez c
    c = int(np.argmin(jumps))
    return (-c) % 256 if kind == "smooth" else (128 - c) % 256


def _determines(V: np.ndarray, c: int) -> bool:
    """Da li kanal c jednoznačno određuje ostala dva kanala."""
    key = V[..., c].reshape(-1).astype(np.int64)
    for c2 in range(3):
        if c2 == c:
            continue
        o = V[..., c2].reshape(-1).astype(np.int64)
        tab = np.zeros(256, np.int64)
        tab[key] = o
        if not np.array_equal(tab[key], o):
            return False
    return True


# ------------------------------------------------------------ uklapanje pravila
def _rolls(F: np.ndarray, diag: bool) -> list:
    offs = ((1, 0), (-1, 0), (0, 1), (0, -1)) + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if diag else ())
    return [np.roll(np.roll(F, dy, -2), dx, -1) for dy, dx in offs]


def _features(kind: str, L: np.ndarray, ts: np.ndarray):
    """(obeležja [n, 3], cilj [n], informativna maska [n], maska bez preloma [n]) za prelaze ts.
    "Bez preloma": raspon oznaka u okolini (sa sledećom vrednošću) < 128 -- samo takve ćelije
    ulaze u regresiju, jer pogrešno postavljen rez oznaka daje ogromna odstupanja."""
    if kind == "smooth":
        s0, y = L[ts], L[ts + 1]
        nbs = _rolls(s0, True)
        n4 = sum(nbs[:4])
        n8 = n4 + sum(nbs[4:])
        F = np.stack([s0, n4, n8], -1).reshape(-1, 3)
        info = ((n8 != 8 * s0) | (y != s0)).reshape(-1)
        allv = [s0, y] + nbs
    else:
        c, p, y = L[ts], L[ts - 1], L[ts + 1]
        nbs = _rolls(c, False)
        n4 = sum(nbs)
        F = np.stack([c, n4, p], -1).reshape(-1, 3)
        info = ((n4 != 4 * c) | (y != c) | (p != c)).reshape(-1)
        allv = [c, p, y] + nbs
    wf = (np.maximum.reduce(allv) - np.minimum.reduce(allv) < 128).reshape(-1)
    return F, y.reshape(-1), info, wf


def _predict(kind: str, F: np.ndarray, coef: np.ndarray, sh: np.ndarray) -> np.ndarray:
    """F [n, 3], coef [m, 3], sh [m] -> predikcije [m, n]."""
    with np.errstate(all="ignore"):
        out = (coef @ F.T) >> sh[:, None]
    return out & 255 if kind == "smooth" else out


def _coef_vec(kind: str, rule) -> np.ndarray:
    if kind == "smooth":
        return np.array(rule[:3], np.int64)
    A, B, P, _ = rule
    return np.array([A, B, -P], np.int64)


def _rule_from(kind: str, cvec, sh) -> tuple:
    a, b, c = (int(v) for v in cvec)
    return (a, b, c, int(sh)) if kind == "smooth" else (a, b, -c, int(sh))


def fit_rule(kind: str, L: np.ndarray, ts: np.ndarray, rng, nmax: int = MAX_SAMPLES) -> tuple[float, tuple | None]:
    """Regresija + celobrojna pretraga nad prelazima ts; vraća (udeo tačnih, pravilo)."""
    F, y, info, wf = _features(kind, L, ts)
    idx = np.flatnonzero(info)
    if len(idx) > nmax:
        idx = np.sort(rng.choice(idx, nmax, replace=False))
    return _fit_feats(kind, F[idx], y[idx], wf=wf[idx])


def _fit_feats(kind: str, Fs: np.ndarray, ys: np.ndarray, fine: bool = True,
               wf: np.ndarray | None = None) -> tuple[float, tuple | None]:
    """Regresija (nad ćelijama bez preloma wf) + celobrojna pretraga skorovana nad svim ćelijama.
    fine=False: samo zaokruženi koeficijenti po sh (brzo, za pretragu q)."""
    if len(ys) < 30:
        return 0.0, None
    A = np.concatenate([Fs.astype(np.float64), np.ones((len(ys), 1))], 1)
    base = np.ones(len(ys), bool) if wf is None or wf.sum() < 30 else wf
    yf = ys.astype(np.float64)
    keep = base.copy()
    coef = None
    # LTS: koraci koncentracije nad boljom polovinom reziduala (izvor i prelomi su izuzeci,
    # kojih ima i do trećine informativnih ćelija), pa prag 1,5 (greška zaokruživanja < 1)
    for it in range(6):
        if keep.sum() < 10:
            return 0.0, None
        coef = np.linalg.lstsq(A[keep], yf[keep], rcond=None)[0]
        r = np.abs(yf - A @ coef)
        thr = max(1.5, float(np.quantile(r[base], 0.5))) if it < 3 else 1.5
        keep = base & (r < thr) if it < 3 else r < thr
    if coef is None or not np.all(np.isfinite(coef)) or np.abs(coef[:3]).max() > 64:
        return 0.0, None
    dl = DELTAS if fine else DELTAS[13:14]
    cands = np.concatenate([np.rint(coef[:3] * (1 << sh)).astype(np.int64)[None, :] + dl
                            for sh in range(10)])
    shs = np.repeat(np.arange(10), len(dl))
    match = (_predict(kind, Fs, cands, shs) == ys[None, :]).sum(1)
    i = np.lexsort((np.abs(cands).sum(1), shs, -match))[0]
    return float(match[i]) / len(ys), _rule_from(kind, cands[i], shs[i])


DELTAS = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T


def neigh_sample(kind: str, U: np.ndarray, ts: np.ndarray, n: int, rng) -> np.ndarray:
    """Okoline nasumičnih informativnih ćelija u sirovim oznakama U: [n, k].
    smooth: s, 4 suseda, 4 dijagonale, sledeće;  wave: prethodno, s, 4 suseda, sledeće."""
    s0 = U[ts]
    parts = [s0]
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        parts.append(np.roll(np.roll(s0, dy, 1), dx, 2))
    if kind == "smooth":
        for dy, dx in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            parts.append(np.roll(np.roll(s0, dy, 1), dx, 2))
    else:
        parts.insert(0, U[ts - 1])
    parts.append(U[ts + 1])
    M = np.stack(parts, -1).reshape(-1, len(parts))
    idx = np.flatnonzero((M != M[:, :1]).any(1))
    if len(idx) > n:
        idx = np.sort(rng.choice(idx, n, replace=False))
    return M[idx]


def sample_feats(kind: str, M: np.ndarray, q: int):
    """(obeležja, cilj, maska bez preloma) uzorka okolina M pri oznaci q."""
    Lm = label(M, q, kind)
    wf = Lm.max(1) - Lm.min(1) < 128
    if kind == "smooth":
        n4 = Lm[:, 1:5].sum(1)
        return np.stack([Lm[:, 0], n4, n4 + Lm[:, 5:9].sum(1)], 1), Lm[:, 9], wf
    return np.stack([Lm[:, 1], Lm[:, 2:6].sum(1), Lm[:, 0]], 1), Lm[:, 6], wf


def refine_q(kind: str, M: np.ndarray, rule, deadline: float) -> list[int]:
    """Svi q sa najviše tačno predviđenih ćelija uzorka M za dato pravilo."""
    coef = _coef_vec(kind, rule)[None, :]
    sh = np.array([rule[3]])
    score = np.zeros(256, np.int64)
    for q in range(256):
        if time.perf_counter() > deadline:
            raise _Timeout
        F, y, _ = sample_feats(kind, M, q)
        score[q] = int((_predict(kind, F, coef, sh)[0] == y).sum())
    return [int(q) for q in np.flatnonzero(score == score.max())]


def refine_q_conservative(U: np.ndarray, rule, deadline: float, cache: dict | None = None) -> list[int] | None:
    """Tačan skor za svaki q nad SVIM prelazima, za konzervativno pravilo (a + 4b + 8d = 2^sh).
    Oznaka q = (U - c) mod 256 sa rezom c = -q. Pravilo je invarijantno na pomeraj, pa ishod
    okoline zavisi samo od toga koje su njene vrednosti ispod reza: za sortirane vrednosti
    v_0 <= ... <= v_9 isti ishod važi za sve c iz (v_j, v_j+1]. Zato se po okolini računa
    najviše 9 slučajeva, a razlike skora se sabiraju po intervalima reza (bincount)."""
    a, b, d, sh = rule
    if a + 4 * b + 8 * d != (1 << sh):
        return None
    cache = {} if cache is None else cache
    if rule in cache:
        return cache[rule]
    if "Nb" not in cache:
        s0 = U[:-1].astype(np.uint8)
        parts = [s0] + _rolls(s0, True) + [U[1:].astype(np.uint8)]
        Nb = np.stack(parts, -1).reshape(-1, len(parts))
        Nb = Nb[Nb.max(1) > Nb.min(1)].astype(np.int32)
        cache["Nb"], cache["Sv"] = Nb, np.sort(Nb, axis=1)
    Nb, Sv = cache["Nb"], cache["Sv"]

    def ok(Lm):
        n4 = Lm[:, 1:5].sum(1)
        pred = ((a * Lm[:, 0] + b * n4 + d * (n4 + Lm[:, 5:9].sum(1))) >> sh) & 255
        return pred == Lm[:, 9]
    m0 = ok(Nb).astype(np.int64)
    diff = np.zeros(258, np.float64)
    for j in range(Nb.shape[1] - 1):
        if time.perf_counter() > deadline:
            raise _Timeout
        lo, hi = Sv[:, j], Sv[:, j + 1]
        r = np.flatnonzero(hi > lo)
        if not len(r):
            continue
        delta = ok((Nb[r] - hi[r, None]) & 255).astype(np.int64) - m0[r]
        nzr = np.flatnonzero(delta)
        if len(nzr):
            diff += np.bincount(lo[r][nzr] + 1, weights=delta[nzr], minlength=258)
            diff -= np.bincount(hi[r][nzr] + 1, weights=delta[nzr], minlength=258)
    score_c = int(m0.sum()) + np.rint(np.cumsum(diff)[:256]).astype(np.int64)
    score = score_c[(256 - np.arange(256)) % 256]          # q = -c mod 256
    cache[rule] = [int(q) for q in np.flatnonzero(score == score.max())]
    return cache[rule]


def scan_q(kind: str, M: np.ndarray, deadline: float) -> tuple[float, int]:
    """Zajednička pretraga q i pravila (za talas): najbolji udeo tačnih i q."""
    best = (-1.0, 0)
    for q in range(256):
        if time.perf_counter() > deadline:
            raise _Timeout
        F, y, wf = sample_feats(kind, M, q)
        sc, _ = _fit_feats(kind, F, y, fine=False, wf=wf)
        if sc > best[0]:
            best = (sc, q)
    return best


# ------------------------------------------------------------ izvor
def fit_track(Mc: np.ndarray, Lmax: int, ts: np.ndarray):
    """Mc [nT, n]: broj neslaganja po koloni/redu; vraća (p0, v, skor) TRI putanje.
    Profil se najpre gladi simetričnim jezgrom: profil diska ima ravan vrh, a glačanje
    pomera maksimum tačno u centar."""
    if Lmax <= 0:
        return 0, 0, 0
    ker = np.array([1, 2, 3, 2, 1], np.int64)
    pad = np.pad(Mc.astype(np.int64), ((0, 0), (2, 2)))
    Mc = sum(ker[j] * pad[:, j:j + Mc.shape[1]] for j in range(5))
    P = 2 * Lmax
    p0 = np.arange(P, dtype=np.int64)
    best = (0, 0, -1)
    for v in range(-min(VMAX, Lmax), min(VMAX, Lmax) + 1):
        pos = D.tri_value(p0[:, None] + v * ts[None, :], Lmax)       # [P, nT]
        sc = Mc[np.arange(len(ts))[None, :], pos].sum(1)
        i = int(np.argmax(sc))
        if sc[i] > best[2]:
            best = (int(p0[i]), v, int(sc[i]))
    return best


def _vote_mod(lo: np.ndarray, width: int, M: int) -> np.ndarray:
    """Glasovi za vrednosti mod M: svaki interval [lo, lo + width) mod M daje 1 glas."""
    diff = np.zeros(M + 1, np.int64)
    lo = np.mod(lo, M)
    hi = lo + width
    wrap = hi > M
    np.add.at(diff, lo, 1)
    np.add.at(diff, np.minimum(hi, M), -1)
    np.add.at(diff, np.zeros(int(wrap.sum()), np.int64), 1)
    np.add.at(diff, hi[wrap] - M, -1)
    return np.cumsum(diff[:M])


def _best_signed(votes: np.ndarray, M: int) -> int:
    """Vrednost sa najviše glasova, predstavnik najmanje apsolutne vrednosti."""
    cand = np.flatnonzero(votes == votes.max())
    signed = np.where(cand >= M // 2, cand - M, cand)
    return int(signed[np.lexsort((signed < 0, np.abs(signed)))[0]])


def fit_smooth_source(L, lin0, rule, ts):
    """Disk-izvor: gruba putanja iz neslaganja, pa zajedničko doterivanje (x0±1, y0±1, R2) i
    amplitude nad prozorom oko putanje (van prozora izvor ne utiče na neslaganja).
    Vraća (izvor, broj neslaganja, broj neslaganja unutar izvora)."""
    a, b, d, sh = rule
    H, W = L.shape[1:]
    obs = L[1:]
    mism = ((lin0 >> sh) & 255) != obs
    base_err = int(mism.sum())
    if base_err == 0:
        return None, 0, 0
    x0, vx, _ = fit_track(mism.sum(1), W - 1, ts)
    y0, vy, _ = fit_track(mism.sum(2), H - 1, ts)
    M = 1 << (sh + 8)
    # prozor: ćelije na rastojanju <= 9 od grube putanje (pokriva R2 <= 50 i pomeraj 1)
    px, py = D.src_positions({"x0": x0, "vx": vx, "y0": y0, "vy": vy}, W, H, ts)
    win = (np.abs(np.arange(H)[None, :, None] - py[:, None, None]) <= 9) & \
          (np.abs(np.arange(W)[None, None, :] - px[:, None, None]) <= 9)
    tt, yy, xx = np.nonzero(win)
    lw, ow = lin0[tt, yy, xx], obs[tt, yy, xx]
    err_out = base_err - int(mism[tt, yy, xx].sum())
    best = (None, base_err, base_err)
    for dx0 in (0, -1, 1):
        for dy0 in (0, -1, 1):
            src0 = {"x0": x0 + dx0 * (1 if vx >= 0 else -1), "vx": vx,
                    "y0": y0 + dy0 * (1 if vy >= 0 else -1), "vy": vy}
            qx, qy = D.src_positions(src0, W, H, ts)
            d2 = (xx - qx[tt]) ** 2 + (yy - qy[tt]) ** 2
            for R2 in R2_CANDS:
                inside = d2 < R2
                if not inside.any():
                    continue
                amp = _best_signed(_vote_mod((ow[inside] << sh) - lw[inside], 1 << sh, M), M)
                bad = (((lw + inside * amp) >> sh) & 255) != ow
                err = err_out + int(bad.sum())
                if err < best[1]:
                    best = (dict(src0, R2=R2, amp=amp), err, int((bad & inside).sum()))
    return best


def fit_wave_source(cur, prev, rule, ts):
    """Tačkasti izvor ISIN(T*w)*amp: putanja iz neslaganja, (w, amp) glasanjem po intervalima."""
    A, B, P, sh = rule
    H, W = cur.shape[1:]
    with np.errstate(all="ignore"):
        lin0 = A * cur[:-1] + B * D.nb4(cur[:-1]) - P * prev[:-1]
    obs = cur[1:]
    mism = (lin0 >> sh) != obs
    base_err = int(mism.sum())
    if base_err == 0:
        return None, 0, 0
    x0, vx, _ = fit_track(mism.sum(1), W - 1, ts)
    y0, vy, _ = fit_track(mism.sum(2), H - 1, ts)
    src = {"x0": x0, "vx": vx, "y0": y0, "vy": vy, "w": 0, "amp": 0}
    px, py = D.src_positions(src, W, H, ts)
    k = np.arange(len(ts))
    lo = (obs[k, py, px] << sh) - lin0[k, py, px]
    hi = lo + (1 << sh) - 1
    Amax = 1 << 14
    tops = []                                  # (glasovi, w, amp) kandidati
    for w in range(1, 256):
        s = D.ISIN_LUT[(ts * w) & 255]
        pos, neg, zer = s > 0, s < 0, s == 0
        alo = np.full(len(ts), -Amax, np.int64)
        ahi = np.full(len(ts), Amax, np.int64)
        alo[pos] = -((-lo[pos]) // s[pos])
        ahi[pos] = hi[pos] // s[pos]
        alo[neg] = -((-hi[neg]) // s[neg])
        ahi[neg] = lo[neg] // s[neg]
        bad = zer & ((lo > 0) | (hi < 0))
        ok = (alo <= ahi) & ~bad
        alo, ahi = np.clip(alo[ok], -Amax, Amax), np.clip(ahi[ok], -Amax, Amax)
        diff = np.zeros(2 * Amax + 2, np.int64)
        np.add.at(diff, alo + Amax, 1)
        np.add.at(diff, ahi + Amax + 1, -1)
        votes = np.cumsum(diff[:-1])
        vmax = int(votes.max())
        cand = np.flatnonzero(votes == vmax) - Amax
        for amp in cand[np.argsort(np.abs(cand), kind="stable")][:4]:
            if amp:
                tops.append((vmax, w, int(amp)))
    # glasanje ne mora da odredi znak/frekvenciju slabog izvora: odlučuje broj preostalih neslaganja
    tops.sort(key=lambda r: (-r[0], abs(r[2]), r[1]))
    best = (None, base_err, base_err)
    mask = D.drive_mask("wave", src, H, W, ts)
    for _, w, amp in tops[:12]:
        cs = dict(src, w=w, amp=amp)
        bad = ((lin0 + mask * D.drive_values("wave", cs, ts)[:, None, None]) >> sh) != obs
        err = int(bad.sum())
        if err < best[1]:
            best = (cs, err, int((bad & (mask > 0)).sum()))
    return best


# ------------------------------------------------------------ INIT
def find_hash_seed(S0: np.ndarray, deadline: float, max_bits: int = 24):
    H, W = S0.shape
    flat = S0.reshape(-1)
    if flat.min() < 0 or flat.max() > 255 or len(flat) < 8 or len(np.unique(flat)) < min(128, len(flat) // 4):
        return None
    chunk = 1 << 20
    for start in range(0, 1 << max_bits, chunk):
        if time.perf_counter() > deadline:
            return None
        s = np.arange(start, start + chunk, dtype=np.int64)
        cand = s[(D.hash64(s) >> D.HASH_SHIFT) == flat[0]]
        for i in range(1, 8):
            cand = cand[(D.hash64(np.int64(i) ^ cand) >> D.HASH_SHIFT) == flat[i]]
        for seed in cand.tolist():
            if np.array_equal(D.hash_init_values(int(seed), H, W), S0):
                return int(seed)
    return None


def init_candidates(kind: str, S0: np.ndarray, deadline: float) -> list[dict]:
    H, W = S0.shape
    vals, cnt = np.unique(S0, return_counts=True)
    if len(vals) == 1:
        return [{"type": "const", "v": int(vals[0])}]
    out = []
    if kind == "smooth":
        seed = find_hash_seed(S0, deadline)
        if seed is not None:
            out.append({"type": "hash", "seed": seed})
    bg = int(vals[np.argmax(cnt)])
    ys, xs = np.nonzero(S0 != bg)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    for (a0, a1, b0, b1) in {(y0, y1, x0, x1), (0, H, 0, W)}:
        blk = S0[a0:a1, b0:b1]
        off = int(min(0, blk.min()))
        if int(blk.max()) - off > 255:
            continue
        out.append({"type": "patch", "y0": a0, "x0": b0, "h": a1 - a0, "w": b1 - b0,
                    "data": (blk - off).reshape(-1), "off": off, "bg": bg})
    return out


# ------------------------------------------------------------ paleta i zvuk
def fit_palette(V: np.ndarray, E: np.ndarray) -> dict | None:
    """V_c = (m*E + o) & 255 po kanalu, inače tabela boja; None ako E ne određuje boju."""
    e8 = (E & 255).reshape(-1)
    tabs = []
    for c in range(3):
        v = V[..., c].reshape(-1).astype(np.int64)
        tab = np.zeros(256, np.int64)
        tab[e8] = v
        if not np.array_equal(tab[e8], v):
            return None
        tabs.append(tab)
    seen = np.unique(e8)
    ms, os_ = [], []
    mm = np.arange(256, dtype=np.int64)
    for c in range(3):
        o = (tabs[c][seen][None, :] - mm[:, None] * seen[None, :]) & 255      # [256, n]
        ok = np.flatnonzero((o == o[:, :1]).all(1))
        if not len(ok):
            ms = None
            break
        sg = np.where(ok >= 128, ok - 256, ok)
        j = int(np.lexsort((sg < 0, np.abs(sg)))[0])
        ms.append(int(sg[j]))
        os_.append(int(o[ok[j], 0]))
    if ms is not None:
        return {"m": ms, "o": os_}
    lut_ = np.stack(tabs, 1).reshape(-1)
    if E.min() >= 0:
        lut_ = lut_[:3 * (int(E.max()) + 1)]
    return {"lut": [int(x) for x in lut_]}


def _tone_ok(a16: np.ndarray, n: np.ndarray, amp: int) -> np.ndarray:
    """ok[f] za f u [0, FMAX): ISIN(((n*f) >> 3)) * amp == uzorci (uint16)."""
    f = np.arange(FMAX, dtype=np.int64)
    k = min(len(n), 12)
    pred = (D.ISIN_LUT[((n[None, :k] * f[:, None]) >> 3) & 255] * amp) & 0xFFFF
    ok = (pred == a16[None, :k]).all(1)
    for fi in np.flatnonzero(ok):
        p = (D.ISIN_LUT[((n * fi) >> 3) & 255] * amp) & 0xFFFF
        ok[fi] = np.array_equal(p, a16)
    return ok


def _clen(v: int) -> int:
    """Dužina konstante v u bajt-kodu KFL."""
    if 0 <= v < 64 or -16 <= v < 0:
        return 1
    u = (v << 1) if v >= 0 else ((-v << 1) - 1)
    return 1 + max(1, (u.bit_length() + 6) // 7)


def audio_candidates(meta, audio: np.ndarray, sums: np.ndarray, deadline: float) -> list:
    """Kandidati izraza zvuka (svi tačni); poslednji je uvek WORDS literal."""
    if not len(audio):
        return [None]
    a16 = audio.view("<u2").astype(np.int64)
    if not a16.any():
        return [None]
    out = []
    nz = np.abs(audio[audio != 0].astype(np.int64))
    g = int(np.gcd.reduce(nz)) if len(nz) else 1
    Tn = meta.frames
    n = np.arange(meta.audio_samples, dtype=np.int64)
    tt = np.clip((n * meta.fps) // meta.audio_rate, 0, Tn - 1) if meta.audio_rate else np.zeros_like(n)
    frames = [t for t in range(Tn) if np.any(tt == t)]
    for amp in (g, -g):
        if time.perf_counter() > deadline:
            break
        ok = np.zeros((Tn, FMAX), bool)
        good = True
        for t in frames:
            sel = tt == t
            ok[t] = _tone_ok(a16[sel], n[sel], amp)
            if not ok[t].any():
                good = False
                break
        if not good:
            continue
        fr = np.array(frames)
        common = np.flatnonzero(ok[fr].all(0))
        if len(common):
            out.append({"f": int(common[0]), "amp": amp})
        f0s = np.flatnonzero(ok[fr[0]])
        best = None
        Rs = np.arange(1, 1025, dtype=np.int64)
        cR = np.array([_clen(int(v)) for v in Rs])
        for k in range(0, 63):
            if time.perf_counter() > deadline:
                break
            r = np.mod((sums[fr] >> k)[None, :], Rs[:, None])            # [nR, nT]
            ck = 0 if k == 0 else 1 + _clen(k)                           # SHR(SUMS, k)
            for f0 in f0s[:4]:
                base = f0 - r[:, 0]
                idx = r + base[:, None]
                valid = (idx.min(1) >= 0) & (idx.max(1) < FMAX)
                hit = np.flatnonzero(valid & ok[fr[None, :], np.clip(idx, 0, FMAX - 1)].all(1))
                for i in hit[np.argsort(cR[hit], kind="stable")][:64]:
                    bs = int(base[i])
                    cost = ck + int(cR[i]) + (0 if bs == 0 else 1 + _clen(bs))
                    if best is None or cost < best[0]:
                        best = (cost, {"k": k, "R": int(Rs[i]), "base": bs, "amp": amp})
        if best is not None:
            out.append(best[1])
    out.append({"words": a16})
    return out


# ------------------------------------------------------------ sastavljanje
def _packed(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    return (cur << D.FIELD) | (prev & D.FMASK)


def build(meta, kind, rule, src, L, prev0, V, audio, deadline):
    """Kompletan program za dato pravilo/izvor i oznake L; None ako nije izrazivo."""
    Tn, H, W = L.shape
    ts = np.arange(Tn - 1, dtype=np.int64)
    if kind == "smooth":
        pred = D.smooth_next(L[:-1], rule, src, ts)
        tgt = L[1:]
        sums = L.reshape(Tn, -1).sum(1)
    else:
        prev = np.concatenate([prev0[None], L[:-1]])
        new = D.wave_new(L[:-1], prev[:-1], rule, src, ts)
        pred = _packed(new, L[:-1])
        Sp = _packed(L, prev)
        tgt = Sp[1:]
        sums = Sp.reshape(Tn, -1).sum(1)
    diff = pred != tgt
    exc = None
    if diff.any():
        vals = (pred ^ tgt)[diff]
        if vals.min() < 0:
            return None
        exc = kfl.Exceptions(np.flatnonzero(diff.reshape(-1)).astype(np.int64), vals.astype(np.int64))
    pal = fit_palette(V, L)
    if pal is None:
        return None
    inits = init_candidates(kind, L[0], deadline)
    if kind == "wave":
        if not prev0.any():
            p0 = "zero"
        elif np.array_equal(prev0, L[0]):
            p0 = "cur"
        elif (prev0 == prev0.flat[0]).all():
            p0 = int(prev0.flat[0])
        else:
            return None
        for it in inits:
            it["prev0"] = p0
            it["nonneg"] = bool(L[0].min() >= 0)
    init = min(inits, key=lambda it: D.blen(D.init_node(kind, it, H, W)))
    auds = audio_candidates(meta, audio, sums, deadline)
    aud = min(auds, key=lambda d: 0 if d is None else D.blen(D.audio_node(d)))
    return D.program_for(kind, rule, src, init, pal, aud, meta, exc_state=exc)


def _structural(meta, payload, deadline):
    """Generator kandidata zapisa (svaki kompletan, još neprovereni)."""
    Tn, H, W = meta.frames, meta.height, meta.width
    V = np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(Tn, H, W, 3)
    audio = np.frombuffer(payload[meta.video_bytes:], "<i2")
    rng = np.random.default_rng(12345)
    chans = [c for c in range(3) if _determines(V, c)]
    ts_s = np.unique(np.linspace(0, Tn - 2, min(FIT_FRAMES, Tn - 1)).astype(np.int64))
    ts_w = np.unique(np.linspace(1, Tn - 2, min(FIT_FRAMES, Tn - 2)).astype(np.int64))
    cands = []
    seen = set()
    for c in chans:
        for k in KINV:
            U = (k * V[..., c].astype(np.int64)) & 255
            # oznake iz različitih kanala koje se razlikuju samo za konstantu su iste (do q)
            sig = ((U[::max(1, Tn // 4)] - U.flat[0]) & 255).tobytes()
            if sig in seen:
                continue
            seen.add(sig)
            for kind, ts in (("smooth", ts_s), ("wave", ts_w)):
                if time.perf_counter() > deadline:
                    raise _Timeout
                q0 = _gap_q(U, kind)
                sc, rule = fit_rule(kind, label(U, q0, kind), ts, rng)
                if rule is not None:
                    cands.append((sc, kind, c, k, rule))
    # drugi krug: za svaku vrstu oznake blizu najbolje, q se traži (talas: zajedno sa pravilom)
    stage = []
    kinds = sorted((("smooth", ts_s), ("wave", ts_w)),
                   key=lambda kt: -max([r[0] for r in cands if r[1] == kt[0]], default=0))
    for kind, ts in kinds:
        kc = sorted([r for r in cands if r[1] == kind], key=lambda r: -r[0])
        if not kc:
            continue
        top = kc[0][0]
        for sc, _, c, k, rule in kc[:4]:
            # skor pri početnom q je samo grub filter (pogrešan q ga jako obara)
            if sc < max(0.3, top - 0.5):
                break
            U = (k * V[..., c].astype(np.int64)) & 255
            M = neigh_sample(kind, U, ts, 3000 if kind == "wave" else 20000, rng)
            if len(M) < 30:
                continue
            if kind == "wave":
                # gruba pretraga (samo zaokruživanje), pa naizmenično pravilo <-> tačan q na uzorku
                _, q = scan_q(kind, M, deadline)
                sc2, rule2 = fit_rule(kind, label(U, q, kind), ts, rng)
                for _ in range(3):
                    if rule2 is None:
                        break
                    qs = refine_q(kind, M, rule2, deadline)
                    if q in qs:
                        break
                    q = min(qs, key=lambda qq: min((qq - q) % 256, (q - qq) % 256))
                    sc2, rule2 = fit_rule(kind, label(U, q, kind), ts, rng)
            else:
                # pravilo i q se uklapaju naizmenično dok se ne ustale (pravilo dobijeno pri
                # pogrešnom q može malo da odstupa, npr. NB4 i NB8 su korelisani)
                def bg_cost(q):
                    v, cnt = np.unique(label(U[0], q, kind), return_counts=True)
                    return 0 if v[np.argmax(cnt)] == 0 else 1
                sc2, rule2, q, qcache = sc, rule, None, {}
                for _ in range(4):
                    qs = refine_q_conservative(U, rule2, deadline, qcache) or refine_q(kind, M, rule2, deadline)
                    q_new = q if q in qs else min(qs[:64], key=bg_cost)
                    sc_new, rule_new = fit_rule(kind, label(U, q_new, kind), ts, rng)
                    if rule_new is None:
                        break
                    stable = q_new == q and rule_new == rule2
                    sc2, rule2, q = sc_new, rule_new, q_new
                    if stable:
                        break
                if q is not None and len(qs) > 1:
                    # stvarna dvosmislenost (polje nikad ne seče rez): bira se q za koji je
                    # INIT HASH šum (seed <= 2^20), inače onaj sa pozadinom 0
                    for qq in sorted(qs[:8], key=bg_cost):
                        L0 = label(U[0], qq, kind)
                        if len(np.unique(L0)) >= 128 and find_hash_seed(L0, deadline, 20) is not None:
                            q = qq
                            break
            if rule2 is not None and q is not None:
                stage.append((sc2, kind, c, k, q, rule2))
    stage.sort(key=lambda r: -r[0])
    # najviše tri kandidata po šablonu; konačni redosled određuje broj preostalih neslaganja
    # na celom filmu (posle izvora), jer slab talas daje skoro isti skor i za obrnut znak oznake
    stage = [r for i, r in enumerate(stage) if sum(x[1] == r[1] for x in stage[:i]) < 3]
    fitted = []
    Tn = meta.frames
    tall = np.arange(Tn - 1, dtype=np.int64)
    for sc, kind, c, k, q, rule in stage:
        if sc < 0.5:
            continue
        if time.perf_counter() > deadline:
            raise _Timeout
        U = (k * V[..., c].astype(np.int64)) & 255
        L = label(U, q, kind)
        prev0 = None
        if kind == "wave":
            # prev0: nula, konstanta (pozadina) ili cur0 -- koja hipoteza bolje objašnjava prvi
            # prelaz. Konstanta je potrebna jer pravila sa δ = 1 imaju tačnu simetriju
            # S -> -(2^sh - 1) - S, pa i "obrnuta" oznaka opisuje film, ali sa prev0 = konst.
            A, B, P, sh = rule
            vals, cnt = np.unique(L[0], return_counts=True)
            hyps = [np.zeros_like(L[0]), np.full_like(L[0], vals[np.argmax(cnt)]), L[0].copy()]
            errs = [int((((A * L[0] + B * D.nb4(L[0]) - P * p0) >> sh) != L[1]).sum()) for p0 in hyps]
            prev0 = hyps[int(np.argmin(errs))]
        # pravilo bez izvora ne razlikuje (a, sh) od (2a, sh+1); izvor može (neparna amplituda),
        # pa se veći sh proba samo ako greške ostaju baš na izvoru
        best_fit = None
        for mult in (1, 2, 4):
            r = tuple(v * mult for v in rule[:3]) + (rule[3] + mult.bit_length() - 1,)
            if kind == "smooth":
                a, b, d, sh = r
                with np.errstate(all="ignore"):
                    lin0 = a * L[:-1] + b * D.nb4(L[:-1]) + (d * D.nb8(L[:-1]) if d else 0)
                src, err, err_in = fit_smooth_source(L, np.asarray(lin0, np.int64), r, tall)
            else:
                prev = np.concatenate([prev0[None], L[:-1]])
                src, err, err_in = fit_wave_source(L, prev, r, tall)
            if best_fit is None or err < best_fit[2]:
                best_fit = (r, src, err)
            if err_in == 0 or time.perf_counter() > deadline:
                break
        fitted.append((best_fit[2], -sc, kind, best_fit[0], best_fit[1], L, prev0))
        r, src = best_fit[0], best_fit[1]
        if kind == "wave" and (1 << r[3]) - (r[0] + 4 * r[1] - r[2]) == 1:
            # tačna simetrija talasa (δ = 1): L -> -(2^sh - 1) - L, amp -> -amp; obe slike
            # opisuju film, ali se razlikuju INIT (prev0) i SUMS (zvuk), pa se grade obe
            cc = (1 << r[3]) - 1
            Lm = -cc - L
            if Lm.min() >= -128 and Lm.max() <= 127:
                srcm = None if src is None else dict(src, amp=-src["amp"])
                fitted.append((best_fit[2], -sc, kind, r, srcm, Lm, -cc - prev0))
    fitted.sort(key=lambda r: (r[0], r[1]))
    # grade se svi kandidati blizu najmanjeg broja neslaganja (simetrične oznake daju zapise
    # različite dužine), najviše šest; MDL bira najkraći
    emin = fitted[0][0] if fitted else 0
    fitted = [r for r in fitted if r[0] <= emin + max(8, emin // 10)][:6]
    for err, _, kind, rule, src, L, prev0 in fitted:
        if time.perf_counter() > deadline:
            raise _Timeout
        prog = build(meta, kind, rule, src, L, prev0, V, audio, deadline)
        if prog is not None:
            yield record.encode_kfl(meta, prog)


def encode(meta, payload, budget_s):
    t0 = time.perf_counter()
    # pretraga staje na 0,8 budžeta; kod malih budžeta ostaje bar 0,5 s za poslednji
    # nedeljivi korak (npr. tabela svih okolina na std), da se budžet ne prekorači
    deadline = t0 + max(0.05, min(0.8 * budget_s, budget_s - 0.5))
    if len(payload) != meta.payload_bytes:
        return None
    best = None

    def offer(rec):
        nonlocal best
        if rec is None or (best is not None and len(rec) >= len(best)):
            return
        try:
            m2, out, used = record.decode(rec)
        except Exception:
            if DEBUG:
                raise
            return
        if out == payload and used == len(rec):
            best = rec

    ok_meta = (meta.frames >= 3 and meta.height >= 3 and meta.width >= 3 and
               (meta.audio_samples == 0 or (meta.audio_channels == 1 and meta.sample_bytes == 2)))
    if ok_meta:
        try:
            kfl.check_meta(meta)
            for rec in _structural(meta, payload, deadline):
                offer(rec)
                if time.perf_counter() > deadline:
                    break
        except _Timeout:
            pass
        except Exception:
            if DEBUG:
                raise
    # obična rezerva (lzma) kada strukturni zapis ne postoji ili nije mnogo manji od sadržaja;
    # samo ako preostalo vreme sigurno pokriva lzma (~1,5 MB/s)
    left = budget_s - (time.perf_counter() - t0)
    if (best is None or len(best) * 100 > meta.payload_bytes) and left > 0.3 + meta.payload_bytes / 1.0e6:
        offer(igc_codec.encode("lzma", meta, payload))
    return best


ENCODERS = [Encoder("diffusion_infer", encode, family_aware=True,
                    note="paleta->stanje, celobrojna regresija pravila, izvor (TRI), HASH seed, izuzeci stanja")]
