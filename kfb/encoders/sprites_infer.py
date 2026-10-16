"""Inverzija porodice SPRITES: pozadina, segmentacija po boji, putanje (TRI), oblik, z-red.

Family-aware: pretpostavlja konvencije porodice sprites (pozadina baza_c + ax_c*X + ay_c*Y,
jednobojni pravougaonici i krugovi, gornji levi ugao TRI(x0 + v*T, W - w), kasniji objekat
iznad ranijeg, ton ISIN((N*f) >> s) * amp), ali SVE parametre izvodi iz bajtova filma.
Koraci: medijana po vremenu -> gradijent (moda razlika mod 256) -> maska prednjeg plana ->
boje objekata -> po boji: veličina i putanja po osi (gruba sila nad (x0, v) sa bodovanjem
ivica, uži izbor po tačnoj grešci oblika, po potrebi sve veličine i naizmenična tačna pretraga
osa) -> oblik (pravougaonik / krug sa R2) -> z-red iz preklapanja -> program; ostatak ide
u izuzetke videa (MDL), zvuk: ton (pretraga f, s, amp) / WORDS / izuzeci. Rezerva: lzma.
"""
from __future__ import annotations

import time

import numpy as np

from igc import codec as igc_codec
from kfb import kfl, record
from kfb.families.sprites import blen, program_for, tone_node
from kfb.kfl import ISIN_LUT, N, words_lit

from . import Encoder

MAX_OBJECTS = 16
MIN_PIXELS = 6          # boja je kandidat za objekat ako u nekom frejmu ima >= ovoliko piksela
TONE_MAX_SHIFT = 7
MAX_EXC_FRAC = 1 / 32   # više izuzetaka od ovog udela bajtova videa: model se odbacuje
LZMA_BPS = 1.0e6        # gruba procena brzine lzma rezerve (bajtova u sekundi)
VIOL = 8                # kazna za frejm u kome vidljivi pikseli izlaze iz pretpostavljenog okvira
SHORT = 8               # najmanji uži izbor hipoteza (veličina, putanja) za proveru oblika
MAX_SHAPE_FRAMES = 128  # frejmova za proveru oblika (uzorak kod dužih filmova)
REFINE_ROUNDS = 2       # krugova naizmenične tačne pretrage osa


def _tri(a, L):
    if L <= 0:
        return np.zeros_like(a)
    return L - np.abs(np.mod(a, 2 * L) - L)


def _mode256(a) -> int:
    a = np.asarray(a).ravel()
    if not a.size:
        return 0
    return int(np.bincount(a & 255, minlength=256).argmax())


def _signed(v: int) -> int:
    return ((int(v) + 128) % 256) - 128


# ------------------------------------------------------------ pozadina
def _fit_bg(video):
    T, H, W, _ = video.shape
    B = np.partition(video, T // 2, axis=0)[T // 2].astype(np.int64)   # (H, W, 3)
    xs, ys = np.arange(W)[None, :], np.arange(H)[:, None]
    base, ax, ay = [], [], []
    for c in range(3):
        a = _signed(_mode256(B[:, 1:, c] - B[:, :-1, c]))
        b = _signed(_mode256(B[1:, :, c] - B[:-1, :, c]))
        base.append(_mode256(B[:, :, c] - a * xs - b * ys))
        ax.append(a)
        ay.append(b)
    bg = {"base": base, "ax": ax, "ay": ay}
    img = np.stack([(base[c] + ax[c] * xs + ay[c] * ys) & 255 for c in range(3)], -1).astype(np.uint8)
    return bg, img


# ------------------------------------------------------------ putanje
def _fit_axis(lo, hi, tp, n, size, top=2):
    """Najbolji (x0, v) za ivicu TRI(x0 + v*t, n - size) po bodovanju ivica.

    Vidljiva ivica koja se poklapa daje +1, frejm u kome vidljivi pikseli izlaze iz okvira
    daje -VIOL (zaklonjena ivica ne daje ništa). Vraća do `top` kandidata (skor, x0, v).
    """
    L = n - size
    if L <= 0:
        pos = np.zeros(len(tp), np.int64)
        sc = int((pos == lo).sum() + (pos + size - 1 == hi).sum()
                 - VIOL * ((pos > lo) | (pos + size - 1 < hi)).sum())
        return [(sc, 0, 0)]
    a = np.arange(2 * L, dtype=np.int64)[:, None]
    res = []
    for v in range(L + 1):
        pos = _tri(a + v * tp[None, :], L)
        end = pos + (size - 1)
        sc = (pos == lo).sum(1) + (end == hi).sum(1) - VIOL * ((pos > lo) | (end < hi)).sum(1)
        for i in np.argsort(-sc, kind="stable")[:top].tolist():
            res.append((int(sc[i]), i, v))
    res.sort(key=lambda r: -r[0])
    return res[:top]


def _size_cands(lo, hi, n, extra=()):
    """Kandidati veličine: najčešće vidljive, najveća (i malo veće, zbog zaklanjanja)."""
    ext = hi - lo + 1
    vals, cnt = np.unique(ext, return_counts=True)
    mx = int(vals.max())
    c = [int(v) for v in vals[np.argsort(-cnt, kind="stable")][:3]] + [mx, mx + 1, mx + 2] + list(extra)
    return [v for v in dict.fromkeys(c) if 1 <= v <= n]


def _box_count(II, t, y0, y1, x0, x1):
    """Broj piksela maske u [y0, y1) x [x0, x1) za frejmove t (integralna slika II)."""
    return II[t, y1, x1] - II[t, y0, x1] - II[t, y1, x0] + II[t, y0, x0]


def _integral(M):
    T, H, W = M.shape
    II = np.zeros((T, H + 1, W + 1), np.int32)
    II[:, 1:, 1:] = M.cumsum(1, dtype=np.int32).cumsum(2, dtype=np.int32)
    return II


def _positions(o, T, W, H):
    t = np.arange(T, dtype=np.int64)
    return _tri(o["x0"] + o["vx"] * t, W - o["w"]), _tri(o["y0"] + o["vy"] * t, H - o["h"])


def _templates(w, h):
    """Kandidati oblika: pravougaonik i (za neparan kvadrat) krugovi sa R2 u [r^2, r^2 + 2r]."""
    out = [({"shape": "rect"}, np.ones((h, w), bool))]
    if w == h and w % 2 == 1 and w >= 3:
        r = (w - 1) // 2
        d = np.arange(w) - r
        d2 = d[:, None] ** 2 + d[None, :] ** 2
        for R2 in range(r * r, r * r + 2 * r + 1):
            out.append(({"shape": "circ", "R2": R2}, d2 <= R2))
    return out


def _search(M, bgvis, tp, edges, ws, hs, W, H):
    """Najbolja hipoteza za date kandidate veličina: ((greška oblika, ključ), parametri)."""
    T = M.shape[0]
    xlo, xhi, ylo, yhi = edges
    xc = [(sc, w, a, v) for w in ws for sc, a, v in _fit_axis(xlo, xhi, tp, W, w)]
    yc = [(sc, h, a, v) for h in hs for sc, a, v in _fit_axis(ylo, yhi, tp, H, h)]
    xc.sort(key=lambda q: -q[0])
    yc.sort(key=lambda q: -q[0])
    xc, yc = xc[:10], yc[:10]
    IM = _integral(M)
    tot = IM[tp, H, W]
    xpos = [_tri(a + v * tp, W - w) for _, w, a, v in xc]
    ypos = [_tri(a + v * tp, H - h) for _, h, a, v in yc]
    cands = []
    for i, (sx, w, _, _) in enumerate(xc):
        px = xpos[i]
        for j, (sy, h, _, _) in enumerate(yc):
            py = ypos[j]
            out = int((tot - _box_count(IM, tp, py, py + h, px, px + w)).sum())
            cy, cx = py + h // 2, px + w // 2
            row = bgvis[tp[:, None], cy[:, None], px[:, None] + np.arange(w)[None, :]]
            col = bgvis[tp[:, None], py[:, None] + np.arange(h)[None, :], cx[:, None]]
            cands.append(((out + int(row.sum()) + int(col.sum()), -(sx + sy)), i, j))
    cands.sort()
    # Uži izbor: sve hipoteze sa najmanjom greškom okvira (zaklonjen objekat ih ima više), bar SHORT.
    short = [q for q in cands if q[0][0] == cands[0][0][0]][:4 * SHORT]
    short += [q for q in cands[:SHORT] if q not in short]
    # Tačna greška oblika na svim frejmovima (i onim gde je objekat potpuno zaklonjen):
    # vidljivi pikseli boje van oblika + vidljiva pozadina u obliku = vis + sum(tm * D),
    # D = (#frejmova sa pozadinom) - (#frejmova sa bojom) po pikselu prozora.
    ft = np.linspace(0, T - 1, min(T, MAX_SHAPE_FRAMES)).astype(np.int64)
    vis = int(M[ft].sum())
    best = None
    for key, i, j in short:
        _, w, a, v = xc[i]
        _, h, b, u = yc[j]
        px, py = _tri(a + v * ft, W - w), _tri(b + u * ft, H - h)
        rr = py[:, None] + np.arange(h)[None, :]
        cc = px[:, None] + np.arange(w)[None, :]
        win = M[ft[:, None, None], rr[:, :, None], cc[:, None, :]].sum(0, dtype=np.int64)
        D = bgvis[ft[:, None, None], rr[:, :, None], cc[:, None, :]].sum(0, dtype=np.int64) - win
        for par, tm in _templates(w, h):
            k2 = (vis + int(D[tm].sum()), key)
            if best is None or k2 < best[0]:
                best = (k2, i, j, par)
    k2, i, j, par = best
    o = {"w": xc[i][1], "h": yc[j][1], "x0": xc[i][2], "vx": xc[i][3], "y0": yc[j][2], "vy": yc[j][3]}
    o.update(par)
    return k2, o


def _axis_given(D, ft, px, w, hmin, n, bound):
    """Tačna pretraga ose y (veličina h >= hmin, putanja (b, u), oblik) uz fiksnu osu x.

    D: (F, H, W) = pozadina - boja na frejmovima ft, px: levi rub po frejmu, w: širina.
    Greška hipoteze je vis + sum_t E[t, py_t]; vraća (e, h, b, u, oblik) sa e < bound ili None.
    """
    F, H, W = D.shape
    P = np.zeros((F, H, W + 1), np.int64)
    P[:, :, 1:] = D.cumsum(2)
    fi = np.arange(F)
    best = None
    # E[t, q]: zbir D u obliku sa gornjim redom q (pravougaonik: preko prefiksnih zbirova)
    rows = P[fi, :, px + w] - P[fi, :, px]                          # (F, H)
    Q = np.zeros((F, H + 1), np.int64)
    Q[:, 1:] = rows.cumsum(1)
    shapes = [(h, {"shape": "rect"}, Q[:, h:] - Q[:, :n - h + 1]) for h in range(max(1, hmin), n + 1)]
    if w % 2 == 1 and 3 <= w <= n and w >= hmin:                    # krug: h = w, red po red
        for par, tm in _templates(w, w)[1:]:
            E = np.zeros((F, n - w + 1), np.int64)
            for dy in range(w):
                xs = np.flatnonzero(tm[dy])
                lo, hi = xs[0], xs[-1] + 1
                seg = P[fi, :, px + hi] - P[fi, :, px + lo]           # (F, H)
                E += seg[:, dy:dy + n - w + 1]
            shapes.append((w, par, E))
    # donja granica po veličini (najbolje mesto u svakom frejmu, bez putanje): obrezivanje
    lbs = [int(E.min(1).sum()) for _, _, E in shapes]
    for k in np.argsort(lbs, kind="stable").tolist():
        h, par, E = shapes[k]
        if lbs[k] >= bound:
            break
        L = n - h
        if L <= 0:
            e, b, u = int(E[:, 0].sum()), 0, 0
        else:
            a = np.arange(2 * L, dtype=np.int64)[:, None]
            e, b, u = None, 0, 0
            for uu in range(L + 1):
                ee = E[fi[None, :], _tri(a + uu * ft[None, :], L)].sum(1)
                j = int(ee.argmin())
                if e is None or ee[j] < e:
                    e, b, u = int(ee[j]), j, uu
        if e < bound:
            bound = e
            best = (e, h, b, u, par)
    return best


def _refine(M, bgvis, err, o, W, H, deadline):
    """Naizmenična tačna pretraga jedne ose uz fiksnu drugu, dok greška oblika opada.

    Za objekat čije se ivice retko vide (npr. vidi se samo uska traka) bodovanje ivica ne
    određuje putanju; ovde se greška oblika računa za SVE (veličina, b, u) jedne ose.
    """
    T = M.shape[0]
    ft = np.linspace(0, T - 1, min(T, MAX_SHAPE_FRAMES)).astype(np.int64)
    D = bgvis[ft].astype(np.int64) - M[ft]
    vis = int(M[ft].sum())
    tp = np.flatnonzero(M.any((1, 2)))
    rows, cols = M[tp].any(2), M[tp].any(1)
    ext_y = int((H - rows[:, ::-1].argmax(1) - rows.argmax(1)).max())
    ext_x = int((W - cols[:, ::-1].argmax(1) - cols.argmax(1)).max())
    for _ in range(REFINE_ROUNDS):
        moved = False
        for ax in "yx":
            if not err or time.perf_counter() > deadline:
                return err, o
            if ax == "y":
                px = _tri(o["x0"] + o["vx"] * ft, W - o["w"])
                r = _axis_given(D, ft, px, o["w"], ext_y, H, err - vis)
                if r is not None:
                    o = {"w": o["w"], "h": r[1], "x0": o["x0"], "vx": o["vx"], "y0": r[2], "vy": r[3], **r[4]}
            else:
                py = _tri(o["y0"] + o["vy"] * ft, H - o["h"])
                r = _axis_given(D.transpose(0, 2, 1), ft, py, o["h"], ext_x, W, err - vis)
                if r is not None:
                    o = {"w": r[1], "h": o["h"], "x0": r[2], "vx": r[3], "y0": o["y0"], "vy": o["vy"], **r[4]}
            if r is not None:
                err, moved = vis + r[0], True
        if not moved:
            break
    return err, o


def _fit_object(M, bgvis, W, H, deadline=float("inf")):
    """M: (T, H, W) maska jedne boje, bgvis: gde se vidi pozadina. Vraća parametre ili None.

    Kandidati (veličina, putanja) po osama se kombinuju i najpre rangiraju po grešci okvira
    (vidljivi pikseli van okvira + vidljiva pozadina na središnjem redu/koloni), a uži izbor
    po tačnoj grešci oblika (vidljiva boja van oblika + vidljiva pozadina u obliku). Obe su
    nemoguće za tačnu hipotezu, nezavisno od zaklanjanja. Ako najbolja hipoteza nije bez
    greške (npr. objekat se nikad ne vidi ceo), pretraga se ponavlja nad svim veličinama,
    pa sledi naizmenična tačna pretraga osa (_refine).
    """
    cols, rows = M.any(1), M.any(2)
    present = cols.any(1)
    if not present.any():
        return None
    tp = np.flatnonzero(present).astype(np.int64)
    c, r = cols[present], rows[present]
    xlo, xhi = c.argmax(1), W - 1 - c[:, ::-1].argmax(1)
    ylo, yhi = r.argmax(1), H - 1 - r[:, ::-1].argmax(1)
    edges = (xlo, xhi, ylo, yhi)
    ws0, hs0 = _size_cands(xlo, xhi, W), _size_cands(ylo, yhi, H)
    ws, hs = _size_cands(xlo, xhi, W, hs0[:2]), _size_cands(ylo, yhi, H, ws0[:2])   # krug je kvadrat
    best = _search(M, bgvis, tp, edges, ws, hs, W, H)
    T = M.shape[0]
    vis = int(M[np.linspace(0, T - 1, min(T, MAX_SHAPE_FRAMES)).astype(np.int64)].sum())
    # Hipoteza sa greškom nije tačna; skuplja pretraga samo ako objašnjava više nego što
    # promašuje (greška < vidljivih piksela), inače je to verovatno tuđ film.
    if 0 < best[0][0] < vis and time.perf_counter() < deadline:
        # veličina nikad ne može biti manja od vidljivog raspona
        wa = list(range(int((xhi - xlo).max()) + 1, W + 1))
        ha = list(range(int((yhi - ylo).max()) + 1, H + 1))
        alt = _search(M, bgvis, tp, edges, wa, ha, W, H)
        if alt[0] < best[0]:
            best = alt
    err, o = best[0][0], best[1]
    if 0 < err < vis and time.perf_counter() < deadline:
        err, o = _refine(M, bgvis, err, o, W, H, deadline)
    return o


def _render_mask(o, T, W, H):
    px, py = _positions(o, T, W, H)
    xs, ys = np.arange(W)[None, :], np.arange(H)[None, :]
    if o["shape"] == "rect":
        inx = (xs >= px[:, None]) & (xs < px[:, None] + o["w"])
        iny = (ys >= py[:, None]) & (ys < py[:, None] + o["h"])
        return iny[:, :, None] & inx[:, None, :]
    r = (o["w"] - 1) // 2
    dx2 = (xs - px[:, None] - r) ** 2
    dy2 = (ys - py[:, None] - r) ** 2
    return dy2[:, :, None] + dx2[:, None, :] <= o["R2"]


def _z_order(video, packed, objs, pred):
    """Redosled crtanja (dno -> vrh) iz vidljivosti u preklapanjima.

    Dokaz je broj piksela tačne boje u preklapanju; pri izjednačenju (npr. preklapanje je
    oštećeno šumom) broje se kanali koji se poklapaju, samo u pikselima koje ne objašnjava
    tačna boja nijednog objekta.
    """
    k = len(objs)
    pc = np.array([(o["color"][0] << 16) | (o["color"][1] << 8) | o["color"][2] for o in objs], np.int64)
    win = np.zeros((k, k), np.int64)     # win[i, j] > 0: i je iznad j
    for i in range(k):
        for j in range(i + 1, k):
            O = pred[i] & pred[j]
            if not O.any():
                continue
            pk = packed[O]
            d = int((pk == pc[i]).sum()) - int((pk == pc[j]).sum())
            if not d:
                px = video[O][~np.isin(pk, pc)]
                d = (int((px == np.array(objs[i]["color"], np.uint8)).sum())
                     - int((px == np.array(objs[j]["color"], np.uint8)).sum()))
            win[i, j], win[j, i] = d, -d
    order, rest = [], list(range(k))
    while rest:
        free = [i for i in rest if not any(win[i, j] > 0 for j in rest if j != i)]
        pick = free[0] if free else min(rest, key=lambda i: sum(max(0, win[i, j]) for j in rest))
        order.append(pick)
        rest.remove(pick)
    return order


def _objects(video, bg_img, deadline):
    T, H, W, _ = video.shape
    v = video.astype(np.int64)
    packed = (v[..., 0] << 16) | (v[..., 1] << 8) | v[..., 2]
    fg = (video != bg_img[None]).any(-1)
    tt = np.broadcast_to(np.arange(T, dtype=np.int64)[:, None, None], fg.shape)[fg]
    keys, cnt = np.unique((tt << 24) | packed[fg], return_counts=True)
    cols = keys & 0xFFFFFF
    big = cnt >= MIN_PIXELS
    ucol = np.zeros(0, np.int64)
    if big.any():
        tot = np.bincount(np.unique(cols[big], return_inverse=True)[1], weights=cnt[big])
        ucol = np.unique(cols[big])[np.argsort(-tot, kind="stable")]
    # slabo vidljiva boja (stalno skoro zaklonjen objekat): >= 2 piksela u bar četvrtini frejmova
    wc, wn = np.unique(cols[cnt >= 2], return_counts=True)
    weak = wc[(wn >= max(3, T // 4)) & ~np.isin(wc, ucol)]
    if len(weak):
        wt = np.bincount(np.searchsorted(weak, cols[np.isin(cols, weak)]), weights=cnt[np.isin(cols, weak)])
        ucol = np.concatenate([ucol, weak[np.argsort(-wt, kind="stable")]])
    ucol = ucol[:MAX_OBJECTS]
    objs = []
    for col in ucol.tolist():
        if time.perf_counter() > deadline:
            break
        M = (packed == col) & fg
        o = _fit_object(M, ~fg, W, H, deadline)
        if o is None:
            continue
        o["color"] = [(col >> 16) & 255, (col >> 8) & 255, col & 255]
        objs.append(o)
    if not objs:
        return []
    pred = [_render_mask(o, T, W, H) for o in objs]
    order = _z_order(video, packed, objs, pred)
    return [objs[i] for i in order]


# ------------------------------------------------------------ zvuk
def _fit_tone(u, deadline):
    """u: uint16 uzorci (int64). Vraća (f, s, amp, tačno) najboljeg tona ili None."""
    NS = len(u)
    a = np.where(u >= 32768, u - 65536, u)
    idx = np.unique(np.concatenate([np.arange(min(NS, 48)), np.linspace(0, NS - 1, 80).astype(np.int64)]))
    tgt, ai = u[idx], a[idx]
    n = np.arange(NS, dtype=np.int64)
    best = None
    for s in range(TONE_MAX_SHIFT + 1):
        if time.perf_counter() > deadline:
            break
        f = np.arange(1, 256 << s, dtype=np.int64)
        if s:
            f = f[f & 1 == 1]       # paran f sa s > 0 je isto što i (f/2, s-1)
        for ch in range(0, len(f), 4096):
            fc = f[ch:ch + 4096]
            isn = ISIN_LUT[((fc[:, None] * idx[None, :]) >> s) & 255]
            den = (isn * isn).sum(1)
            amp = np.rint((isn * ai).sum(1) / np.maximum(den, 1)).astype(np.int64)
            mism = (((isn * amp[:, None]) & 0xFFFF) != tgt).sum(1)
            j = int(mism.argmin())
            if best is None or mism[j] < best[0]:
                best = (int(mism[j]), int(fc[j]), s, int(amp[j]))
            for j in np.flatnonzero(mism == 0)[:8].tolist():
                full = (ISIN_LUT[((n * fc[j]) >> s) & 255] * amp[j]) & 0xFFFF
                if np.array_equal(full, u):
                    return int(fc[j]), s, int(amp[j]), True
    if best is None:
        return None
    return best[1], best[2], best[3], False


def _audio_part(meta, payload, deadline):
    """Najkraći (izraz, izuzeci) za zvuk."""
    u = np.frombuffer(payload[meta.video_bytes:], dtype="<u2").astype(np.int64)
    if not len(u) or not u.any():
        return None, None
    cands = [(words_lit(u, N), None)]
    nz = np.flatnonzero(u)
    if len(nz) <= len(u) // 4:
        cands.append((None, kfl.Exceptions(nz.astype(np.int64), u[nz])))
    tone = _fit_tone(u, deadline)
    if tone is not None:
        f, s, amp, exact = tone
        node = tone_node(f, s, amp)
        if exact:
            cands.append((node, None))
        else:
            n = np.arange(len(u), dtype=np.int64)
            pred = (ISIN_LUT[((n * f) >> s) & 255] * amp) & 0xFFFF
            bad = np.flatnonzero(pred != u)
            if len(bad) <= len(u) // 4:
                cands.append((node, kfl.Exceptions(bad.astype(np.int64), pred[bad] ^ u[bad])))

    def cost(c):
        return (blen(c[0]) if c[0] is not None else 0) + (len(c[1].to_bytes()) if c[1] is not None else 0)
    return min(cands, key=cost)


# ------------------------------------------------------------ glavni tok
def _with_exceptions(meta, payload, prog):
    """Izvršava program; razlike u videu postaju izuzeci. Vraća (zapis, broj izuzetaka) ili None."""
    vb = meta.video_bytes
    out = prog.run(meta)
    if out[vb:] != payload[vb:]:
        return None
    got = np.frombuffer(out[:vb], dtype=np.uint8)
    want = np.frombuffer(payload[:vb], dtype=np.uint8)
    bad = np.flatnonzero(got != want)
    if len(bad) > MAX_EXC_FRAC * vb:    # model ne objašnjava film; rezerva je bolja (i brža)
        return None
    prog.exc_video = (kfl.Exceptions(bad.astype(np.int64), (got[bad] ^ want[bad]).astype(np.int64))
                      if len(bad) else None)
    return record.encode_kfl(meta, prog), len(bad)


def _model_record(meta, payload, soft, hard):
    """Model porodice + izuzeci. soft: rok za pretragu, hard: posle njega ništa skupo."""
    T, H, W = meta.frames, meta.height, meta.width
    video = np.frombuffer(payload[:meta.video_bytes], dtype=np.uint8).reshape(T, H, W, 3)
    anode, aexc = _audio_part(meta, payload, soft)
    bg, bg_img = _fit_bg(video)
    objs = _objects(video, bg_img, soft)
    if time.perf_counter() > hard:
        return None

    def build(ob):
        return _with_exceptions(meta, payload, program_for(bg, ob, W, H, audio=anode, exc_audio=aexc))
    t = time.perf_counter()
    best = build(objs)
    cost = time.perf_counter() - t
    if best is None or not best[1]:
        return None if best is None else best[0]
    # MDL: izbaci objekat koji ne isplati svoj opis (npr. skoro stalno zaklonjen ili lažan)
    for i in range(len(objs) - 1, -1, -1):
        if time.perf_counter() + 2 * cost > hard:
            break
        cand = objs[:i] + objs[i + 1:]
        r = build(cand)
        if r is not None and len(r[0]) < len(best[0]):
            best, objs = r, cand
    return best[0]


def _exact(rec, payload) -> bool:
    try:
        _, pl, used = record.decode(rec)
    except Exception:
        return False
    return pl == payload and used == len(rec)


def encode(meta, payload, budget_s):
    t0 = time.perf_counter()
    soft, hard = t0 + 0.5 * budget_s, t0 + 0.8 * budget_s
    T, H, W = meta.frames, meta.height, meta.width
    if T < 1 or H < 1 or W < 1 or len(payload) != meta.payload_bytes:
        return None
    if meta.audio_samples and (meta.audio_channels != 1 or meta.sample_bytes != 2):
        return None
    best = igc_codec.encode("raw", meta, payload)        # gornja granica, uvek tačna
    try:
        rec = _model_record(meta, payload, soft, hard)
        if rec is not None and len(rec) < len(best) and _exact(rec, payload):
            best = rec
    except Exception:          # tuđ ili neočekivan film: ostaje rezerva
        pass
    # Rezerva (MDL): lzma, osim kad model očigledno pobeđuje (> 1000:1) ili nema vremena.
    if len(best) * 1000 > len(payload) and time.perf_counter() + len(payload) / LZMA_BPS < hard:
        lz = igc_codec.encode("lzma", meta, payload)
        if lz is not None and len(lz) < len(best) and _exact(lz, payload):
            best = lz
    return best


ENCODERS = [Encoder("sprites_infer", encode, family_aware=True,
                    note="pozadina + segmentacija po boji + TRI putanje + oblik + z-red; MDL izuzeci")]
