"""MT19937: generisanje, inverzija tempering-a, "untwist" i inverzija seed-a.

Cilj: pokazati da "nasumičan" film napravljen slabim (nekriptografskim)
generatorom ima malu algoritamsku složenost i da se TAČNO kompresuje na
stanje generatora (2496 B) ili čak na seed (4 B), dok je izlaz kriptografskog
generatora (pod standardnim pretpostavkama) nekompresibilan.

Sve je implementirano nezavisno od Python `random` i NumPy biblioteke;
testovi proveravaju bit-identičnost sa njima.
"""
from __future__ import annotations

import numpy as np

N, M = 624, 397
MATRIX_A = 0x9908B0DF
UPPER, LOWER = 0x80000000, 0x7FFFFFFF
MASK32 = 0xFFFFFFFF
LAG = N - M  # 227: dužina bloka koji se može računati vektorski


# ---------------------------------------------------------------- tempering
def temper(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.uint32).copy()
    y ^= y >> 11
    y ^= (y << 7) & 0x9D2C5680
    y ^= (y << 15) & 0xEFC60000
    y ^= y >> 18
    return y


def _inv_right(y: np.ndarray, s: int) -> np.ndarray:
    r = y.copy()
    for _ in range(32 // s + 1):
        r = y ^ (r >> s)
    return r


def _inv_left(y: np.ndarray, s: int, mask: int) -> np.ndarray:
    r = y.copy()
    for _ in range(32 // s + 1):
        r = y ^ ((r << s) & mask)
    return r


def untemper(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.uint32).copy()
    y = _inv_right(y, 18)
    y = _inv_left(y, 15, 0xEFC60000)
    y = _inv_left(y, 7, 0x9D2C5680)
    y = _inv_right(y, 11)
    return y


# ------------------------------------------------------------- rekurencija
def extend(words: np.ndarray, total: int) -> np.ndarray:
    """Produžava niz NETEMPEROVANIH reči linearnom rekurencijom MT19937.

    x[k+624] = x[k+397] ^ A((x[k] & UPPER) | (x[k+1] & LOWER))
    Važi za bilo kojih 624 uzastopnih reči (poravnanje nije potrebno).
    """
    words = np.asarray(words, dtype=np.uint32)
    if len(words) < N:
        raise ValueError("potrebno je bar 624 reči")
    out = np.empty(max(total, len(words)), dtype=np.uint32)
    out[:len(words)] = words
    t = len(words)
    while t < total:
        n = min(LAG, total - t)
        a = out[t - N:t - N + n]
        b = out[t - N + 1:t - N + 1 + n]
        c = out[t - LAG:t - LAG + n]
        yv = (a & UPPER) | (b & LOWER)
        out[t:t + n] = c ^ (yv >> 1) ^ ((yv & 1) * np.uint32(MATRIX_A))
        t += n
    return out[:total]


def outputs_from_initial_state(state: list[int] | np.ndarray, n_words: int) -> np.ndarray:
    """Temperovani izlazi generatora čije je početno (pre-twist) stanje `state`."""
    x = extend(np.asarray(state, dtype=np.uint32), N + n_words)
    return temper(x[N:])


# ------------------------------------------------------------- inicijalizacija
def init_genrand(seed: int) -> list[int]:
    mt = [0] * N
    mt[0] = seed & MASK32
    for i in range(1, N):
        prev = mt[i - 1]
        mt[i] = (1812433253 * (prev ^ (prev >> 30)) + i) & MASK32
    return mt


def init_by_array(key: list[int]) -> list[int]:
    mt = init_genrand(19650218)
    i, j = 1, 0
    klen = len(key)
    for _ in range(max(N, klen)):
        prev = mt[i - 1]
        mt[i] = ((mt[i] ^ ((prev ^ (prev >> 30)) * 1664525)) + key[j] + j) & MASK32
        i += 1
        j += 1
        if i >= N:
            mt[0] = mt[N - 1]
            i = 1
        if j >= klen:
            j = 0
    for _ in range(N - 1):
        prev = mt[i - 1]
        mt[i] = ((mt[i] ^ ((prev ^ (prev >> 30)) * 1566083941)) - i) & MASK32
        i += 1
        if i >= N:
            mt[0] = mt[N - 1]
            i = 1
    mt[0] = 0x80000000
    return mt


def int_to_key(seed: int) -> list[int]:
    """Kao CPython random.seed(int): apsolutna vrednost, 32-bitni delovi, LE."""
    seed = abs(seed)
    key = []
    while True:
        key.append(seed & MASK32)
        seed >>= 32
        if not seed:
            return key


def key_to_int(key: list[int]) -> int:
    return sum(k << (32 * i) for i, k in enumerate(key))


# ------------------------------------------------------------------ untwist
def untwist(after: list[int]) -> list[int | None]:
    """Iz stanja POSLE prvog twist-a vraća stanje PRE njega.

    Vraća listu od 624 vrednosti; za indeks 0 poznat je samo najviši bit
    (niži bitovi x[0] se ne koriste u twist-u), pa je element 0 = samo UPPER deo.
    """
    x = [0] * (2 * N)
    x[N:] = [int(v) for v in after]
    upper = [0] * (N + 1)
    # Idemo unazad: korak k daje gornji bit x[k] i donjih 31 bit x[k+1].
    # Za k < 227 potrebna je x[k+397], koja je kompletirana u koraku k+396.
    for k in range(N - 1, -1, -1):
        tmp = x[k + N] ^ x[k + M]
        odd = 0
        if tmp & 0x80000000:  # A ima najviši bit 1, a (y >> 1) nema
            tmp ^= MATRIX_A
            odd = 1
        y = ((tmp << 1) | odd) & MASK32
        upper[k] = y & UPPER
        if k + 1 < N:
            x[k + 1] = upper[k + 1] | (y & LOWER)
    x[0] = upper[0]
    return x[:N]


def seed_from_genrand_state(pre: list[int]) -> int | None:
    """Inverzija init_genrand iz x[1] (x[1] = 1812433253*(s^(s>>30)) + 1)."""
    inv = pow(1812433253, -1, 1 << 32)
    v = ((pre[1] - 1) * inv) & MASK32
    s = v ^ (v >> 30)
    cand = init_genrand(s)
    if cand[1:] == pre[1:] and (cand[0] & UPPER) == (pre[0] & UPPER):
        return s
    return None


def key_from_by_array_state(pre: list[int], max_len: int = 620) -> list[int] | None:
    """Inverzija init_by_array (CPython random.seed(int), NumPy seed-niz)."""
    B = pre
    A = [0] * N
    for i in range(3, N):
        prev = B[i - 1]
        A[i] = ((B[i] + i) & MASK32) ^ (((prev ^ (prev >> 30)) * 1566083941) & MASK32)
    C = init_genrand(19650218)
    derived = {}
    for i in range(4, N):
        prev = A[i - 1]
        base = C[i] ^ (((prev ^ (prev >> 30)) * 1664525) & MASK32)
        derived[i] = (A[i] - base) & MASK32  # = key[j] + j, j = (i-1) mod L
    for L in range(1, max_len + 1):
        key = [None] * L
        ok = True
        for i in range(4, N):
            j = (i - 1) % L
            kv = (derived[i] - j) & MASK32
            if key[j] is None:
                key[j] = kv
            elif key[j] != kv:
                ok = False
                break
        if not ok or any(k is None for k in key):
            continue
        # skini vodeće nule (CPython ne pravi ključ sa nultim najvišim delom osim [0])
        cand = init_by_array(key)
        if cand[1:] == B[1:]:
            return key
    return None
