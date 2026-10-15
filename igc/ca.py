"""Film kao dinamički sistem: Life-like ćelijski automat + izuzeci.

Generator filma (izvor):
    F_0      = slučajna "supa" s x s u centru praznog torusa H x W (CSPRNG)
    F_{t+1}  = Phi_rule(F_t) XOR E_t,  E_t ~ Bernoulli(p) i.i.d. (CSPRNG)
    video_t  = boja(F_t),  zvuk_t = trougaoni ton čija frekvencija zavisi od
               broja živih ćelija (celobrojna sinteza -> bit-tačno)

Metoda "rule" (MDL indukcija pravila + korekcije, Seed-and-Correct u
tačnom režimu): enkoder NE zna pravilo ni p; iz frejmova procenjuje pravilo
većinskim glasanjem nad 18 parova (stanje, broj suseda), a zatim kodira:
    pravilo (18 bita) + 2 boje + F_0 (kontekstni aritmetički kod)
    + pozicije izuzetaka (Rice kod razmaka) + 1 bit "zvuk = sinteza(video)".
Očekivana dužina ~ const + H(F_0) + N*h(p) (vidi experiments/e3).
"""
from __future__ import annotations

import hashlib
import os
import zlib

import numpy as np

from .film import Meta, get_varint, put_varint, video_meta
from .rangecoder import COUNTER_INIT, Decoder, Encoder, counter_p12, counter_update

CONWAY = (1 << 3) | (1 << (9 + 2)) | (1 << (9 + 3))   # B3/S23


def rule_table(rule: int) -> np.ndarray:
    return np.array([(rule >> k) & 1 for k in range(18)], dtype=np.uint8)


def neighbors(F: np.ndarray) -> np.ndarray:
    F = F.astype(np.uint8)
    n = np.zeros(F.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                n += np.roll(np.roll(F, dy, axis=-2), dx, axis=-1)
    return n


def step(F: np.ndarray, rule: int) -> np.ndarray:
    return rule_table(rule)[F.astype(np.int64) * 9 + neighbors(F)]


def synth_audio(pops: np.ndarray, audio_rate: int, fps: int) -> bytes:
    spf = audio_rate // fps
    freq = 110 + (pops.astype(np.uint64) % 880)
    inc = (freq << np.uint64(32)) // np.uint64(audio_rate)
    start = np.concatenate([[0], np.cumsum(inc * np.uint64(spf))[:-1]]).astype(np.uint64)
    k = np.arange(1, spf + 1, dtype=np.uint64)
    phase = (start[:, None] + inc[:, None] * k[None, :]) & np.uint64(0xFFFFFFFF)
    v = (phase >> np.uint64(16)).astype(np.int64)
    tri = np.where(v < 32768, v, 65535 - v)
    s = (tri * 2 - 32767) >> 2
    return s.astype("<i2").tobytes()


def life_film(H: int = 128, W: int = 128, T: int = 240, soup: int = 32, p: float = 0.0,
              rule: int = CONWAY, fps: int = 24, audio_rate: int = 7200,
              alive=(240, 220, 90), dead=(16, 16, 40), key: bytes | None = None):
    key = key or os.urandom(32)
    sh = hashlib.shake_256(key + b"life")
    nbits = soup * soup
    nflip = (T - 1) * H * W
    raw = sh.digest((nbits + 7) // 8 + 4 * nflip)
    bits = np.unpackbits(np.frombuffer(raw[:(nbits + 7) // 8], dtype=np.uint8))[:nbits]
    F = np.zeros((T, H, W), dtype=np.uint8)
    y0, x0 = (H - soup) // 2, (W - soup) // 2
    F[0, y0:y0 + soup, x0:x0 + soup] = bits.reshape(soup, soup)
    u = np.frombuffer(raw[(nbits + 7) // 8:], dtype="<u4").reshape(T - 1, H, W)
    thr = np.uint64(min(int(round(p * 2 ** 32)), 2 ** 32))
    flips = (u.astype(np.uint64) < thr).astype(np.uint8)
    for t in range(T - 1):
        F[t + 1] = step(F[t], rule) ^ flips[t]
    meta = video_meta(W, H, T, fps, audio_rate, 1, 2)
    video = np.where(F[..., None].astype(bool), np.array(alive, np.uint8), np.array(dead, np.uint8))
    audio = synth_audio(F.reshape(T, -1).sum(1), audio_rate, fps)
    info = {"exceptions": int(flips.sum()), "N_cells": int(nflip), "soup_bits": nbits}
    return meta, video.astype(np.uint8).tobytes() + audio, info


# ------------------------------------------------------------ bit I/O
class BitWriter:
    def __init__(self) -> None:
        self.acc = 0
        self.n = 0
        self.out = bytearray()

    def write(self, value: int, nbits: int) -> None:
        self.acc = (self.acc << nbits) | value
        self.n += nbits
        while self.n >= 8:
            self.n -= 8
            self.out.append((self.acc >> self.n) & 0xFF)
        self.acc &= (1 << self.n) - 1

    def finish(self) -> bytes:
        if self.n:
            self.out.append((self.acc << (8 - self.n)) & 0xFF)
        return bytes(self.out)


class BitReader:
    def __init__(self, data: bytes, pos: int) -> None:
        self.data, self.pos, self.bit = data, pos, 0

    def read(self) -> int:
        if self.pos >= len(self.data):
            raise ValueError("skraćen bit-tok")
        b = (self.data[self.pos] >> (7 - self.bit)) & 1
        self.bit += 1
        if self.bit == 8:
            self.bit = 0
            self.pos += 1
        return b

    def read_bits(self, k: int) -> int:
        v = 0
        for _ in range(k):
            v = (v << 1) | self.read()
        return v

    def end(self) -> int:
        return self.pos + (1 if self.bit else 0)


def _rice_encode(positions: np.ndarray) -> bytes:
    out = bytearray()
    put_varint(out, len(positions))
    if len(positions) == 0:
        return bytes(out)
    gaps = np.diff(np.concatenate([[-1], positions.astype(np.int64)])) - 1
    best_k = min(range(0, 40), key=lambda k: int(((gaps >> k) + 1 + k).sum()))
    out.append(best_k)
    bw = BitWriter()
    for g in gaps.tolist():
        q = g >> best_k
        while q >= 32:
            bw.write(0xFFFFFFFF, 32)
            q -= 32
        bw.write(((1 << q) - 1) << 1, q + 1)   # q jedinica pa nula
        if best_k:
            bw.write(g & ((1 << best_k) - 1), best_k)
    return bytes(out) + bw.finish()


def _rice_decode(rec: bytes, pos: int) -> tuple[np.ndarray, int]:
    K, pos = get_varint(rec, pos)
    if K == 0:
        return np.zeros(0, dtype=np.int64), pos
    k = rec[pos]
    br = BitReader(rec, pos + 1)
    res = []
    last = -1
    for _ in range(K):
        q = 0
        while br.read():
            q += 1
        g = (q << k) | (br.read_bits(k) if k else 0)
        last = last + 1 + g
        res.append(last)
    return np.array(res, dtype=np.int64), br.end()


# --------------------------------------------- kodovanje F_0 (JBIG-like)
def _ctx_grid(F0: np.ndarray) -> np.ndarray:
    H, W = F0.shape
    P = np.zeros((H + 1, W + 2), dtype=np.uint8)
    P[1:, 1:-1] = F0
    left = P[1:, :-2]
    ul = P[:-1, :-2]
    up = P[:-1, 1:-1]
    ur = P[:-1, 2:]
    return left | (ul << 1) | (up << 2) | (ur << 3)


def _encode_grid(F0: np.ndarray) -> bytes:
    enc = Encoder()
    ctx = _ctx_grid(F0).ravel().tolist()
    bits = F0.ravel().tolist()
    tab = [COUNTER_INIT] * 16
    for c, b in zip(ctx, bits):
        enc.encode(b, counter_p12(tab[c]))
        tab[c] = counter_update(tab[c], b)
    return enc.finish()


def _decode_grid(H: int, W: int, rec: bytes, pos: int) -> tuple[np.ndarray, int]:
    dec = Decoder(rec, pos)
    F0 = np.zeros((H, W), dtype=np.uint8)
    tab = [COUNTER_INIT] * 16
    for y in range(H):
        for x in range(W):
            c = (F0[y, x - 1] if x else 0) | ((F0[y - 1, x - 1] if (x and y) else 0) << 1) \
                | ((F0[y - 1, x] if y else 0) << 2) | ((F0[y - 1, x + 1] if (y and x + 1 < W) else 0) << 3)
            b = dec.decode(counter_p12(tab[c]))
            tab[c] = counter_update(tab[c], b)
            F0[y, x] = b
    return F0, dec.pos


# ----------------------------------------------------- metoda "rule"
def _pack(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]


def rule_encode(meta: Meta, payload: bytes) -> bytes | None:
    T, H, W = meta.frames, meta.height, meta.width
    if T < 2 or H < 1 or W < 1:
        return None
    video = np.frombuffer(payload[:meta.video_bytes], dtype=np.uint8).reshape(T, H, W, 3)
    audio = payload[meta.video_bytes:]
    packed = _pack(video)
    cols = np.unique(packed)
    if len(cols) > 2:
        return None
    cols = [int(c) for c in cols] + ([int(cols[0]) ^ 1] if len(cols) == 1 else [])
    best = None
    for alive, dead in ((cols[0], cols[1]), (cols[1], cols[0])):
        F = (packed == alive).astype(np.uint8)
        idx = F[:-1].astype(np.int64) * 9 + neighbors(F[:-1])
        nxt = F[1:]
        ones = np.bincount(idx[nxt == 1], minlength=18)
        tot = np.bincount(idx.ravel(), minlength=18)
        rule = sum(1 << k for k in range(18) if 2 * ones[k] > tot[k])
        E = nxt ^ rule_table(rule)[idx]
        out = bytearray(rule.to_bytes(3, "little"))
        out += alive.to_bytes(3, "big") + dead.to_bytes(3, "big")
        out += _encode_grid(F[0])
        out += _rice_encode(np.flatnonzero(E.ravel()))
        ok_audio = (meta.audio_channels == 1 and meta.sample_bytes == 2 and meta.fps > 0
                    and meta.audio_rate % meta.fps == 0
                    and synth_audio(F.reshape(T, -1).sum(1), meta.audio_rate, meta.fps) == audio)
        if ok_audio:
            out.append(1)
        else:
            out.append(0)
            c = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
            out += c.compress(audio) + c.flush()
        if best is None or len(out) < len(best):
            best = bytes(out)
    return best


def rule_decode(meta: Meta, rec: bytes, pos: int) -> tuple[bytes, int]:
    T, H, W = meta.frames, meta.height, meta.width
    rule = int.from_bytes(rec[pos:pos + 3], "little")
    alive = int.from_bytes(rec[pos + 3:pos + 6], "big")
    dead = int.from_bytes(rec[pos + 6:pos + 9], "big")
    pos += 9
    F0, pos = _decode_grid(H, W, rec, pos)
    exc, pos = _rice_decode(rec, pos)
    F = np.zeros((T, H, W), dtype=np.uint8)
    F[0] = F0
    per = H * W
    order = np.searchsorted(exc, np.arange(T) * per)
    for t in range(T - 1):
        F[t + 1] = step(F[t], rule)
        e = exc[order[t]:order[t + 1]] - t * per
        if len(e):
            flat = F[t + 1].reshape(-1)
            flat[e] ^= 1
    flag = rec[pos]
    pos += 1
    if flag == 1:
        audio = synth_audio(F.reshape(T, -1).sum(1), meta.audio_rate, meta.fps)
    else:
        d = zlib.decompressobj(-15)
        audio = d.decompress(rec[pos:])
        if not d.eof:
            raise ValueError("skraćen zvuk")
        pos = len(rec) - len(d.unused_data)
    a = np.array([(alive >> 16) & 255, (alive >> 8) & 255, alive & 255], np.uint8)
    b = np.array([(dead >> 16) & 255, (dead >> 8) & 255, dead & 255], np.uint8)
    video = np.where(F[..., None].astype(bool), a, b).astype(np.uint8)
    return video.tobytes() + audio, pos
