"""Format zapisa i sve metode kodovanja.

Zapis (sve se računa u budžet):
    [1 B id metode][metapodaci kao varint-ovi][podaci metode]
Svaki zapis je SAMOOGRANIČAVAJUĆI: dekoder vraća broj pročitanih bajtova i
ne koristi dužinu fajla (test: zapisu se doda smeće i rezultat mora ostati
isti, a broj pročitanih bajtova = dužina zapisa).

Model M = izvorni kod ovog paketa (MODEL_ID = SHA-256 nad igc/*.py). Model je
izuzet iz budžeta, ali je fiksiran PRE generisanja podataka; svaka promena
modela posle viđenja filma menja MODEL_ID i čini rezultat nevažećim.
"""
from __future__ import annotations

import bz2
import hashlib
import lzma
import os
import zlib
from pathlib import Path

import numpy as np

from . import cm, mt19937 as mt
from .film import Meta, get_varint, put_varint
from .sources import WEAK_BITS, weak_shake

PKG_DIR = Path(__file__).resolve().parent
XOR_MODEL_KEY = b"IGC fixed public generator v1"   # javni, nezavisan od podataka
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]

METHODS = {
    "raw": 0, "zlib": 1, "lzma": 2, "bz2": 3, "xor_store": 4, "xor_zlib": 5,
    "cm": 6, "infer": 7, "rule": 8, "leaky": 255,
}
NAMES = {v: k for k, v in METHODS.items()}


def model_id() -> str:
    h = hashlib.sha256()
    for p in sorted(PKG_DIR.glob("*.py")):
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    leak = os.environ.get("IGC_LEAKY_DIR")
    if leak and Path(leak).is_dir():
        for p in sorted(Path(leak).iterdir()):
            h.update(p.name.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


# ------------------------------------------------------------ pomoćne
def _xor_stream(n: int) -> np.ndarray:
    return np.frombuffer(hashlib.shake_256(XOR_MODEL_KEY).digest(n), dtype=np.uint8)


def _xor(payload: bytes) -> bytes:
    a = np.frombuffer(payload, dtype=np.uint8)
    return (a ^ _xor_stream(len(payload))).tobytes()


def _deflate(b: bytes) -> bytes:
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    return c.compress(b) + c.flush()


def _inflate(rec: bytes, pos: int) -> tuple[bytes, int]:
    d = zlib.decompressobj(-15)
    out = d.decompress(rec[pos:])
    if not d.eof:
        raise ValueError("skraćen deflate tok")
    return out, len(rec) - len(d.unused_data)


def _unlzma(rec: bytes, pos: int) -> tuple[bytes, int]:
    d = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    out = d.decompress(rec[pos:])
    if not d.eof:
        raise ValueError("skraćen lzma tok")
    return out, len(rec) - len(d.unused_data)


def _unbz2(rec: bytes, pos: int) -> tuple[bytes, int]:
    d = bz2.BZ2Decompressor()
    out = d.decompress(rec[pos:])
    if not d.eof:
        raise ValueError("skraćen bz2 tok")
    return out, len(rec) - len(d.unused_data)


# ------------------------------------------------- inverzija generatora
def _mt_layout_bytes(tw: np.ndarray, n: int, layout: int) -> bytes:
    full = n // 4
    b = tw[:full].astype("<u4").tobytes()
    rem = n - 4 * full
    if rem:
        if layout == 0:      # NumPy RandomState.bytes: odseca LE bajtove
            b += tw[full:full + 1].astype("<u4").tobytes()[:rem]
        else:                # CPython randbytes: getrandbits uzima GORNJE bitove
            b += (int(tw[full]) >> (32 - 8 * rem)).to_bytes(rem, "little")
    return b


def _mt_from_initial(state: list[int], n: int, layout: int) -> bytes:
    tw = mt.outputs_from_initial_state(state, (n + 3) // 4)
    return _mt_layout_bytes(tw, n, layout)


def _mt_from_outputs(first: bytes, n: int, layout: int) -> bytes:
    u = mt.untemper(np.frombuffer(first, dtype="<u4"))
    tw = mt.temper(mt.extend(u, (n + 3) // 4))
    return _mt_layout_bytes(tw, n, layout)


def infer_encode(payload: bytes, weak_bits: int = WEAK_BITS) -> bytes | None:
    """Pokušava da pronađe generator iz POZNATE porodice koji daje payload."""
    n = len(payload)
    out = bytearray()
    # (a) slabi SHAKE (porodica je deo modela; pretraga 2^weak_bits)
    if n >= 16:
        target = payload[:16]
        from .sources import WEAK_LABEL
        h0 = hashlib.shake_256(WEAK_LABEL)
        for s in range(1 << weak_bits):
            h = h0.copy()
            h.update(s.to_bytes(4, "little"))
            if h.digest(16) == target and weak_shake(s, n) == payload:
                out.append(4)
                put_varint(out, s)
                return bytes(out)
    # (b) MT19937: stanje iz prvih 624 izlaza, zatim provera CELOG toka
    if n >= 4 * mt.N + 4:
        first = payload[:4 * mt.N]
        for layout in (0, 1):
            if _mt_from_outputs(first, n, layout) != payload:
                continue
            u = mt.untemper(np.frombuffer(first, dtype="<u4"))
            pre = mt.untwist(u.tolist())
            seed = mt.seed_from_genrand_state(pre)
            if seed is not None and _mt_from_initial(mt.init_genrand(seed), n, layout) == payload:
                out += bytes([2, layout])
                put_varint(out, seed)
                return bytes(out)
            key = mt.key_from_by_array_state(pre)
            if key is not None and _mt_from_initial(mt.init_by_array(key), n, layout) == payload:
                out += bytes([3, layout])
                put_varint(out, len(key))
                for k in key:
                    put_varint(out, k)
                return bytes(out)
            out += bytes([1, layout]) + first
            return bytes(out)
    return None


def infer_decode(n: int, rec: bytes, pos: int) -> tuple[bytes, int]:
    kind = rec[pos]
    pos += 1
    if kind == 4:
        s, pos = get_varint(rec, pos)
        return weak_shake(s, n), pos
    layout = rec[pos]
    pos += 1
    if kind == 1:
        first = rec[pos:pos + 4 * mt.N]
        if len(first) != 4 * mt.N:
            raise ValueError("skraćeno MT stanje")
        return _mt_from_outputs(first, n, layout), pos + 4 * mt.N
    if kind == 2:
        s, pos = get_varint(rec, pos)
        return _mt_from_initial(mt.init_genrand(s), n, layout), pos
    if kind == 3:
        L, pos = get_varint(rec, pos)
        key = []
        for _ in range(L):
            k, pos = get_varint(rec, pos)
            key.append(k)
        return _mt_from_initial(mt.init_by_array(key), n, layout), pos
    raise ValueError("nepoznat generator")


# -------------------------------------------------------- javni API
def encode(method: str, meta: Meta, payload: bytes, **kw) -> bytes | None:
    """Vraća kompletan zapis ili None ako metoda nije primenljiva."""
    head = bytearray([METHODS[method]]) + meta.to_bytes()
    if method == "raw":
        body = payload
    elif method == "zlib":
        body = _deflate(payload)
    elif method == "lzma":
        body = lzma.compress(payload, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    elif method == "bz2":
        body = bz2.compress(payload, 9)
    elif method == "xor_store":
        body = _xor(payload)
    elif method == "xor_zlib":
        body = _deflate(_xor(payload))
    elif method == "cm":
        body = cm.encode(meta, payload)
    elif method == "infer":
        body = infer_encode(payload, **kw)
    elif method == "rule":
        from . import ca
        body = ca.rule_encode(meta, payload)
    elif method == "leaky":
        # KONTROLA VARANJA: film se upisuje u "model", zapis nosi samo heš.
        leak = Path(kw["leaky_dir"])
        leak.mkdir(parents=True, exist_ok=True)
        hid = hashlib.sha256(payload).digest()[:8]
        (leak / hid.hex()).write_bytes(payload)
        body = hid
    else:
        raise ValueError(method)
    if body is None:
        return None
    return bytes(head) + body


def decode(rec: bytes) -> tuple[Meta, bytes, int]:
    """Vraća (meta, payload, broj_pročitanih_bajtova)."""
    mid = rec[0]
    meta, pos = Meta.from_bytes(rec, 1)
    n = meta.payload_bytes
    name = NAMES.get(mid)
    if name == "raw":
        out, pos = rec[pos:pos + n], pos + n
        if len(out) != n:
            raise ValueError("skraćen zapis")
    elif name == "zlib":
        out, pos = _inflate(rec, pos)
    elif name == "lzma":
        out, pos = _unlzma(rec, pos)
    elif name == "bz2":
        out, pos = _unbz2(rec, pos)
    elif name == "xor_store":
        body = rec[pos:pos + n]
        if len(body) != n:
            raise ValueError("skraćen zapis")
        out, pos = _xor(body), pos + n
    elif name == "xor_zlib":
        body, pos = _inflate(rec, pos)
        out = _xor(body)
    elif name == "cm":
        out, pos = cm.decode(meta, rec, pos)
    elif name == "infer":
        out, pos = infer_decode(n, rec, pos)
    elif name == "rule":
        from . import ca
        out, pos = ca.rule_decode(meta, rec, pos)
    elif name == "leaky":
        hid = rec[pos:pos + 8].hex()
        pos += 8
        leak = os.environ.get("IGC_LEAKY_DIR")
        f = Path(leak) / hid if leak else None
        if f is None or not f.exists():
            raise ValueError("model ne sadrži ovaj film (zamrznuti model)")
        out = f.read_bytes()
    else:
        raise ValueError(f"nepoznata metoda {mid}")
    if len(out) != n:
        raise ValueError("pogrešna dužina dekodovanog sadržaja")
    return meta, bytes(out), pos


def encode_best(meta: Meta, payload: bytes, methods: list[str]) -> tuple[str, bytes]:
    """MDL izbor: najkraći zapis među metodama (id metode je već u zapisu)."""
    best = None
    for m in methods:
        r = encode(m, meta, payload)
        if r is not None and (best is None or len(r) < len(best[1])):
            best = (m, r)
    return best
