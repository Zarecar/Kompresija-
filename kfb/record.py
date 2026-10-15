"""Format zapisa KFB i jedinstveni dekoder (deo referentne mašine M).

    [1 B metoda][8 varint-ova metapodataka][telo]

  0x10        KFL program (kfb.kfl)                    -> "programski" zapis
  0x00-0x08   metode iz igc.codec (raw, zlib, lzma, bz2, xor_*, cm, infer, rule)
  0x20-0x2F   ffmpeg kodeci (kfb.ffcodec: FFV1, x264rgb lossless, ...)
Kontrola varanja `leaky` (0xFF) iz igc NIJE dozvoljena.

Svaki zapis je samoograničavajući: decode() vraća broj pročitanih bajtova.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from igc import codec as igc_codec
from igc.film import Meta

from . import ffcodec, kfl

KFL_METHOD = 0x10
PKG = Path(__file__).resolve().parent
MODEL_FILES = [PKG / "kfl.py", PKG / "record.py", PKG / "ffcodec.py"]


def ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=30).stdout
        return out.splitlines()[0] if out else "ffmpeg: nema"
    except (OSError, subprocess.SubprocessError):
        return "ffmpeg: nema"


def model_id() -> str:
    """SHA-256 referentne mašine: KFL VM + format + igc dekoderi + verzija ffmpeg-a."""
    h = hashlib.sha256()
    files = MODEL_FILES + sorted((PKG.parent / "igc").glob("*.py"))
    for p in files:
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    h.update(ffmpeg_version().encode())
    return h.hexdigest()


def encode_kfl(meta: Meta, program: kfl.Program) -> bytes:
    return bytes([KFL_METHOD]) + meta.to_bytes() + program.to_bytes()


def decode(rec: bytes) -> tuple[Meta, bytes, int]:
    if not rec:
        raise ValueError("prazan zapis")
    m = rec[0]
    if m == KFL_METHOD:
        meta, pos = Meta.from_bytes(rec, 1)
        prog, pos = kfl.Program.from_bytes(rec, pos, meta)
        payload = kfl.run(prog, meta)
        return meta, payload, pos
    if 0x20 <= m <= 0x2F:
        return ffcodec.decode(rec)
    if m in igc_codec.NAMES and igc_codec.NAMES[m] != "leaky":
        return igc_codec.decode(rec)
    raise ValueError(f"nepoznata ili zabranjena metoda 0x{m:02x}")
