"""Standardni kodeci bez gubitaka (ffmpeg CLI) kao generičke osnovne linije.

Svaki kodek je zaseban generički enkoder (family_aware=False) i ne zna ništa o
porodicama. Enkoder proba nekoliko podešavanja kodera (od bržih ka sporijim, dok
traje budžet); svaki kraći kandidat se odmah dekoduje referentnim dekoderom
(kfb.ffcodec.decode) i prihvata samo ako je TAČAN. Zvuk: tišina, FLAC (pretraga
veličine bloka i LPC metode) ili sirovi PCM, šta je kraće. Format zapisa je
opisan u kfb/ffcodec.py.
"""
from __future__ import annotations

import time

from kfb import ffcodec

from . import Encoder

# varijante: {opcija: vrednost} menja ili dodaje opciju, vrednost None je briše
VARIANTS: dict[str, list[dict]] = {
    "ffv1": [{}, {"-context": "0"}, {"-coder": "-2"}, {"-g": "1"}],
    "x264rgb": [{}, {"-preset": "placebo"}],
    "vp9": [{}, {"-deadline": "good", "-cpu-used": "1"}, {"-deadline": "best", "-cpu-used": "0"}],
    # lag > 0: libaom 3.8 ponekad nije bez gubitaka ili puca ("buffer too small"); provera to hvata
    "av1": [{}, {"-lag-in-frames": None, "-arnr-max-frames": "0"},
            {"-cpu-used": "4", "-lag-in-frames": None, "-arnr-max-frames": "0"},
            {"-cpu-used": "3"}],
    "apng": [{}, {"-pred": "paeth"}],
    # x265 3.5 + ffmpeg hevc, 4:4:4: sa subme > 0 i B-frejmovima tok često NIJE bez gubitaka
    # (retke greške predikcije koje se šire); subme=0 je podrazumevano, a bez B-frejmova i
    # samo-intra (keyint=1) su rezerve. Svaki kandidat se ionako proverava.
    "x265": [{"-preset": "medium"}, {}, {"-preset": "placebo"},
             {"-preset": "medium", "-x265-params": ffcodec.X265_PARAMS + ":bframes=0"},
             {"-preset": "medium", "-x265-params": ffcodec.X265_PARAMS + ":keyint=1"}],
}
AUDIO_VARIANTS = [[], ["-lpc_type", "cholesky", "-lpc_passes", "4"]]
AUDIO_FRAMES = [None, 8192, 2048, 16384, 4096, 1024, 32768]


def _args(base, changes: dict) -> list:
    a = list(base)
    for k, v in changes.items():
        if k in a:
            i = a.index(k)
            del a[i:i + 2]
        if v is not None:
            a += [k, v]
    return a


def _exact(rec: bytes, meta, payload: bytes) -> bool:
    try:
        m2, out, used = ffcodec.decode(rec + b"\0" * 8)
    except (ValueError, OSError):
        return False
    return m2 == meta and used == len(rec) and out == payload


def best_audio(meta, audio: bytes, deadline: float) -> tuple[int, bytes]:
    """Najkraći audio tok (tišina / FLAC / sirovi) u okviru roka (time.perf_counter)."""
    if not any(audio):
        return ffcodec.A_SILENCE, b""
    best = (ffcodec.A_RAW, audio)
    if not ffcodec.flac_ok(meta):
        return best
    for fs in AUDIO_FRAMES:
        if fs is not None and fs > 2 * meta.audio_samples:
            continue
        for v in AUDIO_VARIANTS:
            left = deadline - time.perf_counter()
            if left < 0.3:
                return best
            args = ["-compression_level", "12"] + (["-frame_size", str(fs)] if fs else []) + v
            cand = ffcodec.encode_audio(meta, audio, args, timeout=left)
            if len(cand[1]) < len(best[1]):
                best = cand
            if fs is None and not v and best[0] != ffcodec.A_FLAC:
                return best                  # FLAC ne pomaže (npr. šum): ne trošimo vreme
    return best


def _make(name: str):
    method = ffcodec.METHODS[name]
    codec = ffcodec.CODECS[method]

    def encode(meta, payload, budget_s):
        t0 = time.perf_counter()
        if len(payload) != meta.payload_bytes:
            return None
        video, audio = payload[:meta.video_bytes], payload[meta.video_bytes:]
        base = list(codec.enc_args)
        if name == "ffv1" and (meta.width > 352 or meta.height > 288):
            base = _args(base, {"-slices": None})          # FFV1 v3 tada traži >= 4 isečka
        amode, astream = best_audio(meta, audio, t0 + 0.1 * budget_s)
        best = None
        tried = set()
        for ch in VARIANTS[name]:
            args = _args(base, ch)
            if tuple(args) in tried:
                continue
            tried.add(tuple(args))
            left = t0 + budget_s - time.perf_counter()
            if left < 0.5:
                break
            vs = ffcodec.encode_video(method, meta, video, args, timeout=0.75 * left)
            if vs is None:
                continue
            rec = ffcodec.pack(method, meta, vs, amode, astream)
            if (best is None or len(rec) < len(best)) and _exact(rec, meta, payload):
                best = rec
            if meta.video_bytes == 0:
                break
        return best
    encode.__name__ = f"encode_{name}"
    return encode


_NOTES = {
    "ffv1": "FFV1 v3 (bgr0/RCT, range coder, kontekst 1/0, g=T/1, 1 isečak), Matroska omotač u dekoderu",
    "x264rgb": "libx264rgb -qp 0 (veryslow/placebo), Annex B bez SEI",
    "vp9": "libvpx-vp9 -lossless 1 (gbrp, profil 1), cpu-used 4/1/0",
    "av1": "libaom-av1 lossless=1 (gbrp), OBU tok; dekoder libdav1d",
    "apng": "APNG (pred mixed/paeth, zlib 9)",
    "x265": "libx265 lossless=1 subme=0 (gbrp), Annex B bez SEI; medium/veryslow/placebo + rezerve",
}
ENCODERS = [Encoder(n, _make(n), note=_NOTES[n]) for n in ffcodec.METHODS]
