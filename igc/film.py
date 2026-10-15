"""Kanonski oblik "filma" i poštena serijalizacija metapodataka.

Film je par (Meta, payload). Payload = RGB24 frejmovi (red po red, frejm po
frejm) iza kojih slede PCM uzorci (little-endian, interleaved kanali).
Metapodaci su deo originala i UVEK se upisuju u zapis (ulaze u budžet).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


def put_varint(buf: bytearray, n: int) -> None:
    if n < 0:
        raise ValueError("varint mora biti nenegativan")
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            buf.append(b | 0x80)
        else:
            buf.append(b)
            return


def get_varint(data: bytes, pos: int) -> tuple[int, int]:
    n = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("zapis je skraćen (varint)")
        b = data[pos]
        pos += 1
        n |= (b & 0x7F) << shift
        if not b & 0x80:
            return n, pos
        shift += 7


@dataclass(frozen=True)
class Meta:
    width: int
    height: int
    frames: int
    fps: int
    audio_rate: int
    audio_channels: int
    sample_bytes: int
    audio_samples: int

    @property
    def frame_bytes(self) -> int:
        return self.width * self.height * 3

    @property
    def video_bytes(self) -> int:
        return self.frame_bytes * self.frames

    @property
    def audio_bytes(self) -> int:
        return self.audio_samples * self.audio_channels * self.sample_bytes

    @property
    def payload_bytes(self) -> int:
        return self.video_bytes + self.audio_bytes

    def to_bytes(self) -> bytes:
        buf = bytearray()
        for v in (self.width, self.height, self.frames, self.fps, self.audio_rate,
                  self.audio_channels, self.sample_bytes, self.audio_samples):
            put_varint(buf, v)
        return bytes(buf)

    @staticmethod
    def from_bytes(data: bytes, pos: int = 0) -> tuple["Meta", int]:
        vals = []
        for _ in range(8):
            v, pos = get_varint(data, pos)
            vals.append(v)
        return Meta(*vals), pos


def film_bytes(meta: Meta, payload: bytes) -> bytes:
    """Kanonska serijalizacija originala X (metapodaci + sadržaj)."""
    if len(payload) != meta.payload_bytes:
        raise ValueError("payload ne odgovara metapodacima")
    return meta.to_bytes() + payload


def parse_film(blob: bytes) -> tuple[Meta, bytes]:
    meta, pos = Meta.from_bytes(blob, 0)
    payload = blob[pos:]
    if len(payload) != meta.payload_bytes:
        raise ValueError("oštećen film")
    return meta, payload


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def video_meta(width: int, height: int, frames: int, fps: int, audio_rate: int,
               audio_channels: int, sample_bytes: int = 2) -> Meta:
    samples = frames * audio_rate // fps if fps else 0
    return Meta(width, height, frames, fps, audio_rate, audio_channels, sample_bytes, samples)


def bytes_only_meta(n: int) -> Meta:
    """'Film' bez slike, samo n bajtova (koristi se za test granice budžeta)."""
    return Meta(0, 0, 0, 0, 0, 1, 1, n)
