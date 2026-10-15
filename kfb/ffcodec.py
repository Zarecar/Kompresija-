"""ffmpeg kodeci bez gubitaka kao osnovne linije (metode 0x20-0x2F).

INTERFEJS (popunjava se u fazi izgradnje):
    encode(method_id, meta, payload) -> bytes | None   kompletan zapis
    decode(rec) -> (meta, payload, consumed)
Zapis: [id][meta][varint dužina video toka][video tok][varint dužina audio toka][audio tok]
"""
from __future__ import annotations

from igc.film import Meta

METHODS: dict[str, int] = {}


def encode(method_id: int, meta: Meta, payload: bytes) -> bytes | None:
    raise NotImplementedError


def decode(rec: bytes) -> tuple[Meta, bytes, int]:
    raise ValueError("ffmpeg metode još nisu implementirane")
