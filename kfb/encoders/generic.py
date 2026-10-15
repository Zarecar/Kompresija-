"""Generičke osnovne linije iz igc (ne znaju ništa o porodicama)."""
from __future__ import annotations

from igc import codec as igc_codec

from . import Encoder


def _mk(name: str):
    def fn(meta, payload, budget_s):
        return igc_codec.encode(name, meta, payload)
    return fn


ENCODERS = [
    Encoder("raw", _mk("raw"), note="sirovi zapis (gornja granica za sve)"),
    Encoder("zlib", _mk("zlib"), note="raw deflate, nivo 9"),
    Encoder("lzma", _mk("lzma"), note="raw LZMA2, preset 9e"),
    Encoder("bz2", _mk("bz2"), note="bzip2 -9"),
    Encoder("cm", _mk("cm"), tiers=frozenset({"mini"}), note="kontekstni prediktor + mikser (spor, samo mini)"),
    Encoder("mt_infer", _mk("infer"), note="inverzija MT19937 / slabog SHAKE (iz igc)"),
]
