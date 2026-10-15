"""Kontrola: CSPRNG sadržaj (os.urandom preko rng-a NIJE dovoljan -> koristimo secrets).

Nema kratkog opisa: K̄ = sirovi zapis. SVAKI enkoder mora dati >= N bajtova
(Teorema 1); kraći tačan zapis znači bag u merenju ili curenje informacije.
"""
from __future__ import annotations

import secrets

NAME = "control"
LEVELS = {"csprng": {}}


def generate(rng, meta, level):
    return secrets.token_bytes(meta.payload_bytes), {"ideal_bits": 8 * meta.payload_bytes,
                                                     "params": {"source": "secrets.token_bytes"}}
