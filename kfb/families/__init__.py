"""Porodice filmova. Svaki modul u ovom paketu definiše:

    NAME: str                         jedinstveno ime porodice
    LEVELS: dict[str, dict]           ime nivoa -> parametri (šum, težina, ...)
    generate(rng, meta, level) -> (program_ili_payload, info)
        rng   : numpy.random.Generator (jedini izvor slučajnosti!)
        meta  : igc.film.Meta (dimenzije su zadate nivoom benchmarka - "tier")
        level : dict iz LEVELS
        vraća : kfb.kfl.Program (oracle program; K̄ = dužina njegovog zapisa)
                ili bytes (payload bez kratkog opisa, npr. CSPRNG kontrola)
        info  : dict, JSON-serijalizabilan; preporučeno "ideal_bits" (samoinformacija
                slučajnih delova procesa) i "params".
Film se UVEK dobija izvršavanjem oracle programa na VM-u, pa je K̄ proverena
gornja granica složenosti filma u odnosu na mašinu M.
"""
from __future__ import annotations

import importlib
import pkgutil


def load_all() -> dict:
    mods = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        m = importlib.import_module(f"{__name__}.{info.name}")
        for attr in ("NAME", "LEVELS", "generate"):
            if not hasattr(m, attr):
                raise ImportError(f"porodica {info.name} nema {attr}")
        if m.NAME in mods:
            raise ImportError(f"dupla porodica {m.NAME}")
        mods[m.NAME] = m
    return mods
