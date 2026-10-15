"""Enkoderi. Svaki modul u ovom paketu definiše listu ENCODERS:

    Encoder(name, fn, tiers, family_aware, note)
    fn(meta, payload, budget_s) -> bytes | None     KOMPLETAN zapis (kfb.record format)

Enkoder NE SME da čita ground truth (kfb suite truth/), niti da zavisi od
toga koja je porodica generisala film, osim ako je family_aware=True (tada sme
da pretpostavi konvencije SVOJE porodice, ali ne i parametre konkretne instance).
Enkoderi nisu deo modela M; mogu da budu spori, stohastički ili da pretražuju.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Encoder:
    name: str
    fn: Callable
    tiers: frozenset = frozenset({"mini", "std"})
    family_aware: bool = False
    note: str = ""


def load_all() -> dict[str, Encoder]:
    encs: dict[str, Encoder] = {}
    for info in sorted(pkgutil.iter_modules(__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        m = importlib.import_module(f"{__name__}.{info.name}")
        for e in getattr(m, "ENCODERS", []):
            if e.name in encs:
                raise ImportError(f"dupli enkoder {e.name}")
            encs[e.name] = e
    return encs
