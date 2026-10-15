"""E3: Film kao dinamički sistem — globalno pravilo umesto lokalnih korekcija.

Film: Life-like ćelijski automat 128x128, 240 frejmova (10 s @ 24 fps), RGB
+ 10 s zvuka (7200 Hz, s16) izvedenog iz sadržaja = 11 940 491 B.
Šum procesa: svaka ćelija u svakom koraku se invertuje sa verovatnoćom p (CSPRNG).

Tačna entropija procesa (pravilo i boje poznati): H = s^2 + N*h(p) bita, jer je
preslikavanje (F_0, E) -> film bijekcija. Po instanci računamo idealnu dužinu
(samoinformaciju) -log2 P(F_0) - log2 P(E) i poredimo je sa stvarnim zapisom.

Hipoteza (opovrgljiva): zapis metode "rule" = idealno + mala konstanta, za sve p;
generički kompresori su daleko iznad za malo p. Opovrgnuto ako: (a) rule zapis
odstupa od ideala više od ~1 KB + 5 %, ili (b) neki kompresor bez znanja o
pravilu bude kraći za p <= 1e-3, ili (c) zapis padne ispod ideala za više od
~log2(N) bita (empirijski p̂ sme da uštedi O(log N) bita; više od toga => greška u
merenju ili curenje informacije).
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import ROOT, measure, new_workdir  # noqa: E402

from igc import ca, codec  # noqa: E402
from igc.film import film_bytes  # noqa: E402

RULES = {"Conway B3/S23": ca.CONWAY,
         "HighLife B36/S23": (1 << 3) | (1 << 6) | (1 << 11) | (1 << 12),
         "Seeds B2/S": (1 << 2)}


def ideal_bits(soup_bits: int, N: int, K: int, p: float) -> float:
    e = 0.0
    if p > 0:
        e = -K * math.log2(p) - (N - K) * math.log2(1 - p)
    return soup_bits + e


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    mid = codec.model_id()
    wd = new_workdir()
    rows = []
    t0 = time.time()
    plan = [("Conway B3/S23", p) for p in [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 0.2, 0.5]]
    plan += [("HighLife B36/S23", 1e-4), ("Seeds B2/S", 1e-4), ("slučajno pravilo", 1e-4)]
    for rname, p in plan:
        for rep in range(reps):
            rule = RULES.get(rname)
            if rule is None:
                rule = int.from_bytes(os.urandom(3), "little") & ((1 << 18) - 1) & ~1  # bez B0
            meta, payload, info = ca.life_film(p=p, rule=rule)
            film = film_bytes(meta, payload)
            ideal = ideal_bits(info["soup_bits"], info["N_cells"], info["exceptions"], p)
            for m in ["rule", "lzma", "zlib", "bz2"]:
                r = measure(m, film, wd)
                r.update(rule_name=rname, rule=rule, p=p, rep=rep, ideal_bytes=ideal / 8, **info)
                r["model_valid"] = r.get("model_id_at_decode") == mid
                rows.append(r)
                print(f"[{time.time()-t0:6.0f}s] {rname:18s} p={p:<7g} rep{rep} {m:5s} "
                      f"size={r.get('size')} ideal={ideal/8:.0f} exact={r.get('exact')}", flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/e3_results.json").write_text(json.dumps(rows, indent=1))
    (ROOT / "results/e3_summary.md").write_text(summarize(rows))
    print(summarize(rows))


def _f(x: float, nd: int = 0) -> str:
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def summarize(rows: list[dict]) -> str:
    import statistics as st
    L = ["# E3 — pravilo + izuzeci (Life-like CA film, 11 940 491 B)\n",
         "Veličine su srednje vrednosti stvarnih zapisa (B); 'ideal' = samoinformacija procesa "
         "(F_0 + izuzeci) uz poznato pravilo; tačnost = SHA-256 + svi bajtovi, nezavisan dekoder "
         "u zasebnom procesu, 64 B smeća iza zapisa.\n",
         "| pravilo | p | izuzetaka (sr.) | ideal B | rule B | rule/ideal | lzma B | zlib B | bz2 B "
         "| tačno (sve) | odnos (rule) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = list(dict.fromkeys((r["rule_name"], r["p"]) for r in rows))
    for rn, p in keys:
        sub = [r for r in rows if r["rule_name"] == rn and r["p"] == p]
        g = lambda m: [r for r in sub if r["method"] == m]
        sz = lambda m: st.mean(r["size"] for r in g(m))
        ideal = st.mean(r["ideal_bytes"] for r in g("rule"))
        exc = st.mean(r["exceptions"] for r in g("rule"))
        ex = all(r.get("exact") and r.get("self_delimiting") for r in sub)
        orig = sub[0]["orig_bytes"]
        L.append(f"| {rn} | {p:g} | {_f(exc)} | {_f(ideal)} | {_f(sz('rule'))} | {_f(sz('rule') / ideal, 3)} | "
                 f"{_f(sz('lzma'))} | {_f(sz('zlib'))} | {_f(sz('bz2'))} | {ex} | {_f(orig / sz('rule'))}:1 |")
    L.append("\nVreme i memorija (srednje / max preko svih p; zasebni procesi, VmHWM):\n")
    L.append("| metoda | enc s | dec s | enc RSS MB | dec RSS MB |")
    L.append("|---|---|---|---|---|")
    for m in ["rule", "lzma", "zlib", "bz2"]:
        rr = [r for r in rows if r["method"] == m and r.get("size") is not None]
        L.append(f"| {m} | {_f(st.mean(r['enc_time_s'] for r in rr), 2)} | {_f(st.mean(r['dec_time_s'] for r in rr), 2)} | "
                 f"{_f(max(r['enc_rss_mb'] for r in rr))} | {_f(max(r['dec_rss_mb'] for r in rr))} |")
    return "\n".join(L) + "\n"

if __name__ == "__main__":
    main()
