"""E1: nasumični (i kontrolni) video + PCM zvuk — tačna rekonstrukcija.

Za svaku kombinaciju (konfiguracija, izvor, ponavljanje, metoda):
  enkoder i dekoder rade u ODVOJENIM procesima; dekoder dobija samo zapis
  (+64 nasumična bajta "smeća"); poredi se SHA-256 i svaki bajt.
MODEL_ID se beleži PRE generisanja podataka (pre-registracija modela).
Dodatno: kontrola varanja ("leaky" model) i test granice budžeta.

Pokretanje: python3 experiments/e1_random_video.py [--reps 3] [--quick]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import BUDGETS, ROOT, baseline_rss_mb, measure, new_workdir  # noqa: E402

from igc import codec, sources  # noqa: E402
from igc.film import bytes_only_meta, film_bytes, video_meta  # noqa: E402

CONFIGS = {
    # 160x120 RGB, 48 frejmova (2 s @ 24 fps) + 2 s 48 kHz stereo s16 = 3 148 811 B
    "veliki": (video_meta(160, 120, 48, 24, 48000, 2),
               ["raw", "zlib", "lzma", "bz2", "xor_store", "xor_zlib", "infer"]),
    # 32x24 RGB, 24 frejma (1 s) + 1 s 8 kHz mono s16 = 71 307 B (dovoljno malo za CM)
    "mali": (video_meta(32, 24, 24, 24, 8000, 1),
             ["raw", "zlib", "lzma", "bz2", "xor_store", "xor_zlib", "infer", "cm"]),
}
BOUNDARY_N = [512, 1000, 1013, 1024, 1100, 4096, 16384, 65536]


def log2_p_bound(B_bytes: int, N_bits: int) -> float:
    """log2 gornje granice P(tačno) <= min(1, 2^(B+1-N)) za uniformni ulaz."""
    return min(0.0, 8 * B_bytes + 1 - N_bits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    mid = codec.model_id()               # pre-registracija: PRE podataka
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    wd = new_workdir()
    rows = []
    t0 = time.time()
    cfgs = {"mali": CONFIGS["mali"]} if a.quick else CONFIGS
    for cname, (meta, methods) in cfgs.items():
        for src in sources.SOURCES:
            for rep in range(a.reps):
                payload, info = sources.make(src, meta)
                film = film_bytes(meta, payload)
                for m in methods:
                    r = measure(m, film, wd)
                    r.update(config=cname, source=src, rep=rep, truth=str(info))
                    r["model_valid"] = r.get("model_id_at_decode") == mid
                    rows.append(r)
                    print(f"[{time.time()-t0:7.1f}s] {cname:6s} {src:16s} rep{rep} {m:9s} "
                          f"size={r.get('size')} exact={r.get('exact')}", flush=True)
    # --- kontrola varanja: film sakriven u "modelu"
    meta, _ = CONFIGS["mali"]
    payload, _ = sources.make("csprng", meta)
    film = film_bytes(meta, payload)
    leak_dir = wd / "leaky_model"
    r1 = measure("leaky", film, wd, extra_enc_args=["--leaky-dir", str(leak_dir)],
                 dec_env={"IGC_LEAKY_DIR": str(leak_dir)})
    r1.update(config="mali", source="csprng", rep=0, scenario="dekoder sa 'modelom' koji sadrži film")
    r1["model_valid"] = r1.get("model_id_at_decode") == mid
    r2 = measure("leaky", film, wd, extra_enc_args=["--leaky-dir", str(wd / "leaky2")])
    r2.update(config="mali", source="csprng", rep=0, scenario="dekoder sa zamrznutim (pre-registrovanim) modelom")
    r2["model_valid"] = r2.get("model_id_at_decode") == mid
    leaky = [r1, r2]
    # --- granica budžeta: CSPRNG bajtovi bez slike
    boundary = []
    for N in BOUNDARY_N:
        bmeta = bytes_only_meta(N)
        for rep in range(a.reps):
            payload, _ = sources.make("csprng", bmeta)
            film = film_bytes(bmeta, payload)
            for m in ["raw", "zlib", "lzma", "bz2", "cm", "infer"]:
                r = measure(m, film, wd)
                r.update(N=N, rep=rep)
                boundary.append(r)
        print(f"granica N={N} gotovo", flush=True)
    res = {"model_id_preregistered": mid, "baseline_rss_mb": baseline_rss_mb(), "rows": rows,
           "leaky_control": leaky, "boundary": boundary, "budgets": BUDGETS,
           "wall_time_s": time.time() - t0, "python": sys.version}
    (out_dir / "e1_results.json").write_text(json.dumps(res, indent=1))
    (out_dir / "e1_summary.md").write_text(summarize(res))
    print(summarize(res))


def _fmt(x: float, nd: int = 0) -> str:
    """Srpski format: razmak za hiljade, zarez za decimale."""
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def summarize(res: dict) -> str:
    rows = res["rows"]
    L = [f"# E1 — rezultati (MODEL_ID `{res['model_id_preregistered'][:16]}…`)\n",
         f"Osnovni RSS procesa (Python+NumPy+igc, bez rada): {_fmt(res['baseline_rss_mb'], 1)} MB. "
         f"Ukupno vreme: {_fmt(res['wall_time_s'] / 60, 1)} min.\n",
         "Veličina = srednja vrednost stvarnog zapisa u bajtovima (sa zaglavljem i metapodacima). "
         "Tačno = broj ponavljanja sa identičnim SHA-256 i svim bajtovima / broj ponavljanja. "
         "Svaki zapis je dekodovan u zasebnom procesu uz 64 B dodatog smeća.\n"]
    for cname in dict.fromkeys(r["config"] for r in rows):
        sub = [r for r in rows if r["config"] == cname]
        methods = list(dict.fromkeys(r["method"] for r in sub))
        orig = sub[0]["orig_bytes"]
        L.append(f"\n## Konfiguracija `{cname}` — original {_fmt(orig)} B\n")
        L.append("| izvor | " + " | ".join(methods) + " | najbolji: odnos | ≤1K/4K/16K/64K |")
        L.append("|---" * (len(methods) + 3) + "|")
        for src in dict.fromkeys(r["source"] for r in sub):
            cells, best = [], None
            for m in methods:
                rr = [r for r in sub if r["source"] == src and r["method"] == m]
                ok = [r for r in rr if r.get("size") is not None]
                if not ok:
                    cells.append("n/p")
                    continue
                sz = st.mean(r["size"] for r in ok)
                ex = sum(1 for r in ok if r.get("exact") and r.get("self_delimiting"))
                cells.append(f"{_fmt(sz)} ({ex}/{len(rr)})")
                if ex == len(rr) and (best is None or sz < best):
                    best = sz
            fits = "/".join("✔" if best is not None and best <= b else "✘" for b in res["budgets"])
            L.append(f"| {src} | " + " | ".join(cells) + f" | {_fmt(orig / best, 1)}:1 | {fits} |")
        L.append("\nVreme enkodovanja / dekodovanja (s, srednje; zasebni procesi) i vršni RSS (MB, max; VmHWM):\n")
        L.append("| metoda | enc s | dec s | enc RSS MB | dec RSS MB |")
        L.append("|---|---|---|---|---|")
        for m in methods:
            rr = [r for r in sub if r["method"] == m and r.get("size") is not None]
            if rr:
                L.append(f"| {m} | {_fmt(st.mean(r['enc_time_s'] for r in rr), 2)} | "
                         f"{_fmt(st.mean(r['dec_time_s'] for r in rr), 2)} | "
                         f"{_fmt(max(r['enc_rss_mb'] for r in rr))} | {_fmt(max(r['dec_rss_mb'] for r in rr))} |")
    L.append("\n## Kontrola varanja (film sakriven u modelu)\n")
    L.append("| scenario | zapis B | tačno | MODEL_ID isti kao pre-registrovan | važeće |")
    L.append("|---|---|---|---|---|")
    for r in res["leaky_control"]:
        L.append(f"| {r['scenario']} | {r.get('size')} | {r.get('exact')} | {r['model_valid']} | "
                 f"{'DA' if r.get('exact') and r['model_valid'] else 'NE'} |")
    L.append("\n## Granica budžeta: CSPRNG sadržaj od N bajtova (najkraći tačan zapis)\n")
    L.append("Granica se računa samo za N·8 nasumičnih bitova (metapodaci su fiksni). Naš univerzalni "
             "zapis nosi ~10 B zaglavlja (id metode + metapodaci); specijalizovan dekoder bi uštedeo "
             "tih ~10 B, ali ni bit više.\n")
    L.append("| N (B) | najkraći zapis B (metoda) | ≤1 KB | log2 gornje granice P(uspeh) za B=1 KB |")
    L.append("|---|---|---|---|")
    for N in dict.fromkeys(r["N"] for r in res["boundary"]):
        rr = [r for r in res["boundary"] if r["N"] == N and r.get("exact")]
        b = min(rr, key=lambda r: r["size"])
        Nbits = 8 * N
        L.append(f"| {N} | {b['size']} ({b['method']}) | {'✔' if b['size'] <= 1024 else '✘'} | "
                 f"{log2_p_bound(1024, Nbits):.0f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
