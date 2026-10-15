"""E5: Proračun granica (aritmetika, ne merenje). Pretpostavke su eksplicitne.

Ispisuje markdown tabelu u results/e5_bounds.md.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = 8192  # 1 KB u bitovima


def th(x: float) -> str:
    """Hiljade razdvojene razmakom."""
    return f"{x:,.0f}".replace(",", " ")


def dc(x: float, nd: int) -> str:
    """Decimalni zarez."""
    return f"{x:.{nd}f}".replace(".", ",")


def main() -> None:
    secs = 2 * 3600
    fps = 24
    frames = secs * fps
    raw = 1920 * 1080 * 1.5 * frames + 48000 * 2 * 2 * secs       # 8-bit 4:2:0 + PCM stereo 16 bit
    shots_lo, shots_hi = 1000, 3000                                  # pretpostavka: tipičan broj kadrova
    lloyd_ops = 1e120
    rows = [
        ("Sirovi 2 h 1080p24 (8 bit 4:2:0) + PCM 48 kHz stereo", f"{th(raw/1e9)} GB"),
        ("Tipičan distribuirani fajl (~5 Mbit/s)", f"{dc(5e6*secs/8/1e9, 1)} GB"),
        ("Generativni video kodek pri 0,005 bpp (red veličine iz radova 2025.)",
         f"{th(1920*1080*0.005*frames/8/1e6)} MB"),
        ("Budžet 1 KB po sekundi filma", f"{dc(B/secs, 2)} bit/s"),
        ("Budžet 1 KB po frejmu", f"{dc(B/frames, 4)} bit/frejm"),
        (f"Budžet 1 KB po kadru ({shots_lo}–{shots_hi} kadrova)", f"{dc(B/shots_hi, 1)}–{dc(B/shots_lo, 1)} bit/kadar"),
        ("Broj različitih zapisa ≤ 1 KB (bajt-poravnatih)", f"≈ 2^{8*1024}·256/255 ≈ 10^{8192*math.log10(2):.0f}"),
        ("Fizička granica pretrage (Lloyd 2002: ~10^120 operacija u istoriji svemira)",
         f"≈ 2^{math.log2(lloyd_ops):.0f} kandidata → najviše ~{math.log2(lloyd_ops):.0f} bita 'pogođenog' seed-a"),
        ("Dijalog filma (pretp. 10–15 hiljada reči, ~6 znakova/reč, 0,7–1,0 bit/znak uz jak LM)",
         f"{dc(10000*6*0.7/8/1024, 1)}–{dc(15000*6*1.0/8/1024, 1)} KB"),
        ("Tekst koji staje u 1 KB pri 0,7–1,0 bit/znak (≈ 'treatment')",
         f"{th(B/1.0)}–{th(B/0.7)} znakova ≈ {th(B/1.0/6)}–{th(B/0.7/6)} reči"),
        ("Šum senzora: zaokružen Gaus σ=2 → entropija po uzorku", "≈ 3,06 bita (vidi igc.sources)"),
        ("Donja granica tačnog zapisa sirovog filma sa takvim šumom (samo luma)",
         f"≥ {th(1920*1080*frames*3.06/8/1e9)} GB"),
    ]
    import sys
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "experiments"))
    from bounds_util import structured_noisy_entropy_bits
    from igc.film import video_meta
    for name, meta in [("mali", video_meta(32, 24, 24, 24, 8000, 1)),
                       ("veliki", video_meta(160, 120, 48, 24, 48000, 2))]:
        hb = structured_noisy_entropy_bits(meta, 0)
        rows.append((f"E1 `structured_noisy` ({name}): tačna entropija šuma sa odsecanjem (faza 0)",
                     f"{th(hb / 8)} B = donja granica E[dužine] tačnog koda"))
    L = ["# E5 — granice i budžeti (proračun, ne merenje)\n", "| veličina | vrednost |", "|---|---|"]
    L += [f"| {a} | {b} |" for a, b in rows]
    L.append("\nOpšta granica (Teorema 1 u IZVESTAJ.md): P(tačno) ≤ 2^(B+1−H_∞(X)); za uniformni "
             "N-bitni ulaz H_∞ = N. Za očekivanu dužinu prefiksnog koda: E[L] ≥ H(X).")
    text = "\n".join(L) + "\n"
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/e5_bounds.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
