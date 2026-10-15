# E5 — granice i budžeti (proračun, ne merenje)

| veličina | vrednost |
|---|---|
| Sirovi 2 h 1080p24 (8 bit 4:2:0) + PCM 48 kHz stereo | 539 GB |
| Tipičan distribuirani fajl (~5 Mbit/s) | 4,5 GB |
| Generativni video kodek pri 0,005 bpp (red veličine iz radova 2025.) | 224 MB |
| Budžet 1 KB po sekundi filma | 1,14 bit/s |
| Budžet 1 KB po frejmu | 0,0474 bit/frejm |
| Budžet 1 KB po kadru (1000–3000 kadrova) | 2,7–8,2 bit/kadar |
| Broj različitih zapisa ≤ 1 KB (bajt-poravnatih) | ≈ 2^8192·256/255 ≈ 10^2466 |
| Fizička granica pretrage (Lloyd 2002: ~10^120 operacija u istoriji svemira) | ≈ 2^399 kandidata → najviše ~399 bita 'pogođenog' seed-a |
| Dijalog filma (pretp. 10–15 hiljada reči, ~6 znakova/reč, 0,7–1,0 bit/znak uz jak LM) | 5,1–11,0 KB |
| Tekst koji staje u 1 KB pri 0,7–1,0 bit/znak (≈ 'treatment') | 8 192–11 703 znakova ≈ 1 365–1 950 reči |
| Šum senzora: zaokružen Gaus σ=2 → entropija po uzorku | ≈ 3,06 bita (vidi igc.sources) |
| Donja granica tačnog zapisa sirovog filma sa takvim šumom (samo luma) | ≥ 137 GB |
| E1 `structured_noisy` (mali): tačna entropija šuma sa odsecanjem (faza 0) | 24 200 B = donja granica E[dužine] tačnog koda |
| E1 `structured_noisy` (veliki): tačna entropija šuma sa odsecanjem (faza 0) | 1 125 974 B = donja granica E[dužine] tačnog koda |

Opšta granica (Teorema 1 u IZVESTAJ.md): P(tačno) ≤ 2^(B+1−H_∞(X)); za uniformni N-bitni ulaz H_∞ = N. Za očekivanu dužinu prefiksnog koda: E[L] ≥ H(X).
