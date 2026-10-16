# KFB v0: rezultati

MODEL_ID (pre-registrovan): `469c1a4a619ecb94…`; skup napravljen sa `469c1a4a619ecb94…`. Zapisa: 2109, tačnih i važećih: 1683. Ukupno vreme: 51,9 min (paralelno 3 procesa).

K̄ je dužina oracle zapisa (proverena gornja granica složenosti u odnosu na mašinu M). L/K̄ je geometrijska sredina po instancama u grupi; ≤K̄ je broj instanci gde je najbolji zapis klase ne duži od oracle-a. Klase su definisane u zaglavlju `kfb/report.py`.

## Ukupno po klasi enkodera (sve instance osim CSPRNG kontrole)

| klasa | instanci sa tačnim zapisom | L/K̄ (geom.) | ≤ K̄ | ≤ 1,1·K̄ | ≤ 1 KB |
|---|---|---|---|---|---|
| generički | 114/114 | 216,29 | 0 | 0 | 9 |
| KFL-porodica | 90/114 | 1,00 | 90 | 90 | 81 |
| KFL-pretraga | 20/114 | 429,35 | 7 | 7 | 7 |
| LLM | 8/114 | 2,51 | 6 | 6 | 7 |
| **najbolji ukupno** | 114/114 | 2,40 | 98 | 98 | 91 |

Oracle (K̄) staje u 1 KB na 105/114 instanci.

## Po porodici, nivou i veličini

| porodica | nivo | tier | n | N (B) | K̄ (B) | generički L (L/K̄) | KFL-porodica L (L/K̄) | KFL-pretraga L (L/K̄) | LLM L (L/K̄) | najbolji (enkoder) | ≤K̄ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ca | conway_p0 | mini | 3 | 89 738 | 111 | 4 294 (38,54) | 111 (1,00) | — | 68 (0,61) [1/3] | ca_infer, ext:llm3 | 3/3 |
| ca | conway_p0 | std | 3 | 2 718 219 | 231 | 28 334 (122,26) | 231 (1,00) | — | — | ca_infer | 3/3 |
| ca | conway_p1e-3 | mini | 3 | 89 738 | 147 | 3 907 (26,48) | 147 (1,00) | — | — | ca_infer | 3/3 |
| ca | conway_p1e-3 | std | 3 | 2 718 219 | 1 497 | 30 080 (20,06) | 1 497 (1,00) | — | — | ca_infer | 3/3 |
| ca | random_rule_p1e-4 | mini | 3 | 89 738 | 117 | 5 672 (45,41) | 117 (1,00) | — | — | ca_infer | 3/3 |
| ca | random_rule_p1e-4 | std | 3 | 2 718 219 | 390 | 123 245 (315,72) | 390 (1,00) | — | — | ca_infer | 3/3 |
| control | csprng | mini | 3 | 89 738 | 89 739 | 89 739 (1,00) | — | — | 89 761 (1,00) [1/3] | raw→raw | 3/3 |
| control | csprng | std | 3 | 2 718 219 | 2 718 220 | 2 718 220 (1,00) | — | 2 718 248 (1,00) | — | raw→raw | 3/3 |
| diffusion | smooth | mini | 3 | 89 738 | 116 | 10 859 (90,52) | 116 (1,00) | — | 87 (0,96) [1/3] | diffusion_infer, ext:llm2 | 3/3 |
| diffusion | smooth | std | 3 | 2 718 219 | 116 | 172 610 (1 111,34) | 116 (1,00) | — | — | diffusion_infer | 3/3 |
| diffusion | smooth_noise | mini | 3 | 89 738 | 100 | 9 482 (78,13) | 100 (1,00) | — | — | diffusion_infer | 3/3 |
| diffusion | smooth_noise | std | 3 | 2 718 219 | 346 | 190 085 (181,65) | 346 (1,00) | — | — | diffusion_infer | 3/3 |
| diffusion | wave | mini | 3 | 89 738 | 143 | 8 754 (61,22) | 143 (1,00) | — | — | diffusion_infer | 3/3 |
| diffusion | wave | std | 3 | 2 718 219 | 156 | 168 699 (1 066,38) | 156 (1,00) | — | — | diffusion_infer | 3/3 |
| prng | seed16 | mini | 3 | 89 738 | 51 | 89 739 (1 759,59) | 51 (1,00) | — | 89 761 (1 760,02) [1/3] | prng_infer | 3/3 |
| prng | seed16 | std | 3 | 2 718 219 | 52 | 2 718 220 (52 273,46) | 52 (1,00) | 2 718 248 (52 274,00) | — | prng_infer | 3/3 |
| prng | seed24 | mini | 3 | 89 738 | 52 | 89 739 (1 725,75) | 52 (1,00) | — | — | prng_infer | 3/3 |
| prng | seed24 | std | 3 | 2 718 219 | 54 | 2 718 220 (50 652,02) | 54 (1,00) | 2 718 248 (50 652,55) | — | prng_infer | 3/3 |
| prng | seed64 | mini | 3 | 89 738 | 53 | 89 739 (1 693,19) | — | — | — | raw→raw | 0/3 |
| prng | seed64 | std | 3 | 2 718 219 | 54 | 2 718 220 (50 337,41) | — | 2 718 248 (50 337,93) | — | raw→raw | 0/3 |
| randprog | n15 | mini | 3 | 89 738 | 32 | 1 915 (60,36) | — | — | 60 (1,76) [1/3] | diffusion_infer→lzma, ext:llm1 | 0/3 |
| randprog | n15 | std | 3 | 2 718 219 | 36 | 14 501 (266,08) | — | — | — | diffusion_infer→lzma | 0/3 |
| randprog | n5 | mini | 3 | 89 738 | 17 | 274 (16,38) | — | 17 (1,00) | 16 (1,00) [1/3] | enum_search | 3/3 |
| randprog | n5 | std | 3 | 2 718 219 | 17 | 4 957 (167,08) | — | 17 (1,00) | — | enum_search | 3/3 |
| randprog | n9 | mini | 3 | 89 738 | 28 | 660 (20,43) | — | 23 (1,00) [1/3] | 31 (1,00) [1/3] | diffusion_infer→lzma, enum_search, ext:llm0 | 2/3 |
| randprog | n9 | std | 3 | 2 718 219 | 27 | 9 402 (327,76) | — | — | — | diffusion_infer→lzma | 0/3 |
| shader | one_term | mini | 3 | 89 738 | 54 | 2 445 (43,09) | 54 (1,00) | — | 41 (0,95) [1/3] | ext:llm1, shader_infer | 3/3 |
| shader | one_term | std | 3 | 2 718 219 | 49 | 2 435 (50,03) | 49 (1,00) | — | — | shader_infer | 3/3 |
| shader | rings_xor | mini | 3 | 89 738 | 57 | 25 677 (427,42) | 57 (1,00) | — | — | shader_infer | 3/3 |
| shader | rings_xor | std | 3 | 2 718 219 | 57 | 1 131 714 (19 843,05) | 57 (1,00) | — | — | shader_infer | 3/3 |
| shader | three_terms | mini | 3 | 89 738 | 89 | 52 881 (593,66) | 89 (1,00) | 74 785 (849,83) [1/3] | — | shader_infer | 3/3 |
| shader | three_terms | std | 3 | 2 718 219 | 85 | 1 132 923 (13 112,74) | 85 (1,00) | — | — | shader_infer | 3/3 |
| shader | three_terms_noise | mini | 3 | 89 738 | 264 | 46 357 (175,09) | 264 (1,00) | 74 448 (282,39) | — | shader_infer | 3/3 |
| shader | three_terms_noise | std | 3 | 2 718 219 | 6 532 | 1 210 644 (185,14) | 6 532 (1,00) | — | — | shader_infer | 3/3 |
| sprites | k2 | mini | 3 | 89 738 | 90 | 1 943 (15,76) | 90 (1,00) | — | 77 (0,90) [1/3] | ext:llm0, sprites_infer | 3/3 |
| sprites | k2 | std | 3 | 2 718 219 | 111 | 11 269 (85,84) | 111 (1,00) | — | — | sprites_infer | 3/3 |
| sprites | k3_noise | mini | 3 | 89 738 | 292 | 1 799 (5,58) | 292 (1,00) | — | — | sprites_infer | 3/3 |
| sprites | k3_noise | std | 3 | 2 718 219 | 6 589 | 14 510 (2,20) | 6 589 (1,00) | — | — | sprites_infer | 3/3 |
| sprites | k6 | mini | 3 | 89 738 | 199 | 1 895 (9,49) | 199 (1,00) | — | — | sprites_infer | 3/3 |
| sprites | k6 | std | 3 | 2 718 219 | 253 | 31 575 (110,09) | 253 (1,00) | — | — | sprites_infer | 3/3 |

## Kontrole

- CSPRNG kontrola: nijedan tačan zapis nije kraći od sadržaja (OK).
- Zapisi dekodovani sa drugim MODEL_ID: 0.
- Tačni, ali ne samoograničavajući (nevažeći): 0.
- Dekodovani, ali NETAČNI zapisi: 0.
- Zapisi koje dekoder nije mogao da pročita: 0.
- Prekoračenje vremena enkodera: 0.

## Po enkoderu

| enkoder | family-aware | pokušaja | tačnih | od toga KFL | najbolji na instanci | sr. enc s | max enc s | sr. dec s | max enc RSS MB |
|---|---|---|---|---|---|---|---|---|---|
| apng | ne | 120 | 120 | 0 | 0 | 1,39 | 8,1 | 0,11 | 53 |
| av1 | ne | 120 | 120 | 0 | 0 | 4,88 | 21,1 | 0,18 | 61 |
| bz2 | ne | 120 | 120 | 0 | 0 | 0,24 | 0,9 | 0,07 | 48 |
| ca_infer | da | 120 | 18 | 18 | 17 | 0,16 | 1,1 | 0,04 | 107 |
| cm | ne | 60 | 60 | 0 | 0 | 4,21 | 6,5 | 4,18 | 63 |
| diffusion_infer | da | 120 | 120 | 21 | 26 | 3,12 | 20,9 | 0,02 | 253 |
| enum_search | ne | 120 | 120 | 23 | 7 | 33,33 | 53,8 | 0,03 | 1 012 |
| ext:llm0 | ne | 2 | 2 | 2 | 2 | 1 100,00 | 1 500,0 | 0,01 | — |
| ext:llm1 | ne | 2 | 2 | 2 | 2 | 1 650,00 | 2 400,0 | 0,01 | — |
| ext:llm2 | ne | 2 | 2 | 2 | 1 | 945,00 | 1 650,0 | 0,01 | — |
| ext:llm3 | ne | 1 | 1 | 1 | 1 | — | — | 0,01 | — |
| ext:llm4 | ne | 2 | 2 | 2 | 0 | — | — | 0,01 | — |
| ffv1 | ne | 120 | 120 | 0 | 0 | 1,38 | 3,4 | 0,18 | 66 |
| lzma | ne | 120 | 120 | 0 | 0 | 0,52 | 2,5 | 0,02 | 129 |
| mt_infer | ne | 120 | 0 | 0 | 0 | 1,05 | 1,3 | — | 43 |
| prng_infer | da | 120 | 12 | 12 | 12 | 0,54 | 0,9 | 0,04 | 78 |
| raw | ne | 120 | 120 | 0 | 12 | 0,00 | 0,0 | 0,00 | 38 |
| shader_infer | da | 120 | 24 | 24 | 23 | 2,80 | 48,1 | 0,03 | 374 |
| sprites_infer | da | 120 | 120 | 18 | 17 | 1,95 | 27,9 | 0,02 | 185 |
| vp9 | ne | 120 | 120 | 0 | 0 | 4,88 | 19,6 | 0,17 | 73 |
| x264rgb | ne | 120 | 120 | 0 | 0 | 1,33 | 4,5 | 0,14 | 53 |
| x265 | ne | 120 | 120 | 0 | 0 | 7,76 | 26,6 | 0,19 | 65 |
| zlib | ne | 120 | 120 | 0 | 0 | 0,09 | 2,1 | 0,01 | 42 |
