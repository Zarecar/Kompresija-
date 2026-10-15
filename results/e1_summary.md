# E1 — rezultati (MODEL_ID `69dba75c5f848aaa…`)

Osnovni RSS procesa (Python+NumPy+igc, bez rada): 30,2 MB. Ukupno vreme: 9,5 min.

Veličina = srednja vrednost stvarnog zapisa u bajtovima (sa zaglavljem i metapodacima). Tačno = broj ponavljanja sa identičnim SHA-256 i svim bajtovima / broj ponavljanja. Svaki zapis je dekodovan u zasebnom procesu uz 64 B dodatog smeća.


## Konfiguracija `veliki` — original 3 148 813 B

| izvor | raw | zlib | lzma | bz2 | xor_store | xor_zlib | infer | najbolji: odnos | ≤1K/4K/16K/64K |
|---|---|---|---|---|---|---|---|---|---|
| csprng | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 099 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | n/p | 1,0:1 | ✘/✘/✘/✘ |
| shake_keyed | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 085 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | n/p | 1,0:1 | ✘/✘/✘/✘ |
| np_pcg64 | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 196 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | n/p | 1,0:1 | ✘/✘/✘/✘ |
| np_legacy | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 259 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | 21 (3/3) | 149 943,5:1 | ✔/✔/✔/✔ |
| py_random | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 525 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | 22 (3/3) | 143 127,9:1 | ✔/✔/✔/✔ |
| py_random_os | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 396 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | 2 512 (3/3) | 1 253,5:1 | ✘/✔/✔/✔ |
| shake_weak20 | 3 148 814 (3/3) | 3 149 299 (3/3) | 3 148 971 (3/3) | 3 163 313 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | 18 (3/3) | 174 934,1:1 | ✔/✔/✔/✔ |
| structured | 3 148 814 (3/3) | 2 296 508 (3/3) | 847 998 (3/3) | 2 516 763 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | n/p | 3,7:1 | ✘/✘/✘/✘ |
| structured_noisy | 3 148 814 (3/3) | 3 078 831 (3/3) | 2 668 780 (3/3) | 2 939 502 (3/3) | 3 148 814 (3/3) | 3 149 299 (3/3) | n/p | 1,2:1 | ✘/✘/✘/✘ |

Vreme enkodovanja / dekodovanja (s, srednje; zasebni procesi) i vršni RSS (MB, max; VmHWM):

| metoda | enc s | dec s | enc RSS MB | dec RSS MB |
|---|---|---|---|---|
| raw | 0,00 | 0,01 | 37 | 40 |
| zlib | 0,08 | 0,02 | 41 | 46 |
| lzma | 1,01 | 0,05 | 132 | 49 |
| bz2 | 0,42 | 0,25 | 47 | 49 |
| xor_store | 0,01 | 0,02 | 40 | 46 |
| xor_zlib | 0,09 | 0,02 | 45 | 46 |
| infer | 0,86 | 0,04 | 46 | 43 |

## Konfiguracija `mali` — original 71 306 B

| izvor | raw | zlib | lzma | bz2 | xor_store | xor_zlib | infer | cm | najbolji: odnos | ≤1K/4K/16K/64K |
|---|---|---|---|---|---|---|---|---|---|---|
| csprng | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 975 (3/3) | 71 307 (3/3) | 71 322 (3/3) | n/p | 72 209 (3/3) | 1,0:1 | ✘/✘/✘/✘ |
| shake_keyed | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 973 (3/3) | 71 307 (3/3) | 71 322 (3/3) | n/p | 72 204 (3/3) | 1,0:1 | ✘/✘/✘/✘ |
| np_pcg64 | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 984 (3/3) | 71 307 (3/3) | 71 322 (3/3) | n/p | 72 202 (3/3) | 1,0:1 | ✘/✘/✘/✘ |
| np_legacy | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 969 (3/3) | 71 307 (3/3) | 71 322 (3/3) | 18 (3/3) | 72 195 (3/3) | 3 961,4:1 | ✔/✔/✔/✔ |
| py_random | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 971 (3/3) | 71 307 (3/3) | 71 322 (3/3) | 19 (3/3) | 72 196 (3/3) | 3 752,9:1 | ✔/✔/✔/✔ |
| py_random_os | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 963 (3/3) | 71 307 (3/3) | 71 322 (3/3) | 2 509 (3/3) | 72 206 (3/3) | 28,4:1 | ✘/✔/✔/✔ |
| shake_weak20 | 71 307 (3/3) | 71 322 (3/3) | 71 314 (3/3) | 71 985 (3/3) | 71 307 (3/3) | 71 322 (3/3) | 15 (3/3) | 72 188 (3/3) | 4 753,7:1 | ✔/✔/✔/✔ |
| structured | 71 307 (3/3) | 48 941 (3/3) | 19 099 (3/3) | 46 880 (3/3) | 71 307 (3/3) | 71 322 (3/3) | n/p | 3 677 (3/3) | 19,4:1 | ✘/✔/✔/✔ |
| structured_noisy | 71 307 (3/3) | 66 904 (3/3) | 55 410 (3/3) | 63 167 (3/3) | 71 307 (3/3) | 71 322 (3/3) | n/p | 33 507 (3/3) | 2,1:1 | ✘/✘/✘/✔ |

Vreme enkodovanja / dekodovanja (s, srednje; zasebni procesi) i vršni RSS (MB, max; VmHWM):

| metoda | enc s | dec s | enc RSS MB | dec RSS MB |
|---|---|---|---|---|
| raw | 0,00 | 0,00 | 31 | 31 |
| zlib | 0,00 | 0,00 | 31 | 31 |
| lzma | 0,06 | 0,00 | 96 | 31 |
| bz2 | 0,01 | 0,01 | 31 | 31 |
| xor_store | 0,00 | 0,00 | 31 | 31 |
| xor_zlib | 0,00 | 0,00 | 31 | 31 |
| infer | 0,87 | 0,00 | 31 | 31 |
| cm | 3,71 | 3,74 | 59 | 60 |

## Kontrola varanja (film sakriven u modelu)

| scenario | zapis B | tačno | MODEL_ID isti kao pre-registrovan | važeće |
|---|---|---|---|---|
| dekoder sa 'modelom' koji sadrži film | 19 | True | False | NE |
| dekoder sa zamrznutim (pre-registrovanim) modelom | 19 | False | True | NE |

## Granica budžeta: CSPRNG sadržaj od N bajtova (najkraći tačan zapis)

Granica se računa samo za N·8 nasumičnih bitova (metapodaci su fiksni). Naš univerzalni zapis nosi ~10 B zaglavlja (id metode + metapodaci); specijalizovan dekoder bi uštedeo tih ~10 B, ali ni bit više.

| N (B) | najkraći zapis B (metoda) | ≤1 KB | log2 gornje granice P(uspeh) za B=1 KB |
|---|---|---|---|
| 512 | 522 (raw) | ✔ | 0 |
| 1000 | 1010 (raw) | ✔ | 0 |
| 1013 | 1023 (raw) | ✔ | 0 |
| 1024 | 1034 (raw) | ✘ | 0 |
| 1100 | 1110 (raw) | ✘ | -607 |
| 4096 | 4106 (raw) | ✘ | -24575 |
| 16384 | 16395 (raw) | ✘ | -122879 |
| 65536 | 65547 (raw) | ✘ | -516095 |
