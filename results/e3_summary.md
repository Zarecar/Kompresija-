# E3 — pravilo + izuzeci (Life-like CA film, 11 940 491 B)

Veličine su srednje vrednosti stvarnih zapisa (B); 'ideal' = samoinformacija procesa (F_0 + izuzeci) uz poznato pravilo; tačnost = SHA-256 + svi bajtovi, nezavisan dekoder u zasebnom procesu, 64 B smeća iza zapisa.

| pravilo | p | izuzetaka (sr.) | ideal B | rule B | rule/ideal | lzma B | zlib B | bz2 B | tačno (sve) | odnos (rule) |
|---|---|---|---|---|---|---|---|---|---|---|
| Conway B3/S23 | 0 | 0 | 128 | 203 | 1,586 | 154 504 | 181 286 | 131 570 | True | 58 820:1 |
| Conway B3/S23 | 1e-06 | 6 | 144 | 225 | 1,555 | 159 020 | 186 295 | 136 556 | True | 53 148:1 |
| Conway B3/S23 | 1e-05 | 30 | 197 | 274 | 1,390 | 154 279 | 179 366 | 130 906 | True | 43 525:1 |
| Conway B3/S23 | 0.0001 | 402 | 867 | 947 | 1,092 | 144 355 | 166 572 | 122 409 | True | 12 613:1 |
| Conway B3/S23 | 0.001 | 3 902 | 5 694 | 5 813 | 1,021 | 163 434 | 188 034 | 142 610 | True | 2 054:1 |
| Conway B3/S23 | 0.01 | 39 098 | 39 624 | 39 846 | 1,006 | 273 228 | 316 878 | 271 387 | True | 300:1 |
| Conway B3/S23 | 0.05 | 195 719 | 140 274 | 141 773 | 1,011 | 626 673 | 774 966 | 661 004 | True | 84:1 |
| Conway B3/S23 | 0.2 | 782 682 | 353 373 | 361 653 | 1,023 | 668 490 | 821 204 | 699 688 | True | 33:1 |
| Conway B3/S23 | 0.5 | 1 957 283 | 489 600 | 489 678 | 1,000 | 679 975 | 827 326 | 709 574 | True | 24:1 |
| HighLife B36/S23 | 0.0001 | 396 | 856 | 937 | 1,094 | 157 617 | 182 612 | 135 257 | True | 12 748:1 |
| Seeds B2/S | 0.0001 | 394 | 852 | 935 | 1,097 | 489 964 | 609 785 | 518 997 | True | 12 766:1 |
| slučajno pravilo | 0.0001 | 384 | 837 | 920 | 1,100 | 123 619 | 179 652 | 121 659 | True | 12 974:1 |

Vreme i memorija (srednje / max preko svih p; zasebni procesi, VmHWM):

| metoda | enc s | dec s | enc RSS MB | dec RSS MB |
|---|---|---|---|---|
| rule | 1,03 | 0,38 | 209 | 121 |
| lzma | 6,09 | 0,07 | 210 | 67 |
| zlib | 3,76 | 0,05 | 53 | 56 |
| bz2 | 2,88 | 0,35 | 53 | 59 |
