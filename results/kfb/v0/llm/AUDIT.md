# LLM faza KFB v0: izolacija i revizija

- **Enkoderi**: 5 agenata (Claude), svaki sa 2 mini filma, raspoređena tako da nijedan agent nema dva filma iste porodice (`selection.json`).
- **Okruženje**: sandbox sa `kfb/kfl.py`, `igc/film.py`, `KFL_SPEC.md`, `check.py` i 10 filmova pod anonimnim `iid`. Bez generatora porodica, enkodera, ground truth-a i rezultata. Uputstvo: raditi samo u sandbox-u.
- **Revizija dnevnika** (svi pozivi alata, 206 ukupno): nijedan pristup repozitorijumu, skupu instanci, `truth/`, rezultatima ni seed-u. Jedina čitanja van sandbox-a (po jedno kod 3 agenta) bila su fajlovi u kojima sistem čuva preveliki izlaz **sopstvene** komande agenta. Sva tri su ispis `kfb/kfl.py` iz sandbox-a (34 KB, 0 pominjanja porodica).
- **Prekid**: agenti `llm:3` i `llm:4` su zaustavljeni limitom sesije pre završetka. Zapisi koje su stigli da upišu su verifikovani i uključeni. Za `f056574d802a` (shader rings_xor) zapis ne postoji, a PRNG film je ostao na sirovom zapisu.
- **Verifikacija**: svih 9 zapisa dekodovano je istim nezavisnim dekoderom (`kfb.run --extra-records`, zaseban proces, 64 B smeća, SHA-256 + svi bajtovi): 9/9 tačno, 9/9 samoograničavajuće, MODEL_ID isti.

| iid | porodica / nivo | K̄ | LLM | najbolji automatski (KFL) | najbolji generički |
|---|---|---|---|---|---|
| 5375d1a5b48f | ca / conway_p0 | 111 | **68** | 111 (ca_infer) | 4 108 |
| 930ac01b795f | sprites / k2 | 86 | **77** | 86 (sprites_infer) | 768 |
| 8c40d7277068 | diffusion / smooth | 91 | **87** | 91 (diffusion_infer) | 15 718 |
| 441637081961 | shader / one_term | 43 | **41** | 43 (shader_infer) | 1 337 |
| 8622f7e6f343 | randprog / n9 | 31 | **31** | — (enum_search nije našao) | 343 |
| fd85cff6f0cf | randprog / n5 | 16 | 16 | 16 (enum_search) | 289 |
| cdbfc27e5f16 | randprog / n15 | 34 | 60 | — | 1 822 |
| ac0594a0ca56 | prng / seed16 | 51 | 89 761 (sirovo; agent prekinut) | 51 (prng_infer) | 89 739 |
| a2200e114bec | control / csprng | 89 739 | 89 761 (sirovo, ispravno) | — | 89 739 |

Primeri otkrivene strukture (dekodovani zapisi):
- **CA**: tabela Conway pravila `LUT[18](9S + NB8(S))` zamenjena je formulom `EQ(OR(NB8(S), S), 3)`, koja je *tačno* Conway pravilo. `MOD(SUMS, 61)` je izbačen jer na ovom filmu nema efekta.
- **sprites**: brojač `S = 2T + 24` u stanju, simetrija `TRI(NOT S, ·)` i indeks palete preko `ISIN(C)`.
- **diffusion**: izvor koji se kreće (`X + T − 25`) umesto TRI odbijanja, koje se u 24 frejma nikad ne dogodi, i ekvivalentan kernel `NB8 + NB4 + 4S`.
