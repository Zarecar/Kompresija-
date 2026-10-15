# Kompresija: može li se film rekonstruisati iz 1 KB?

Istraživački prototip za koncept **IGC / Seed-and-Correct** (Inverse Generative Compression). Cilj je pronaći najkraći opis koji omogućava **tačnu** rekonstrukciju filma uz unapred poznat model, i eksperimentalno utvrditi gde su stvarne granice.

Pun izveštaj (literatura, teoreme, alternativni pravci, plan, predviđanja): **[IZVESTAJ.md](IZVESTAJ.md)**.

## Glavni izmereni rezultati

Sve je merila tačna rekonstrukcija (SHA-256 + svi bajtovi): 375/375 zapisa u E1 i 144/144 u E3, 3 ponavljanja. Dekoder je zaseban proces koji dobija samo zapis; zapis je samoograničavajući i u budžet ulaze zaglavlje i metapodaci.

| film | original | najbolji tačan zapis | metoda |
|---|---|---|---|
| CSPRNG video + PCM (`os.urandom`) | 3 148 813 B | 3 148 814 B (+1 B) | sirovo; **ništa ne pomaže** |
| SHAKE-256 sa tajnim ključem, PCG64 | 3 148 813 B | 3 148 814 B | sirovo |
| „nasumičan" video iz `np.random.RandomState(seed)` | 3 148 813 B | **21 B** | inverzija MT19937 → seed |
| „nasumičan" video iz `random.Random(seed)` | 3 148 813 B | **22 B** | inverzija MT19937 → ključ |
| `random.Random()` (seed iz OS-a, 19 968 bita) | 3 148 813 B | 2 512 B | stanje MT19937 |
| proceduralni video + sinusni zvuk | 71 306 B | 3 677 B | CM (kontekst + logistički mikser) |
| isti + šum σ=2 („senzor") | 71 306 B | 33 507 B | CM; tačna donja granica (entropija šuma) ≈ 24,1 KB |
| Life-like ćelijski automat 128×128×240 + zvuk, bez šuma | 11 940 491 B | **203 B** | MDL pravilo + izuzeci |
| isti, šum p = 10⁻⁴ (≈ 400 događaja) | 11 940 491 B | **947 B** | MDL pravilo + izuzeci (lzma: 144 355 B) |

Pretraga seed-a + XOR rezidual na nasumičnom ulazu: neto dobitak je **uvek negativan** (−5 do −13 bita, 33/33 pokušaja). Za kompresiju sa gubicima ista ideja radi (Šenonov slučajni kodni rečnik, kanalna simulacija). Detalji su u `results/` i u odeljku 8 izveštaja.

## Pokretanje

Potrebni su Python ≥ 3.11 i NumPy ≥ 2.0 (za `np.bitwise_count`).

```bash
python3 -m unittest discover -s tests -v                       # 14 testova
IGC_TMP=/tmp python3 experiments/e1_random_video.py --reps 3   # nasumični video, ~15 min
python3 experiments/e2_seed_search.py 3                         # pretraga seed-a, ~1 min
IGC_TMP=/tmp python3 experiments/e3_rule_and_exception.py 3     # film kao dinamički sistem
python3 experiments/e4_lossy_seed.py                            # gde seed radi (sa gubicima)
python3 experiments/e5_bounds.py                                # granice i budžeti
```

## Struktura

- `igc/`: model M (njegov SHA-256 je `MODEL_ID` i pre-registruje se pre generisanja podataka)
  - `codec.py`: format zapisa i metode `raw`, `zlib`, `lzma`, `bz2`, `xor_store`, `xor_zlib`, `cm`, `infer`, `rule` i kontrola varanja `leaky`
  - `mt19937.py`: generisanje, untemper, untwist, inverzija `init_genrand` / `init_by_array`
  - `cm.py`, `rangecoder.py`: celobrojni aritmetički koder i adaptivni prediktor
  - `ca.py`: film kao ćelijski automat; indukcija pravila + izuzeci
  - `sources.py`: izvori testnih filmova (CSPRNG naspram ponovljivih PRNG-ova naspram strukture)
- `tools/`: enkoder i nezavisni dekoder kao zasebni procesi
- `experiments/`: E1–E5
- `results/`: sirovi JSON i sažeci u markdown-u

## Ograničenja (iskreno)

- Ovo **nije** kompresor filmova. To je merni instrument za granice tačne kompresije.
- Ekstremni odnosi postoje samo za filmove koji *jesu* izlaz kratkog procesa iz porodice koju model poznaje (slab PRNG, ćelijski automat). Za snimljene filmove Teorema 1 iz izveštaja isključuje tačan zapis od 1 KB.
- CM je čist Python (~20 KB/s) i koristi se samo na malim filmovima.
- Pretrenirani veliki AI model nije korišćen, jer u ovom okruženju nije bio dostupan na ponovljiv način. Za uniformni ulaz granica važi za svaki model.
