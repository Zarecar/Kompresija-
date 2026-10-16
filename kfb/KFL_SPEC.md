# KFL v1.1: Kolmogorov Film Language (specifikacija)

Izmene u odnosu na v1 (pre prvog zvaničnog skupa): vrednosti izuzetaka vmode 4 i 5 i vmode 3 kao uint64, ograničenja rada i memorije (MAX_WORK, MAX_DIM, MAX_RATE), ograničeni varint-ovi, javni `evaluate`, preciziranja ispod.

KFL je referentna mašina **M** benchmarka KFB. Program je niz bajtova koji deterministički proizvodi ceo film. **Mera je dužina kompletnog zapisa u bajtovima** (zapis = 1 B metode + metapodaci + program). Interpreter je u `kfb/kfl.py` i on je normativan: ako se ovaj dokument i kod razlikuju, važi kod (i to je bag u dokumentu).

## 1. Film
- Metapodaci (`igc.film.Meta`): `width W, height H, frames T, fps, audio_rate, audio_channels=1, sample_bytes=2, audio_samples NS`.
- Video: `T × H × W × 3` bajtova (RGB, red po red, frejm po frejm).
- Zvuk: `NS` uzoraka, int16 little-endian, mono.
- Payload = video bajtovi, pa zvuk. Kanonski film = `meta.to_bytes() + payload`.

## 2. Zapis
```
[0x10][8 varint-ova: W H T fps audio_rate audio_channels sample_bytes NS][program]
```
Varint je LEB128 (7 bita po bajtu, najniži prvo) i ima **najviše 10 bajtova**; duži se odbacuje. Zapis je samoograničavajući: dekoder ne koristi dužinu fajla.
Ostale metode (`0x00–0x08` igc, `0x20–0x2F` ffmpeg) su osnovne linije i nisu KFL.

## 3. Program
```
flags (1 B):  bit0 STATE   (slede INIT i STEP izrazi)
              bit1 AUDIO   (sledi AUDIO izraz; inače je zvuk nula)
              bit2 EXC_STATE, bit3 EXC_VIDEO, bit4 EXC_AUDIO  (blokovi izuzetaka)
[INIT][STEP]   ako STATE
[VIDEO]        uvek
[AUDIO]        ako AUDIO
[izuzeci stanja][izuzeci videa][izuzeci zvuka]   po zastavicama, tim redom
```

## 4. Izrazi
Izraz je **prefiksni bajt-kod**: opkod, pa (za literale) ugrađeni podaci, pa argumenti redom. Vrednosti su int64 nizovi sa NumPy broadcasting-om; aritmetika se prelama (wrap-around mod 2⁶⁴).

### Listovi (arnost 0)
| bajt | ime | značenje |
|---|---|---|
| 0x80+v | konstanta 0..63 | 1 bajt |
| 0xC0+k | konstanta −1..−16 | `−(k+1)` |
| 0x00 | CONST | sledi zigzag varint (proizvoljan int64) |
| 0x01 | T | indeks frejma |
| 0x02 | Y | red (0..H−1) |
| 0x03 | X | kolona (0..W−1) |
| 0x04 | C | kanal boje 0=R, 1=G, 2=B |
| 0x05 | N | indeks uzorka zvuka |
| 0x06 | S | stanje (H×W int64) u tekućem frejmu |
| 0x07 | SUMS | zbir stanja preko celog frejma kome pripada uzorak |
| 0x08 / 0x09 / 0x0A | WID / HEI / NFR | W / H / T |

Zigzag: `0→0, −1→1, 1→2, −2→3, …`

### Unarni (arnost 1)
| bajt | ime | semantika |
|---|---|---|
| 0x10 | NEG | −a |
| 0x11 | ABS | \|a\| |
| 0x12 | NOT | ~a (bitski) |
| 0x13 | ISIN | `ISIN_LUT[a & 255]`, ≈ round(127·sin(2πa/256)), opseg −127..127. Tabela tačno zadovoljava ISIN(−a) = −ISIN(a), ISIN(a+128) = −ISIN(a) i ISIN(128−a) = ISIN(a). |
| 0x14 | HASH | splitmix64 završnica nad (a + 0x9E3779B97F4A7C15), rezultat ≫ 1 (nenegativan) |
| 0x15 | ISQRT | ⌊√max(a,0)⌋, ulaz se seče na 2⁵². Računa se IEEE-754 sqrt (tačno zaokružen) uz celobrojnu korekciju ±1, pa je rezultat egzaktan. |
| 0x16 | BOOL | a ≠ 0 → 1, inače 0 |
| 0x17 | SQ | a·a |
| 0x18 | NB8 | zbir 8 suseda na torusu (prostorne ose) |
| 0x19 | NB4 | zbir 4 suseda (gore, dole, levo, desno) na torusu |

### Binarni (arnost 2)
| bajt | ime | semantika |
|---|---|---|
| 0x20–0x22 | ADD SUB MUL | a+b, a−b, a·b |
| 0x23 | DIV | ⌊a/b⌋ (Python floor), b=0 → 0 |
| 0x24 | MOD | a mod b (znak kao b, Python), b=0 → 0 |
| 0x25–0x27 | AND OR XOR | bitski |
| 0x28 / 0x29 | SHL / SHR | pomeraj sečen na [0,63]; SHR je aritmetički |
| 0x2A / 0x2B | MIN / MAX | |
| 0x2C–0x2F | LT LE EQ NE | 1 / 0 |
| 0x30 / 0x31 | ROLLY / ROLLX | `np.roll(a, b)` po osi y / x na torusu; b mora biti skalar |
| 0x32 | TRI | „odbijanje": za L=b>0: `L − |a mod 2L − L|`, inače 0 |

### Ternarni (arnost 3)
| bajt | ime | semantika |
|---|---|---|
| 0x38 | WHERE | `WHERE(c, a, b) = (c ≠ 0) ? a : b`; uslov je PRVI argument |
| 0x39 | CLIP | `min(max(a, lo), hi)` |

### Literali i tabele (arnost 1; podaci ugrađeni posle opkoda, pre argumenta)
| bajt | ime | kodiranje | vrednost |
|---|---|---|---|
| 0x3A | LUT | varint L, L zigzag varint-ova (svaki u opsegu int64) | `tab[a mod L]` |
| 0x3B | BITS | varint nbits, ⌈nbits/8⌉ bajtova (MSB prvi) | `bit[a]` za 0 ≤ a < nbits, inače 0 |
| 0x3C | BYTES | varint L, L bajtova | `byte[a mod L]` |
| 0x3D | WORDS | varint L, 2L bajtova (uint16 LE) | `word[a mod L]` |

## 5. Sekcije i promenljive
| sekcija | oblik | dozvoljeni listovi | izlaz |
|---|---|---|---|
| INIT | H×W | Y, X, WID, HEI, NFR, konstante | S₀ |
| STEP | H×W | S, T, Y, X, … | S_{t+1} = STEP(S_t, t), zatim izuzeci stanja |
| VIDEO | H×W×3 po frejmu | T, Y, X, C, S (ako STATE) | `v & 255` |
| AUDIO | NS | N, T = ⌊N·fps/audio_rate⌋ (sečeno na T−1), SUMS (ako STATE) | `v & 0xFFFF` kao int16 |

U VIDEO sekciji `Y` ima oblik (H,1,1), `X` (1,W,1), `C` (1,1,3), a `S` (H,W,1). Prostorni operatori (NB8, NB4, ROLLY, ROLLX) najpre proširuju argument na pun oblik sekcije, a u AUDIO sekciji su zabranjeni. Izraz se proverava pre izvršenja, pa nedozvoljen list znači neispravan program.

## 6. Izuzeci (Seed-and-Correct unutar jezika)
```
varint K, [ako K>0: 1 B Rice parametar k (0..62), Rice-kodovani razmaci, poravnanje na bajt, 1 B vmode, vrednosti]
```
- Pozicije su strogo rastuće. Razmak `g_i = pos_i − pos_{i−1} − 1` (sa `pos_{−1} = −1`) kodira se kao Rice(k): `g≫k` jedinica, jedna nula, pa k nižih bitova.
- vmode: `0` znači da su sve XOR vrednosti 1 (bez podataka); `1` daje uint8 po izuzetku; `2` daje uint16 LE; `3` daje varint kao uint64 mod 2⁶⁴, tumačen kao int64 (pa je izraziv i XOR bita znaka); `4` daje zigzag varint (označeni int64); `5` daje 4-bitne vrednosti 0..15, dve po bajtu (viši nibl prvi). Enkoder iz `kfl.py` bira najkraći primenljiv način.
- Svaki razmak mora biti manji od granice pozicija, inače je zapis neispravan.
- **Stanje**: pozicije nad `(T−1)·H·W`. Pozicija `p` pogađa frejm `1 + p // (H·W)` i ćeliju `p mod (H·W)`, na koju se primenjuje `S ^= v`.
- **Video**: pozicije nad `T·H·W·3` bajtova videa: `bajt ^= v`.
- **Zvuk**: pozicije nad NS uzoraka: `uint16 ^= v`.

Javni evaluator za alate je `kfl.evaluate(node, env, shape)` (ne proverava sekciju).

## 7. Ograničenja mašine
- W, H ≤ 2¹⁶ pojedinačno i W·H ≤ 512²; T ≤ 2048; T·H·W·3 ≤ 2²⁸ bajtova; NS ≤ 2²⁴; fps, audio_rate < 2³¹.
- ≤ 8192 čvorova ukupno; dubina: koren je na dubini 0, najveća dozvoljena je 255 (256 nivoa); LUT ≤ 65 536 elemenata; literal ≤ 2²⁴. Konstante i elementi LUT-a su int64.
- **Rad**: pre izvršenja se računa `work_estimate` = čvorovi(INIT)·HW + čvorovi(STEP)·HW·(T−1) + dinamički čvorovi(VIDEO)·3·HW·T + statički čvorovi(VIDEO)·3·HW + čvorovi(AUDIO)·NS (NB8/NB4 se broje kao 9). Program sa `work_estimate > 2³⁶` je neispravan.
- **Memorija**: stanje se ne čuva za sve frejmove (O(H·W)); keš statičkih podizraza je ograničen na 2²⁸ bajtova (rezultat ne zavisi od keša).
- Granični slučajevi: T = 0 daje prazan video, a SUMS je tada 0. audio_rate = 0 daje T = 0 u AUDIO sekciji. `N·fps` se računa u int64.
- Svaki neispravan zapis daje grešku (ValueError u `kfb.record.decode`); dekoder nikada ne vraća pogrešan film.

## 8. Primeri (Python graditelj iz `kfb.kfl`)
```python
from kfb import kfl, record
from kfb.kfl import T, Y, X, C, N, S, SUMS, op, lut, bits, const

# gradijent koji se pomera, sa tonom od ≈ 440 Hz pri 8 kHz: 8000·225/(16·256) ≈ 439,5 Hz
p = kfl.Program(video=(X * 4 + T * 3 + C * 85), audio=op("ISIN", (N * 225) >> 4) * 200)

# Conway na torusu sa supom
p2 = kfl.Program(init=op("WHERE", (Y < 8) & (X < 8), bits([1,0,1,1]*16, Y * 8 + X), 0),
                 step=lut([(0x1808 >> k) & 1 for k in range(18)], S * 9 + op("NB8", S)),
                 video=lut([0, 0, 0, 255, 255, 255], S * 3 + C))

rec = record.encode_kfl(meta, p)          # kompletan zapis; len(rec) je mera
meta2, payload, used = record.decode(rec)  # nezavisna provera
```

## 9. Bodovanje u KFB
- **Tačnost** je obavezna: SHA-256 i svi bajtovi kanonskog filma, uz dekodovanje u zasebnom procesu sa 64 B smeća iza zapisa.
- **Dužina** L je broj bajtova zapisa. **K̄** je dužina oracle zapisa generatora instance (proverena gornja granica složenosti u odnosu na M).
- Izveštava se `L / K̄` (geometrijska sredina) i broj instanci sa `L ≤ K̄`. Kod CSPRNG kontrole nijedan tačan zapis ne sme biti kraći od sadržaja.
