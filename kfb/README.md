# KFB: Kolmogorovljev filmski benchmark

KFB meri koliko kratak **tačan** opis filma enkoder može da pronađe kada kratak opis zaista postoji, i koliko daleko od njega ostaju generički kompresori. Benchmark proizlazi iz odeljaka 5 i 10 u `IZVESTAJ.md`: zapis od 1 KB je moguć samo za filmove koji su izlaz kratkog procesa, a otvoreno pitanje je da li se taj proces može **pronaći**.

## Ideja u jednoj rečenici
Svaka instanca je film koji je proizveo kratak program u jeziku KFL. Dužina tog *oracle* programa daje **proverenu gornju granicu K̄** složenosti filma u odnosu na fiksiranu mašinu M (KFL VM). Enkoder dobija samo bajtove filma, a meri se `L / K̄`, uz obaveznu bit-tačnost.

## Sastav
| deo | fajl | uloga |
|---|---|---|
| mašina M | `kfl.py`, `record.py`, `ffcodec.py` (+ `igc/` dekoderi) | jedini dekoder; heš = MODEL_ID |
| jezik | `KFL_SPEC.md` | specifikacija za ljude i LLM enkodere |
| porodice | `families/*.py` | generatori instanci (oracle programi) |
| enkoderi | `encoders/*.py` | generički i family-aware enkoderi |
| skup | `suite.py` | instance iz tajnog master seed-a, anonimni `iid` |
| merenje | `run.py`, `report.py`, `tools/` | zasebni procesi, 64 B smeća, SHA-256, gap prema K̄ |
| LLM sandbox | `tools/make_sandbox.py` | izolovano okruženje za spoljne enkodere |
| protokol | `tools/freeze.py` | zapisnik zamrzavanja pre pravljenja seed-a |

## Protokol (vremensko razdvajanje)
1. **Zamrzavanje.** Mašina M i svi enkoderi se komituju. `python3 -m kfb.tools.freeze` beleži MODEL_ID, heš enkodera i porodica i git commit, a zapisnik se komituje i objavljuje (push).
2. **Tek posle toga** pravi se master seed (`os.urandom(32)`), a iz njega skup instanci. Enkoderi ga ranije nisu mogli videti.
3. Automatski enkoderi rade u zasebnim procesima sa budžetom vremena (mini 20 s, std 60 s).
4. Spoljni (LLM) enkoderi rade u sandbox-u koji sadrži samo VM, specifikaciju i filmove, bez generatora, enkodera i ground truth-a. Njihove zapise proverava isti nezavisni dekoder.
5. Seed i ground truth se objavljuju na kraju, pa svako može da regeneriše skup i proveri sve brojeve.

## Kontrole
- **CSPRNG kontrola** (`families/control.py`): nijedan tačan zapis ne sme biti kraći od sadržaja (Teorema 1 iz izveštaja).
- **MODEL_ID** mora biti isti pri generisanju, kodovanju i dekodovanju.
- **Samoograničavanje**: zapisu se doda 64 B smeća, a dekoder mora pročitati tačno `len(zapis)`.
- **Ground truth** se nalazi u `truth/` skupa i enkoderi ga ne čitaju.

## Pokretanje
```bash
head -c 32 /dev/urandom > seed.bin                       # TEK posle zamrzavanja enkodera
python3 -m kfb.suite --master-seed-file seed.bin --out SUITE --reps 2
python3 -m kfb.run --suite SUITE --out results/kfb --jobs 3
python3 -m unittest discover -s tests -p "test_kfb*.py"
```

## Dodavanje porodice ili enkodera
Napravite modul u `families/` ili `encoders/` po interfejsu opisanom u `families/__init__.py` i `encoders/__init__.py`. Registri ih otkrivaju automatski. Porodica sme da koristi samo dati `rng`. Enkoder ne sme da čita ground truth. `family_aware=True` znači da enkoder sme da pretpostavi konvencije svoje porodice, ali ne i parametre instance.

## Rezultati v0
Pun izveštaj je u `results/kfb/v0/summary.md` (i `IZVESTAJ.md`, odeljak 11). Sirovi redovi su u `rows.json`, ground truth i oracle zapisi u `truth/`, a LLM zapisi i revizija izolacije u `llm/`. Svih 114 nekontrolnih filmova se bit-identično regeneriše iz `master_seed.bin`.
