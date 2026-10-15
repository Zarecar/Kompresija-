# Može li se ceo film rekonstruisati iz 1 KB?

**IGC / Seed-and-Correct: istraživački izveštaj, teorija i eksperimenti**

Datum: 15. 10. 2026. Svi brojevi u odeljcima 7 i 8 su **merenja** iz ovog repozitorijuma (skripte u `experiments/`, sirovi podaci u `results/`). Sve ostalo je izvođenje, citat ili **procena/predviđanje** i tako je označeno. Literaturu sam proveravao pretragom weba (oktobar 2026). Stavke označene sa ✔ proverio sam direktno, a ostale su standardne reference iz oblasti.

---

## 0. Kratak odgovor

| pitanje | odgovor | osnova |
|---|---|---|
| Tačna rekonstrukcija **proizvoljnog** (ili kamerom snimljenog) filma iz 1 KB? | **Ne.** Ovo je teorema, a ne inženjerski problem. | Teorema 1 (brojanje, min-entropija); šum senzora ≳ 10¹¹ bita za 2 h (E5) |
| Tačna rekonstrukcija **ograničene klase** iz < 1 KB? | **Da, izmereno.** Uslov je da je film zaista nastao kratkim procesom iz porodice koju model poznaje. | E1: 3,1 MB „nasumičnog" videa iz NumPy/Python PRNG-a → **21–22 B**. E3: 11,9 MB filma (ćelijski automat + zvuk) → **203 B** (bez šuma) do **947 B** (p = 10⁻⁴). |
| Pomaže li Seed-and-Correct sa generatorom nezavisnim od podataka u tačnom režimu? | **Ne, dokazivo i izmereno.** XOR sa pseudoslučajnim P je šifra (one-time pad), a ne kompresija. | E2: neto dobitak uvek < 0 (−5 do −13 bita); pristrasan izvor: 3 897 B → 8 194 B |
| Gde ista ideja *radi*? | Kod kompresije sa gubicima i perceptivnog cilja: indeks u zajednički slučajni kodni rečnik ili kanalna simulacija | E4: slučajni rečnik D = 0,299 (Lloyd-Max 0,363, Šenon 0,25); PFR indeks ≈ KL bita |
| Perceptivno „veoma slično" za igrani film na 1 KB? | **Ne** | 1 KB / 2 h = **1,14 bit/s**, odnosno 2,7–8,2 bita po kadru |
| Semantički na 1 KB? | **Grubo da**: ≈ 1 400–2 000 reči „treatment-a". Doslovni dijalog ne staje. | procena (E5) |
| Da li je IGC novo? | Ne u ovom obliku. To je kombinacija dvodelnog MDL koda, analize-sintezom (CELP, 1985), Šenonovog slučajnog kodiranja i relativnog entropijskog kodiranja. | odeljak 2 |
| Šta bi moglo biti originalno? | „Kolmogorovljev filmski benchmark": filmovi sa *poznatom* gornjom granicom K, šumom kao parametrom, kriptografskim negativnim kontrolama i kontrolom curenja kroz model. | odeljak 5; javni ekvivalent za video nisam našao |

### 0.1 Odgovori redom na 10 istraživačkih pitanja

1. **Postojeće metode:** model + aritmetičko kodovanje (PAQ/cmix, NNCP, LLM kao kompresor), MDL dvodelni kodovi i bits-back, neuralni video kodeci (DCVC), INR po filmu (NeRV), generativni i perceptivni kodeci (HiFiC, PerCo, video-difuzioni kodeci 2025–26), analiza-sintezom (CELP), slučajni kodni rečnici i relativno entropijsko kodiranje, sinteza programa (KoLMogorov Test), dinamički modeli videa (dinamičke teksture, DMD), fraktali i gramatike, replay i demoscena (odeljak 2).
2. **Najbliže cilju:** CELP (indeks u slučajni kodni rečnik pobuđuje poznat sintetizator), dvodelni MDL, SeedLM/PRANC (seed + koeficijenti), relativno entropijsko kodiranje (indeks u zajedničkom nizu kao poruka), KoLMogorov Test (najkraći program), replay kodovanje.
3. **Stvarne granice:** Teorema 1 (`P ≤ 2^{B+1−H_∞}`), Teorema 2 (`E[L] ≥ H`), neizračunljivost K, Liu–Pass (pronalaženje kratkih opisa ≈ razbijanje kriptografije), Lloyd (≤ ~2³⁹⁹ pokušaja), Brudno (dinamika ne stvara informaciju besplatno) i entropija šuma senzora (odeljak 3, tabela 2.12).
4. **Novo?** Ne u predloženom obliku: IGC je kombinacija poznatih ideja. Potencijalno originalno: benchmark sa poznatom K za video, kriptografske i „leaky" kontrole kao standard merenja, identifikacija generatora kao faza kompresije i „bitovi identifikacije" kao semantička metrika (odeljak 5).
5. **Najperspektivniji pravac:** za Cilj A identifikacija generatora za sintetički i proceduralni sadržaj (P1–P3); za Cilj B kanalna simulacija nad generativnim priorom (P4); za Cilj C hijerarhijska lestvica sa merenjem bitova identifikacije (P5).
6. **Hipoteza sa najmanje resursa:** H0 iz E2 („pretraga seed-a skraćuje tačan zapis nasumičnog ulaza"). Testira se za ≈ 1 minut na jednom CPU jezgru. Nije potvrđena: 0/33 pozitivnih ishoda.
7. **Najbrži dokaz beskorisnosti:** E2 zajedno sa `xor_zlib` na strukturisanom izvoru iz E1. Za manje od minuta pokazuju da fiksni generator + XOR rezidual ne daje ništa na nasumičnom ulazu i da kvari strukturisani (3,1 MB: lzma 843 KB, a XOR + zlib 3,15 MB).
8. **Tačno ispod 1 KB za ograničenu klasu:** da, izmereno za filmove iz ćelijskog automata sa retkim izuzecima (E3) i za sadržaj iz slabih PRNG-ova (E1). Korisne klase su (predviđanje) replay snimci determinističkih igara i proceduralna animacija, a demoscena to radi „unapred". Snimljeni sadržaj nije u toj klasi.
9. **Perceptivna rekonstrukcija:** granica se pomera sa „nemoguće" na „moguće za kratke klipove uskog domena". Za igrani film je 1 KB ≈ 1,14 bit/s, što nije dovoljno ni za „veoma slično". Današnji generativni kodeci rade na ≈ 10⁻³–10⁻² bpp, što je ~10⁵× više. Semantički 1 KB ≈ 1 400–2 000 reči.
10. **Drugačiji matematički pristupi:** kanalna simulacija (P4), uzastopno profinjavanje sa bitovima identifikacije (P5), pogađanje na strani dekodera i Rényijeva entropija (P7), kriptoanalitička kompresija kao računski, a ne informacioni pogled (P1).

---

## 1. Sažetak istraživačkog problema

Neka je `M` unapred poznat model (dekoder) i `G(M, z)` deterministička rekonstrukcija. Za **tačni** cilj tražimo

```
L_M(X) = min { |z| : G(M, z) = X }        (|z| u bitovima, svi bitovi koje dekoder dobija)
```

Ovo je, doslovno, **Kolmogorovljeva složenost filma X relativno prema mašini M**, `K_M(X)`. Predložena optimizacija `z* = argmin_z [D(X, G(M,z)) + λ L(z)]` je Lagranžijan problema stopa–distorzija (Everett 1963; Shoham & Gersho 1988; Sullivan & Wiegand 1998, RDO u video kodecima). Za `D = ∞·1[X̂ ≠ X]` svodi se na `K_M(X)`.

Tri cilja razdvajam formalno:

- **A, tačno**: `X̂ = X` bit po bit (Hamming-distorzija nula).
- **B, perceptivno**: `d_perc(X, X̂) ≤ ε` i (idealno) `X̂` izgleda kao uzorak iz raspodele prirodnih filmova (realizam).
- **C, semantičko**: `S(X̂) = S(X)` za ekstraktor `S` (likovi, događaji, redosled). Operativno merilo predlažem u 4.5 (bitovi identifikacije).

Preformulacija koja vodi kroz ceo izveštaj: zapis od ≤ 1 KB postoji za film X **ako i samo ako X pripada skupu `S_M = {G(M,z) : |z| ≤ 8192}`**, koji ima najviše ≈ 2⁸¹⁹³ elemenata i koji je **fiksiran pre nego što je film viđen**. Zato pitanje nije „kako kompresovati bolje", već „koji skup filmova model sme da pokriva i da li konkretan film pripada tom skupu".

---

## 2. Relevantna literatura i postojeći pristupi

### 2.1 Teorijski temelji
- Shannon (1948), teorema o kodovanju izvora: `E[L] ≥ H(X)`. Shannon (1959), funkcija stopa–distorzija `R(D)`, dokaz preko **slučajnog kodnog rečnika**.
- Solomonoff (1964), Kolmogorov (1965), Chaitin (1966): algoritamska složenost. Li & Vitányi, *An Introduction to Kolmogorov Complexity*.
- Rissanen (1978), MDL; Grünwald (2007), *The MDL Principle*. Dvodelni kodovi (model + podaci uz model) su formalni oblik „seed + korekcija".
- Levin (1973), univerzalna pretraga i `Kt` složenost.
- **Brudno (1983)**: za ergodičnu meru je složenost trajektorija skoro svih tačaka po jedinici vremena jednaka Kolmogorov–Sinajevoj entropiji. Pesin: `h = Σ λᵢ⁺`. Ovo je ključ za pravac „film kao dinamički sistem".
- Alon & Orlitsky (1994): donja granica za kodove koji nisu prefiksni.
- Massey (1994), Arikan (1996): *guesswork*. Očekivani broj pogađanja je određen Rényijevom entropijom `H_{1/2}`.
- ✔ **Liu & Pass, FOCS 2020**, „On One-way Functions and Kolmogorov Complexity": jednosmerne funkcije postoje **ako i samo ako** je vremenski ograničena složenost `K^t` blago teška u prosečnom slučaju. Pronalaženje kratkog opisa je, u opštem slučaju, ekvivalentno razbijanju kriptografije. ([ePrint 2020/423](https://eprint.iacr.org/2020/423), [arXiv 2009.11514](https://arxiv.org/abs/2009.11514))
- ✔ **Lloyd (2002)**, „Computational capacity of the universe", PRL 88: ≈ 10¹²⁰ operacija nad ≈ 10⁹⁰ bita (10¹²⁰ uz gravitaciju) od Velikog praska. To je fizička granica svake pretrage. ([arXiv quant-ph/0110141](https://arxiv.org/abs/quant-ph/0110141))

### 2.2 Model + entropijsko kodovanje (tačna kompresija sa naučenim modelima)
- Aritmetičko kodovanje (Rissanen & Langdon 1979; Witten, Neal & Cleary 1987), ANS (Duda 2009/2013). Kodna dužina je `−log₂ P_M(X)`, što je optimalno za dati model.
- PAQ/cmix (Mahoney; Knoll), NNCP (Bellard). ✔ **Hutter Prize**: rekord za enwik9 (1 GB) je 110 793 128 B (fx2-cmix, Orav & Knoll, 3. 9. 2024), **uključujući dekompresor**. ([hutter1.net](https://hutter1.net/prize/index.htm), [LTCB](https://mattmahoney.net/dc/text.html))
- ✔ **Delétang i sar., ICLR 2024**, „Language Modeling Is Compression": Chinchilla 70B kompresuje ImageNet isečke na 43,4 % (PNG 58,5 %) i LibriSpeech na 16,4 % (FLAC 30,3 %). **Kada se uračuna veličina modela, odnos postaje katastrofalan.** To je tačno ona zamka „model izuzet iz budžeta". ([ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html), [arXiv 2309.10668](https://arxiv.org/abs/2309.10668v1))
- Bits-back: Frey & Hinton (1997), Townsend i sar. (BB-ANS, 2019). Vraćaju „cenu seed-a" u dvodelnom kodu. Integer Discrete Flows (Hoogeboom i sar. 2019) daju tačnu kompresiju kroz bijektivan latentni prostor.

### 2.3 Neuralni video kodeci (sa gubicima, metrika fidelnosti)
- DVC (Lu i sar. 2019), porodica DCVC. ✔ **DCVC-RT (CVPR 2025)**: ≈ 21 % manje bita od H.266/VTM uz 1080p u realnom vremenu (autori navode ≈ 125 fps enkodovanja i ≈ 113 fps dekodovanja). ([arXiv 2502.20762](https://arxiv.org/html/2502.20762v2))

### 2.4 Implicitne neuralne reprezentacije (model prilagođen filmu)
- COIN (Dupont i sar. 2021), NeRV (Chen i sar. 2021), HNeRV (2023), Deep Image Prior (Ulyanov i sar. 2018). **Težine su podaci specifični za film**: ulaze u zapis u celosti (kvantizovane i entropijski kodovane).

### 2.5 Generativna i perceptivna kompresija (cilj B/C)
- Blau & Michaeli (2018) daju kompromis percepcija–distorzija, a (2019) stopa–distorzija–percepcija. Theis & Agustsson (2021) pišu o prednostima stohastičkih enkodera, Wagner (2022) o ulozi zajedničke slučajnosti.
- HiFiC (Mentzer i sar. 2020), PerCo (Careil i sar., ICLR 2024), Text+Sketch (Lei i sar. 2023). Kod videokonferencija, kodeci zasnovani na ključnim tačkama lica (Wang, Mallya & Liu 2021).
- ✔ Video-difuzioni kodeci 2025–26: GNVC-VD (< 0,01 bpp, [arXiv 2512.05016](https://arxiv.org/html/2512.05016v2)), GVC (≈ 0,005 bpp, [arXiv 2512.24300](https://arxiv.org/pdf/2512.24300v2)), T-GVC ([2507.07633](https://arxiv.org/pdf/2507.07633v2)), Free-GVC ([2602.09868](https://arxiv.org/pdf/2602.09868)), aktivno uzorkovanje ([2605.02849](https://arxiv.org/pdf/2605.02849)). Tvrdnje autora nisam reprodukovao. **Čak i pri 0,005 bpp, 2 h 1080p filma je ≈ 224 MB**, što je 2·10⁵ puta više od 1 KB.

### 2.6 Analiza-sintezom i slučajni kodni rečnici (najbliži srodnici IGC-a)
- **Šenonov slučajni kodni rečnik**: 2^{nR} reči iz zajedničke raspodele, šalje se indeks najbliže. To je asimptotski optimalno **za kodiranje sa gubicima**.
- **CELP (Schroeder & Atal 1985)**: stohastički (slučajni Gausov) kodni rečnik pobuđuje poznat LPC sintezni filtar. Enkoder traži indeks koji minimizuje perceptivno težinsku grešku, a šalje indeks, pojačanje i parametre filtra. Ovo je **IGC iz 1985. godine**, za govor, sa gubicima.
- Vektorska kvantizacija (Gray 1984). Matching pursuit (Mallat & Zhang 1993) i *MP video kodovanje na veoma niskim brzinama* (Neff & Zakhor 1997): pohlepni izbor „korekcija" po smanjenju greške po bitu, isto kao predloženi `Sᵢ`.
- ✔ **SeedLM (ICLR 2025)**: blokovi težina LLM-a kao seed LFSR generatora + nekoliko koeficijenata. Ovo je Seed-and-Correct za težine, **sa gubicima**. ([arXiv 2410.10714](https://arxiv.org/pdf/2410.10714)) PRANC (2023) i NOLA (2024) su slični pristupi. MIRACLE (Havasi i sar. 2019).
- Relativno entropijsko kodiranje / kanalna simulacija. ✔ **Li & El Gamal (2018)**: uzorak iz `P_{Y|X}` se može preneti uz zajedničku slučajnost za `I(X;Y) + log(I(X;Y)+1) + 4` bita ([arXiv 1701.02827](https://arxiv.org/pdf/1701.02827)). Pooštrenje je dao Li (2025, [arXiv 2501.13736](https://arxiv.org/html/2501.13736v3)). ✔ Flamich (NeurIPS 2023): GPRS i GRC, brzi algoritmi ([GPRS](https://mlanthology.org/neurips/2023/flamich2023neurips-greedy), [GRC](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html)). Theis & Yosri (2022). **Ovo je matematički ispravan oblik ideje „seed kao poruka".**

### 2.7 Program kao opis
- Demoscena: kategorije intro programa od 256 B, 1 KB, 4 KB i 64 KB, sa minutima audio-vizuelnog sadržaja koji se reprodukuju deterministički na istoj platformi. Primeri su 4k „Elevated" (2009) i 96 KB igra *.kkrieger*. **„Film od 1 KB" već postoji, ali u suprotnom smeru**: prvo nastaje program, pa iz njega film.
- ✔ **The KoLMogorov Test (Yoran i sar., ICLR 2025)**: LLM treba da napiše najkraći program koji reprodukuje niz. Vodeći modeli (GPT-4o, Llama-3.1-405B) prolaze loše na prirodnim i sintetičkim nizovima. Postoje samo audio, tekst, DNK i celobrojni nizovi, **bez videa**. ([arXiv 2503.13992](https://arxiv.org/pdf/2503.13992))
- ✔ Noviji preprint o pronalaženju programa za elementarne CA ([arXiv 2606.13799](https://arxiv.org/pdf/2606.13799)) i IEICE 2009 rad o kompresiji zvuka bez gubitaka pomoću CA pravila ([IEICE](https://www.ieice.org/ken/paper/20090228Oajv/eng/)) su srodni našem E3, ali nisu isti. Status recenzije preprinta nisam proverio.

### 2.8 Dinamički sistemi i matematičke strukture
- *Dynamic textures* (Doretto, Chiuso, Wu & Soatto 2003): video kao linearni dinamički sistem `x_{t+1} = A x_t + v_t`, `y_t = C x_t + w_t`. *Video textures* (Schödl i sar. 2000). DMD (Schmid 2010; za video Grosek & Kutz 2014), Koopmanov operator. Neural CA (Mordvintsev i sar. 2020).
- Fraktalna kompresija (Barnsley; Jacquin 1992; Fisher 1995): teorema kolaža je inverzni problem. U praksi je slična JPEG-u, uz veoma sporo kodovanje. Reklamirani odnosi „10 000:1" odnosili su se na pojedinačne slike, sa gubicima.
- Gramatička kompresija (Sequitur 1997, Re-Pair 2000): globalna pravila umesto lokalnih ponavljanja.

### 2.9 Deterministički simulatori i „replay"
- Doom demo (.lmp) fajlovi i TAS filmovi (TASVideos) daju **tačan** video iz zapisa ulaza, uz isti engine i ROM. Ako se verzija engine-a razlikuje, nastaje *desync*. Ovo je stvarna klasa „filmova" čiji je opis mali.

### 2.10 Kriptoanaliza PRNG-a
- MT19937: stanje iz 624 izlaza (poznato). ✔ PCG (podrazumevan u NumPy): rekonstrukcija seed-a iz 512 B izlaza, do ≈ 20 000 CPU-sati (Bouillaguet, Martinez & Sauvage, ToSC 2020). ([ToSC](https://tosc.iacr.org/index.php/ToSC/article/view/8700))

### 2.11 Istorija tvrdnji
- ✔ **Jan Sloot** (Holandija, 1995–1999) je tvrdio da „Sloot Digital Coding System" smešta igrani film u 8 KB (u nekim izvorima 1 KB). Tvrdnja **nikada nije nezavisno proverena**, a izvorni kod nije pronađen. Prema Boing Boing-u (2026), inženjeri su utvrdili da je demonstracioni uređaj sadržao hard disk. Roel Pieper je princip poredio sa PostScript-om („pošiljalac i primalac znaju recepte"). Pieter Spronck je odgovorio da je Sloot tvrdio da kodira *bilo koji* video. ([Wikipedia](https://en.Wikipedia.com/wiki/Jan_Sloot), [Boing Boing](https://boingboing.net/2026/07/13/sloot-digital-coding.html), [CoRecursive](https://corecursive.com/sloot-digital-coding-system/), [EDN](https://www.edn.com/?p=4006347)) Disk na strani dekodera je upravo `I(X;M) > 0`, scenario koji hvata naša kontrola „leaky" (odeljak 8).
- ✔ Goldman izazov (2001): Patrick Craig je „kompresovao" nasumičan fajl tako što ga je podelio na 218–219 fajlova. Informacija je sakrivena u granicama fajlova (metapodaci fajl-sistema). ([Boing Boing 2025](https://boingboing.net/2025/01/14/coding-challenge-goes-wrong-creator-refuses-to-pay-5000-prize-money.html)) Zato su svi naši zapisi **samoograničavajući**.
- Iz comp.compression FAQ-a: WEB Technologies „DataFiles/16" (1992) i ZeoSync (2002) tvrdili su rekurzivnu kompresiju nasumičnih podataka. Obe tvrdnje su opovrgnute argumentom brojanja.

### 2.12 Najbliže našem cilju i stvarne granice
| pristup | šta deli sa IGC | stvarna granica |
|---|---|---|
| model + aritmetičko kodovanje | „generator" je prediktor, „korekcija" je `−log P_M(X)` | `E[L] ≥ H(X)`; dobitak samo za strukturisane izvore |
| dvodelni MDL / bits-back | seed + korekcija | ne može ispod mešavinskog koda `−log Σ_s 2^{−L(s)} P(X|s)` |
| CELP / slučajni rečnik / SeedLM | pretraga indeksa u zajedničkom generatoru | samo sa gubicima; pretraga raste kao 2^R |
| relativno entropijsko kodiranje | indeks u zajedničkom nizu kao poruka | `≈ I(X;Y)` bita; vreme ≈ 2^{KL} |
| INR / fine-tuning | model prilagođen filmu | težine ulaze u budžet |
| generativni kodeci | „model zna svet" | ≥ 10⁻³ bpp za prepoznatljiv sadržaj, i dalje stotine MB za igrani film |
| program synthesis (KT) | najkraći program | neizračunljivo u opštem slučaju; LLM-ovi su danas slabi |
| replay / demoscena | film = trajektorija programa | samo za sintetički (ili proceduralni) sadržaj |

---

## 3. Matematičke granice

### Teorema 1 (brojanje, uopšteno na min-entropiju)
Neka je `D` proizvoljan deterministički dekoder (model `M` je deo `D`), neka je `X` slučajan film raspodele `P`, **nezavisan od `D`**, i neka enkoder (proizvoljno moćan, i randomizovan uz zajedničku slučajnost `R ⊥ X`) šalje kod od najviše `B` bita. Tada

```
P(D(Z) = X) ≤ (2^{B+1} − 1) · max_x P(x) = (2^{B+1} − 1) · 2^{−H_∞(X)}.
```

*Dokaz.* Uslovno na `R = r`, skup mogućih izlaza `S_r = D_r({0,1}^{≤B})` ima najviše `2^{B+1} − 1` elemenata, pa je `P(X ∈ S_r) ≤ |S_r| max_x P(x)`. Usrednjavanje po `r` daje tvrdnju. ∎

- Za uniformni N-bitni ulaz je `H_∞ = N`, pa dobijamo vašu granicu `min(1, 2^{B+1−N})`. **Ona je tačna** i važi nezavisno od računarske snage, veštačke inteligencije, seed-ova i zajedničke slučajnosti, *dokle god je sve što dekoder zna osim `z` nezavisno od X*.
- Za prefiksno-slobodne kodove (samoograničavajuće zapise, kao kod nas) imamo `2^B`, a ne `2^{B+1}−1`. Razlika je 1 bit, pa to nije rupa. Bajt-poravnatih zapisa dužine ≤ 1024 B ima `Σ 256^k ≈ 2^{8192,006}`.
- **Pretpostavke koje se krše u praksi** (i to su načini varanja): (a) dekoder dobija skrivene kanale: dužinu fajla, broj i imena fajlova, vreme (Goldman/Craig); (b) model je izabran **posle** viđenja filma (Sloot); (c) „uspeh" se meri perceptivno umesto bit po bit.
- **Implikacija za realne filmove**: min-entropiju određuje nesvodljiv šum (senzor, zrno filma, kvantizacija enkodera). Za šum σ = 2 to je ≈ 3,06 bita po uzorku, odnosno ≥ 137 GB samo za lumu 2 h 1080p (E5, procena).

### Posledica 1 (curenje kroz model)
Ako je `X = G(M, Z)`, onda važi `H(X|M) ≤ H(Z)`, pa

```
E|Z| ≳ H(X) − I(X; M)          (do logaritamskih članova za kodove koji nisu prefiksni)
```

Izuzimanje modela iz budžeta je pošteno **samo ako je `I(X;M) ≈ 0`**, to jest ako je model fiksiran pre nastanka filma. Operativno: heš modela se objavljuje pre generisanja ili izbora test filmova (*vremensko razdvajanje*). U algoritamskoj verziji, 1 KB zapis znači da model „zna" skoro ceo film: `I_K(M : X) ≥ K(X) − 8192 − O(log)`.

### Teorema 2 (očekivana dužina)
Za prefiksne kodove važi `E[L] ≥ H(X)`, a za jedan-na-jedan kodove `E[L] ≥ H(X) − log₂(H(X)+1) − log₂ e` (Alon–Orlitsky). Za uniformni ulaz nijedan izbor modela, seed-a ili pretrage ne daje prosečnu uštedu. E2 to meri direktno.

### Kolmogorov, neizračunljivost i „sertifikat nekompresibilnosti"
Manje od `2^{−c}` svih n-bitnih nizova ima `K < n − c`. `K` nije izračunljivo, pa se nekompresibilnost reziduala **nikada ne može dokazati**, već samo neuspeh *naše porodice modela*. E1 to demonstrira: izlaz MT19937 prolazi statističke testove i odolijeva zlib, lzma, bz2 i CM, ali ima `K` ≈ 32 bita. Teorema invarijantnosti (`K_U ≤ K_V + c`) ne govori ništa o konkretnom filmu kada je `c` veličine modela, odnosno gigabajti.

### Računske granice
- CSPRNG `G: {0,1}^s → {0,1}^n` ima `K(G(seed)) ≤ s + O(1)`. Ipak, nijedan kompresor polinomijalnog vremena ne može primetno kompresovati njegov izlaz, jer bi time postao razlikovač (Yao). Prema Liu–Pass rezultatu, efikasno nalaženje kratkih opisa u prosečnom slučaju je ekvivalentno nepostojanju jednosmernih funkcija.
- Prema Lloyd-ovoj granici, svaka pretraga grubom silom ispituje najviše ≈ 2³⁹⁹ kandidata. **Nijedan proces u našem svemiru ne može „pogoditi" seed duži od ~400 bita**, a 1 KB ima 8 192 bita.
- Prema Arikanovoj granici, dekoder koji pogađa (enumeriše kandidate po verovatnoći modela) pravi očekivano ≈ `2^{H_{1/2}(X|M)}` pokušaja. To je izvodljivo samo ako je `H_{1/2}(X|M) ≲ 60–80` bita u praksi (≤ 399 fizički).

### Dinamički sistemi
Po Brudnu, opis „početno stanje + pravilo evolucije" haotičnog sistema košta ≈ `K(pravilo) + h·T` bita, gde je `h` KS-entropija. **Dinamički sistem ne stvara besplatnu kompresiju**: ušteda postoji samo kada je stvarna stopa proizvodnje informacije `h` mala na rezoluciji posmatranja. U E3 je to tačno vidljivo: `H(film) = s² + N·h(p)`.

### Sa gubicima, perceptivno
`R(D) = min I(X;Y)`. Blau & Michaeli: savršen realizam za MSE košta najviše dvostruku distorziju pri istoj stopi. Uz zajedničku slučajnost se tačan uzorak iz `P_{Y|X}` prenosi za ≈ `I(X;Y)` bita (Li–El Gamal). **Za realizam ne treba prenositi detalje, ali ono što se ne prenese nije original.**

### Budžet 1 KB za igrani film (E5, proračun)
- 1 KB / 2 h = **1,14 bit/s** = 0,047 bita po frejmu = **2,7–8,2 bita po kadru** (uz pretpostavku od 1 000–3 000 kadrova).
- Doslovni dijalog (10–15 hiljada reči, uz jak LM entropijski koder od 0,7–1,0 bit/znak) zauzima ≈ **5–11 KB**, dakle ne staje u 1 KB.
- U 1 KB staje ≈ 8 200–11 700 znakova ≈ **1 400–2 000 reči** proze koju model smatra verovatnom.

### Razlike između pojmova (traženo u zadatku)
| pojam | šta zapis nosi | kada je legitimno |
|---|---|---|
| **kompresija** | `≈ H(X|M)` bita, uz `I(X;M) ≈ 0` | uvek, ako je model zamrznut pre filma |
| **prenos informacije** | poruku kroz kanal; ne može preneti više od svojih bita | uvek |
| **generisanje verovatnog sadržaja** | 0 bita o konkretnom X | to nije rekonstrukcija; sličnost je slučajna (`≤ 2^{−H_∞}`) |
| **rekonstrukcija spoljnim znanjem** | indeks u arhivu koja *sadrži* X (≈ 20 bita za ~10⁶ igranih filmova; magnet link 160 bita) | kao *adresiranje*, a ne kompresija |
| **skrivanje u modelu** | malo, jer je informacija prebačena u M posle viđenja X | **nikada**; detektuje se promenom heša modela |
| **prilagođavanje modela** | razlika težina ΔM je deo z | legitimno samo ako je ΔM u budžetu |

---

## 4. Alternativni pristupi

### 4.0 Šta tačno vredi Seed-and-Correct (odgovori na pitanja iz zadatka)
- **Da li korekcije mogu biti efikasnije od originala?** Da, ako i samo ako `−log₂ P_M(X|s) + L(s) < −log₂ P_0(X)`, odnosno ako model dodeljuje originalu veću verovatnoću od osnovnog modela za više bita nego što košta seed. Za uniformni X je očekivani dobitak ≤ 0 (Gibbsova nejednakost). E2 meri gubitak od ≈ log₂ N bita.
- **XOR rezidual sa fiksnim generatorom** (`R = X ⊕ P`, P nezavisan od X) je **one-time pad**: R je uniforman za *svaki* X. Na nasumičnom ulazu nema dobitka, a na strukturisanom ulazu struktura nestaje (E1 `xor_zlib` i E2 pristrasni izvor). XOR ima smisla samo ako je P *predikcija* (P ≈ X). Čak i tada je bolje kodirati X aritmetički uz prediktivnu raspodelu nego XOR-ovati tačkastu predikciju.
- **Kada rezidual postaje nekompresibilan?** Kada je algoritamski slučajan u odnosu na sve što dekoder zna. To se ne može sertifikovati; operativno merilo je da „nijedna porodica u modelu ne dodeljuje veću verovatnoću od uniformne". U E1 je CM na CSPRNG-u **veći** od originala (+1,3 %), što je cena učenja ničega.
- **Pikseli, blokovi, frekvencije, događaji ili latenti?** Bijekcija ne menja entropiju (`H(T(X)) = H(X)`). Transformacija pomaže samo tako što olakšava posao *modelu* (dekorelacija, retkost). Za tačnost su potrebne celobrojno reverzibilne transformacije (lifting talasići, RCT, celobrojni DCT, IDF tokovi), a latentne korekcije samo uz bijektivan dekoder ili bits-back. **Događaji** su najbolji kad su promene retke u vremenu (E3: izuzeci su događaji).
- **Pohlepni kriterijum `Sᵢ = ΔD/bᵢ`**: za tačan cilj je irelevantan. Sve korekcije moraju biti poslate, a ukupna dužina je `−log P_M(X|z)` nezavisno od redosleda. Za cilj sa gubicima to je pravilo jednakih nagiba (Lagranžijan). Optimalno je za nezavisne stavke sa konveksnom R-D krivom, a uz interakcije se ponovna procena svodi na OMP / matching pursuit (Neff & Zakhor 1997). Ima smisla za progresivno (ugrađeno) kodovanje.
- **Globalna pravila umesto lokalnih korekcija**: da, MDL izborom pravila iz porodice (E3). Jedno pravilo od 18 bita zamenjuje ≈ 3,9 miliona lokalnih „korekcija".

Ispod je sedam pravaca. Suštinski su različiti po tome *odakle* dolazi ušteda: računarska (P1), dinamička (P2, P3), stohastička (P4), hijerarhijska (P5), referencijalna (P6) i ušteda na strani dekodera (P7).

### P1. Inverzija generatora (kriptoanalitička kompresija), implementirano
- **Definicija.** Porodica `𝒢 = {G_θ}` je deo modela. `L(X) = ⌈log₂ |porodica|⌉ + |θ|` ako postoji θ takvo da je `G_θ = X`. Inače se koristi rezervna metoda.
- **Ključna pretpostavka.** X je zaista izlaz generatora iz `𝒢`, a θ se može naći u dostupnom vremenu.
- **Algoritam.** Untempering i linearna rekurencija MT19937 daju stanje iz 624 izlaza, pa provera celog toka. Zatim sledi *untwist* i inverzija `init_genrand` / `init_by_array`, što vraća seed. Za slabe seed-ove radi i gruba sila (`2^20`).
- **Prototip.** `igc/mt19937.py`, `igc/codec.py` (metoda `infer`).
- **Kontrole.** CSPRNG, SHAKE sa tajnim ključem i PCG64 moraju propasti, odnosno ostati ≥ N.
- **Kriterijum uspeha.** Tačno, ≤ 32 B za seed od 32 bita, ≈ 2,5 KB za MT seed-ovan iz OS-a, nezavisno od dužine filma.
- **Prepreke.** Pravi filmovi nisu izlazi PRNG-a. Za CSPRNG je ovo ekvivalentno razbijanju kriptografije.
- **Opovrgnuto ako** se CSPRNG izlaz kompresuje (tada je to bag ili, malo verovatno, razbijena kriptografija).
- **Zašto je bitno:** **„nasumičan video" iz `np.random` ili `random` nije težak test**, već jedan od najlakših. Test „potpuno nasumičnog" sadržaja mora koristiti CSPRNG ili `os.urandom`.

### P2. MDL indukcija pravila + izuzeci (film kao dinamički sistem), implementirano
- **Definicija.** `L = L(Φ) + L(F₀) + Σₜ L(Eₜ)`, gde je `F_{t+1} = Φ(Fₜ) ⊕ Eₜ` i Φ iz porodice lokalnih pravila. Generalizacije su celobrojni linearni dinamički sistemi (tačne dinamičke teksture), čestični i fizički sistemi.
- **Pretpostavka.** Postoji pravilo iz porodice koje objašnjava skoro sve prelaze; izuzeci su retki.
- **Algoritam.** Većinsko glasanje nad 18 parova (stanje, broj suseda), zatim izuzeci kao Rice-kodovani razmaci, `F₀` kontekstnim aritmetičkim kodom, i provera hipoteze „zvuk = sinteza(video)".
- **Prototip.** `igc/ca.py`, eksperiment `e3`.
- **Kontrola.** Idealna dužina procesa `s² + Σ(−log₂ P(Eₜ))`. Zapis ne sme pasti ispod nje za više od O(log N) bita (koliko sme uštedeti empirijska procena p).
- **Kriterijum.** `rule ≤ ideal + 1 KB + 5 %` za sve p, i bolji rezultat od svakog generičkog kompresora za p ≤ 10⁻³.
- **Prepreke.** Za snimljeni sadržaj porodica pravila ne objašnjava šum. Prostor porodica raste kombinatorno (program synthesis).
- **Opovrgnuto ako** ne važi kriterijum, ili ako je zapis primetno ispod ideala (bag ili curenje).

### P3. Film kao trajektorija simulatora pod upravljanjem („replay" kodovanje), plan
- **Definicija.** `X = Render(Sim(s₀, u₁…u_T))`, a zapis je `(id scene, s₀, entropijski kodovani uₜ)`.
- **Pretpostavka.** Engine je deterministički i bit-identičan na enkoderu i dekoderu (verzija i heš su deo modela).
- **Algoritam.** Ulazi se kodiraju aritmetičkim koderom uz prediktor ulaza. Izuzeci iz E3 su upravo ovo, sa „ulazima" na mestu šuma.
- **Kontrola.** Desync test na drugoj mašini i drugoj verziji engine-a.
- **Kriterijum.** Zapis ≈ entropija ulaza ≪ FFV1 / x264 lossless.
- **Opovrgnuto ako** tačnost ne preživi drugu mašinu ili ako je kodovani ulaz veći od lossless videa.
- **Predviđanje:** vidi PRED-3.

### P4. Kanalna simulacija / zajednička slučajnost (cilj B), demonstrirano u E4
- **Definicija.** Enkoder bira indeks K u zajedničkom nizu `U₁, U₂, … ~ p` tako da je `U_K ~ q(·|X)` (posterior generativnog modela).
- **Pretpostavka.** Postoji dobar generativni prior `p` i posterior `q` sa malim `KL(q‖p)` po bloku.
- **Algoritam.** PFR / GPRS / GRC po blokovima od 10–20 bita KL-a.
- **Kontrola.** Raspodela izabranih uzoraka mora biti jednaka `q`, a cena mora biti `≤ KL + log₂(KL+1) + 4`.
- **Kriterijum.** Ispunjene obe kontrole (E4: ispunjene).
- **Prepreke.** Vreme raste kao ≈ 2^{KL} po bloku, pa je potrebna faktorizacija na mnogo malih blokova. Dobija se realističan uzorak, a ne original.
- **Opovrgnuto ako** je izmerena cena iznad granice ili ako uzorci nisu iz `q`.

### P5. Hijerarhijska semantička lestvica sa uzastopnim profinjavanjem, plan
- **Definicija.** Lestvica nivoa: logline → sinopsis → lista kadrova → ključni frejmovi → latenti → pikseli. Bitovi se raspodeljuju po nivoima u okviru teorije *successive refinement* (Equitz & Cover 1991).
- **Operativno semantičko merilo (predlog):** **bitovi identifikacije**. Ako sudija (čovek ili model) iz rekonstrukcije prepoznaje original među `2^k` kandidata sa tačnošću `a`, rekonstrukcija nosi `≳ k − h(a) − (1−a)·k` bita o identitetu filma (Fano). Time „semantičku sličnost" pretvaramo u bitove uporedive sa budžetom.
- **Kontrola.** Rekonstrukcija bez zapisa (bezuslovno generisanje) mora dati tačnost ≈ 2^{−k}.
- **Opovrgnuto ako** zapis od 1 KB ne nosi više bitova identifikacije od, na primer, naslova filma (≈ 100 bita).

### P6. Uslovno adresiranje (K(X|arhiva)), kontrolni pravac
Indeks u javnu arhivu (ISAN, heš sadržaja, magnet link) od 20–160 bita „rekonstruiše" svaki film iz arhive. Ovo je validno kao *adresiranje* i poštena interpretacija Pieperove „PostScript" analogije, ali **ne kao kompresija**. Test: vremensko razdvajanje (film nastao posle zamrzavanja modela ili arhive) daje neuspeh.

### P7. Pogađanje na strani dekodera sa heš-verifikacijom, analitički
- **Definicija.** Zapis je `(opis klase, k-bitni heš)`. Dekoder enumeriše kandidate po opadajućoj `P_M` i vraća prvog sa odgovarajućim hešom.
- **Dužina.** ≈ `log₂ rang(X) + rezerva`, što je asimptotski isto kao `−log₂ P_M(X)` i ne nadmašuje entropijsko kodiranje.
- **Cena.** `E[G] ≈ 2^{H_{1/2}}` (Arikan).
- **Šta pokazuje.** Ovo je *pošten oblik* ideje „seed pretrage" i tačno pokazuje šta bi tvrdnje iz 1990-ih zahtevale: `H_{1/2}(film | model) ≲ 400` bita.

---

## 5. Najperspektivnija hipoteza

**H\*.** Tačan zapis ≤ 1 KB postoji za X *ako i samo ako* X leži u skupu `S_M` filmova koje model generiše iz ≤ 8 192 bita. Za klasu filmova koji su **trajektorije determinističkih procesa sa retkim nedeterminističkim događajima** optimalna dužina je

```
L*(X) ≈ L(Φ | M) + H(s₀) + Σₜ H(Eₜ),
```

i MDL pretraga po porodicama pravila je dostiže do konstante od ~100 B. Za snimljene filmove u `Σₜ H(Eₜ)` dominira šum senzora (≥ 10¹¹ bita za 2 h), pa je Cilj A nedostižan *za svaki* M koji je nezavisan od filma.

Najperspektivniji *istraživački* pravac stoga nije „bolji generator za proizvoljne filmove", već **identifikacija generatora** (P1–P3): kako, za film koji *ima* kratak opis, taj opis pronaći, i koliko se kratki opisi mogu proširiti od CA ka igrama, proceduralnoj animaciji i renderovanim scenama.

**Šta bi bio originalan doprinos (predlog):** *Kolmogorovljev filmski benchmark (KFB)*.
1. Porodice filmova sa **poznatom gornjom granicom** `K̄` (dužina generatorskog programa + seed): CA, celobrojni LDS (dinamičke teksture), sprajtovi i fizika, proceduralni šejderi u malom DSL-u, PRNG šum (slab i jak).
2. Šum `p` i neusklađenost modela kao kontrolisani parametri.
3. Obavezne **kriptografske negativne kontrole** (CSPRNG mora ostati ≥ N) i **kontrola curenja** (pre-registrovani heš modela, vremensko razdvajanje).
4. Metrika `gap = L_found − K̄` (i tačnost i računsko vreme).

KoLMogorov Test (ICLR 2025) pokriva 1-D nizove (tekst, audio, DNK, celobrojni nizovi), bez videa, bez šuma kao parametra i bez kriptografskih kontrola. Ciljanom pretragom (oktobar 2026) nisam našao javni ekvivalent za video, ali **to treba temeljno proveriti pre bilo kakve tvrdnje o novini**.

---

## 6. Konkretan eksperimentalni plan

| faza | sadržaj | resursi | kriterijum uspeha | kriterijum neuspeha |
|---|---|---|---|---|
| **0 (urađeno)** | E1–E5 u ovom repozitorijumu | 1 CPU, ~30 min | vidi odeljak 8 | — |
| **1: KFB v0** | 5 porodica × 4 nivoa šuma × 3 ponavljanja. Enkoderi: xz, zstd `--long -19`, FFV1, x264 lossless (qp 0), CM, indukcija pravila (CA, LDS), LLM sinteza programa (lokalni otvoreni model, fiksiran heš) | 1–2 nedelje, CPU + 1 GPU | `gap/K̄ ≤ 0,1` za porodice u modelu; CSPRNG ≥ N; svi zapisi tačni na drugoj mašini | LLM sinteza ne smanji `gap` ni za jednu porodicu van CA |
| **2: replay** | deterministički engine (otvoreni emulator + homebrew ROM ili Freedoom), snimljeni ulazi → tačni frejmovi | 1 nedelja | zapis ≈ H(ulaza), tačno na drugoj mašini | desync ili zapis > lossless video |
| **3: lestvica stopa na realnom filmu** | otvoreni film (Big Buck Bunny / Sintel, CC). Lossless (FFV1) → DCVC-RT na 3 tačke → generativni kodek na najnižim → semantički nivo (LLM-kodovan sinopsis od 256 B, 1 KB, 4 KB, 16 KB → text-to-video). Meri se: PSNR, MS-SSIM, LPIPS, DISTS, FVD, VMAF, vremenska konzistencija (warping greška), bitovi identifikacije, ljudska procena (MOS, 2AFC) | 2–4 nedelje, GPU | kriva „bitovi ↔ nivo očuvanja" na jednom grafiku za A/B/C | semantički nivo od 1 KB ne nosi više bitova identifikacije od naslova |
| **3b** | izmeriti veličinu javnih produkcionih fajlova BBB-a (scene, teksture, animacija) kao *gornju granicu K* za renderovan film | 1 dan | — | — |

Svaka faza: pre-registracija heša modela, nezavisna mašina za dekodovanje, SHA-256 i poređenje bajtova, ≥ 3 ponavljanja sa novim seed-ovima i objavljeni sirovi JSON.

---

## 7. Izvršiv prototip

```
igc/film.py        kanonski film (RGB24 + PCM) i varint metapodaci (ulaze u budžet)
igc/rangecoder.py  binarni aritmetički koder, celobrojan, samoograničavajući (flush 4 B)
igc/cm.py          kontekstni prediktor + logistički mikser (minimalni neuronski model)
igc/mt19937.py     MT19937: temper/untemper, rekurencija, untwist, inverzija seed-a
igc/ca.py          Life-like CA film + metoda "rule" (MDL pravilo + izuzeci)
igc/sources.py     izvori: csprng, shake_keyed, np_pcg64, np_legacy, py_random,
                   py_random_os, shake_weak20, structured, structured_noisy
igc/codec.py       format zapisa, sve metode, MODEL_ID, kontrola "leaky"
tools/encode_one.py, tools/decode_one.py   enkoder i NEZAVISNI dekoder (zasebni procesi)
experiments/e1…e5  eksperimenti; rezultati u results/
tests/test_igc.py  14 testova (bit-identičnost sa NumPy/Python, povratak, skraćivanje)
```

Format zapisa: `[1 B metoda][8 varint-ova metapodataka][podaci metode]`. **Ništa se ne čita iz dužine fajla.** Dekoder vraća broj pročitanih bajtova, a eksperiment mu dodaje 64 B nasumičnog smeća i proverava da je pročitao tačno `len(zapis)`.

Pokretanje:
```bash
python3 -m unittest discover -s tests -v
IGC_TMP=/tmp python3 experiments/e1_random_video.py --reps 3   # ~15 min
python3 experiments/e2_seed_search.py 3                         # ~1 min
IGC_TMP=/tmp python3 experiments/e3_rule_and_exception.py 3     # ~15 min
python3 experiments/e4_lossy_seed.py                            # ~10 s
python3 experiments/e5_bounds.py
```

Napomena o „AI modelu" (stavka 6 iz zadatka): veliki pretrenirani model nije bio dostupan u okruženju na ponovljiv način (bez GPU-a i bez lokalnih težina). Ulogu prediktora ima CM sa online logističkim mikserom (jedan neuron, kao u PAQ porodici). Za uniformni ulaz Teorema 2 važi za *svaki* model, uključujući LLM, pa veći model može samo da plati veću „cenu učenja ničega". Za strukturisane izvore pravi LLM bi bio relevantan, i to je deo Faze 1 i Faze 3.

---

## 8. Kontrolni eksperimenti, metrike i izmereni rezultati

### 8.0 Revizija početnog prototipa (shake_256 + XOR + zlib)
| moguća slabost | nalaz | gde je provereno |
|---|---|---|
| Da li je rezidual kompresibilan za nasumičan ulaz? | **Ne.** `xor_zlib` je *duži* od sirovog zapisa: +15 B (71 KB) i +485 B (3,1 MB). | E1 |
| Da li zaglavlje i metapodaci ulaze u budžet? | Moraju. Ako se meri samo `len(zlib.compress(R))`, nedostaju id generatora i verzija, seed, dužina, format i zvučni parametri. Kod nas je to 10–14 B unutar zapisa. | `igc/film.py`, `codec.py` |
| Da li je generator nezavisan od podataka? | Kod nas je ključ javna konstanta (`XOR_MODEL_KEY`). Rizik je seed izabran ili izveden *iz* X (npr. `hash(X)`) koji se ne prenese, ili se ne uračuna. | `codec.py`, MODEL_ID |
| Da li je veličina merena posle svih koraka? | Meri se dužina konačnog fajla zapisa koji dekoder čita. Zapis je samoograničavajući, pa nema skrivenog kanala kroz dužinu fajla. | `tools/`, test sa smećem |
| Da li rezultat ostaje identičan pri nezavisnoj rekonstrukciji? | Da, za svih 375 primenljivih zapisa u E1 (zaseban proces, SHA-256 + svi bajtovi). | E1 |
| Da li je poređenje sa standardnim kompresorima pošteno? | zlib (raw deflate, nivo 9), LZMA2 (raw, 9e) i bz2 (9) rade nad istim bajtovima. Korišćeni su formati sa minimalnim zaglavljem, što je u korist standardnih kompresora. | `codec.py` |
| **Suštinska slabost** | `R = X ⊕ SHAKE(k)` je one-time pad: za svaki X je R uniformno raspodeljen. To je šifrovanje, a ne kompresija. | E2 (pristrasan izvor) |

### Kontrole
1. **Negativna kriptografska kontrola**: `csprng` (os.urandom) i `shake_keyed` (ključ odbačen) moraju dati zapis ≥ N kod *svake* metode.
2. **„Nije razbijeno od nas"**: `np_pcg64` je nekriptografski generator, ali ga ne invertujemo, pa mora pasti na sirovi zapis.
3. **Kontrola curenja kroz model (`leaky`)**: film se upiše u „model", a zapis nosi 8-bajtni heš. Važi samo ako je MODEL_ID u trenutku dekodovanja jednak pre-registrovanom.
4. **Samoograničavanje**: 64 B smeća iza zapisa; mora biti pročitano tačno `len(zapis)`.
5. **Nezavisna rekonstrukcija**: enkoder i dekoder su odvojeni procesi; dekoder dobija samo zapis.
6. **Tačnost**: SHA-256 i poređenje svih bajtova (broj razlika se beleži).
7. **Analitičke donje granice**: `structured_noisy` (entropija šuma) i E3 (samoinformacija procesa). Zapis ispod granice znači bag ili curenje.
8. **Ponavljanja**: 3 nezavisna izvora po konfiguraciji, svaki put novi seed ili ključ.

### Metrike
Stvarna veličina zapisa (B, sa zaglavljem), odnos kompresije `|X| / |zapis|` (X = metapodaci + sadržaj), tačnost (k/n), vreme enkodovanja i dekodovanja (s, zasebni procesi), vršni RSS procesa (MB; uključuje ≈ 39 MB osnovnog Python+NumPy procesa).

### E1: nasumičan i kontrolni video + PCM, tačna rekonstrukcija (3 ponavljanja, novi seed/ključ svaki put)

Svih **375/375** primenljivih zapisa je tačno (SHA-256 + svi bajtovi), samoograničavajuće (64 B smeća) i dekodovano uz pre-registrovani MODEL_ID `69dba75c…`. Pun izveštaj: `results/e1_summary.md`, a sirovi podaci su u `results/e1_results.json`. U tabeli je najkraći tačan zapis (srednja vrednost) i metoda koja ga daje.

| izvor | veliki: 160×120×48 + 2 s 48 kHz stereo (3 148 813 B) | mali: 32×24×24 + 1 s 8 kHz (71 306 B) | ≤ 1 KB? |
|---|---|---|---|
| `csprng` (os.urandom) | 3 148 814 (raw) | 71 307 (raw) | ✘ |
| `shake_keyed` (CSPRNG, ključ odbačen) | 3 148 814 (raw) | 71 307 (raw) | ✘ |
| `np_pcg64` (nekriptografski, ali ga ne invertujemo) | 3 148 814 (raw) | 71 307 (raw) | ✘ |
| `np_legacy` (MT19937, seed 32 b) | **21** (infer → seed), 149 944:1 | **18** | ✔ |
| `py_random` (MT19937, seed 32 b) | **22** (infer → ključ), 143 128:1 | **19** | ✔ |
| `py_random_os` (MT19937, seed iz OS-a) | 2 512 (infer → stanje) | 2 509 | ✘ (≤ 4 KB ✔) |
| `shake_weak20` (SHAKE, seed 20 b) | **18** (gruba sila 2²⁰) | **15** | ✔ |
| `structured` (proceduralni) | 847 998 (lzma) | 3 677 (cm) | ✘ |
| `structured_noisy` (σ = 2) | 2 668 780 (lzma); granica 1 125 974 | 33 507 (cm); granica 24 200 | ✘ |

Šta se vidi:
- Na CSPRNG sadržaju **sve** metode daju ≥ N (zlib +486 B, lzma +158 B, bz2 ≈ +14 KB, CM +902 B na malom). Nijedna rezervna metoda ne „pobeđuje" raw.
- `xor_store` je *identičan* raw-u po veličini, a `xor_zlib` je *uvek* veći: na `structured` 3 149 299 B naspram 847 998 B (lzma). XOR je one-time pad.
- Isti „nasumičan" sadržaj je ili nekompresibilan ili kompresibilan 150 000:1, **zavisno samo od toga koji je generator upotrebljen**. Statistički testovi to ne razlikuju; razlikuje ga samo pretpostavka o porodici generatora.
- `structured`: program koji ga generiše ima ~30 linija, ali lzma postiže samo 3,7:1. Generički kompresori ne pronalaze programe (PRED-2).
- `structured_noisy`: zapisi su iznad tačne entropije šuma (CM 1,38×, lzma 2,37×), kako i mora biti. To je kontrola da merenje ne „curi".
- Kontrola varanja: zapis od 19 B je tačan **samo** uz model koji sadrži film, a tada se MODEL_ID promeni (`važeće: NE`). Sa zamrznutim modelom dekodovanje ne uspeva.
- Granica budžeta (CSPRNG, bez slike, 3 ponavljanja): N = 1 013 B → zapis 1 023 B ✔. N = 1 024 B → 1 034 B ✘ (≈ 10 B univerzalnog zaglavlja; po Teoremi 1 specijalizovan dekoder bi mogao da stane, ali ni bit ispod N). N = 1 100 B → gornja granica uspeha za 1 KB je 2⁻⁶⁰⁷. N = 65 536 B → 2⁻⁵¹⁶⁰⁹⁵.
- Vreme i memorija (zasebni procesi, VmHWM; osnovni proces ≈ 30 MB): dekodovanje 0,00–0,26 s i ≤ 49 MB za sve metode na velikom filmu. `infer` kodira ≈ 0,9 s (najviše za grubu silu 2²⁰ SHAKE-a), a dekodira 0,04 s. CM (čist Python) radi ≈ 19 KB/s u oba smera, sa ≈ 60 MB.

Napomena o merenju memorije: prvo pokretanje je merilo `ru_maxrss`, koji na Linuxu posle fork+exec nasleđuje RSS roditelja i pokazivao je lažnih 229 MB. Prešao sam na `VmHWM` i ceo E1/E3 ponovo izvršio. Veličine i tačnost su u oba pokretanja bili isti.

### E3: film kao dinamički sistem (pravilo + izuzeci), 11 940 491 B, 3 ponavljanja

Svih **144/144** zapisa je tačno, samoograničavajuće i sa važećim MODEL_ID. Enkoder **ne zna** pravilo ni p: procenjuje ih iz frejmova. Pun izveštaj: `results/e3_summary.md`.

| pravilo | p | izuzetaka | ideal B | **rule B** | rule/ideal | lzma B | bz2 B | odnos (rule) |
|---|---|---|---|---|---|---|---|---|
| Conway B3/S23 | 0 | 0 | 128 | **203** | 1,59 | 154 504 | 131 570 | 58 820:1 |
| Conway | 10⁻⁶ | 6 | 144 | **225** | 1,56 | 159 020 | 136 556 | 53 148:1 |
| Conway | 10⁻⁵ | 30 | 197 | **274** | 1,39 | 154 279 | 130 906 | 43 525:1 |
| Conway | 10⁻⁴ | 402 | 867 | **947** | 1,09 | 144 355 | 122 409 | 12 613:1 |
| Conway | 10⁻³ | 3 902 | 5 694 | 5 813 | 1,02 | 163 434 | 142 610 | 2 054:1 |
| Conway | 10⁻² | 39 098 | 39 624 | 39 846 | 1,006 | 273 228 | 271 387 | 300:1 |
| Conway | 0,05 | 195 719 | 140 274 | 141 773 | 1,011 | 626 673 | 661 004 | 84:1 |
| Conway | 0,2 | 782 682 | 353 373 | 361 653 | 1,023 | 668 490 | 699 688 | 33:1 |
| Conway | 0,5 | 1 957 283 | 489 600 | 489 678 | 1,000 | 679 975 | 709 574 | 24:1 |
| HighLife B36/S23 | 10⁻⁴ | 396 | 856 | **937** | 1,09 | 157 617 | 135 257 | 12 748:1 |
| Seeds B2/S | 10⁻⁴ | 394 | 852 | **935** | 1,10 | 489 964 | 518 997 | 12 766:1 |
| slučajno pravilo | 10⁻⁴ | 384 | 837 | **920** | 1,10 | 123 619 | 121 659 | 12 974:1 |

- **Hipoteza P2 nije opovrgnuta.** Zapis = ideal + ≈ 75–80 B konstante (zaglavlje 11 B, pravilo 3 B, boje 6 B, učenje konteksta za F₀ i varint-ovi) za malo p, odnosno ≤ 2,3 % iznad ideala za veliko p. Ni u jednom slučaju zapis nije pao ispod ideala.
- **Tačno ispod 1 KB** važi do p ≈ 10⁻⁴ (≈ 400 nepredvidivih događaja u 3,9 miliona prelaza), za sva četiri pravila. Film od 11,9 MB staje u 920–947 B.
- Generički kompresori su 760× (p = 0) do 150× (p = 10⁻⁴) iznad. Pri p = 0,5 sadržaj je stvarno slučajan i *sve* metode konvergiraju ka N bita: rule je na idealu 1,000, a lzma 1,39×.
- Brudno u praksi: posle F₀ deterministička dinamika ne proizvodi nijedan bit; svaki bit zapisa je ili F₀ ili događaj (izuzetak).

### E2: pretraga seed-a za tačnu kompresiju nasumičnog ulaza (3 ponavljanja po tački)

| N (bita) | k (bita seed-a) | min težina reziduala (sr.) | neto dobitak (sr., bita) | najbolji neto | zlib(rezidual) |
|---|---|---|---|---|---|
| 4 096 | 0 | 2 032 | −6,3 | −5 | 523 B (> 512) |
| 4 096 | 8 | 1 970 | −11,0 | −10 | 523 B |
| 4 096 | 20 | 1 893 | −10,3 | −10 | 523 B |
| 65 536 | 0 | 32 745 | −7,7 | −6 | 8 203 B (> 8 192) |
| 65 536 | 16 | 32 237 | −12,7 | −11 | 8 203 B |

Ušteda u rezidualu raste tačno kao ≈ k bita, ali seed košta k bita, pa je neto ≈ −log₂ N. Enumerativni kod je stvarno izvršen i dekodovan za N = 4 096 (18/18 tačno). **Pristrasan izvor** (P(1) ≈ 0,1, N = 65 536 bita): direktan enumerativni kod je 3 897 B, a seed + XOR 8 194 B. zlib direktno daje 4 715 B, a zlib(XOR) 8 203 B. **XOR sa generatorom nezavisnim od podataka uništava strukturu.** H0 iz E2 **nije opovrgnuta**: ni u jednom od 33 pokušaja neto dobitak nije bio pozitivan.

### E4: gde seed ideja radi
| d (dim. bloka) | kandidata po bloku | D po dim. (r = 1 bit/dim) |
|---|---|---|
| 1 | 2 | 0,444 |
| 8 | 256 | 0,356 |
| 16 | 65 536 | 0,302 |
| 20 | 1 048 576 | **0,299** |
| Lloyd-Max (skalarni, 1 bit) | — | 0,363 |
| Šenon D(1) | ∞ | 0,250 |

| q = N(μ, s²) | KL (bita) | E[log₂ K] | Elias-δ (bita) | Li–El Gamal granica | uzorak: sr / std |
|---|---|---|---|---|---|
| N(1; 0,5²) | 1,18 | 1,06 | 3,28 | 6,30 | 0,968 / 0,500 |
| N(2; 0,3²) | 3,97 | 3,43 | 7,20 | 10,28 | 2,013 / 0,291 |
| N(3; 0,1²) | 9,10 | 8,27 | 14,31 | 16,44 | 2,996 / 0,099 |
| N(4; 0,03²) | 15,88 | 15,17 | 22,88 | 23,96 | 4,001 / 0,029 |

Slučajni kodni rečnik iz zajedničkog seed-a nadmašuje skalarni kvantizator već od d ≈ 8 i približava se Šenonovoj granici, ali uz eksponencijalnu pretragu. Kanalna simulacija šalje **tačan uzorak iz posteriora** za ≈ KL bita.

---

## 9. Očekivani rezultati (PREDVIĐANJA, nisu merenja)

- **PRED-1.** Nijedna metoda, uključujući bilo koji AI model, neće postići tačan zapis CSPRNG sadržaja kraći za c bita sa učestalošću većom od 2^{−c}. *Opovrgava se:* jednim ponovljivim kontraprimerom (tada je bag ili razbijena kriptografija).
- **PRED-2.** Na KFB-u će generički kompresori ostaviti `gap` od 10²–10⁴× za filmove sa malim šumom (E3 to već pokazuje: lzma je ≈ 700× iznad pri p = 10⁻⁶). Indukcija pravila će ga zatvoriti na ≤ 1,1× kada je porodica u modelu. LLM sinteza programa će ga zatvoriti samo za najjednostavnije porodice, u skladu sa rezultatima KoLMogorov testa.
- **PRED-3.** Replay kodovanje: 5 minuta tačnog snimka igre staje u 1–10 KB (≈ 0,5–4 bita po frejmu za ulaze sa prediktorom). Ispod 1 KB staje samo uz jak prediktor ulaza ili za kraće segmente.
- **PRED-4.** Lossless kompresija realnog filma neće pasti ispod ~30–50 % sirove veličine (šum i zrno), dakle ≥ 10¹¹ bita za 2 h.
- **PRED-5.** Perceptivno na 1 KB za 2 h: kvalitet na nivou frejma biće statistički nerazlikovan od bezuslovnog generisanja sa istim žanrovskim promptom. Na nivou filma moguće je eventualno ≈ 10 bitova identifikacije (izbor među ~1 000 filmova).
- **PRED-6.** Semantički na 1 KB: ≈ 1 400–2 000 reči „treatment-a" čuva likove, glavne događaje i redosled. Doslovni dijalog se gubi (treba mu 5–11 KB).
- **PRED-7.** Za klip od 10–30 s „veoma slično" na 1 KB je moguće samo za usko ograničen sadržaj (npr. govornik sa *već poznatim* identitetom u modelu), i to samo ako se bitovi identiteta pošteno uračunaju.
- **PRED-8.** Javni produkcioni fajlovi renderovanog filma (BBB) su ≥ 10⁸ B. Čak i potpuno sintetički film ima veliku K kada je umetnička informacija ručno napravljena, a ne proceduralna.

---

## 10. Sledeći korak koji najviše povećava znanje

**Napraviti KFB v0 i na njemu pustiti LLM sintezu programa.** Ovo je jedini režim u kome je 1 KB moguć, a sada ne znamo da li „inverzno generativno" pretraživanje može da *pronađe* kratak opis kada on postoji izvan ručno pisanih porodica (CA, PRNG). Ishod je informativan u oba smera:
- ako LLM ili pretraga zatvori `gap` za porodice koje nije video, IGC ima realan put za sintetički i proceduralni sadržaj;
- ako ne zatvori, granica je računska (Liu–Pass, Levin), a ne informaciona, i to treba objaviti uz brojeve.

Konkretno, prvi zadatak: dodati 2 porodice (celobrojni LDS i sprajt-fizika), enkoder koji traži program u malom DSL-u (enumeracija + LLM predlozi) i meriti `gap`, tačnost i vreme na 3 nivoa šuma. Paralelno, za Cilj C, izmeriti bitove identifikacije za LLM-kodovane sinopsise od 256 B do 16 KB na 100 filmova (Faza 3, semantički deo). To je jeftino, a direktno meri koliko „filma" zaista staje u 1 KB.
