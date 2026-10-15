"""E4: Gde "seed" ideja zaista radi — kompresija sa gubicima i uzorkovanje.

A) Šenonov slučajni kodni rečnik (1959): 2^R kodnih reči iz zajedničkog seed-a,
   enkoder šalje indeks najbliže. Gausov izvor N(0,1), r = 1 bit/dim.
   Teorijska granica D(r) = 2^(-2r) = 0.25; skalarni 1-bitni Lloyd-Max: 1-2/pi.
   Predviđanje: D opada ka 0.25 sa dimenzijom bloka d, uz cenu 2^d poređenja.

B) Kanalna simulacija (Poisson functional representation, Li & El Gamal 2018):
   iz zajedničkog niza U_i ~ p = N(0,1) bira se indeks K tako da je U_K TAČAN
   uzorak iz q = N(mu, s^2). Cena: E[log2 K] <= KL(q||p) + log2(KL+1) + O(1).
   Ovo je matematički ispravna verzija "seed-a" za perceptivni cilj: šalje se
   ~I(X;Y) bita, a dekoder dobija realističan uzorak iz posteriora.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def codebook_chunk(seed: int, d: int, start: int, count: int, sigma: float) -> np.ndarray:
    g = np.random.Generator(np.random.PCG64([seed, d, start]))
    return g.standard_normal((count, d)) * sigma


def random_codebook_vq(d: int, n_samples: int = 4000, seed: int = 2024, rate: float = 1.0) -> dict:
    R = int(round(rate * d))
    D_star = 2 ** (-2 * rate)
    sigma = math.sqrt(1 - D_star)
    nb = max(1, n_samples // d)
    x = np.random.default_rng(7).standard_normal((nb, d))
    best = np.full(nb, np.inf)
    chunk = 1 << 15
    t = time.perf_counter()
    for start in range(0, 1 << R, chunk):
        C = codebook_chunk(seed, d, start, min(chunk, (1 << R) - start), sigma)
        dist = (x * x).sum(1)[:, None] - 2 * x @ C.T + (C * C).sum(1)[None, :]
        best = np.minimum(best, dist.min(1))
    return {"d": d, "R_bits": R, "D_per_dim": float(best.mean() / d), "D_shannon": D_star,
            "D_lloyd_max_1bit": 1 - 2 / math.pi, "search_per_block": 1 << R,
            "time_s": time.perf_counter() - t}


def elias_delta_len(k: int) -> int:
    L = k.bit_length()
    return L - 1 + 2 * (L.bit_length() - 1) + 1


def pfr_index(mu: float, s: float, rng: np.random.Generator, chunk: int = 4096) -> tuple[int, float]:
    """Vraća (K, U_K) za q = N(mu, s^2), p = N(0,1), s < 1."""
    # log r(u) = log q(u) - log p(u)
    def log_r(u):
        return -math.log(s) - (u - mu) ** 2 / (2 * s * s) + u * u / 2
    u_star = mu / (1 - s * s)
    log_rmax = log_r(u_star)
    best, bestK, bestU = np.inf, None, None
    T = 0.0
    i = 0
    while True:
        e = rng.exponential(size=chunk)
        u = rng.standard_normal(chunk)
        Ts = T + np.cumsum(e)
        score = np.log(Ts) - log_r(u)              # log(T_i / r(U_i))
        j = int(score.argmin())
        if score[j] < best:
            best, bestK, bestU = float(score[j]), i + j + 1, float(u[j])
        T = float(Ts[-1])
        i += chunk
        if math.log(T) - log_rmax > best:          # ni jedan kasniji ne može biti bolji
            return bestK, bestU


def kl_bits(mu: float, s: float) -> float:
    return (-math.log(s) + (s * s + mu * mu) / 2 - 0.5) / math.log(2)


def main() -> None:
    out = {"vq": [], "pfr": []}
    print("A) slučajni kodni rečnik, r = 1 bit/dim")
    for d in [1, 2, 4, 8, 12, 16, 20]:
        r = random_codebook_vq(d)
        out["vq"].append(r)
        print(f"  d={d:2d}  2^{r['R_bits']:<2d} kandidata/blok  D={r['D_per_dim']:.4f}  "
              f"(Šenon {r['D_shannon']:.3f}, Lloyd-Max {r['D_lloyd_max_1bit']:.4f})  t={r['time_s']:.1f}s", flush=True)
    print("B) kanalna simulacija (PFR): cena indeksa vs KL")
    cases = [(0.5, 0.8), (1.0, 0.5), (2.0, 0.3), (3.0, 0.1), (4.0, 0.03)]
    for mu, s in cases:
        rng = np.random.default_rng(11)
        Ks, Us = [], []
        trials = 400
        for _ in range(trials):
            K, U = pfr_index(mu, s, rng)
            Ks.append(K)
            Us.append(U)
        kl = kl_bits(mu, s)
        lg = float(np.mean(np.log2(Ks)))
        ed = float(np.mean([elias_delta_len(k) for k in Ks]))
        row = {"mu": mu, "s": s, "KL_bits": kl, "E_log2K": lg, "E_elias_delta_bits": ed,
               "li_elgamal_bound": kl + math.log2(kl + 1) + 4,
               "sample_mean": float(np.mean(Us)), "sample_std": float(np.std(Us))}
        out["pfr"].append(row)
        print(f"  q=N({mu},{s}^2): KL={kl:5.2f} b  E[log2 K]={lg:5.2f}  Elias-δ={ed:5.2f} b  "
              f"granica={row['li_elgamal_bound']:5.2f}  uzorak: sr={row['sample_mean']:.3f} std={row['sample_std']:.3f}",
              flush=True)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/e4_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
