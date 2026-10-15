"""E2: Može li pretraga seed-a + korekcija smanjiti TAČAN zapis nasumičnog ulaza?

Hipoteza H0 (koju pokušavamo da opovrgnemo): za uniformni X od N bita,
izbor najboljeg od 2^k pseudonasumičnih kandidata P_s (fiksni generator) i
kodovanje reziduala R = X xor P_s daje ukupni zapis kraći od N bita.

Zapis: [k bita: s][ceil(log2(N+1)) bita: težina w = |R|][ceil(log2 C(N,w)) bita: rang R]
Rang je enumerativni kod (kombinatorni brojni sistem) — optimalan za R date
težine. Za N=4096 kod se stvarno izvršava i dekoduje; za veće N dužina se
računa tačno preko math.comb (bez aproksimacija).

Analitičko predviđanje: min_s w_s ~ N/2 - sqrt(N k ln2 / 2), pa ušteda u
rezidualu ~ k bita, koja se troši na k bita seed-a -> neto ~ -log2(N+1).
Dodatno: (a) originalni prototip (SHAKE + zlib nad rezidualom); (b) pristrasan
izvor Bernoulli(0.1) — XOR sa generatorom nezavisnim od podataka UNIŠTAVA strukturu.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def candidates(k_start: int, count: int, nbytes: int) -> np.ndarray:
    h0 = hashlib.shake_256(b"IGC-E2")
    buf = bytearray()
    for s in range(k_start, k_start + count):
        h = h0.copy()
        h.update(s.to_bytes(4, "little"))
        buf += h.digest(nbytes)
    return np.frombuffer(bytes(buf), dtype=np.uint8).reshape(count, nbytes)


def best_seed(x: np.ndarray, k: int, chunk: int = 1 << 13) -> tuple[int, int]:
    nb = x.size
    best_w, best_s = None, None
    total = 1 << k
    for start in range(0, total, chunk):
        c = candidates(start, min(chunk, total - start), nb)
        w = np.bitwise_count(c ^ x).sum(axis=1, dtype=np.int64)
        i = int(w.argmin())
        if best_w is None or w[i] < best_w:
            best_w, best_s = int(w[i]), start + i
    return best_s, best_w


def enum_rank(bits: list[int]) -> int:
    rank, k = 0, sum(bits)
    for n in range(len(bits) - 1, -1, -1):
        if bits[n]:
            rank += math.comb(n, k)
            k -= 1
    return rank


def enum_unrank(rank: int, N: int, w: int) -> list[int]:
    bits, k = [0] * N, w
    for n in range(N - 1, -1, -1):
        if k and rank >= math.comb(n, k):
            bits[n] = 1
            rank -= math.comb(n, k)
            k -= 1
    return bits


def code_bits(N: int, k: int, w: int) -> int:
    return k + math.ceil(math.log2(N + 1)) + (math.comb(N, w) - 1).bit_length()


def run_trial(N: int, k: int, rng_bytes: bytes, verify: bool) -> dict:
    x = np.frombuffer(rng_bytes, dtype=np.uint8)
    t = time.perf_counter()
    s, w = best_seed(x, k)
    t_search = time.perf_counter() - t
    total = code_bits(N, k, w)
    row = {"N": N, "k": k, "seed": s, "w": w, "total_bits": total,
           "net_gain_bits": N - total, "search_s": t_search}
    resid = (candidates(s, 1, x.size)[0] ^ x)
    row["zlib_resid_bytes"] = len(zlib.compress(resid.tobytes(), 9))
    if verify:  # stvarno enumerativno kodovanje + dekodovanje
        bits = np.unpackbits(resid).tolist()
        rank = enum_rank(bits)
        nrank = (math.comb(N, w) - 1).bit_length()
        packed = s.to_bytes(4, "little") + w.to_bytes(2, "little") + rank.to_bytes((nrank + 7) // 8, "little")
        s2 = int.from_bytes(packed[:4], "little")
        w2 = int.from_bytes(packed[4:6], "little")
        r2 = int.from_bytes(packed[6:], "little")
        rec = np.packbits(np.array(enum_unrank(r2, N, w2), dtype=np.uint8)) ^ candidates(s2, 1, x.size)[0]
        row["roundtrip_exact"] = bool((rec == x).all())
    return row


def main() -> None:
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = {"uniform": [], "biased": []}
    plan = [(4096, [0, 4, 8, 12, 16, 20]), (65536, [0, 4, 8, 12, 16])]
    for N, ks in plan:
        for k in ks:
            for r in range(reps):
                row = run_trial(N, k, os.urandom(N // 8), verify=(N == 4096))
                row["pred_w"] = N / 2 - math.sqrt(N * k * math.log(2) / 2)
                out["uniform"].append(row)
            rr = [x for x in out["uniform"] if x["N"] == N and x["k"] == k]
            print(f"N={N:6d} k={k:2d} w_min={np.mean([x['w'] for x in rr]):9.1f} "
                  f"(pred {rr[0]['pred_w']:9.1f}) neto={np.mean([x['net_gain_bits'] for x in rr]):6.1f} b "
                  f"max_neto={max(x['net_gain_bits'] for x in rr):4d} "
                  f"zlib(R)={np.mean([x['zlib_resid_bytes'] for x in rr]):.0f}B vs N/8={N//8} "
                  f"t={np.mean([x['search_s'] for x in rr]):.2f}s", flush=True)
    # pristrasni izvor: p(1)=0.1
    N = 65536
    for r in range(reps):
        u = np.frombuffer(os.urandom(N), dtype=np.uint8)
        bits = (u < 26).astype(np.uint8)          # P(1) = 26/256 ~ 0.1016
        x = np.packbits(bits)
        w = int(bits.sum())
        direct = math.ceil(math.log2(N + 1)) + (math.comb(N, w) - 1).bit_length()
        s, wr = best_seed(x, 12)
        out["biased"].append({"N": N, "w": w, "direct_enum_bits": direct, "seed_xor_bits": code_bits(N, 12, wr),
                              "zlib_direct_B": len(zlib.compress(x.tobytes(), 9)),
                              "zlib_xor_B": len(zlib.compress((x ^ candidates(s, 1, x.size)[0]).tobytes(), 9))})
    b = out["biased"]
    print(f"pristrasan izvor N={N}: direktno {np.mean([x['direct_enum_bits'] for x in b])/8:.0f} B, "
          f"seed+XOR {np.mean([x['seed_xor_bits'] for x in b])/8:.0f} B, zlib direktno "
          f"{np.mean([x['zlib_direct_B'] for x in b]):.0f} B, zlib(XOR) {np.mean([x['zlib_xor_B'] for x in b]):.0f} B")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results/e2_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
