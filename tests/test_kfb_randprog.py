"""Testovi porodice randprog i generičkog enkodera enum_search.

    python3 -m unittest tests.test_kfb_randprog -v
"""
import json
import random
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from igc import codec as igc_codec  # noqa: E402
from igc.film import Meta  # noqa: E402
from kfb import encoders, families, kfl, record  # noqa: E402
from kfb.encoders import enum_search  # noqa: E402
from kfb.families import ca, prng, randprog  # noqa: E402
from kfb.kfl import C, T, X, Y, op  # noqa: E402
from kfb.suite import TIERS  # noqa: E402

MINI, STD = TIERS["mini"], TIERS["std"]


def make(meta, level, seed):
    prog, info = randprog.generate(np.random.default_rng(seed), meta, randprog.LEVELS[level])
    return prog, info, record.encode_kfl(meta, prog)


def split(meta, payload):
    v = np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(meta.frames, meta.height, meta.width, 3)
    return v, np.frombuffer(payload[meta.video_bytes:], "<i2")


def encode_timed(meta, payload, budget):
    t0 = time.perf_counter()
    rec = enum_search.encode(meta, payload, budget)
    return rec, time.perf_counter() - t0


class TestFamily(unittest.TestCase):
    def test_registered(self):
        self.assertEqual(randprog.NAME, "randprog")
        self.assertIn("randprog", families.load_all())
        self.assertEqual(set(randprog.LEVELS), {"n5", "n9", "n15"})
        enc = encoders.load_all()["enum_search"]
        self.assertFalse(enc.family_aware)
        self.assertEqual(enc.tiers, frozenset({"mini", "std"}))

    def test_oracle_decodes_all_levels_mini(self):
        for lname, lv in randprog.LEVELS.items():
            for seed in (11, 12, 13):
                prog, info, rec = make(MINI, lname, seed)
                payload = prog.run(MINI)
                m2, pl2, used = record.decode(rec + b"\xff" * 64)
                self.assertEqual(used, len(rec), lname)
                self.assertEqual(m2, MINI)
                self.assertEqual(pl2, payload, lname)
                json.dumps(info)
                self.assertIsInstance(info["ideal_bits"], float)
                self.assertGreater(info["ideal_bits"], 0)
                n = lv["nodes"]
                self.assertTrue(n - 1 <= info["params"]["nodes"] <= n + 1, (lname, info["params"]))
                self.assertIsNone(prog.init)                          # bez stanja
                v, a = split(MINI, payload)
                self.assertGreaterEqual(len(np.unique(v)), 16, lname)
                self.assertFalse(np.array_equal(v[1:], v[:-1]), lname)   # menja se u vremenu
                self.assertFalse(np.array_equal(v, np.broadcast_to(v[:, :1, :1], v.shape)), lname)
                if not lv["audio"]:
                    self.assertIsNone(prog.audio)
                    self.assertFalse(a.any())
                elif prog.audio is not None:
                    self.assertGreaterEqual(len(np.unique(a)), 16, lname)

    def test_deterministic(self):
        for lname in randprog.LEVELS:
            _, i1, r1 = make(MINI, lname, 7)
            random.seed(123)
            np.random.seed(123)                                       # globalno stanje ne sme da utiče
            _, i2, r2 = make(MINI, lname, 7)
            self.assertEqual(r1, r2, lname)
            self.assertEqual(json.dumps(i1), json.dumps(i2))
            self.assertNotEqual(r1, make(MINI, lname, 8)[2], lname)

    def test_std_instances_fast(self):
        for lname in randprog.LEVELS:
            t0 = time.perf_counter()
            prog, info, rec = make(STD, lname, 3)
            payload = prog.run(STD)
            self.assertLess(time.perf_counter() - t0, 2.0, lname)
            self.assertEqual(record.decode(rec)[1], payload, lname)
            self.assertLess(len(rec), 80, lname)
            v, _ = split(STD, payload)
            self.assertGreaterEqual(len(np.unique(v)), 16)

    def test_audio_levels_use_audio(self):
        kinds = set()
        for seed in range(40):
            prog, info, _ = make(MINI, "n9", seed)
            kinds.add(info["params"]["audio_kind"])
        self.assertTrue({"silence", "tone"} <= kinds, kinds)
        self.assertTrue(len(kinds) >= 3, kinds)

    def test_rejects_degenerate_and_reducible(self):
        env = randprog._domain(MINI)
        ok = lambda e: randprog.video_ok(e, env)  # noqa: E731
        self.assertTrue(ok(op("XOR", op("MUL", X, 7), T)))
        self.assertFalse(ok(op("XOR", X, X)))                      # konstantan
        self.assertFalse(ok(op("ADD", X, Y)))                      # svi frejmovi isti
        self.assertFalse(ok(op("ISIN", T)))                        # ravni frejmovi
        self.assertFalse(ok(op("ADD", op("MIN", X, 50), T)))       # MIN(X, 50) == X na 32 kolone
        self.assertFalse(ok(op("AND", op("ADD", X, T), 1)))        # samo 2 vrednosti
        self.assertFalse(ok(op("WHERE", op("LT", X, T), T, X)))    # == MAX(X, T), kraće
        e = op("ADD", op("SUB", X, T), op("ADD", T, T))           # == ADD(X, T): 7 B umesto 3 B
        self.assertTrue(randprog._tight(randprog.evaluate(e, env, (24, 32, 32, 3))[1], (24, 32, 32, 3)))
        self.assertFalse(ok(e))                                    # odbija ga samo provera svodljivosti
        # ISIN(T) != 0 <=> T != 0 na 24 frejma: unutrašnji čvor zamenljiv detetom (WHERE(T, T, Y))
        e = op("WHERE", op("ISIN", T), T, Y)
        shape = (24, 32, 32, 3)
        v, out = randprog.evaluate(e, env, shape)
        v = np.broadcast_to(v & 255, shape).astype(np.uint8)
        self.assertTrue(randprog._tight(out, shape))
        self.assertFalse(randprog._reducible(out, env, shape, v))
        self.assertTrue(randprog._child_reducible(e, env, shape, v))
        self.assertFalse(ok(e))
        # AND sa 37 vidi samo niže bitove: MUL(202, .) == MUL(10, .), 2 B kraće
        e = op("ADD", op("SQ", T), op("AND", op("MUL", 202, op("SHR", Y, T)), 37))
        v, out = randprog.evaluate(e, env, shape)
        v = np.broadcast_to(v & 255, shape).astype(np.uint8)
        self.assertTrue(randprog._child_reducible(e, env, shape, v))
        self.assertFalse(ok(e))
        # bitan je samo niži bajt: 1211 -> HASH(53) (isti niži bajt), 1 B kraće
        e = op("XOR", op("ADD", T, 1211), op("SQ", Y))
        v, out = randprog.evaluate(e, env, shape)
        v = np.broadcast_to(v & 255, shape).astype(np.uint8)
        self.assertTrue(randprog._child_reducible(e, env, shape, v))
        self.assertFalse(ok(e))

    def test_audio_expr_not_reducible(self):
        n = np.arange(MINI.audio_samples, dtype=np.int64)
        for e, want in ((op("SHR", op("XOR", 12, op("ADD", kfl.N, kfl.N)), 5), True),   # XOR 12 nevidljiv
                        (op("SHR", op("XOR", 100, op("ADD", kfl.N, kfl.N)), 5), False)):
            a, _ = randprog.evaluate(e, {"N": n}, n.shape)
            a = np.broadcast_to(a & 0xFFFF, n.shape)
            self.assertEqual(randprog._audio_reducible(e, n, a), want, kfl.to_text(e))

    def test_ideal_bits_large_constant(self):
        """Velika konstanta K nosi -log2 P(K) bita (8..14), ne paušalno."""
        rng = np.random.default_rng(3)
        p_k = randprog.LEAF_W["K"] / sum(randprog.LEAF_W.values())
        seen = 0
        for _ in range(3000):
            sm = randprog._Sampler(rng)
            n = randprog._leaf(sm)
            if n.op == "CONST" and n.value >= 64:
                seen += 1
                want = -np.log2(p_k) - np.log2((np.log2(n.value + 1) - np.log2(n.value)) / 6)
                self.assertAlmostEqual(sm.bits, want, places=6)
                self.assertGreater(sm.bits + np.log2(p_k), 8.0)
        self.assertGreater(seen, 20)


class TestEncoder(unittest.TestCase):
    def test_ops_match_vm(self):
        """Tabela operacija enkodera daje iste vrednosti kao VM (na nasumičnim izrazima porodice)."""
        meta = Meta(5, 4, 3, 24, 8000, 1, 2, 0)
        tt, yy, xx, cc = np.meshgrid(np.arange(3), np.arange(4), np.arange(5), np.arange(3), indexing="ij")
        env = {"T": tt.ravel(), "Y": yy.ravel(), "X": xx.ravel(), "C": cc.ravel(),
               "WID": np.int64(5), "HEI": np.int64(4), "NFR": np.int64(3)}
        rng = np.random.default_rng(0)
        for _ in range(300):
            sm = randprog._Sampler(rng)
            e = randprog._gen(sm, int(rng.integers(3, 12)))
            want = np.frombuffer(kfl.run(kfl.Program(video=e), meta), np.uint8)
            with np.errstate(all="ignore"):
                got = np.broadcast_to(enum_search.evaluate(e, env), want.shape) & 255
            self.assertTrue(np.array_equal(got, want), kfl.to_text(e))
        for name in enum_search.UNARY + enum_search.BINARY + enum_search.TERNARY:
            ar = kfl.ARITY[kfl.NAME_TO_CODE[name]][1]
            vals = [0, 1, -1, 7, -7, 63, 64, 200, -129, 1 << 40, -(1 << 62)]
            for _ in range(30):
                kids = [int(rng.choice(vals)) for _ in range(ar)]
                e = kfl.Node(name, [kfl.const(v) for v in kids])
                want = kfl.run(kfl.Program(video=e), meta)[0]
                with np.errstate(all="ignore"):
                    got = int(np.asarray(enum_search.evaluate(e, env)).reshape(-1)[0]) & 255
                self.assertEqual(got, want, (name, kids))

    def test_exc_size_matches_serialization(self):
        rng = np.random.default_rng(1)
        for K in (1, 5, 300):
            pos = np.sort(rng.choice(100000, K, replace=False)).astype(np.int64)
            for vals, vb in ((None, 0), (rng.integers(1, 256, K), 1), (rng.integers(256, 65536, K), 2)):
                gaps = np.diff(np.concatenate([[-1], pos])) - 1
                n = len(kfl.Exceptions(pos, vals).to_bytes())
                self.assertEqual(enum_search.exc_size(K, gaps, vb), n)

    def test_exact_on_fresh_n5_mini(self):
        hits = 0
        for seed in (101, 102, 103):
            prog, info, oracle = make(MINI, "n5", seed)
            payload = prog.run(MINI)
            rec, dt = encode_timed(MINI, payload, 20)
            self.assertIsNotNone(rec)
            self.assertEqual(record.decode(rec)[1], payload)
            self.assertLess(dt, 20)
            hits += len(rec) <= len(oracle)
        self.assertGreaterEqual(hits, 2)                # mali programi se nalaze (L <= K̄)

    def test_exact_and_budget_n9_n15(self):
        for lname, seed in (("n9", 21), ("n15", 22)):
            prog, info, oracle = make(MINI, lname, seed)
            payload = prog.run(MINI)
            rec, dt = encode_timed(MINI, payload, 4)
            self.assertLess(dt, 4.0, lname)            # budžet poštovan
            self.assertIsNotNone(rec)
            self.assertEqual(record.decode(rec)[1], payload, lname)
            self.assertLessEqual(len(rec), MINI.payload_bytes + 64)
            # MDL rezerva: nikad lošije od bz2/lzma
            for name in ("bz2", "lzma"):
                self.assertLessEqual(len(rec), len(igc_codec.encode(name, MINI, payload)), (lname, name))

    def test_video_yields_time_to_audio(self):
        """Tačan video (9 B) ne sme da potroši vreme zvuka: oba se nalaze, L <= K̄."""
        prog = kfl.Program(video=op("XOR", op("XOR", op("ISIN", T), op("TRI", op("ISIN", X), 20)), 36),
                           audio=op("MUL", op("ISIN", op("SHR", op("MUL", kfl.N, 25), 2)), 79))
        payload = prog.run(MINI)
        rec, dt = encode_timed(MINI, payload, 20)
        self.assertLess(dt, 20)
        self.assertEqual(record.decode(rec)[1], payload)
        self.assertLessEqual(len(rec), len(record.encode_kfl(MINI, prog)))

    def test_std_instance(self):
        prog, info, oracle = make(STD, "n5", 3)
        payload = prog.run(STD)
        rec, dt = encode_timed(STD, payload, 60)
        self.assertLess(dt, 60)
        self.assertEqual(record.decode(rec)[1], payload)
        self.assertLessEqual(len(rec), len(oracle) + 4)

    def test_noisy_film_uses_exceptions(self):
        prog, info, oracle = make(MINI, "n5", 3)
        pl = bytearray(prog.run(MINI))
        rng = np.random.default_rng(5)
        for p in rng.choice(MINI.video_bytes, 20, replace=False):
            pl[p] ^= int(rng.integers(1, 256))
        pl = bytes(pl)
        rec, dt = encode_timed(MINI, pl, 6)
        self.assertEqual(record.decode(rec)[1], pl)
        self.assertLess(len(rec), len(oracle) + 20 * 4)   # program + izuzeci, ne literal

    def test_audio_tone_and_literal(self):
        meta = MINI
        N = kfl.N
        for audio, max_len in ((op("MUL", op("ISIN", op("MUL", N, 7)), 100), 40),
                               (None, 20)):
            prog = kfl.Program(video=op("XOR", op("MUL", X, 3), T), audio=audio)
            payload = prog.run(meta)
            rec, _ = encode_timed(meta, payload, 8)
            self.assertEqual(record.decode(rec)[1], payload)
            self.assertLessEqual(len(rec), max_len)

    def test_other_families_no_crash(self):
        cases = [(ca, "conway_p1e-3"), (prng, "seed16")]
        for fam, lname in cases:
            prog, _ = fam.generate(np.random.default_rng(4), MINI, fam.LEVELS[lname])
            payload = prog.run(MINI)
            rec, dt = encode_timed(MINI, payload, 3)
            self.assertLess(dt, 3.0)
            if rec is not None:
                self.assertEqual(record.decode(rec)[1], payload, fam.NAME)
        noise = np.random.default_rng(9).integers(0, 256, MINI.payload_bytes, dtype=np.uint8).tobytes()
        rec, _ = encode_timed(MINI, noise, 3)
        if rec is not None:
            self.assertEqual(record.decode(rec)[1], noise)
            self.assertGreaterEqual(len(rec), len(noise))     # šum se ne sme "kompresovati"

    def test_edge_metas(self):
        for meta in (Meta(1, 1, 1, 24, 8000, 1, 2, 0), Meta(0, 0, 0, 24, 8000, 1, 2, 50),
                     Meta(3, 2, 2, 24, 8000, 1, 2, 5)):
            for pl in (bytes(meta.payload_bytes), bytes(range(256)) * (meta.payload_bytes // 256 + 1)):
                pl = pl[:meta.payload_bytes]
                rec = enum_search.encode(meta, pl, 2)
                self.assertIsNotNone(rec)
                self.assertEqual(record.decode(rec)[1], pl)
        self.assertIsNone(enum_search.encode(MINI, b"\0" * 10, 2))     # pogrešna dužina

    def test_box_literal_periodic(self):
        prog = kfl.Program(video=op("ADD", op("MOD", X, 4), op("MUL", op("MOD", T, 3), 16)) * 5 + C)
        meta = Meta(16, 8, 9, 24, 8000, 1, 2, 0)
        v, _ = split(meta, prog.run(meta))
        lit = enum_search._box_literal(v, meta)
        self.assertEqual(kfl.run(kfl.Program(video=lit), meta), v.tobytes())
        self.assertEqual(len(lit.data), 3 * 4 * 3)                   # kutija T=3, X=4, C=3


if __name__ == "__main__":
    unittest.main()
