"""Testovi porodice SHADER i enkodera shader_infer.

    python3 -m unittest tests.test_kfb_shader -v
"""
import json
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kfb import encoders, families, kfl, record  # noqa: E402
from kfb.encoders import shader_infer  # noqa: E402
from kfb.families import shader  # noqa: E402
from kfb.suite import TIERS  # noqa: E402

MINI, STD = TIERS["mini"], TIERS["std"]


def make(meta, level, seed):
    prog, info = shader.generate(np.random.default_rng(seed), meta, shader.LEVELS[level])
    return prog, info, record.encode_kfl(meta, prog)


def video_of(meta, payload):
    return np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(meta.frames, meta.height, meta.width, 3)


class TestFamily(unittest.TestCase):
    def test_registered(self):
        self.assertIn("shader", families.load_all())
        self.assertIn("shader_infer", encoders.load_all())
        self.assertTrue(encoders.load_all()["shader_infer"].family_aware)
        for lv in ("one_term", "three_terms", "rings_xor", "three_terms_noise"):
            self.assertIn(lv, shader.LEVELS)

    def test_oracle_decodes_all_levels_mini(self):
        for lname in shader.LEVELS:
            for seed in (11, 12):
                prog, info, rec = make(MINI, lname, seed)
                payload = prog.run(MINI)
                m2, pl2, used = record.decode(rec + b"\xff" * 64)
                self.assertEqual(used, len(rec), lname)
                self.assertEqual(m2, MINI)
                self.assertEqual(pl2, payload, lname)
                self.assertEqual(len(payload), MINI.payload_bytes)
                json.dumps(info)
                self.assertIsInstance(info["ideal_bits"], float)
                v = video_of(MINI, payload)
                self.assertGreater(len(np.unique(v)), 16, lname)        # nije degenerisan
                self.assertFalse(np.array_equal(v[0], v[-1]), lname)    # pomera se u vremenu
                a = np.frombuffer(payload[MINI.video_bytes:], "<i2")
                self.assertTrue(a.any(), lname)

    def test_deterministic(self):
        for lname in shader.LEVELS:
            _, i1, r1 = make(MINI, lname, 7)
            _, i2, r2 = make(MINI, lname, 7)
            self.assertEqual(r1, r2, lname)
            self.assertEqual(json.dumps(i1), json.dumps(i2))
            _, _, r3 = make(MINI, lname, 8)
            self.assertNotEqual(r1, r3, lname)

    def test_std_instances_fast(self):
        for lname in ("three_terms", "rings_xor", "three_terms_noise"):
            t0 = time.perf_counter()
            prog, info, rec = make(STD, lname, 3)
            payload = prog.run(STD)
            self.assertLess(time.perf_counter() - t0, 2.0, lname)
            self.assertEqual(record.decode(rec)[1], payload, lname)
            if lname == "three_terms_noise":
                K = info["params"]["exceptions"]
                self.assertTrue(0.0005 * STD.video_bytes < K < 0.002 * STD.video_bytes)
            else:
                self.assertLess(len(rec), 200)

    def test_noise_exceptions_are_xor_nonzero(self):
        prog, info, rec = make(MINI, "three_terms_noise", 5)
        clean = kfl.Program(video=prog.video, audio=prog.audio)
        a = np.frombuffer(prog.run(MINI)[:MINI.video_bytes], np.uint8)
        b = np.frombuffer(clean.run(MINI)[:MINI.video_bytes], np.uint8)
        self.assertEqual(int((a != b).sum()), info["params"]["exceptions"])

    def test_isin_identities_used_by_canonical_forms(self):
        t = np.arange(256)
        L = kfl.ISIN_LUT
        self.assertTrue(np.all(L[(-t) & 255] == -L[t]))
        self.assertTrue(np.all(L[(t + 128) & 255] == -L[t]))
        self.assertTrue(np.all(L[(128 - t) & 255] == L[t]))


class TestEncoder(unittest.TestCase):
    def check(self, meta, lname, seed, budget, slack=16):
        prog, info, rec = make(meta, lname, seed)
        payload = prog.run(meta)
        t0 = time.perf_counter()
        out = shader_infer.encode(meta, payload, budget)
        dt = time.perf_counter() - t0
        self.assertIsNotNone(out, (lname, seed))
        m2, pl2, used = record.decode(out + b"\x00" * 64)
        self.assertEqual(pl2, payload, (lname, seed))
        self.assertEqual(used, len(out))
        self.assertLessEqual(len(out), len(rec) + slack, (lname, seed, len(out), len(rec)))
        self.assertLess(dt, budget + 2.0)
        return len(out), len(rec)

    def test_exact_mini_all_levels(self):
        for lname in shader.LEVELS:
            for seed in (101, 102, 103):
                self.check(MINI, lname, seed, 20)

    def test_exact_std(self):
        self.check(STD, "three_terms", 201, 60)
        self.check(STD, "rings_xor", 202, 60)

    def test_budget_respected(self):
        prog, _, _ = make(STD, "three_terms_noise", 9)
        payload = prog.run(STD)
        for budget in (0.5, 2.0):
            t0 = time.perf_counter()
            out = shader_infer.encode(STD, payload, budget)
            dt = time.perf_counter() - t0
            self.assertLess(dt, budget + 1.5, budget)
            if out is not None:
                self.assertEqual(record.decode(out)[1], payload)

    def test_foreign_films_do_not_crash(self):
        fams = families.load_all()
        for fname in ("ca", "prng"):
            fam = fams[fname]
            lname = next(iter(fam.LEVELS))
            obj, _ = fam.generate(np.random.default_rng(1), MINI, fam.LEVELS[lname])
            payload = kfl.run(obj, MINI)
            out = shader_infer.encode(MINI, payload, 10)
            if out is not None:
                self.assertEqual(record.decode(out)[1], payload, fname)
        rnd = np.random.default_rng(2).integers(0, 256, MINI.payload_bytes, dtype=np.uint8).tobytes()
        out = shader_infer.encode(MINI, rnd, 10)
        if out is not None:
            self.assertEqual(record.decode(out)[1], rnd)

    def test_audio_inference(self):
        n = np.arange(8000, dtype=np.int64)
        for tones in ([{"A": 200, "p": 56, "q": 2}],
                      [{"A": 90, "p": 37, "q": 2}, {"A": -45, "p": 101, "q": 2}]):
            a = sum(t["A"] * kfl.ISIN_LUT[((n * t["p"]) >> t["q"]) & 255] for t in tones)
            a16 = (a & 0xFFFF).astype(np.uint16).view("<i2")
            cands = shader_infer.infer_audio(a16, time.perf_counter() + 10)
            best = min(((shader.nbytes(e) + (len(x.to_bytes()) if x else 0), e, x) for e, x in cands
                        if e is not None), key=lambda r: r[0])
            self.assertLess(best[0], 30)
            meta = MINI
            prog = kfl.Program(video=kfl.const(0), audio=best[1], exc_audio=best[2])
            got = np.frombuffer(prog.run(meta)[meta.video_bytes:], "<i2")
            self.assertTrue(np.array_equal(got, a16))


if __name__ == "__main__":
    unittest.main()
