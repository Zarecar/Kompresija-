"""Testovi porodice diffusion i enkodera diffusion_infer. python3 -m unittest tests.test_kfb_diffusion -v"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from igc.film import video_meta  # noqa: E402
from kfb import encoders, families, kfl, record  # noqa: E402
from kfb.encoders import diffusion_infer  # noqa: E402
from kfb.families import ca, diffusion, prng  # noqa: E402
from kfb.suite import TIERS  # noqa: E402

BUDGET = {"mini": 20, "std": 60}


def make(level, tier, seed, **over):
    meta = TIERS[tier]
    lv = dict(diffusion.LEVELS[level], **over)
    prog, info = diffusion.generate(np.random.default_rng(seed), meta, lv)
    return meta, prog, info


def split(meta, payload):
    v = np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(meta.frames, meta.height, meta.width, 3)
    return v, np.frombuffer(payload[meta.video_bytes:], "<i2")


def state_from_video(meta, payload, info):
    """Stanje (smooth: S, wave: cur) pročitano iz kanala 0 pomoću poznate palete."""
    v, _ = split(meta, payload)
    pal = info["params"]["palette"]
    inv = pow(pal["m"][0] % 256, -1, 256)
    U = ((v[..., 0].astype(np.int64) - pal["o"][0]) * inv) & 255
    return U if info["params"]["kind"] == "smooth" else ((U + 128) & 255) - 128


def initial_field(meta, info):
    it = dict(info["params"]["init"])
    if it["type"] == "hash":
        return diffusion.hash_init_values(it["seed"], meta.height, meta.width)
    it["data"] = np.frombuffer(bytes.fromhex(it["data"]), np.uint8).astype(np.int64)
    return diffusion._patch_field(it, meta.height, meta.width)


class TestFamily(unittest.TestCase):
    def test_module_interface(self):
        self.assertEqual(diffusion.NAME, "diffusion")
        self.assertTrue({"smooth", "wave", "smooth_noise"} <= set(diffusion.LEVELS))
        enc = {e.name: e for e in diffusion_infer.ENCODERS}
        self.assertIn("diffusion_infer", enc)
        self.assertTrue(enc["diffusion_infer"].family_aware)

    def test_registries(self):
        try:
            fams, encs = families.load_all(), encoders.load_all()
        except Exception as e:   # tuđ modul u izradi ne sme da obori ovaj test
            self.skipTest(f"registar se ne učitava zbog drugog modula: {e}")
        self.assertIs(fams["diffusion"], diffusion)
        self.assertIn("diffusion_infer", encs)

    def test_oracle_decodes_and_is_deterministic(self):
        cases = [(lv, "mini", s) for lv in diffusion.LEVELS for s in (1, 2)]
        cases += [(lv, "std", 3) for lv in diffusion.LEVELS]
        for lv, tier, seed in cases:
            with self.subTest(level=lv, tier=tier, seed=seed):
                meta, prog, info = make(lv, tier, seed)
                rec = record.encode_kfl(meta, prog)
                m2, pl, used = record.decode(rec + b"\xff" * 64)
                self.assertEqual(used, len(rec))
                self.assertEqual(m2, meta)
                self.assertEqual(pl, prog.run(meta))
                _, prog2, info2 = make(lv, tier, seed)
                self.assertEqual(record.encode_kfl(meta, prog2), rec)
                self.assertEqual(info2, info)
                _, prog3, _ = make(lv, tier, seed + 100)
                self.assertNotEqual(record.encode_kfl(meta, prog3), rec)
                json.dumps(info)
                self.assertGreater(info["ideal_bits"], 0)
                self.assertLess(len(rec), meta.payload_bytes // 50)

    def test_films_are_meaningful(self):
        for lv in diffusion.LEVELS:
            for seed in range(4):
                meta, prog, info = make(lv, "mini", seed)
                pl = prog.run(meta)
                v, a = split(meta, pl)
                packed = (v[..., 0].astype(np.int64) << 16) | (v[..., 1].astype(np.int64) << 8) | v[..., 2]
                with self.subTest(level=lv, seed=seed):
                    for t in range(meta.frames):
                        self.assertGreater(len(np.unique(packed[t])), 3)          # nije konstanta
                    self.assertTrue(all((v[t] != v[t + 1]).any() for t in range(meta.frames - 1)))
                    S = state_from_video(meta, pl, info)
                    self.assertGreater(float(S[-1].std()), 0.5)                    # ne umire u konstantu
                    if diffusion.LEVELS[lv]["audio"] == "sums":
                        self.assertGreater(int(np.abs(a).max()), 1000)
                    else:
                        self.assertFalse(a.any())
                    self.assertEqual(info["params"]["palette"]["m"][0] % 2, 1)     # stanje čitljivo

    def test_replica_matches_vm(self):
        # NumPy replika dinamike = stanje pročitano iz videa (bez šuma), i za std
        for lv, tier, seed in (("smooth", "mini", 5), ("wave", "mini", 5), ("smooth", "std", 6),
                               ("wave", "std", 6), ("smooth", "mini", 7), ("smooth", "mini", 9)):
            meta, prog, info = make(lv, tier, seed)
            pp = info["params"]
            sim = diffusion.simulate(pp["kind"], tuple(pp["rule"]), pp["src"], initial_field(meta, info),
                                     meta.frames)
            with self.subTest(level=lv, tier=tier, seed=seed):
                self.assertTrue(np.array_equal(sim, state_from_video(meta, prog.run(meta), info)))
                if pp["kind"] == "wave":
                    self.assertTrue(sim.min() >= -128 and sim.max() <= 127)

    def test_init_types_and_levels_vary(self):
        kinds = {make("smooth", "mini", s)[2]["params"]["init"]["type"] for s in range(12)}
        self.assertEqual(kinds, {"hash", "patch"})
        rules = {tuple(make("wave", "mini", s)[2]["params"]["rule"]) for s in range(12)}
        self.assertGreater(len(rules), 6)
        for r in diffusion.WAVE_RULES + diffusion.SMOOTH_RULES:
            self.assertTrue((r[0] | r[1] | r[2]) & 1, r)                          # nesvodljiva pravila

    def test_noise_level(self):
        meta, prog, info = make("smooth_noise", "std", 5)
        K, p = info["params"]["exceptions"], info["params"]["p"]
        n = (meta.frames - 1) * meta.width * meta.height
        self.assertLess(abs(K - n * p), 5 * (n * p) ** 0.5)
        self.assertEqual(len(prog.exc_state.positions), K)
        self.assertTrue((prog.exc_state.values >= 1).all() and (prog.exc_state.values <= 15).all())
        clean = kfl.Program(video=prog.video, init=prog.init, step=prog.step).run(meta)
        self.assertNotEqual(clean, prog.run(meta))

    def test_generation_fast_on_std(self):
        for lv in diffusion.LEVELS:
            for seed in (11, 12):
                t = time.perf_counter()
                meta, prog, _ = make(lv, "std", seed)
                prog.run(meta)
                self.assertLess(time.perf_counter() - t, 2.0, lv)

    def test_builder_equivalences(self):
        meta = video_meta(13, 11, 40, 24, 0, 1)
        rng = np.random.default_rng(0)
        for _ in range(30):
            L = int(rng.integers(1, 40))
            p0, v = int(rng.integers(-200, 200)), int(rng.integers(-9, 10))
            vid, _ = split(meta, kfl.Program(video=diffusion.tri_node(p0, v, L)).run(meta))
            want = diffusion.tri_value(p0 + v * np.arange(meta.frames), L) & 255
            self.assertEqual(vid[:, 0, 0, 0].tolist(), want.tolist(), (p0, v, L))
        H, W = meta.height, meta.width
        for _ in range(30):
            h, w = int(rng.integers(1, H + 1)), int(rng.integers(1, W + 1))
            it = {"type": "patch", "y0": int(rng.integers(0, H - h + 1)), "x0": int(rng.integers(0, W - w + 1)),
                  "h": h, "w": w, "data": rng.integers(0, 256, h * w), "off": int(rng.integers(-5, 3)),
                  "bg": int(rng.integers(0, 3))}
            vid, _ = split(meta, kfl.Program(video=diffusion.init_node("smooth", it, H, W)).run(meta))
            self.assertTrue(np.array_equal(vid[0, :, :, 0], diffusion._patch_field(it, H, W) & 255), it)
        for R2 in diffusion.DISK_R2:
            self.assertEqual(diffusion.canon_r2(R2), R2)
            src = {"x0": 3, "vx": 2, "y0": 17, "vy": -1, "R2": R2}
            vid, _ = split(meta, kfl.Program(video=diffusion.disk_node(src, W, H)).run(meta))
            mask = diffusion.drive_mask("smooth", src, H, W, np.arange(meta.frames))
            self.assertTrue(np.array_equal(vid[..., 0], mask))
        vid, _ = split(meta, kfl.Program(video=diffusion.hash_init_node(77)).run(meta))
        self.assertTrue(np.array_equal(diffusion.hash_init_values(77, H, W), vid[0, :, :, 0]))

    def test_suite_make(self):
        from kfb import suite
        try:
            suite.families.load_all()
        except Exception as e:
            self.skipTest(f"registar se ne učitava zbog drugog modula: {e}")
        with tempfile.TemporaryDirectory() as d:
            man = suite.make(b"diffusion-test", Path(d), reps=1, tiers=("mini",), only={"diffusion"})
            self.assertEqual(len(man), len(diffusion.LEVELS))


class TestEncoder(unittest.TestCase):
    def check(self, meta, prog, budget):
        pl = prog.run(meta)
        t = time.perf_counter()
        rec = diffusion_infer.encode(meta, pl, budget)
        dt = time.perf_counter() - t
        self.assertIsNotNone(rec)
        m2, out, used = record.decode(rec + b"\x00" * 16)
        self.assertEqual(out, pl)
        self.assertEqual(used, len(rec))
        self.assertLess(dt, budget)
        return rec, dt

    def test_exact_and_reaches_kbar_mini(self):
        for lv in diffusion.LEVELS:
            for seed in (1000, 1001):
                with self.subTest(level=lv, seed=seed):
                    meta, prog, _ = make(lv, "mini", seed)
                    rec, _ = self.check(meta, prog, BUDGET["mini"])
                    self.assertEqual(rec[0], record.KFL_METHOD)
                    self.assertLessEqual(len(rec), len(record.encode_kfl(meta, prog)))

    def test_exact_std(self):
        for lv in diffusion.LEVELS:
            with self.subTest(level=lv):
                meta, prog, _ = make(lv, "std", 2000)
                rec, _ = self.check(meta, prog, BUDGET["std"])
                self.assertLessEqual(len(rec), len(record.encode_kfl(meta, prog)))

    def test_denser_noise(self):
        # 10x gušći šum: i dalje tačno, sa izuzecima stanja umesto neuspeha
        for seed in (1, 2):
            meta, prog, _ = make("smooth_noise", "mini", seed, p=1e-3)
            rec, _ = self.check(meta, prog, BUDGET["mini"])
            self.assertEqual(rec[0], record.KFL_METHOD)
            self.assertLess(len(rec), 1.3 * len(record.encode_kfl(meta, prog)))

    def test_budget_respected(self):
        meta, prog, _ = make("wave", "std", 3000)
        pl = prog.run(meta)
        for budget in (0.5, 2.0):
            t = time.perf_counter()
            rec = diffusion_infer.encode(meta, pl, budget)
            self.assertLess(time.perf_counter() - t, budget + 1.0)
            if rec is not None:
                self.assertEqual(record.decode(rec)[1], pl)

    def test_foreign_films(self):
        meta = TIERS["mini"]
        films = []
        for fam, lv in ((ca, "conway_p1e-3"), (ca, "random_rule_p1e-4"), (prng, "seed16")):
            obj, _ = fam.generate(np.random.default_rng(4), meta, fam.LEVELS[lv])
            films.append((meta, obj.run(meta)))
        obj, _ = ca.generate(np.random.default_rng(4), TIERS["std"], ca.LEVELS["conway_p0"])
        films.append((TIERS["std"], obj.run(TIERS["std"])))
        films.append((meta, np.random.default_rng(9).bytes(meta.payload_bytes)))
        for m, pl in films:
            t = time.perf_counter()
            rec = diffusion_infer.encode(m, pl, 20)
            self.assertLess(time.perf_counter() - t, 20)
            if rec is not None:
                self.assertEqual(record.decode(rec)[1], pl)
        if rec is not None:
            self.assertGreaterEqual(len(rec), meta.payload_bytes)   # nasumičan sadržaj se ne sažima


if __name__ == "__main__":
    unittest.main()
