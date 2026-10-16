"""Testovi porodice sprites i enkodera sprites_infer. python3 -m unittest tests.test_kfb_sprites -v"""
import json
import secrets
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from igc.film import video_meta  # noqa: E402
from kfb import encoders, families, kfl, record  # noqa: E402
from kfb.encoders import sprites_infer  # noqa: E402
from kfb.families import ca, prng, sprites  # noqa: E402
from kfb.kfl import N, op, words_lit  # noqa: E402
from kfb.suite import TIERS  # noqa: E402

BUDGET = {"mini": 20, "std": 60}


def make(level, tier, seed):
    meta = TIERS[tier]
    prog, info = sprites.generate(np.random.default_rng(seed), meta, sprites.LEVELS[level])
    return meta, prog, info


def split(meta, payload):
    v = np.frombuffer(payload[:meta.video_bytes], np.uint8).reshape(meta.frames, meta.height, meta.width, 3)
    return v, np.frombuffer(payload[meta.video_bytes:], "<i2")


class TestFamily(unittest.TestCase):
    def test_module_interface(self):
        self.assertEqual(sprites.NAME, "sprites")
        self.assertTrue({"k2", "k6", "k3_noise"} <= set(sprites.LEVELS))
        enc = {e.name: e for e in sprites_infer.ENCODERS}
        self.assertIn("sprites_infer", enc)
        self.assertTrue(enc["sprites_infer"].family_aware)

    def test_registries(self):
        try:
            fams, encs = families.load_all(), encoders.load_all()
        except Exception as e:   # tuđ modul u izradi ne sme da obori ovaj test
            self.skipTest(f"registar se ne učitava zbog drugog modula: {e}")
        self.assertIs(fams["sprites"], sprites)
        self.assertIn("sprites_infer", encs)

    def test_oracle_decodes_and_is_deterministic(self):
        cases = [(lv, "mini", s) for lv in sprites.LEVELS for s in (1, 2)]
        cases += [(lv, "std", 3) for lv in sprites.LEVELS]
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
        for lv in sprites.LEVELS:
            for seed in range(4):
                meta, prog, info = make(lv, "mini", seed)
                v, a = split(meta, prog.run(meta))
                k = sprites.LEVELS[lv]["k"]
                packed = (v[..., 0].astype(np.int64) << 16) | (v[..., 1].astype(np.int64) << 8) | v[..., 2]
                cols = {tuple(o["color"]) for o in info["params"]["objects"]}
                seen = {(c >> 16, (c >> 8) & 255, c & 255) for c in np.unique(packed).tolist()}
                with self.subTest(level=lv, seed=seed):
                    self.assertEqual(len(cols), k)                      # različite boje objekata
                    self.assertGreaterEqual(len(cols & seen), k - 1)    # objekti se vide
                    for t in range(meta.frames):
                        self.assertGreater(len(np.unique(packed[t])), 1)
                    self.assertTrue((v[0] != v[-1]).any())              # kretanje
                    if sprites.LEVELS[lv]["audio"] == "tone":
                        self.assertGreater(int(np.abs(a).max()), 1000)
                    else:
                        self.assertFalse(a.any())

    def test_noise_level(self):
        meta, prog, info = make("k3_noise", "std", 5)
        K, p = info["params"]["exceptions"], info["params"]["p"]
        n = meta.video_bytes
        self.assertLess(abs(K - n * p), 5 * (n * p) ** 0.5)
        self.assertEqual(len(prog.exc_video.positions), K)
        self.assertTrue((prog.exc_video.values > 0).all() and (prog.exc_video.values < 256).all())
        clean = kfl.Program(video=prog.video, audio=prog.audio).run(meta)
        diff = np.frombuffer(clean, np.uint8) != np.frombuffer(prog.run(meta), np.uint8)
        self.assertEqual(int(diff.sum()), K)

    def test_generation_fast_on_std(self):
        for lv in sprites.LEVELS:
            t = time.perf_counter()
            meta, prog, _ = make(lv, "std", 11)
            prog.run(meta)
            self.assertLess(time.perf_counter() - t, 2.0, lv)

    def test_builder_equivalences(self):
        rng = np.random.default_rng(0)
        meta = video_meta(4, 4, 64, 64, 8000, 1)
        for _ in range(40):
            L, p0, v = int(rng.integers(1, 120)), int(rng.integers(-300, 300)), int(rng.integers(-9, 10))
            vid, _ = split(meta, kfl.Program(video=sprites.tri_node(p0, v, L)).run(meta))
            want = [sprites.tri_value(p0 + v * t, L) & 255 for t in range(meta.frames)]
            self.assertEqual(vid[:, 0, 0, 0].tolist(), want, (p0, v, L))
        for f, s, amp in [(48, 3, 100), (7, 0, 3), (200, 5, -90)]:
            raw = op("ISIN", (N * f) >> s) * amp
            a1 = split(meta, kfl.Program(video=kfl.const(0), audio=raw).run(meta))[1]
            node = sprites.tone_node(f, s, amp)
            a2 = split(meta, kfl.Program(video=kfl.const(0), audio=node).run(meta))[1]
            self.assertTrue(np.array_equal(a1, a2))
            self.assertLessEqual(sprites.blen(node), sprites.blen(raw))
        for r in range(1, 12):
            d = np.arange(-r, r + 1)
            d2 = d[:, None] ** 2 + d[None, :] ** 2
            for R2 in range(r * r, r * r + 2 * r + 1):
                self.assertTrue(np.array_equal(d2 <= R2, d2 <= sprites.canon_r2(r, R2)))

    def test_suite_make(self):
        from kfb import suite
        try:
            suite.families.load_all()
        except Exception as e:
            self.skipTest(f"registar se ne učitava zbog drugog modula: {e}")
        with tempfile.TemporaryDirectory() as d:
            man = suite.make(b"sprites-test", Path(d), reps=1, tiers=("mini",), only={"sprites"})
            self.assertEqual(len(man), len(sprites.LEVELS))


class TestEncoder(unittest.TestCase):
    def check(self, meta, prog, budget):
        pl = prog.run(meta)
        t = time.perf_counter()
        rec = sprites_infer.encode(meta, pl, budget)
        dt = time.perf_counter() - t
        self.assertIsNotNone(rec)
        m2, out, used = record.decode(rec + b"\x00" * 16)
        self.assertEqual(out, pl)
        self.assertEqual(used, len(rec))
        self.assertLess(dt, budget)
        return rec, dt

    def test_exact_and_reaches_kbar_mini(self):
        for lv in sprites.LEVELS:
            for seed in (1000, 1001, 1002):
                with self.subTest(level=lv, seed=seed):
                    meta, prog, _ = make(lv, "mini", seed)
                    rec, _ = self.check(meta, prog, BUDGET["mini"])
                    self.assertEqual(rec[0], record.KFL_METHOD)
                    self.assertLessEqual(len(rec), len(record.encode_kfl(meta, prog)))

    def test_exact_std(self):
        for lv in sprites.LEVELS:
            with self.subTest(level=lv):
                meta, prog, _ = make(lv, "std", 2000)
                rec, _ = self.check(meta, prog, BUDGET["std"])
                self.assertLessEqual(len(rec), len(record.encode_kfl(meta, prog)))

    def test_occlusion_regressions(self):
        # zaklonjen krug (vidi se samo deo, veličina/oblik dvosmisleni po okviru), pravougaonik
        # čija se puna visina nikad ne vidi, krug od koga se vidi samo kolona od 3 piksela,
        # objekat sa <= 5 vidljivih piksela po frejmu, z-red čije je jedino preklapanje
        # oštećeno šumom
        cases = [("k6", 7355851), ("k6", 9100100), ("k6", 20000046), ("k6", 20000630),
                 ("k6", 20000336), ("k6", 20000955), ("k3_noise", 20000106), ("k3_noise", 9100100)]
        for lv, seed in cases:
            with self.subTest(level=lv, seed=seed):
                meta, prog, _ = make(lv, "mini", seed)
                rec, _ = self.check(meta, prog, BUDGET["mini"])
                self.assertLessEqual(len(rec), len(record.encode_kfl(meta, prog)))

    def test_harder_than_levels(self):
        # više objekata, gušći šum: i dalje tačno, sa izuzecima umesto neuspeha
        meta = TIERS["mini"]
        for seed in (1, 2):
            prog, _ = sprites.generate(np.random.default_rng(seed), meta, {"k": 9, "p": 1e-2, "audio": "tone"})
            rec, _ = self.check(meta, prog, BUDGET["mini"])
            self.assertLess(len(rec), 2 * len(record.encode_kfl(meta, prog)))

    def test_non_tone_audio(self):
        meta, prog, info = make("k2", "mini", 7)
        noise = np.random.default_rng(1).integers(0, 65536, meta.audio_samples)
        prog.audio = words_lit(noise, N)
        rec, _ = self.check(meta, prog, BUDGET["mini"])
        self.assertLess(len(rec), 2 * meta.audio_samples + 400)

    def test_budget_respected(self):
        meta, prog, _ = make("k6", "std", 3000)
        pl = prog.run(meta)
        for budget in (0.3, 1.0):
            t = time.perf_counter()
            rec = sprites_infer.encode(meta, pl, budget)
            self.assertLess(time.perf_counter() - t, budget + 1.0)
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
        films.append((meta, secrets.token_bytes(meta.payload_bytes)))      # CSPRNG kontrola
        for m, pl in films:
            t = time.perf_counter()
            rec = sprites_infer.encode(m, pl, 20)
            self.assertLess(time.perf_counter() - t, 20)
            if rec is not None:
                self.assertEqual(record.decode(rec)[1], pl)
        self.assertGreaterEqual(len(rec), meta.payload_bytes)   # nasumičan sadržaj se ne sažima


if __name__ == "__main__":
    unittest.main()
