"""Testovi ffmpeg osnovnih linija (kfb.ffcodec, kfb.encoders.ffmpeg_enc).

    python3 -m unittest tests.test_kfb_ffmpeg -v
"""
import os
import secrets
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from igc.film import Meta, bytes_only_meta, put_varint, video_meta  # noqa: E402
from kfb import encoders, families, ffcodec, kfl, record, suite  # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None
NAMES = list(ffcodec.METHODS)
MINI = suite.TIERS["mini"]


def film(family: str, level: str, tier: str = "mini", seed: int = 0) -> tuple[Meta, bytes]:
    meta = suite.TIERS[tier]
    if family == "csprng":
        return meta, secrets.token_bytes(meta.payload_bytes)
    fam = families.load_all()[family]
    obj, _ = fam.generate(np.random.default_rng(seed), meta, fam.LEVELS[level])
    return meta, kfl.run(obj, meta) if isinstance(obj, kfl.Program) else bytes(obj)


def exact(rec: bytes, meta: Meta, payload: bytes) -> bool:
    m2, out, used = record.decode(rec + os.urandom(64))
    return m2 == meta and out == payload and used == len(rec)


def leftovers() -> set:
    return {p for p in os.listdir(tempfile.gettempdir()) if p.startswith("kfb_ff")}


@unittest.skipUnless(HAVE_FFMPEG, "nema ffmpeg-a")
class TestFormat(unittest.TestCase):
    def test_registry(self):
        encs = encoders.load_all()
        for n, m in ffcodec.METHODS.items():
            self.assertTrue(0x20 <= m <= 0x2F)
            self.assertIn(n, encs)
            self.assertFalse(encs[n].family_aware)

    def test_mkv_wrap_parse(self):
        pk = [bytes([i]) * (i * 37 + 1) for i in range(30)]
        cid, priv, got = ffcodec.mkv_parse(ffcodec.mkv_wrap("V_FFV1", MINI, b"xyz", pk))
        self.assertEqual((cid, priv, got), ("V_FFV1", b"xyz", pk))
        vs = ffcodec.pack_packets(b"ab", pk)
        self.assertEqual(ffcodec.unpack_packets(vs, 100), (b"ab", pk))

    def test_audio_modes(self):
        meta = MINI
        n = meta.audio_samples
        tone = (np.round(3000 * np.sin(np.arange(n) * 0.05))).astype("<i2").tobytes()
        noise = secrets.token_bytes(meta.audio_bytes)
        self.assertEqual(ffcodec.encode_audio(meta, bytes(meta.audio_bytes)), (ffcodec.A_SILENCE, b""))
        self.assertEqual(ffcodec.encode_audio(meta, noise)[0], ffcodec.A_RAW)
        mode, st = ffcodec.encode_audio(meta, tone)
        self.assertEqual(mode, ffcodec.A_FLAC)
        self.assertLess(len(st), len(tone) // 2)
        video = bytes(range(256)) * (meta.video_bytes // 256)
        vs = ffcodec.encode_video(ffcodec.METHODS["ffv1"], meta, video)
        silence = bytes(meta.audio_bytes)
        for a, mode_st in ((tone, (mode, st)), (noise, (ffcodec.A_RAW, noise)), (silence, (0, b""))):
            rec = ffcodec.pack(ffcodec.METHODS["ffv1"], meta, vs, *mode_st)
            self.assertTrue(exact(rec, meta, video + a))

    def test_default_encode_api(self):
        meta, pl = film("ca", "conway_p1e-3", seed=3)
        rec = ffcodec.encode(ffcodec.METHODS["ffv1"], meta, pl)
        self.assertIsNotNone(rec)
        self.assertTrue(exact(rec, meta, pl))
        self.assertIsNone(ffcodec.encode(ffcodec.METHODS["ffv1"], meta, pl[:-1]))

    def test_malformed_raises(self):
        meta, pl = film("ca", "conway_p0", seed=1)
        rec = encoders.load_all()["ffv1"].fn(meta, pl, 20)
        rng = np.random.default_rng(9)
        bad = [b"", bytes([0x2F]) + rec[1:], rec[:len(rec) // 2], rec[:5], rec[:1] + rng.bytes(200)]
        buf = bytearray(rec[:1] + meta.to_bytes())          # audio način 3
        put_varint(buf, 0)
        put_varint(buf, 3)
        bad.append(bytes(buf))
        buf = bytearray(rec[:1] + meta.to_bytes())          # nema videa iako ga meta traži
        put_varint(buf, 0)
        put_varint(buf, 0)
        bad.append(bytes(buf))
        buf = bytearray(rec[:1] + Meta(1 << 20, 1 << 20, 9, 24, 8000, 1, 2, 0).to_bytes())   # prevelik film
        put_varint(buf, 0)
        bad.append(bytes(buf))
        for b in bad:
            t = time.perf_counter()
            with self.assertRaises(ValueError):
                record.decode(b)
            self.assertLess(time.perf_counter() - t, 30)
        # slučajne izmene: greška ili deterministički rezultat, nikad zaglavljivanje
        for k in range(12):
            r = bytearray(rec)
            for _ in range(1 + k % 4):
                r[int(rng.integers(12, len(r)))] = int(rng.integers(256))
            t = time.perf_counter()
            try:
                a = record.decode(bytes(r))[1]
                self.assertEqual(a, record.decode(bytes(r))[1])
            except ValueError:
                pass
            self.assertLess(time.perf_counter() - t, 30)

    def test_non_rgb_or_changing_streams_rejected(self):
        # YUV/sivi tok ili promena dimenzija bi prošli kroz swscale (zavisi od SIMD-a) -> greška
        meta = MINI
        rng = np.random.default_rng(5)

        def h264(v, size, args):
            with tempfile.TemporaryDirectory() as d:
                src, dst = Path(d, "v.rgb"), Path(d, "v.h264")
                src.write_bytes(v)
                ffcodec._run(ffcodec._BASE + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-s", size, "-r", "24", "-i", src]
                             + args + ["-f", "h264", dst], 60)
                return dst.read_bytes()
        fb = meta.frame_bytes
        v = rng.integers(0, 256, meta.video_bytes, dtype=np.uint8).tobytes()
        rgb = ["-pix_fmt", "rgb24", "-c:v", "libx264rgb", "-qp", "0"]
        ok = ffcodec.pack(0x21, meta, h264(v, "32x32", rgb), 0, b"")
        self.assertTrue(exact(ok, meta, v + bytes(meta.audio_bytes)))
        yuv = h264(v, "32x32", ["-pix_fmt", "yuv444p", "-c:v", "libx264", "-qp", "0"])
        # 12 frejmova 32x32 + 12 frejmova 32x64: isti broj bajtova kao 24 x 32x32 posle skaliranja
        mixed = h264(v[:12 * fb], "32x32", rgb) + h264(v[:24 * fb], "32x64", rgb)
        for vs in (yuv, mixed):
            with self.assertRaisesRegex(ValueError, "dekodovan frejm"):
                record.decode(ffcodec.pack(0x21, meta, vs, 0, b""))
        # FLAC sa promenom učestanosti usred toka (ffmpeg bi tiho ubacio resampler)
        def flac(rate, n):
            m = Meta(0, 0, 0, 0, rate, 1, 2, n)
            a = rng.integers(-3000, 3000, n).astype("<i2").tobytes()
            mode, st = ffcodec.encode_audio(m, a, ["-compression_level", "5"])
            self.assertEqual(mode, ffcodec.A_FLAC)
            return st
        m = Meta(0, 0, 0, 0, 8000, 1, 2, 5424)
        with self.assertRaisesRegex(ValueError, "FLAC"):
            record.decode(ffcodec.pack(0x20, m, b"", ffcodec.A_FLAC, flac(8000, 4000) + flac(16000, 4000)))
        # predugačak varint u zaglavlju: brza greška, bez kvadratnog parsiranja
        t = time.perf_counter()
        with self.assertRaises(ValueError):
            record.decode(bytes([0x20]) + meta.to_bytes() + b"\xff" * 300000 + b"\x01")
        self.assertLess(time.perf_counter() - t, 1.0)

    def test_no_video_and_odd_shapes(self):
        encs = encoders.load_all()
        rng = np.random.default_rng(4)
        for meta in (bytes_only_meta(300), video_meta(33, 17, 3, 24, 8000, 2)):
            pl = np.clip(rng.normal(128, 20, meta.payload_bytes), 0, 255).astype(np.uint8).tobytes()
            for n in ("ffv1", "x264rgb"):
                rec = encs[n].fn(meta, pl, 10)
                self.assertIsNotNone(rec)
                self.assertTrue(exact(rec, meta, pl))


@unittest.skipUnless(HAVE_FFMPEG, "nema ffmpeg-a")
class TestEncoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encs = encoders.load_all()
        # privatni tempdir: drugi procesi (kfb.run, paralelni testovi) ne utiču na proveru ostataka
        cls.old_tempdir = tempfile.tempdir
        cls.tmp = tempfile.mkdtemp(prefix="kfbtest_ff_")
        tempfile.tempdir = cls.tmp
        cls.before = leftovers()

    @classmethod
    def tearDownClass(cls):
        tempfile.tempdir = cls.old_tempdir
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def check(self, name, meta, pl, budget, must=True):
        t = time.perf_counter()
        rec = self.encs[name].fn(meta, pl, budget)
        dt = time.perf_counter() - t
        self.assertLess(dt, budget + 5, name)
        if must:
            self.assertIsNotNone(rec, name)
        if rec is not None:
            self.assertEqual(rec[0], ffcodec.METHODS[name])
            self.assertTrue(exact(rec, meta, pl), name)
        return rec

    def test_mini_ca_prng_csprng(self):
        films = [film("ca", "random_rule_p1e-4", seed=11), film("prng", "seed16", seed=12),
                 film("csprng", "", seed=13), film("sprites", "k6", seed=14) if "sprites" in families.load_all()
                 else film("ca", "conway_p0", seed=14)]
        for name in NAMES:
            for i, (meta, pl) in enumerate(films):
                with self.subTest(codec=name, film=i):
                    # x265 (4:4:4) ponekad nije bez gubitaka; enkoder tada nalazi rezervu ili vraća None
                    rec = self.check(name, meta, pl, 20, must=name != "x265")
                    if rec is not None and i == 2:
                        self.assertGreater(len(rec), len(pl))     # CSPRNG se ne sme "komprimovati"

    def test_fresh_instances_ffv1_x264(self):
        for seed in (21, 22, 23):
            meta, pl = film("ca", "conway_p1e-3", seed=seed)
            for name in ("ffv1", "x264rgb"):
                self.check(name, meta, pl, 20)

    def test_std_instance(self):
        meta, pl = film("ca", "conway_p1e-3", "std", seed=31)
        for name in NAMES:
            with self.subTest(codec=name):
                self.check(name, meta, pl, 15, must=name in ("ffv1", "x264rgb", "apng"))

    def test_budget_respected(self):
        meta, pl = film("csprng", "", "std")
        for name in ("vp9", "av1", "x265"):
            t = time.perf_counter()
            rec = self.encs[name].fn(meta, pl, 2.0)
            self.assertLess(time.perf_counter() - t, 2.0 + 3.0, name)
            if rec is not None:
                self.assertTrue(exact(rec, meta, pl))
        self.assertIsNone(self.encs["ffv1"].fn(meta, pl, 0.1))

    def test_z_no_temp_files_left(self):
        self.assertLessEqual(leftovers(), self.before)


if __name__ == "__main__":
    unittest.main()
