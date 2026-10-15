"""Testovi: python3 -m unittest discover -s tests -v"""
import os
import random
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from igc import ca, codec, mt19937 as mt, sources  # noqa: E402
from igc.film import Meta, bytes_only_meta, film_bytes, get_varint, parse_film, put_varint, video_meta  # noqa: E402
from igc.rangecoder import Decoder, Encoder  # noqa: E402

SMALL = video_meta(16, 12, 6, 24, 8000, 1)


class TestMT(unittest.TestCase):
    def test_numpy_legacy_and_seed_inversion(self):
        for s in [0, 1, 2**32 - 1, 123456789]:
            ref = np.random.RandomState(s).bytes(4001)
            self.assertEqual(codec._mt_from_initial(mt.init_genrand(s), 4001, 0), ref)
            pre = mt.untwist(mt.untemper(np.frombuffer(ref[:2496], dtype="<u4")).tolist())
            self.assertEqual(mt.seed_from_genrand_state(pre), s)

    def test_python_random_and_key_inversion(self):
        for s in [0, 7, 2**32 - 1, 2**64 + 3]:
            ref = random.Random(s).randbytes(3001)
            self.assertEqual(codec._mt_from_initial(mt.init_by_array(mt.int_to_key(s)), 3001, 1), ref)
            pre = mt.untwist(mt.untemper(np.frombuffer(ref[:2496], dtype="<u4")).tolist())
            self.assertEqual(mt.key_to_int(mt.key_from_by_array_state(pre)), s)

    def test_state_only_recovery(self):
        ref = random.Random().randbytes(6000)      # seed iz OS-a (624 reči)
        rec = codec.infer_encode(ref, weak_bits=4)
        self.assertEqual(rec[0], 1)
        self.assertEqual(codec.infer_decode(len(ref), rec, 0)[0], ref)


class TestCodec(unittest.TestCase):
    def roundtrip(self, method, meta, payload):
        rec = codec.encode(method, meta, payload)
        if rec is None:
            return None
        junk = os.urandom(50)
        m2, out, used = codec.decode(rec + junk)
        self.assertEqual(m2, meta)
        self.assertEqual(out, payload)
        self.assertEqual(used, len(rec), f"{method} nije samoograničavajući")
        return rec

    def test_all_methods_all_sources(self):
        for src in sources.SOURCES:
            payload, _ = sources.make(src, SMALL)
            for m in ["raw", "zlib", "lzma", "bz2", "xor_store", "xor_zlib", "cm"]:
                self.roundtrip(m, SMALL, payload)

    def test_infer_on_weak_sources(self):
        meta = video_meta(32, 24, 6, 24, 8000, 1)
        for src, kind in [("np_legacy", 2), ("py_random", 3), ("shake_weak20", 4)]:
            payload, _ = sources.make(src, meta)
            rec = self.roundtrip("infer", meta, payload)
            self.assertIsNotNone(rec)
            self.assertLess(len(rec), 32)

    def test_infer_rejects_csprng(self):
        meta = video_meta(32, 24, 6, 24, 8000, 1)
        payload, _ = sources.make("csprng", meta)
        self.assertIsNone(codec.encode("infer", meta, payload))

    def test_truncation_detected(self):
        payload, _ = sources.make("structured", SMALL)
        for m in ["raw", "zlib", "lzma", "bz2", "cm"]:
            rec = codec.encode(m, SMALL, payload)
            with self.assertRaises(Exception):
                codec.decode(rec[:-3])

    def test_film_serialization(self):
        payload, _ = sources.make("csprng", SMALL)
        self.assertEqual(parse_film(film_bytes(SMALL, payload)), (SMALL, payload))
        b = bytearray()
        for v in [0, 1, 127, 128, 2**40]:
            put_varint(b, v)
        pos, vals = 0, []
        for _ in range(5):
            v, pos = get_varint(bytes(b), pos)
            vals.append(v)
        self.assertEqual(vals, [0, 1, 127, 128, 2**40])


class TestCA(unittest.TestCase):
    def test_rule_roundtrip_and_inference(self):
        for p in [0.0, 1e-3, 0.3]:
            meta, payload, info = ca.life_film(H=32, W=40, T=20, soup=12, p=p)
            rec = codec.encode("rule", meta, payload)
            m2, out, used = codec.decode(rec + b"junk")
            self.assertEqual(out, payload)
            self.assertEqual(used, len(rec))
            if p == 0:
                self.assertLess(len(rec), 80)
                inferred = int.from_bytes(rec[1 + len(meta.to_bytes()):][:3], "little")
                self.assertEqual(inferred, ca.CONWAY)

    def test_rule_rejects_non_ca(self):
        payload, _ = sources.make("structured", SMALL)
        self.assertIsNone(codec.encode("rule", SMALL, payload))


class TestCoders(unittest.TestCase):
    def test_binary_coder(self):
        rng = np.random.default_rng(1)
        bits = (rng.random(20000) < 0.03).astype(int).tolist()
        enc = Encoder()
        for b in bits:
            enc.encode(b, 123)
        data = enc.finish()
        dec = Decoder(data + b"\x00\x11", 0)
        self.assertEqual([dec.decode(123) for _ in bits], bits)
        self.assertEqual(dec.pos, len(data))

    def test_enumerative_code(self):
        from e2_seed_search import enum_rank, enum_unrank
        rng = np.random.default_rng(3)
        bits = (rng.random(300) < 0.4).astype(int).tolist()
        r = enum_rank(bits)
        self.assertEqual(enum_unrank(r, 300, sum(bits)), bits)


class TestBounds(unittest.TestCase):
    def test_counting_bound(self):
        from e1_random_video import log2_p_bound
        self.assertEqual(log2_p_bound(1024, 8192), 0.0)
        self.assertEqual(log2_p_bound(1024, 8203), -10)

    def test_bytes_only_meta(self):
        m = bytes_only_meta(1024)
        self.assertEqual(m.payload_bytes, 1024)
        self.assertIsInstance(Meta.from_bytes(m.to_bytes())[0], Meta)


if __name__ == "__main__":
    unittest.main()
