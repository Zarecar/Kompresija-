"""Testovi jezgra KFB (VM, format zapisa, suite). python3 -m unittest tests.test_kfb_core -v"""
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from igc.film import video_meta  # noqa: E402
from kfb import encoders, families, kfl, record  # noqa: E402
from kfb.kfl import C, N, S, SUMS, T, X, Y, bits, bytes_lit, const, lut, op, words_lit  # noqa: E402

META = video_meta(8, 6, 5, 24, 8000, 1)


def video_of(prog, meta=META):
    pl = prog.run(meta)
    return np.frombuffer(pl[:meta.video_bytes], np.uint8).reshape(meta.frames, meta.height, meta.width, 3), \
        np.frombuffer(pl[meta.video_bytes:], "<i2")


def py_ref(name, a, b=None, c=None):
    M = 1 << 64

    def w(v):  # int64 wrap
        v %= M
        return v - M if v >= 1 << 63 else v
    if name == "ADD": return w(a + b)
    if name == "SUB": return w(a - b)
    if name == "MUL": return w(a * b)
    if name == "DIV": return 0 if b == 0 else w(a // b)
    if name == "MOD": return 0 if b == 0 else a % b
    if name == "AND": return a & b
    if name == "OR": return a | b
    if name == "XOR": return a ^ b
    if name == "SHL": return w(a << min(max(b, 0), 63))
    if name == "SHR": return a >> min(max(b, 0), 63)
    if name == "MIN": return min(a, b)
    if name == "MAX": return max(a, b)
    if name == "LT": return int(a < b)
    if name == "LE": return int(a <= b)
    if name == "EQ": return int(a == b)
    if name == "NE": return int(a != b)
    if name == "TRI": return (b - abs(a % (2 * b) - b)) if b > 0 else 0
    if name == "NEG": return w(-a)
    if name == "ABS": return w(abs(a))
    if name == "NOT": return ~a
    if name == "BOOL": return int(a != 0)
    if name == "SQ": return w(a * a)
    if name == "WHERE": return b if a != 0 else c
    if name == "CLIP": return min(max(a, b), c)
    raise KeyError(name)


class TestOps(unittest.TestCase):
    def eval_scalar(self, node):
        # izraz bez promenljivih -> isti bajt u celom videu; poredimo nisku 8 bita
        v, _ = video_of(kfl.Program(video=node))
        return int(v[0, 0, 0, 0])

    def test_binary_and_unary_vs_python(self):
        rnd = random.Random(7)
        vals = [0, 1, -1, 2, 7, -7, 63, 64, -17, 255, 256, 1 << 40, -(1 << 40), (1 << 62), -(1 << 63)]
        for name in ["ADD", "SUB", "MUL", "DIV", "MOD", "AND", "OR", "XOR", "SHL", "SHR", "MIN", "MAX",
                     "LT", "LE", "EQ", "NE", "TRI"]:
            for _ in range(40):
                a, b = rnd.choice(vals), rnd.choice(vals)
                if name == "TRI" and b > (1 << 61):
                    continue
                if name in ("DIV",) and a == -(1 << 63) and b == -1:
                    continue  # prelivanje int64 deljenja (dokumentovano: definisano NumPy-jem)
                got = self.eval_scalar(op(name, a, b))
                self.assertEqual(got, py_ref(name, a, b) & 255, (name, a, b))
        for name in ["NEG", "ABS", "NOT", "BOOL", "SQ"]:
            for a in vals:
                self.assertEqual(self.eval_scalar(op(name, a)), py_ref(name, a) & 255, (name, a))
        for a, b, c in [(0, 3, 4), (5, 3, 4), (-2, 9, 1)]:
            self.assertEqual(self.eval_scalar(op("WHERE", a, b, c)), py_ref("WHERE", a, b, c) & 255)
            self.assertEqual(self.eval_scalar(op("CLIP", a, b, c)), py_ref("CLIP", a, b, c) & 255)

    def test_isin_hash_isqrt(self):
        self.assertEqual(self.eval_scalar(op("ISIN", 64)), 127)
        self.assertEqual(self.eval_scalar(op("ISIN", 192)), (-127) & 255)
        self.assertEqual(self.eval_scalar(op("ISQRT", 1000000)), 1000 & 255)
        self.assertEqual(self.eval_scalar(op("ISQRT", -5)), 0)
        # HASH je deterministički i nenegativan; referenca splitmix64
        def sm(z):
            z = (z + 0x9E3779B97F4A7C15) % (1 << 64)
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) % (1 << 64)
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) % (1 << 64)
            return (z ^ (z >> 31)) >> 1
        for a in [0, 1, 12345, -1, 1 << 50]:
            self.assertEqual(self.eval_scalar(op("HASH", a)), sm(a % (1 << 64)) & 255)

    def test_coordinates_and_audio(self):
        v, a = video_of(kfl.Program(video=X + Y * 10 + C * 100 + T * 50, audio=N * 3 - 7))
        t, y, x, c = 3, 4, 5, 2
        self.assertEqual(v[t, y, x, c], (x + 10 * y + 100 * c + 50 * t) & 255)
        self.assertEqual(int(a[100]), 300 - 7)
        self.assertEqual(len(a), META.audio_samples)

    def test_state_nb_roll_and_sums(self):
        init = op("EQ", Y * 8 + X, 9)            # jedna živa ćelija (1,1)
        step = op("ROLLX", S, 1)                 # pomera se udesno, torus
        p = kfl.Program(init=init, step=step, video=S * 200, audio=SUMS)
        v, a = video_of(p)
        for t in range(META.frames):
            self.assertEqual(v[t, 1, (1 + t) % 8, 0], 200)
            self.assertEqual(int(v[t].sum()), 200 * 3)
        self.assertTrue((a == 1).all())
        nb = kfl.Program(init=init, step=S, video=op("NB8", S) + op("NB4", S) * 10)
        v, _ = video_of(nb)
        self.assertEqual(v[0, 0, 0, 0], 1)        # dijagonalni sused
        self.assertEqual(v[0, 0, 1, 0], 11)       # i NB8 i NB4

    def test_literals(self):
        b = [1, 0, 1, 1, 0, 0, 1]
        p = kfl.Program(video=bits(b, X) + bytes_lit(b"\x05\x06", Y) * 0 + lut([3, 4], C) * 0)
        v, _ = video_of(p)
        self.assertEqual(list(v[0, 0, :, 0]), b + [0])
        p = kfl.Program(video=bytes_lit(b"\x05\x06\x07", X), audio=words_lit([1000, 65535], N))
        v, a = video_of(p)
        self.assertEqual(list(v[0, 0, :4, 0]), [5, 6, 7, 5])
        self.assertEqual(int(a[1]), -1)

    def test_exceptions_all_kinds(self):
        init = const(0)
        p = kfl.Program(init=init, step=S, video=S, audio=const(0),
                        exc_state=kfl.Exceptions(np.array([0, 50]), np.array([7, 9])),
                        exc_video=kfl.Exceptions(np.array([3, 4, 700])),
                        exc_audio=kfl.Exceptions(np.array([2]), np.array([40000])))
        rec = record.encode_kfl(META, p)
        m, pl, used = record.decode(rec + b"\xff" * 9)
        self.assertEqual(used, len(rec))
        v, a = video_of(p)
        self.assertEqual(v[1, 0, 0, 0], 7)                 # pozicija 0 -> frejm 1, ćelija 0
        self.assertEqual(v[2, 0, 0, 0], 7)                 # STEP = S čuva izuzetak
        self.assertEqual(v[2, 0, 2, 1], 9)                 # pozicija 50 -> frejm 1 + 50//48, ćelija 2
        self.assertEqual(v[0, 0, 1, 0], 1)                 # video bajt 3 = piksel 1 kanal 0
        self.assertEqual(int(a[2]), np.uint16(40000).view(np.int16))


class TestFormat(unittest.TestCase):
    def test_roundtrip_and_text(self):
        progs = [kfl.Program(video=X * 4 + T * 3 + C * 85, audio=op("ISIN", (N * 14) >> 3) * 200),
                 kfl.Program(video=const(-(1 << 63)) + const((1 << 63) - 1) + 1000000),
                 kfl.Program(init=op("WHERE", (Y < 3) & (X < 3), bits([1, 0, 1] * 3, Y * 3 + X), 0),
                             step=lut([(0x1808 >> k) & 1 for k in range(18)], S * 9 + op("NB8", S)),
                             video=lut([0, 0, 0, 255, 255, 255], S * 3 + C))]
        for p in progs:
            b = p.to_bytes()
            q, pos = kfl.Program.from_bytes(b, 0, META)
            self.assertEqual(pos, len(b))
            self.assertEqual(q.to_bytes(), b)
            self.assertEqual(q.run(META), p.run(META))

    def test_rejects_malformed(self):
        bad = [b"", b"\x00", b"\x00\x20\x81", b"\x00\xff", b"\x40\x81", b"\x01\x81\x81",
               bytes([0, 0x3B, 0xff, 0xff, 0xff, 0x7f, 0x81]), bytes([2, 0x86])]
        for b in bad:
            with self.assertRaises(Exception, msg=b.hex()):
                q, _ = kfl.Program.from_bytes(b, 0, META)
                kfl.run(q, META)
        # list N u VIDEO sekciji, S bez stanja
        for node in (N, S):
            with self.assertRaises(ValueError):
                kfl.run(kfl.Program(video=node), META)
        with self.assertRaises(ValueError):   # prostorni operator u zvuku
            kfl.run(kfl.Program(video=const(0), audio=op("NB4", N)), META)

    def test_record_rejects_leaky_and_unknown(self):
        with self.assertRaises(ValueError):
            record.decode(bytes([0xFF]) + META.to_bytes() + b"x" * 8)
        with self.assertRaises(ValueError):
            record.decode(bytes([0x55]))

    def test_fuzz_no_hang(self):
        rnd = random.Random(3)
        for _ in range(400):
            b = bytes(rnd.randrange(256) for _ in range(rnd.randrange(1, 24)))
            try:
                q, _ = kfl.Program.from_bytes(b, 0, META)
                kfl.run(q, META)
            except Exception:
                pass


class TestSuite(unittest.TestCase):
    def test_make_and_oracles(self):
        from kfb import suite
        with tempfile.TemporaryDirectory() as d:
            man = suite.make(b"test-seed", Path(d), reps=1, tiers=("mini",), only={"ca", "prng", "control"})
            self.assertGreaterEqual(len(man), 7)
            again = Path(d) / "again"
            man2 = suite.make(b"test-seed", again, reps=1, tiers=("mini",), only={"ca", "prng"})
            for m in man2:   # determinizam (osim CSPRNG kontrole)
                self.assertEqual((Path(d) / "films" / f"{m['iid']}.film").read_bytes(),
                                 (again / "films" / f"{m['iid']}.film").read_bytes())

    def test_registries(self):
        f = families.load_all()
        e = encoders.load_all()
        self.assertIn("ca", f)
        self.assertIn("lzma", e)
        self.assertEqual(len(record.model_id()), 64)


if __name__ == "__main__":
    unittest.main()
