import pathlib
import tempfile
import unittest

import gguf_lite as g


class GgufLiteTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip(self):
        kv = {"general.architecture": (g.T_STR, "qwen4exp"),
              "qwen4exp.block_count": (g.T_U32, 48),
              "qwen4exp.attention.compress_ratios": (g.T_ARR, (g.T_U32, [0, 4] * 24)),
              "tokenizer.ggml.tokens": (g.T_ARR, (g.T_STR, ["a", "bc", ""]))}
        blob = bytes(range(256)) * 2
        tensors = [{"name": "t.q8", "dims": [32, 4], "type": 8, "nbytes": 4 * 34, "data": blob[:136]},
                   {"name": "t.f32", "dims": [3], "type": 0, "nbytes": 12, "data": blob[:12]}]
        shas = g.write(self.p / "a.gguf", kv, tensors, 32)
        r = g.Reader(self.p / "a.gguf")
        self.assertEqual(r.kv, kv)
        self.assertEqual([t["name"] for t in r.tensors], ["t.q8", "t.f32"])
        self.assertEqual([t["offset"] % 32 for t in r.tensors], [0, 0])
        with open(self.p / "a.gguf", "rb") as f:
            f.seek(r.tensors[1]["abs"])
            self.assertEqual(f.read(12), blob[:12])
        self.assertEqual(set(shas), {"t.q8", "t.f32"})

    def test_copy_from_source(self):
        src = self.p / "src.bin"
        src.write_bytes(b"x" * 100 + bytes(range(64)))
        t = [{"name": "c", "dims": [16], "type": 0, "nbytes": 64, "src": (str(src), 100)}]
        g.write(self.p / "b.gguf", {}, t, 32)
        r = g.Reader(self.p / "b.gguf")
        with open(self.p / "b.gguf", "rb") as f:
            f.seek(r.tensors[0]["abs"])
            self.assertEqual(f.read(64), bytes(range(64)))

    def test_nbytes(self):
        self.assertEqual(g.nbytes(21, [2560, 640, 512]), 2560 // 256 * 110 * 640 * 512)
        self.assertEqual(g.nbytes(20, [640, 2560]), 640 // 32 * 18 * 2560)
        self.assertEqual(g.nbytes(42, [640, 2560]), 640 // 64 * 18 * 2560)
        with self.assertRaises(ValueError):
            g.nbytes(99, [32])

    def test_not_gguf(self):
        (self.p / "x").write_bytes(b"NOPE" + bytes(20))
        with self.assertRaises(ValueError):
            g.Reader(self.p / "x")


if __name__ == "__main__":
    unittest.main()
