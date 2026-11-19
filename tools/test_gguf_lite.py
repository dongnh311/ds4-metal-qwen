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
        self.assertEqual(g.nbytes(6, [640, 2560]), 640 // 32 * 22 * 2560)      # Q5_0 (ffn_down_shexp)
        self.assertEqual(g.nbytes(11, [6144, 2560]), 6144 // 256 * 110 * 2560)  # Q3_K (ssm_out)
        with self.assertRaises(ValueError):
            g.nbytes(99, [32])

    def test_nocache_write_is_identical(self):
        src = self.p / "src.bin"
        src.write_bytes(bytes(range(256)) * 4)
        def tensors():
            return [{"name": "c", "dims": [64], "type": 0, "nbytes": 256, "src": (str(src), 0)},
                    {"name": "t", "dims": [320, 2], "type": 12, "nbytes": 2 * 192, "trim": (str(src), 0, 432, 192)}]
        src.write_bytes(bytes(i & 0xFF for i in range(2 * 432)))
        a = g.write(self.p / "a.gguf", {}, tensors(), 32)
        b = g.write(self.p / "b.gguf", {}, tensors(), 32, nocache=True)
        self.assertEqual(a, b)
        self.assertEqual((self.p / "a.gguf").read_bytes(), (self.p / "b.gguf").read_bytes())

    def test_q4k_short_final_block(self):
        self.assertEqual(g.q4k_row_bytes(768), 432)
        self.assertEqual(g.q4k_row_bytes(640), 368)
        self.assertEqual(g.q4k_row_bytes(320), 192)
        self.assertEqual(g.q4k_row_bytes(704), 400)
        self.assertEqual(g.q4k_row_bytes(672), 0)
        self.assertEqual(g.nbytes(12, [640, 3, 2]), 368 * 6)
        self.assertEqual(g.nbytes(12, [768, 3]), 432 * 3)
        with self.assertRaises(ValueError):
            g.nbytes(12, [672, 3])

    def test_q2k_short_final_block(self):
        self.assertEqual(g.q2k_row_bytes(768), 252)
        self.assertEqual(g.q2k_row_bytes(640), 216)
        self.assertEqual(g.q2k_row_bytes(704), 248)
        self.assertEqual(g.q2k_row_bytes(576), 216)
        self.assertEqual(g.q2k_row_bytes(672), 0)
        self.assertEqual(g.nbytes(10, [640, 3, 2]), 216 * 6)
        self.assertEqual(g.nbytes(10, [768, 3]), 252 * 3)
        with self.assertRaises(ValueError):
            g.nbytes(10, [672, 3])

    def test_q2k_trim_rows(self):
        # one padded row (768 physical) with nonzero padding
        row = bytes((i * 7 + 3) & 0xFF for i in range(252))
        out = g.q2k_trim_rows(row, 252, 216, 640)
        self.assertEqual(len(out), 216)
        self.assertEqual(out[:168], row[:168])          # two full blocks
        self.assertEqual(out[168:176], row[168:176])    # real scales
        self.assertEqual(out[176:180], row[248:252])    # d/dmin moved into the header tail
        self.assertEqual(out[184:216], row[184:216])    # real qs bytes
        self.assertEqual(out[180:184], b"\0\0\0\0")
        # 704 values keeps 12 scales and all 64 qs bytes, d/dmin at 12
        out2 = g.q2k_trim_rows(row, 252, 248, 704)
        self.assertEqual(out2[180:184], row[248:252])
        self.assertEqual(out2[184:248], row[184:248])

    def test_not_gguf(self):
        (self.p / "x").write_bytes(b"NOPE" + bytes(20))
        with self.assertRaises(ValueError):
            g.Reader(self.p / "x")


if __name__ == "__main__":
    unittest.main()
