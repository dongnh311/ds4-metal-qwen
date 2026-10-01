import hashlib
import json
import pathlib
import tempfile
import unittest
import warnings

import numpy as np

import gguf_lite as g
import ista_hc_to_f16 as hc


def f32(values):
    return np.asarray(values, dtype=np.float32).tobytes()


class HcToF16Test(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)
        self.kv = {"general.architecture": (g.T_STR, "qwen4exp"), "general.alignment": (g.T_U32, 32),
                   "qwen4exp.block_count": (g.T_U32, 2)}
        self.exps = bytes(range(36))
        tensors = [
            {"name": "blk.0.ffn_down_exps.weight", "dims": [64, 2, 1], "type": 42, "nbytes": 36, "data": self.exps},
            {"name": "blk.0.ffn_gate_inp.weight", "dims": [32], "type": 0, "nbytes": 128, "data": f32([1.0] * 32)},
            {"name": "blk.0.hc_attn_up.weight", "dims": [32], "type": 0, "nbytes": 128,
             "data": f32([0.5, -2.0, 1.5, 0.25] * 8)},
            {"name": "blk.0.hc_ffn_inject.weight", "dims": [32], "type": 0, "nbytes": 128,
             "data": f32([1e-7] + [1.0] * 31)},
            {"name": "blk.0.hc_ffn_up.weight", "dims": [32], "type": 0, "nbytes": 128,
             "data": f32([69632.0] + [1.0] * 31)},
            {"name": "blk.0.hc_attn_down.weight", "dims": [32], "type": 30, "nbytes": 64, "data": bytes(range(64))},
            {"name": "blk.1.hc_attn_up.weight", "dims": [32], "type": 1, "nbytes": 64,
             "data": np.full(32, 0.75, dtype=np.float16).tobytes()},
        ]
        g.write(self.p / "in.gguf", self.kv, tensors, 32)
        self.prior_sources = {"ista": {"path": "/x/ista.gguf", "bytes": 47039860096}}
        (self.p / "in.gguf.json").write_text(json.dumps({"sources": self.prior_sources, "tensors": {
            "blk.0.ffn_gate_inp.weight": {"converted_from": "BF16"},
            "blk.0.hc_attn_up.weight": {"converted_from": "BF16"}}}))

    def tearDown(self):
        self.dir.cleanup()

    def run_tool(self):
        hc.main(["x", str(self.p / "in.gguf"), str(self.p / "out.gguf")])
        return g.Reader(self.p / "out.gguf"), json.loads((self.p / "out.gguf.json").read_text())

    def data(self, r, name):
        t = next(t for t in r.tensors if t["name"] == name)
        with open(r.path, "rb") as f:
            f.seek(t["abs"])
            return t, f.read(t["nbytes"])

    def test_hc_f32_to_f16(self):
        r, _ = self.run_tool()
        t, d = self.data(r, "blk.0.hc_attn_up.weight")
        self.assertEqual(t["type"], 1)
        self.assertEqual(list(np.frombuffer(d, dtype=np.float16)), [0.5, -2.0, 1.5, 0.25] * 8)

    def test_tiny_values_round_within_bound(self):
        r, _ = self.run_tool()
        t, d = self.data(r, "blk.0.hc_ffn_inject.weight")
        self.assertEqual(t["type"], 1)
        err = np.abs(np.frombuffer(d, dtype=np.float16).astype(np.float32) - np.frombuffer(f32([1e-7] + [1.0] * 31),
                                                                                          dtype=np.float32))
        self.assertLessEqual(float(err.max()), hc.rp.HC_F16_MAX_ABS_ERR)

    def test_overflow_stays_f32(self):
        r, _ = self.run_tool()
        t, d = self.data(r, "blk.0.hc_ffn_up.weight")
        self.assertEqual(t["type"], 0)
        self.assertEqual(d, f32([69632.0] + [1.0] * 31))

    def test_other_tensors_byte_identical(self):
        r, _ = self.run_tool()
        self.assertEqual(r.kv, g.Reader(self.p / "in.gguf").kv)
        self.assertEqual(self.data(r, "blk.0.ffn_down_exps.weight")[1], self.exps)
        t, d = self.data(r, "blk.0.ffn_gate_inp.weight")       # F32 but not an hc mixer weight
        self.assertEqual((t["type"], d), (0, f32([1.0] * 32)))
        self.assertEqual(self.data(r, "blk.0.hc_attn_down.weight")[1], bytes(range(64)))
        t, d = self.data(r, "blk.1.hc_attn_up.weight")          # already F16
        self.assertEqual((t["type"], d), (1, np.full(32, 0.75, dtype=np.float16).tobytes()))

    def test_manifest(self):
        r, man = self.run_tool()
        self.assertEqual(man["sources"]["input"]["path"], str(self.p / "in.gguf"))
        self.assertEqual(man["sources"]["input"]["sources"], self.prior_sources)   # provenance kept
        self.assertEqual(man["hc_f16_max_abs_err"], hc.rp.HC_F16_MAX_ABS_ERR)
        self.assertEqual(man["bytes"], (self.p / "out.gguf").stat().st_size)
        ent = man["tensors"]
        self.assertEqual(ent["blk.0.hc_attn_up.weight"]["converted_from"], "BF16")   # carried from the input
        self.assertEqual(ent["blk.0.hc_ffn_inject.weight"]["converted_from"], "F32")
        self.assertEqual(ent["blk.0.ffn_gate_inp.weight"]["converted_from"], "BF16")
        self.assertIsNone(ent["blk.0.ffn_down_exps.weight"]["converted_from"])
        self.assertEqual(ent["blk.0.hc_attn_up.weight"]["type"], 1)
        _, d = self.data(r, "blk.0.ffn_down_exps.weight")
        self.assertEqual(ent["blk.0.ffn_down_exps.weight"]["sha256"], hashlib.sha256(d).hexdigest())

    def test_nan_stays_f32(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self.assertIsNone(hc.rp.hc_f16(np.array([np.nan, 1.0], dtype=np.float32)))
            self.assertIsNone(hc.rp.hc_f16(np.array([np.inf, 1.0], dtype=np.float32)))

    def test_nothing_to_convert_refused(self):
        self.run_tool()
        with self.assertRaises(SystemExit):
            hc.main(["x", str(self.p / "out.gguf"), str(self.p / "again.gguf")])
        self.assertFalse((self.p / "again.gguf").exists())

    def test_usage_on_missing_arguments(self):
        with self.assertRaises(SystemExit):
            hc.main(["x"])

    def test_truncated_input_refused(self):
        data = (self.p / "in.gguf").read_bytes()
        (self.p / "in.gguf").write_bytes(data[:-40])
        with self.assertRaises(ValueError):
            self.run_tool()
        self.assertFalse((self.p / "out.gguf").exists())
        self.assertFalse((self.p / "out.gguf.partial").exists())


if __name__ == "__main__":
    unittest.main()
