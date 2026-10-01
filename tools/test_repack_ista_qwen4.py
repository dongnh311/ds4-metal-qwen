import json
import pathlib
import struct
import tempfile
import unittest

import numpy as np

import gguf_lite as g
import repack_ista_qwen4 as rp


def bf16(values):
    return (np.asarray(values, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()


class RepackTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)
        ista_kv = {"general.architecture": (g.T_STR, "qwen4exp"), "qwen4exp.block_count": (g.T_U32, 2),
                   "qwen4exp.attention.compress_ratios": (g.T_ARR, (g.T_U32, [0, 4])),
                   "split.count": (g.T_U16, 2), "split.no": (g.T_U16, 0), "split.tensors.count": (g.T_I32, 5)}
        self.hc = bf16([0.5, -2.0, 1.5, 0.25] * 8)
        ista_t = [
            {"name": "blk.0.ffn_down_exps.weight", "dims": [64, 2, 1], "type": 42, "nbytes": 36, "data": bytes(range(36))},
            {"name": "blk.0.ffn_gate_inp.weight", "dims": [32], "type": 30, "nbytes": 64, "data": bf16([1.0] * 32)},
            {"name": "blk.0.hc_attn_up.weight", "dims": [32], "type": 30, "nbytes": 64, "data": self.hc},
            {"name": "blk.0.hc_ffn_inject.weight", "dims": [32], "type": 30, "nbytes": 64,
             "data": bf16([1e-7] + [1.0] * 31)},   # 1e-7 is subnormal in F16: rounds by ~2e-8
            {"name": "blk.0.hc_ffn_up.weight", "dims": [32], "type": 30, "nbytes": 64,
             "data": bf16([70000.0] + [1.0] * 31)},   # past F16's 65504
            {"name": "output_hc_down.weight", "dims": [32], "type": 30, "nbytes": 64, "data": bf16([3.0] * 32)},
            {"name": "output_hc_up.weight", "dims": [32], "type": 30, "nbytes": 64, "data": self.hc},
        ]
        g.write(self.p / "ista.gguf", ista_kv, ista_t, 32)
        ivan_kv = {"general.architecture": (g.T_STR, "qwen4exp"), "qwen4exp.block_count": (g.T_U32, 3),
                   "qwen4exp.attention.compress_ratios": (g.T_ARR, (g.T_U32, [0, 4, 4])),
                   "general.alignment": (g.T_U32, 32), "qwen4exp.nextn_predict_layers": (g.T_U32, 1),
                   "qwen4exp.vocab_size": (g.T_U32, 248320), "qwen4exp.ple.seed": (g.T_U32, 1234),
                   "qwen4exp.ple.vocab_base": (g.T_U32, 20000000), "qwen4exp.ple.vocab_divisor": (g.T_U32, 128),
                   "qwen4exp.ple.row_count": (g.T_U64, 320001536), "qwen4exp.ple.row_dimension": (g.T_U32, 160),
                   "ds4.iq2.imatrix_sha256": (g.T_STR, "x")}
        ivan_t = [{"name": "blk.1.ffn_down_exps.weight", "dims": [32], "type": 0, "nbytes": 128, "data": b"\0" * 128},
                  {"name": "blk.2.nextn.eh_proj.weight", "dims": [32, 2], "type": 8, "nbytes": 68, "data": bytes(68)}]
        g.write(self.p / "ivan.gguf", ivan_kv, ivan_t, 32)

    def tearDown(self):
        self.dir.cleanup()

    def repack(self):
        rp.main(["x", str(self.p / "ista.gguf"), str(self.p / "ivan.gguf"), str(self.p / "out.gguf")])
        return g.Reader(self.p / "out.gguf"), json.loads((self.p / "out.gguf.json").read_text())

    def test_metadata(self):
        r, _ = self.repack()
        self.assertEqual(r.kv["qwen4exp.block_count"], (g.T_U32, 3))
        self.assertEqual(r.kv["qwen4exp.attention.compress_ratios"][1], (g.T_U32, [0, 4, 4]))
        self.assertEqual(r.kv["ds4.qwen4.down.logical_input"], (g.T_U32, 640))
        self.assertEqual(r.kv["ds4.qwen4.down.physical_input"], (g.T_U32, 640))
        for k in ("general.alignment", "qwen4exp.nextn_predict_layers", "qwen4exp.vocab_size",
                  "qwen4exp.ple.row_count", "qwen4exp.ple.seed"):
            self.assertIn(k, r.kv)
        for k in ("split.count", "split.no", "split.tensors.count", "ds4.iq2.imatrix_sha256"):
            self.assertNotIn(k, r.kv)

    def test_tensors_copied_and_grafted(self):
        r, man = self.repack()
        names = [t["name"] for t in r.tensors]
        self.assertIn("blk.2.nextn.eh_proj.weight", names)            # the MTP layer (index >= ISTA's count)
        self.assertNotIn("blk.1.ffn_down_exps.weight", names)         # Ivan's trunk is not taken
        t = next(t for t in r.tensors if t["name"] == "blk.0.ffn_down_exps.weight")
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            self.assertEqual(f.read(36), bytes(range(36)))
        self.assertEqual(man["tensors"]["blk.0.ffn_down_exps.weight"]["converted_from"], None)

    def test_router_to_f32(self):
        r, man = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.ffn_gate_inp.weight")
        self.assertEqual(t["type"], 0)
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            self.assertEqual(struct.unpack("<32f", f.read(128)), (1.0,) * 32)
        self.assertEqual(man["tensors"]["blk.0.ffn_gate_inp.weight"]["converted_from"], "BF16")

    def test_hc_to_f16_when_exact(self):
        r, _ = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.hc_attn_up.weight")
        self.assertEqual(t["type"], 1)
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            self.assertEqual(list(np.frombuffer(f.read(64), dtype=np.float16)), [0.5, -2.0, 1.5, 0.25] * 8)

    def test_output_hc_up_to_f16(self):
        # the output head's hc mixer takes the same F16/F32/Q8_0 kernel as the layers' hc up
        r, man = self.repack()
        t = next(t for t in r.tensors if t["name"] == "output_hc_up.weight")
        self.assertEqual(t["type"], 1)
        self.assertEqual(man["tensors"]["output_hc_up.weight"]["converted_from"], "BF16")

    def test_hc_tiny_rounding_to_f16(self):
        # values below F16's normal range round by at most ~3e-8; that is kept as F16
        r, _ = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.hc_ffn_inject.weight")
        self.assertEqual(t["type"], 1)
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(t["abs"])
            got = np.frombuffer(f.read(64), dtype=np.float16).astype(np.float32)
        want = np.frombuffer(bf16([1e-7] + [1.0] * 31), dtype=np.uint16).astype(np.uint32) << 16
        self.assertLessEqual(float(np.abs(got - want.view(np.float32)).max()), rp.HC_F16_MAX_ABS_ERR)

    def test_hc_overflow_falls_back_to_f32(self):
        r, _ = self.repack()
        t = next(t for t in r.tensors if t["name"] == "blk.0.hc_ffn_up.weight")
        self.assertEqual(t["type"], 0)

    def test_other_bf16_untouched(self):
        r, _ = self.repack()
        self.assertEqual(next(t for t in r.tensors if t["name"] == "output_hc_down.weight")["type"], 30)

    def test_truncated_source_refused(self):
        data = (self.p / "ista.gguf").read_bytes()
        (self.p / "ista.gguf").write_bytes(data[:-40])
        with self.assertRaises(ValueError):
            self.repack()
        self.assertFalse((self.p / "out.gguf").exists())


if __name__ == "__main__":
    unittest.main()
