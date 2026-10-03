import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

import gguf_lite as g
import qwen4_trim_down_pad as tr

HERE = pathlib.Path(__file__).resolve().parent


def padded_rows(rows, seed):
    """rows x 432 bytes: three Q4_K blocks per row; the padding bytes are nonzero too."""
    return bytes((i * 7 + seed) & 0xFF for i in range(rows * 432))


class TrimTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = pathlib.Path(self.dir.name)
        self.kv = {"general.architecture": (g.T_STR, "qwen4exp"),
                   "qwen4exp.expert_feed_forward_length": (g.T_U32, 640)}
        # 2 experts x 3 output rows per expert: dims [768, 3, 2]
        self.down = padded_rows(6, 1)
        self.other = bytes(range(64))
        self.tensors = [
            {"name": "blk.0.ffn_down_exps.weight", "dims": [768, 3, 2], "type": 12, "nbytes": 6 * 432,
             "data": self.down},
            {"name": "blk.0.attn_norm.weight", "dims": [16], "type": 0, "nbytes": 64, "data": self.other},
            {"name": "blk.1.ffn_down_exps.weight", "dims": [640, 3, 2], "type": 8, "nbytes": 6 * 20 * 34,
             "data": bytes(6 * 20 * 34)},   # MTP-like Q8_0 down: copied as is
        ]
        g.write(self.p / "in.gguf", self.kv, self.tensors, 32)

    def tearDown(self):
        self.dir.cleanup()

    def run_tool(self, *args):
        return subprocess.run([sys.executable, str(HERE / "qwen4_trim_down_pad.py"), *map(str, args)],
                              capture_output=True, text=True)

    def test_trim_rows(self):
        self.assertEqual(tr.trim_rows(self.down, 432, 368),
                         b"".join(self.down[r * 432:r * 432 + 368] for r in range(6)))

    def test_converts_q4k_down_only(self):
        r = self.run_tool(self.p / "in.gguf", self.p / "out.gguf")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = g.Reader(self.p / "out.gguf")
        by = {t["name"]: t for t in out.tensors}
        down = by["blk.0.ffn_down_exps.weight"]
        self.assertEqual((down["dims"], down["type"], down["nbytes"]), ([640, 3, 2], 12, 6 * 368))
        with open(self.p / "out.gguf", "rb") as f:
            f.seek(down["abs"])
            self.assertEqual(f.read(down["nbytes"]), tr.trim_rows(self.down, 432, 368))
            f.seek(by["blk.0.attn_norm.weight"]["abs"])
            self.assertEqual(f.read(64), self.other)
        self.assertEqual(by["blk.1.ffn_down_exps.weight"]["dims"], [640, 3, 2])
        self.assertEqual(out.kv, self.kv)

    def test_manifest(self):
        self.run_tool(self.p / "in.gguf", self.p / "out.gguf")
        m = json.loads((self.p / "out.gguf.json").read_text())
        t = m["tensors"]["blk.0.ffn_down_exps.weight"]
        self.assertEqual((t["bytes"], t["trimmed_from"]), (6 * 368, [768, 3, 2]))
        self.assertIsNone(m["tensors"]["blk.0.attn_norm.weight"]["trimmed_from"])
        self.assertEqual(m["bytes"], (self.p / "out.gguf").stat().st_size)

    def test_already_trimmed_refused(self):
        self.run_tool(self.p / "in.gguf", self.p / "out.gguf")
        r = self.run_tool(self.p / "out.gguf", self.p / "again.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("nothing to trim", r.stderr)
        self.assertFalse((self.p / "again.gguf").exists())

    def test_non_qwen4_refused(self):
        kv = dict(self.kv, **{"general.architecture": (g.T_STR, "llama")})
        g.write(self.p / "llama.gguf", kv, self.tensors, 32)
        r = self.run_tool(self.p / "llama.gguf", self.p / "o.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("qwen4exp", r.stderr)
        self.assertFalse((self.p / "o.gguf").exists())

    def test_bad_ff_refused(self):
        kv = dict(self.kv, **{"qwen4exp.expert_feed_forward_length": (g.T_U32, 672)})
        g.write(self.p / "bad.gguf", kv, self.tensors, 32)
        r = self.run_tool(self.p / "bad.gguf", self.p / "o.gguf")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("multiple of 64", r.stderr)

    def test_usage(self):
        r = self.run_tool()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("usage", r.stderr)


if __name__ == "__main__":
    unittest.main()
