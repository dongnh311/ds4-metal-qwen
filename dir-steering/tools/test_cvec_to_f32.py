import json
import math
import pathlib
import struct
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cvec_to_f32  # noqa: E402

W = 4
DIRS = {"direction.1": [2.0, 0.0, 0.0, 0.0], "direction.2": [0.0, 3.0, 4.0, 0.0],
        "direction.3": [0.0, 0.0, 0.0, 1.0]}


def _str(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def gguf(tensors, arch="controlvector", align=None, extra_array=False, kind=0):
    """A small GGUF v3 control vector: tensors maps a name to its float values."""
    kv = [(_str("general.architecture"), 8, _str(arch)),
          (_str("controlvector.layer_count"), 4, struct.pack("<I", len(tensors)))]
    if align:
        kv.append((_str("general.alignment"), 4, struct.pack("<I", align)))
    if extra_array:
        kv.append((_str("general.tags"), 9, struct.pack("<IQ", 8, 2) + _str("a") + _str("bc")))
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    head += b"".join(k + struct.pack("<I", t) + v for k, t, v in kv)
    payload = b""
    for name, values in tensors.items():
        head += _str(name) + struct.pack("<I", 1) + struct.pack("<Q", len(values))
        head += struct.pack("<IQ", kind, len(payload))
        payload += struct.pack("<%df" % len(values), *values)
        payload += b"\0" * (-len(payload) % (align or 32))
    head += b"\0" * (-len(head) % (align or 32))
    return head + payload


class Convert(unittest.TestCase):
    def test_row_n_is_direction_n_inside_the_range(self):
        rows, norms = cvec_to_f32.convert(gguf(DIRS), (2, 3), 4, W)
        self.assertEqual(rows[0], [0.0] * W)
        self.assertEqual(rows[1], [0.0] * W)  # direction.1 exists but is outside 2-3
        self.assertEqual(rows[2], [0.0, 0.6, 0.8, 0.0])
        self.assertEqual(rows[3], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(norms, {2: 5.0, 3: 1.0})

    def test_rows_are_unit_length(self):
        rows, norms = cvec_to_f32.convert(gguf(DIRS), (1, 1), 4, W)
        self.assertEqual(rows[1], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(norms, {1: 2.0})

    def test_alignment_and_array_keys_are_honoured(self):
        rows, _ = cvec_to_f32.convert(gguf(DIRS, align=64, extra_array=True), (2, 2), 4, W)
        self.assertEqual(rows[2], [0.0, 0.6, 0.8, 0.0])

    def test_refusals(self):
        cases = [
            (gguf(DIRS, arch="llama"), (1, 3), "controlvector"),
            (gguf(DIRS, kind=1), (1, 3), "expected f32"),
            (gguf({"direction.1": [1.0, 0.0]}), (1, 1), "expected f32 [4]"),
            (gguf(DIRS), (0, 3), "direction.0 is missing"),
            (gguf(DIRS), (2, 4), "outside"),
            (gguf(dict(DIRS, **{"direction.2": [0.0] * W})), (1, 3), "norm"),
        ]
        for data, layers, message in cases:
            with self.subTest(message=message), self.assertRaises(ValueError) as cm:
                cvec_to_f32.convert(data, layers, 4, W)
            self.assertIn(message, str(cm.exception))


class Cli(unittest.TestCase):
    def test_writes_rows_and_sidecar(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = pathlib.Path(d) / "v.gguf", pathlib.Path(d) / "v.f32"
            src.write_bytes(gguf(DIRS))
            cvec_to_f32.main(["--in", str(src), "--out", str(out), "--layers", "2-3",
                              "--n-layers", "4", "--width", "4"])
            values = struct.unpack("<16f", out.read_bytes())
            meta = json.loads((pathlib.Path(d) / "v.f32.json").read_text())
        self.assertEqual(values[:8], (0.0,) * 8)
        self.assertAlmostEqual(math.hypot(*values[8:12]), 1.0, places=6)
        self.assertEqual(meta["layers"], [2, 3])
        self.assertEqual(meta["source_norms"], {"2": 5.0, "3": 1.0})
        self.assertEqual(len(meta["source_sha256"]), 64)
        self.assertEqual(len(meta["output_sha256"]), 64)

    def test_bad_input_exits_with_a_message(self):
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d) / "v.gguf"
            src.write_bytes(gguf(DIRS, arch="llama"))
            with self.assertRaises(SystemExit) as cm:
                cvec_to_f32.main(["--in", str(src), "--out", str(pathlib.Path(d) / "v.f32"),
                                  "--layers", "1-3", "--n-layers", "4", "--width", "4"])
        self.assertIn("controlvector", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
