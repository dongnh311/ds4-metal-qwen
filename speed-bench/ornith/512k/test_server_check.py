"""Unit tests for server_check.py (python3 -m unittest, no model)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import server_check as sc


class KeyedDirTest(unittest.TestCase):
    def test_found(self):
        log = "x\nds4-server: keyed kv cache directory /kv/yarn-2\ny\n"
        self.assertEqual(sc.keyed_dir(log), "/kv/yarn-2")

    def test_absent(self):
        self.assertIsNone(sc.keyed_dir("ds4-server: listening\n"))


class CachedTest(unittest.TestCase):
    def test_reads_usage(self):
        resp = {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 96}}}
        self.assertEqual(sc.cached(resp), (96, 100))

    def test_missing_details_is_zero(self):
        self.assertEqual(sc.cached({"usage": {"prompt_tokens": 7}}), (0, 7))


class VerdictTest(unittest.TestCase):
    GOOD = [
        {"phase": "262k-store", "ctx": 262144, "dir": None, "cached": 0, "prompt": 30000},
        {"phase": "512k-cold", "ctx": 524288, "dir": "/kv/yarn-2", "cached": 0, "prompt": 30000},
        {"phase": "512k-restore", "ctx": 524288, "dir": "/kv/yarn-2", "cached": 29900, "prompt": 30000},
    ]

    def test_good(self):
        self.assertEqual(sc.verdict(self.GOOD), (True, []))

    def test_cross_rope_restore_fails(self):
        bad = [dict(p) for p in self.GOOD]
        bad[1]["cached"] = 29900
        ok, why = sc.verdict(bad)
        self.assertFalse(ok)
        self.assertIn("512k-cold restored", why[0])

    def test_no_restore_fails(self):
        bad = [dict(p) for p in self.GOOD]
        bad[2]["cached"] = 0
        self.assertFalse(sc.verdict(bad)[0])

    def test_unkeyed_dir_fails(self):
        bad = [dict(p) for p in self.GOOD]
        bad[1]["dir"] = None
        self.assertFalse(sc.verdict(bad)[0])


if __name__ == "__main__":
    unittest.main()
