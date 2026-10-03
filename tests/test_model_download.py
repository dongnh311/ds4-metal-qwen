#!/usr/bin/env python3
"""Offline downloader checks with small, genuinely hashed artifact fixtures."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
Q2 = "DeepSeek-V4.1-Flash-Q2.gguf"
Q4 = "DeepSeek-V4.1-Flash-Q4.gguf"
PART1, PART2 = Q4 + ".part1", Q4 + ".part2"
VISION = "DeepSeek-V4.1-Flash-Vision.gguf"
EQ4 = "DeepSeek-V4.1-Flash-Q2-EngramQ4.gguf"
EQ4_PARTS = [EQ4 + ".part" + str(i) for i in range(1, 7)]
QWEN_Q2 = "Qwen3.8-Flash-Next-Q2.gguf"
QWEN_Q4 = "Qwen3.8-Flash-Next-Q4.gguf"
ARTIFACTS = {
    QWEN_Q2: (147207127040, "b1b93fa69aca5f187b0fb813aca8f3ec1beb5cf8cf0bd38cf041b93e0b6ccac9"),
    QWEN_Q4: (177280286720, "680944460a8cbe93ba8b6d7b6107213ffb7e22320bd913000e563ca0a0f25a8a"),
    Q2: (365713686528, "1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42"),
    Q4: (518596067328, "a5e2e2c3ada4b2e98d9f9e4b50f6d9c2a12c2c96f5da165c07e13aff9264984e"),
    PART1: (480000000000, "6442b1f9224079662c02003c0ef9ef6be6e2aff509510f681dab9e6cc41df246"),
    PART2: (38596067328, "7c3e10646c918eeaffbc39305a75ec96117450262c61454ff194cef00d7617f0"),
    VISION: (970555552, "cc283f032b3e8b8d78aeb5fccaa14e97b859b0c53aae3cd6bffa690ddf0e9e15"),
    EQ4: (267406761552, "311f35981bf14ef8e49968ef942e13263bf9011ffdc28c7b6d00a1de45d2f719"),
    EQ4_PARTS[0]: (45097156608, "e3a25ed2c0498eea95f4a7b5605f41eeeac9855486414f647b7ddc002e8fa635"),
    EQ4_PARTS[1]: (45097156608, "19673581014fb8ee8f0f9f466107a0168fd97859346474c00b05f7e45e5d8dcc"),
    EQ4_PARTS[2]: (45097156608, "1512b3d02258b979c8a5a5fec8fca473ae37c134c832daf71ef4546a1b6c5929"),
    EQ4_PARTS[3]: (45097156608, "3125d0fae3a2a79ee79c0bdda59e1fb9bdcb0067f49d201549a1a9affbe7d227"),
    EQ4_PARTS[4]: (45097156608, "936b8b67e7be53cb09e60d6ca969ef53c494d83b03cb7d84cd4325a4886aeec8"),
    EQ4_PARTS[5]: (41920978512, "d485a71b28ea3e9795f1849c06fb8101f60da26de50b58f222678e4b045d3c73"),
}


def payload(name):
    if name == Q4:
        return payload(PART1) + payload(PART2)
    if name == EQ4:
        return b"".join(payload(part) for part in EQ4_PARTS)
    return (name + "\n").encode()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve() / "source tree"
        self.root.mkdir()
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.out = self.root / "gguf files"
        self.script = self.root / "download_model.sh"
        text = (ROOT / "download_model.sh").read_text()
        for name, (size, sha) in ARTIFACTS.items():
            data = payload(name)
            # Several parts share a size, so swap each size and SHA pair as one unit.
            pinned = "expected_bytes=%d\n            expected_sha=%s\n" % (size, sha)
            self.assertEqual(text.count(pinned), 1, name)
            text = text.replace(pinned, "expected_bytes=%d\n            expected_sha=%s\n"
                                % (len(data), hashlib.sha256(data).hexdigest()))
        self.script.write_text(text)
        hf = self.bin / "hf"
        hf.write_text("""#!/usr/bin/env python3
import os
from pathlib import Path
import sys
args = sys.argv[1:]
assert args[0] == 'download'
assert args[1] == ('antirez/qwen3.8-flash-next-gguf' if args[2].startswith('Qwen')
                   else 'dongnhdev/DeepSeek-V4.1-Flash-Q2-EngramQ4-GGUF' if 'EngramQ4' in args[2]
                   else 'antirez/deepseek-v4.1-flash-gguf')
if os.environ.get('HF_LOG'):
    with open(os.environ['HF_LOG'], 'a') as log:
        log.write(args[2] + '\\n')
if os.environ.get('FAIL_DOWNLOAD'):
    sys.exit(7)
if os.environ.get('HF_RACE_SRC'):
    Path(os.environ['HF_RACE_DST']).write_bytes(Path(os.environ['HF_RACE_SRC']).read_bytes())
out = Path(args[args.index('--local-dir') + 1])
out.mkdir(parents=True, exist_ok=True)
(out / args[2]).write_bytes((args[2] + '\\n').encode())
""")
        hf.chmod(0o755)
        self.env = dict(os.environ, HOME=str(self.root / "home"), HF_TOKEN="",
                        PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
                        DS4_GGUF_DIR=str(self.out))

    def run_download(self, target, ok=True):
        result = subprocess.run(["sh", str(self.script), target], env=self.env,
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode == 0, ok, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_quants_vision_and_existing(self):
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q2"))
        link = self.root / "ds4flash.gguf"
        self.assertEqual(link.resolve(), self.out / Q2)
        self.assertIn("Already downloaded", self.run_download("ds41f-q2"))
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q4"))
        self.assertEqual(link.resolve(), self.out / Q4)
        self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))
        for name in (PART1, PART2, Q4 + ".assembling"):
            self.assertFalse((self.out / name).exists())
        self.assertIn("Already downloaded", self.run_download("ds41f-q4"))
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-vision"))
        self.assertEqual(link.resolve(), self.out / Q4)

    def test_truncated_and_corrupt_artifacts(self):
        self.out.mkdir()
        for target, name in (("ds41f-q2", Q2), ("ds41f-q4", Q4), ("ds41f-q2-eq4", EQ4),
                             ("ds41f-vision", VISION), ("qwen38-q2", QWEN_Q2), ("qwen38-q4k", QWEN_Q4)):
            with self.subTest(target=target):
                path = self.out / name
                path.write_bytes(b"short")
                self.assertIn("Incorrect file size", self.run_download(target, ok=False))
                self.assertFalse((self.root / "ds4flash.gguf").exists())
                path.write_bytes(b"x" * len(payload(name)))
                self.assertIn("Checksum mismatch", self.run_download(target, ok=False))
                self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_qwen_is_one_verified_file(self):
        for target, name in (("qwen38-q2", QWEN_Q2), ("qwen38-q4k", QWEN_Q4)):
            self.assertIn("Verifying SHA-256", self.run_download(target))
            self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / name)
            self.assertEqual((self.out / name).read_bytes(), payload(name))
        self.assertEqual({p.name for p in self.out.iterdir()}, {QWEN_Q2, QWEN_Q4})
        self.assertIn("Already downloaded", self.run_download("qwen38-iq2"))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / QWEN_Q2)
        self.assertNotIn("--ple", self.run_download("--help"))

    def downloaded(self):
        log = self.root / "hf.log"
        names = log.read_text().split() if log.exists() else []
        log.unlink(missing_ok=True)
        return names

    def fresh_out(self):
        """Empties the download directory and the fetch log, so a failed subtest cannot leak into the next."""
        shutil.rmtree(self.out, ignore_errors=True)
        self.out.mkdir()
        self.downloaded()

    def test_engram_q4_joins_six_verified_parts(self):
        self.env["HF_LOG"] = str(self.root / "hf.log")
        self.assertIn("Verifying SHA-256", self.run_download("ds41f-q2-eq4"))
        self.assertEqual(self.downloaded(), EQ4_PARTS)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / EQ4)
        self.assertEqual((self.out / EQ4).read_bytes(), payload(EQ4))
        self.assertEqual({p.name for p in self.out.iterdir()}, {EQ4, EQ4 + ".assembling.lock"})
        self.assertIn("Already downloaded", self.run_download("ds41f-q2-eq4"))
        self.assertEqual(self.downloaded(), [])
        self.assertIn("ds41f-q2-eq4", self.run_download("--help"))

    def test_engram_q4_resumes_without_fetching_joined_parts(self):
        self.env["HF_LOG"] = str(self.root / "hf.log")
        pending = self.out / (EQ4 + ".assembling")
        for joined in range(1, 6):
            for tail in (b"", payload(EQ4_PARTS[joined])[:7]):
                with self.subTest(joined=joined, tail=len(tail)):
                    self.fresh_out()
                    # Joined parts are gone; the next one may be half appended or not yet downloaded.
                    pending.write_bytes(b"".join(payload(p) for p in EQ4_PARTS[:joined]) + tail)
                    if tail:
                        (self.out / EQ4_PARTS[joined]).write_bytes(payload(EQ4_PARTS[joined]))
                    self.run_download("ds41f-q2-eq4")
                    self.assertEqual(self.downloaded(), EQ4_PARTS[joined + 1 if tail else joined:])
                    self.assertEqual((self.out / EQ4).read_bytes(), payload(EQ4))
                    self.assertEqual({p.name for p in self.out.iterdir()}, {EQ4, pending.name + ".lock"})

    def test_engram_q4_keeps_joined_data_when_a_part_copy_is_left_over(self):
        self.env["HF_LOG"] = str(self.root / "hf.log")
        pending = self.out / (EQ4 + ".assembling")
        # joined = parts in the assembly; strays = copies still on disk (another run, or a crash before unlink).
        for joined, strays in ((4, (2, 3)), (3, (3,)), (5, (1, 5))):
            with self.subTest(joined=joined, strays=strays):
                self.fresh_out()
                pending.write_bytes(b"".join(payload(p) for p in EQ4_PARTS[:joined]))
                for k in strays:
                    (self.out / EQ4_PARTS[k - 1]).write_bytes(payload(EQ4_PARTS[k - 1]))
                self.run_download("ds41f-q2-eq4")
                self.assertEqual(self.downloaded(), EQ4_PARTS[joined:])
                self.assertEqual((self.out / EQ4).read_bytes(), payload(EQ4))
                self.assertEqual({p.name for p in self.out.iterdir()}, {EQ4, pending.name + ".lock"})

    def test_engram_q4_removes_parts_left_by_an_overlapping_run(self):
        self.out.mkdir()
        done = self.root / "joined-by-other-run"
        done.write_bytes(payload(EQ4))
        # Another run finishes the file while this one downloads its parts.
        self.env.update(HF_RACE_SRC=str(done), HF_RACE_DST=str(self.out / EQ4))
        self.run_download("ds41f-q2-eq4")
        self.assertEqual((self.out / EQ4).read_bytes(), payload(EQ4))
        self.assertEqual({p.name for p in self.out.iterdir()}, {EQ4, EQ4 + ".assembling.lock"})
        # A finished file with parts still beside it is checked and the parts are removed.
        del self.env["HF_RACE_SRC"], self.env["HF_RACE_DST"]
        for name in (EQ4_PARTS[1], EQ4_PARTS[4]):
            (self.out / name).write_bytes(payload(name))
        self.assertIn("Already downloaded", self.run_download("ds41f-q2-eq4"))
        self.assertEqual({p.name for p in self.out.iterdir()}, {EQ4, EQ4 + ".assembling.lock"})

    def test_engram_q4_rejects_a_corrupt_part(self):
        self.run_download("ds41f-q2")
        self.out.joinpath(EQ4_PARTS[3]).write_bytes(b"x" * len(payload(EQ4_PARTS[3])))
        self.assertIn("Checksum mismatch", self.run_download("ds41f-q2-eq4", ok=False))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        self.assertFalse((self.out / EQ4).exists())
        self.assertFalse((self.out / (EQ4 + ".assembling")).exists())

    def test_failure_does_not_replace_link(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD"] = "1"
        self.run_download("ds41f-q4", ok=False)
        self.run_download("ds41f-vision", ok=False)
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)

    def test_partial_requires_explicit_cleanup(self):
        self.out.mkdir()
        (self.out / (Q2 + ".part")).write_bytes(b"partial")
        self.assertIn("cannot resume", self.run_download("ds41f-q2", ok=False))
        self.assertFalse((self.root / "ds4flash.gguf").exists())

    def test_q4_resumes_interrupted_assembly(self):
        self.run_download("ds41f-q2")
        self.env["FAIL_DOWNLOAD"] = "1"
        pending = self.out / (Q4 + ".assembling")
        for tail in (b"", payload(PART2)[:7], payload(PART2)):
            with self.subTest(tail=len(tail)):
                pending.write_bytes(payload(PART1) + tail)
                (self.out / PART2).write_bytes(payload(PART2))
                self.run_download("ds41f-q4")
                self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))
                self.assertFalse(pending.exists())
                self.assertFalse((self.out / PART2).exists())
                (self.out / Q4).unlink()

    def test_q4_rejects_corrupt_parts_and_assembly(self):
        self.run_download("ds41f-q2")
        link = self.root / "ds4flash.gguf"
        for name in (PART1, PART2):
            with self.subTest(part=name):
                for other in (PART1, PART2):
                    (self.out / other).write_bytes(payload(other))
                (self.out / name).write_bytes(b"x" * len(payload(name)))
                self.assertIn("Checksum mismatch", self.run_download("ds41f-q4", ok=False))
                self.assertEqual(link.resolve(), self.out / Q2)
                self.assertFalse((self.out / Q4).exists())
        (self.out / PART1).unlink()
        (self.out / PART2).write_bytes(payload(PART2))
        pending = self.out / (Q4 + ".assembling")
        pending.write_bytes(b"x" * len(payload(PART1)))
        self.env["FAIL_DOWNLOAD"] = "1"
        self.assertIn("Checksum mismatch", self.run_download("ds41f-q4", ok=False))
        self.assertEqual(link.resolve(), self.out / Q2)
        self.assertFalse((self.out / Q4).exists())
        self.assertTrue(pending.exists())

    def test_q4_no_space_preserves_resumable_parts(self):
        self.run_download("ds41f-q2")
        for name in (PART1, PART2):
            (self.out / name).write_bytes(payload(name))
        (self.bin / "sitecustomize.py").write_text(
            "import shutil\nfrom collections import namedtuple\n"
            "shutil.disk_usage = lambda _: namedtuple('usage', 'total used free')(1, 1, 0)\n")
        self.env["PYTHONPATH"] = str(self.bin)
        self.assertIn("Not enough disk space", self.run_download("ds41f-q4", ok=False))
        self.assertEqual((self.out / (Q4 + ".assembling")).read_bytes(), payload(PART1))
        self.assertEqual((self.out / PART2).read_bytes(), payload(PART2))
        self.assertEqual((self.root / "ds4flash.gguf").resolve(), self.out / Q2)
        del self.env["PYTHONPATH"]
        self.env["FAIL_DOWNLOAD"] = "1"
        self.run_download("ds41f-q4")
        self.assertEqual((self.out / Q4).read_bytes(), payload(Q4))

    def test_help_and_invalid_target(self):
        help_text = self.run_download("--help")
        self.assertIn("ds41f-q2", help_text)
        self.assertIn("ds41f-q4", help_text)
        self.assertIn("ds41f-vision", help_text)
        self.assertIn("Unknown model", self.run_download("nonexistent", ok=False))
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
