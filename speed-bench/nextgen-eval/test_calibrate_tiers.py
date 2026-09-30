import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibrate_tiers  # noqa: E402


class Search(unittest.TestCase):
    def test_finds_chars_for_a_target_token_count(self):
        count = lambda chars: int(chars / 3.5)  # noqa: E731  deep ds4.c tokenizes at ~3.5 chars/token
        chars = calibrate_tiers.search(count, 480000, lo=1_000_000, hi=2_000_000, tol=1000)
        self.assertLessEqual(abs(count(chars) - 480000), 1000)

    def test_refuses_a_target_outside_the_bracket(self):
        with self.assertRaises(ValueError):
            calibrate_tiers.search(lambda c: c // 4, 960000, lo=1_000_000, hi=2_000_000, tol=1000)

    def test_counts_the_first_line_of_a_token_dump(self):
        self.assertEqual(calibrate_tiers.count_dump("[1, 2, 3]\n     1  a\n"), 3)


if __name__ == "__main__":
    unittest.main()
