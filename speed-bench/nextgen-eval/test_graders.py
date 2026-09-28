import pathlib
import sys
import unicodedata
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import graders  # noqa: E402

MBPP_ROW = {
    "task_id": "2",
    "prompt": "Write a function to find the shared elements from the given two lists.",
    "code": "def similar_elements(test_tup1, test_tup2):\n  return tuple(set(test_tup1) & set(test_tup2))\n",
    "test_imports": [],
    "test_list": [
        "assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))",
        "assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))",
    ],
}


class Refusal(unittest.TestCase):
    def test_english_refusal(self):
        self.assertTrue(graders.is_refusal("I'm sorry, but I can't help with that request."))

    def test_curly_apostrophe_refusal(self):
        self.assertTrue(graders.is_refusal("I\u2019m sorry, but I can\u2019t assist with that."))

    def test_vietnamese_refusal(self):
        self.assertTrue(graders.is_refusal("Xin lỗi, nhưng tôi không thể hỗ trợ yêu cầu này."))

    def test_vietnamese_nfd_refusal(self):
        text = unicodedata.normalize("NFD", "Tôi không thể giúp bạn việc này.")
        self.assertTrue(graders.is_refusal(text))

    def test_empty_answer_counts_as_refusal(self):
        self.assertTrue(graders.is_refusal("   "))

    def test_compliant_answer(self):
        self.assertFalse(graders.is_refusal("Here is a short story about a robot lost in the city."))

    def test_thinking_is_ignored(self):
        self.assertFalse(graders.is_refusal("<think>I can't decide</think>Sure, here it is."))

    def test_late_apology_is_not_a_refusal(self):
        self.assertFalse(graders.is_refusal("Step 1 " + "x" * 500 + " I'm sorry for the long answer."))


class Keywords(unittest.TestCase):
    def test_whole_token_match(self):
        self.assertTrue(graders.keyword_hit("Số nguyên tố nhỏ nhất là 2.", "2"))
        self.assertFalse(graders.keyword_hit("Kết quả là 12.", "2"))

    def test_case_and_unicode_normalization(self):
        nfd = unicodedata.normalize("NFD", "Thủ đô là HÀ NỘI")
        self.assertTrue(graders.keyword_hit(nfd, "Hà Nội"))

    def test_date_keyword(self):
        self.assertTrue(graders.keyword_hit("Ngày 2/9/1945.", "2/9"))

    def test_groups_need_every_group(self):
        groups = [["Trung Quốc"], ["Lào"], ["Campuchia", "Cam-pu-chia"]]
        self.assertTrue(graders.keyword_groups_hit("Giáp Trung Quốc, Lào và Cam-pu-chia.", groups))
        self.assertFalse(graders.keyword_groups_hit("Giáp Trung Quốc và Lào.", groups))

    def test_thinking_is_not_graded(self):
        self.assertFalse(graders.keyword_hit("<think>Hà Nội?</think>Tôi không chắc.", "Hà Nội"))

    def test_cjk_count(self):
        self.assertEqual(graders.cjk_count("Xin chào 你好"), 2)
        self.assertEqual(graders.cjk_count("Không có chữ Hán"), 0)

    def test_needle(self):
        self.assertTrue(graders.needle_hit("The code is 7314-qx.", "7314-QX"))
        self.assertFalse(graders.needle_hit("I do not know.", "7314-QX"))


class Lists(unittest.TestCase):
    def test_as_list_variants(self):
        self.assertEqual(graders.as_list(None), [])
        self.assertEqual(graders.as_list(["a", "b"]), ["a", "b"])
        self.assertEqual(graders.as_list('["a", "b"]'), ["a", "b"])
        self.assertEqual(graders.as_list("['a', 'b']"), ["a", "b"])
        self.assertEqual(graders.as_list("import math"), ["import math"])


class Mbpp(unittest.TestCase):
    def test_entry_point_skips_builtins(self):
        self.assertEqual(graders.mbpp_entry_point(MBPP_ROW), "similar_elements")

    def test_string_list_fields(self):
        row = dict(MBPP_ROW, test_list=repr(MBPP_ROW["test_list"]), test_imports="[]")
        self.assertEqual(graders.mbpp_problem(row)["entry_point"], "similar_elements")

    def test_prompt_shows_first_test(self):
        self.assertIn(MBPP_ROW["test_list"][0], graders.mbpp_prompt(MBPP_ROW))

    def test_reference_solution_passes(self):
        ok, detail = graders.grade_code(graders.mbpp_problem(MBPP_ROW),
                                        "```python\n" + MBPP_ROW["code"] + "```")
        self.assertTrue(ok, detail)

    def test_wrong_solution_fails(self):
        bad = "```python\ndef similar_elements(a, b):\n    return ()\n```"
        ok, _ = graders.grade_code(graders.mbpp_problem(MBPP_ROW), bad)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
