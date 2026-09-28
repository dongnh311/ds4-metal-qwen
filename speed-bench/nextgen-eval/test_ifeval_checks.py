import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ifeval_checks  # noqa: E402

CASES = [
    ("punctuation:no_comma", {}, "No commas here.", True),
    ("punctuation:no_comma", {}, "One, two.", False),
    ("length_constraints:number_words", {"relation": "at least", "num_words": 3}, "one two three", True),
    ("length_constraints:number_words", {"relation": "less than", "num_words": 3}, "one two three", False),
    ("length_constraints:number_sentences", {"relation": "less than", "num_sentences": 3}, "One. Two.", True),
    ("length_constraints:number_sentences", {"relation": "at least", "num_sentences": 3}, "One. Two.", False),
    ("keywords:forbidden_words", {"forbidden_words": ["rock"]}, "I like jazz.", True),
    ("keywords:forbidden_words", {"forbidden_words": ["rock"]}, "Rock music!", False),
    ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* and **b**", True),
    ("detectable_format:number_highlighted_sections", {"num_highlights": 2}, "*a* only", False),
    ("keywords:frequency", {"relation": "at least", "keyword": "story", "frequency": 2},
     "A story. Another Story.", True),
    ("keywords:frequency", {"relation": "less than", "keyword": "story", "frequency": 2},
     "A story. Another Story.", False),
    ("startend:quotation", {}, '"quoted"', True),
    ("startend:quotation", {}, 'not "quoted"', False),
    ("change_case:english_lowercase", {}, "all lower case.", True),
    ("change_case:english_lowercase", {}, "Not all lower.", False),
    ("keywords:existence", {"keywords": ["alpha", "beta"]}, "Alpha and beta.", True),
    ("keywords:existence", {"keywords": ["alpha", "beta"]}, "Only alpha.", False),
    ("detectable_format:title", {}, "<<My Title>>\nBody", True),
    ("detectable_format:title", {}, "My Title\nBody", False),
    ("keywords:letter_frequency", {"letter": "a", "let_relation": "at least", "let_frequency": 3}, "banana", True),
    ("keywords:letter_frequency", {"letter": "a", "let_relation": "less than", "let_frequency": 3}, "banana", False),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "* one\n* two", True),
    ("detectable_format:number_bullet_lists", {"num_bullets": 2}, "- one\n- two\n- three", False),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] at [address]", True),
    ("detectable_content:number_placeholders", {"num_placeholders": 2}, "[name] only", False),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "First.\n***\nSecond.", True),
    ("length_constraints:number_paragraphs", {"num_paragraphs": 2}, "First.\n***\n\n***\nSecond.", False),
    ("startend:end_checker", {"end_phrase": "Is there anything else?"}, "Done. Is there anything else?", True),
    ("startend:end_checker", {"end_phrase": "Is there anything else?"}, "Is there anything else? Done.", False),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Hi.\nP.S. see you", True),
    ("detectable_content:postscript", {"postscript_marker": "P.S."}, "Hi. See you.", False),
    ("change_case:english_capital", {}, "ALL CAPS.", True),
    ("change_case:english_capital", {}, "Not caps.", False),
    ("change_case:capital_word_frequency", {"capital_relation": "at least", "capital_frequency": 2},
     "THE BIG dog", True),
    ("change_case:capital_word_frequency", {"capital_relation": "less than", "capital_frequency": 2},
     "THE BIG dog", False),
    ("detectable_format:json_format", {}, '```json\n{"a": 1}\n```', True),
    ("detectable_format:json_format", {}, 'Here: {"a": 1}', False),
    ("detectable_format:multiple_sections", {"section_spliter": "SECTION", "num_sections": 2},
     "SECTION 1\na\nSECTION 2\nb", True),
    ("detectable_format:multiple_sections", {"section_spliter": "SECTION", "num_sections": 2},
     "SECTION 1\na", False),
    ("detectable_format:constrained_response", {}, "My answer is yes.", True),
    ("detectable_format:constrained_response", {}, "Yes.", False),
]


class Checks(unittest.TestCase):
    def test_every_supported_id_has_cases(self):
        self.assertEqual({c[0] for c in CASES}, set(ifeval_checks.SUPPORTED))

    def test_cases(self):
        for iid, kwargs, response, expected in CASES:
            with self.subTest(iid=iid, response=response):
                self.assertEqual(ifeval_checks.check_instruction(iid, kwargs, response), expected)

    def test_none_kwargs_are_ignored(self):
        self.assertTrue(ifeval_checks.check_instruction(
            "punctuation:no_comma", {"num_words": None}, "fine"))

    def test_unsupported_id_raises(self):
        with self.assertRaises(KeyError):
            ifeval_checks.check_instruction("language:response_language", {"language": "kn"}, "x")

    def test_prompt_needs_every_instruction(self):
        row = {"instruction_id_list": ["punctuation:no_comma", "change_case:english_lowercase"],
               "kwargs": [{}, {}]}
        self.assertEqual(ifeval_checks.check_prompt(row, "all lower no comma"), (True, [True, True]))
        self.assertEqual(ifeval_checks.check_prompt(row, "Lower, not"), (False, [False, False]))


if __name__ == "__main__":
    unittest.main()
