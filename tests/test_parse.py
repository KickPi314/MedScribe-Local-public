"""The parser's withhold-don't-guess contract, on synthetic model replies.

No scans, no fixtures, no Ollama: every input below is a made-up string shaped
like a model reply, parsed with the real config/extract.toml.
"""
import unittest

from src.extract.config import load_extract_config
from src.extract.parse import (
    ABSENT,
    MISSING_KEY,
    NO_JSON,
    OK,
    OUT_OF_RANGE,
    UNPARSEABLE,
    parse_reply,
    parse_vitals,
)

CFG = load_extract_config()


def one(reply: str, name: str):
    return parse_reply(reply, CFG, (name,)).fields[name]


class VitalsTest(unittest.TestCase):
    def test_clean_reply_with_units_and_aliases(self):
        r = parse_vitals('{"wt": "14.2 kg", "temp": 99.1, "spo2": 98, "hr": "112"}', CFG)
        self.assertEqual(r.fields["wt_kg"].value, 14.2)
        self.assertEqual(r.fields["wt_kg"].source_key, "wt")  # found via alias
        self.assertEqual(r.fields["temp_f"].value, 99.1)
        self.assertEqual(r.fields["spo2_pct"].value, 98)
        self.assertIsInstance(r.fields["hr"].value, int)
        self.assertEqual(r.fields["hr"].value, 112)

    def test_out_of_range_is_withheld_not_rounded(self):
        # A saturation cannot exceed 100; 100.1 must not become 100.
        f = parse_vitals('{"spo2": "100.1"}', CFG).fields["spo2_pct"]
        self.assertEqual(f.status, OUT_OF_RANGE)
        self.assertIsNone(f.value)
        self.assertEqual(f.raw_value, "100.1")  # evidence kept for the reviewer

    def test_date_digits_read_as_weight_are_withheld(self):
        f = parse_vitals('{"wt": "2741202"}', CFG).fields["wt_kg"]
        self.assertEqual(f.status, OUT_OF_RANGE)
        self.assertIsNone(f.value)

    def test_words_are_unparseable(self):
        f = parse_vitals('{"hr": "ninety"}', CFG).fields["hr"]
        self.assertEqual(f.status, UNPARSEABLE)
        self.assertIsNone(f.value)

    def test_null_is_absent_and_omitted_is_missing(self):
        r = parse_vitals('{"mac_cm": null}', CFG)
        self.assertEqual(r.fields["mac_cm"].status, ABSENT)
        self.assertEqual(r.fields["abd_cm"].status, MISSING_KEY)
        self.assertTrue(r.fields["abd_cm"].failed)
        self.assertFalse(r.fields["mac_cm"].failed)  # absent is an answer, not a failure

    def test_non_json_reply_nulls_every_field(self):
        r = parse_vitals("I cannot read this image.", CFG)
        self.assertFalse(r.json_ok)
        self.assertTrue(all(f.status == NO_JSON and f.value is None for f in r.fields.values()))

    def test_json_wrapped_in_prose_is_salvaged(self):
        r = parse_vitals('Sure! ```json\n{"hr": 120}\n```', CFG)
        self.assertTrue(r.json_ok)
        self.assertTrue(r.json_salvaged)
        self.assertEqual(r.fields["hr"].value, 120)


class DateTest(unittest.TestCase):
    def test_written_date_is_reordered_to_iso(self):
        f = one('{"date": "27/4/2025"}', "date")
        self.assertEqual((f.status, f.value), (OK, "2025-04-27"))
        self.assertEqual(f.raw_value, "27/4/2025")

    def test_run_together_digits_are_not_split_by_guessing(self):
        f = one('{"date": "2742025"}', "date")
        self.assertNotEqual(f.status, OK)
        self.assertIsNone(f.value)


class ListTest(unittest.TestCase):
    def test_complaints_normalise_unit_spellings(self):
        f = one('{"complaints": [{"text": "fever", "duration": 2, "duration_unit": "HRS"},'
                ' {"text": "cough", "duration": null, "duration_unit": null}]}', "complaints")
        self.assertEqual(f.status, OK)
        self.assertEqual(f.value, [
            {"text": "fever", "duration": 2, "duration_unit": "hours"},
            {"text": "cough", "duration": None, "duration_unit": None},
        ])

    def test_unknown_unit_withholds_the_whole_list(self):
        # Lists are all-or-nothing: one bad item must not leave a partial list
        # that looks complete.
        f = one('{"complaints": [{"text": "cough", "duration": 3, "duration_unit": "days"},'
                ' {"text": "fever", "duration": 10, "duration_unit": "years"}]}', "complaints")
        self.assertEqual(f.status, UNPARSEABLE)
        self.assertIsNone(f.value)
        self.assertIn("whole list", f.reason)

    def test_empty_list_is_a_real_answer(self):
        f = one('{"investigations": []}', "investigations")
        self.assertEqual((f.status, f.value), (ABSENT, []))


if __name__ == "__main__":
    unittest.main()
