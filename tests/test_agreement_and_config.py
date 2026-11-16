"""Two-pass agreement semantics and the loopback-only network guard."""
import tempfile
import unittest
from pathlib import Path

from src.extract.agreement import TwoPassField, values_agree
from src.extract.config import DEFAULT_EXTRACT_CONFIG_PATH, load_extract_config
from src.preprocess.config import ConfigError


class AgreementTest(unittest.TestCase):
    def test_int_and_float_agree_but_text_is_exact(self):
        self.assertTrue(values_agree(5, 5.0))
        self.assertFalse(values_agree("Paracetamol", "paracetamol"))  # no case folding, no fuzzy match

    def test_disagreement_has_no_winner(self):
        f = TwoPassField("hr", 112, 172, "ok", "ok")
        self.assertFalse(f.agree)
        self.assertFalse(hasattr(f, "value"))  # deliberately no single final value
        self.assertIn("pass A 112", f.detail)

    def test_empty_list_counts_with_nulls(self):
        f = TwoPassField("investigations", [], None, "absent", "no_json")
        self.assertTrue(f.both_null)
        self.assertFalse(f.both_read)
        self.assertFalse(f.status_agree)  # same emptiness, different reason


class LoopbackGuardTest(unittest.TestCase):
    def _load_with_host(self, host: str):
        text = DEFAULT_EXTRACT_CONFIG_PATH.read_text(encoding="utf8")
        self.assertEqual(text.count('host = "127.0.0.1"'), 1)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "extract.toml"
            p.write_text(text.replace('host = "127.0.0.1"', f'host = "{host}"'), encoding="utf8")
            return load_extract_config(p)

    def test_shipped_config_is_loopback(self):
        self.assertEqual(load_extract_config().get("ollama", "host"), "127.0.0.1")

    def test_non_loopback_host_is_rejected(self):
        for host in ("0.0.0.0", "192.168.1.20", "api.example.com"):
            with self.subTest(host=host), self.assertRaises(ConfigError):
                self._load_with_host(host)


if __name__ == "__main__":
    unittest.main()
