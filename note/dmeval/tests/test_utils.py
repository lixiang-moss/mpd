import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dmeval.utils import flatten_mapping, format_template, nanmean, nanstd, to_float


class TestUtils(unittest.TestCase):
    def test_format_template_missing_key(self):
        self.assertEqual(format_template("{x}", {"y": "1"}), "{x}")

    def test_to_float_basic(self):
        self.assertEqual(to_float(1), 1.0)
        self.assertEqual(to_float(1.5), 1.5)
        self.assertTrue(math.isnan(to_float(None)))

    def test_flatten_mapping(self):
        d = {"a": {"b": 1}, "c": 2}
        flat = flatten_mapping(d)
        self.assertEqual(flat["a.b"], 1)
        self.assertEqual(flat["c"], 2)
        flat_p = flatten_mapping(d, prefix="cfg")
        self.assertEqual(flat_p["cfg.a.b"], 1)
        self.assertEqual(flat_p["cfg.c"], 2)

    def test_nanmean_nanstd(self):
        self.assertTrue(math.isnan(nanmean([float("nan")])))
        self.assertEqual(nanmean([1.0, float("nan"), 3.0]), 2.0)
        self.assertTrue(math.isnan(nanstd([1.0])))
        self.assertAlmostEqual(nanstd([1.0, 3.0]), math.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
