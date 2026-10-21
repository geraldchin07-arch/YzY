import unittest
import numpy as np
import pandas as pd
from src.matsci_lens import build_model, clean_xy, evaluate_model

class MatSciLensTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.x = pd.DataFrame({"feature_a": rng.normal(size=120), "feature_b": rng.uniform(0, 1, size=120), "feature_missing": [np.nan if i % 17 == 0 else float(i) for i in range(120)]})
        self.y = pd.Series(4.0 * self.x["feature_a"] + self.x["feature_b"] + 10.0)

    def test_clean_xy_drops_invalid_target_and_empty_columns(self):
        x = pd.DataFrame({"a": [1,2,3,4,5,6,7,8,9,10], "empty": [np.nan]*10})
        clean_x, clean_y = clean_xy(x, pd.Series([1,2,3,4,5,6,7,8,9,np.nan]))
        self.assertEqual(len(clean_x), 9); self.assertEqual(list(clean_x.columns), ["a"]); self.assertEqual(len(clean_y), 9)

    def test_non_numeric_target_rows_are_dropped(self):
        clean_x, clean_y = clean_xy(pd.DataFrame({"a": range(10)}), pd.Series([1,2,"bad",4,5,6,7,8,9,10]))
        self.assertEqual(len(clean_x), 9); self.assertEqual(len(clean_y), 9)

    def test_raises_if_too_few_valid_rows(self):
        with self.assertRaises(ValueError): clean_xy(pd.DataFrame({"a": [1,2,3]}), pd.Series([1,np.nan,3]))

    def test_builds_supported_models(self):
        self.assertIsNone(build_model("Median Baseline")); self.assertIsNotNone(build_model("Ridge Regression")); self.assertIsNotNone(build_model("Extra Trees"))

    def test_unknown_model_raises(self):
        with self.assertRaises(ValueError): build_model("Mystery Model")

    def test_evaluation_produces_finite_metrics_for_all_models(self):
        for model in ["Median Baseline", "Ridge Regression", "Extra Trees"]:
            result = evaluate_model(self.x, self.y, model, random_state=7)
            self.assertTrue(np.isfinite([result.mae, result.rmse, result.r2]).all())
            self.assertGreater(len(result.y_test), 0); self.assertEqual(len(result.y_test), len(result.y_pred))

    def test_reproducible_split(self):
        first = evaluate_model(self.x, self.y, "Ridge Regression", random_state=18)
        second = evaluate_model(self.x, self.y, "Ridge Regression", random_state=18)
        self.assertAlmostEqual(first.mae, second.mae, places=9); np.testing.assert_allclose(first.y_pred, second.y_pred)

    def test_invalid_test_fraction_raises(self):
        with self.assertRaises(ValueError): evaluate_model(self.x, self.y, "Median Baseline", test_size=0.9)

if __name__ == "__main__": unittest.main()
