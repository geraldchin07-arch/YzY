import unittest
import numpy as np
import pandas as pd
from src.matsci_lens import (
    build_model, clean_xy, composition_features, evaluate_model,
    find_tx_column, parse_composition, prepare_dataset, predict_tx,
)

class CrystalCastTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.raw_x = pd.DataFrame({
            "Fe (at.%)": rng.uniform(50, 80, size=120),
            "B (at.%)": rng.uniform(5, 25, size=120),
            "Si (at.%)": rng.uniform(0, 20, size=120),
        })
        self.x = composition_features(self.raw_x)
        self.y = pd.Series(650 + 0.7 * self.raw_x["Fe (at.%)"] - 0.2 * self.raw_x["B (at.%)"] + rng.normal(0, 5, 120))

    def test_parse_formula_to_normalized_atomic_percent(self):
        result = parse_composition("Fe72B20Si8")
        self.assertAlmostEqual(sum(result.values()), 100.0)
        self.assertAlmostEqual(result["Fe"], 72.0)
        self.assertAlmostEqual(result["B"], 20.0)

    def test_parse_rejects_non_formula(self):
        self.assertEqual(parse_composition("unknown sample"), {})

    def test_detects_crystallization_target(self):
        raw = pd.DataFrame({"Fe": [70] * 10, "B": [20] * 10, "Tx [K]": [750] * 10})
        self.assertEqual(find_tx_column(raw), "Tx [K]")

    def test_extracts_numeric_element_columns(self):
        features = composition_features(self.raw_x)
        self.assertEqual(set(features.columns), {"Fe", "B", "Si"})

    def test_prepare_dataset_uses_composition_only(self):
        raw = self.raw_x.copy()
        raw["Tx [K]"] = self.y
        raw["Tg [K]"] = self.y - 30
        raw["Reference"] = ["paper"] * len(raw)
        x, y = prepare_dataset(raw)
        self.assertEqual(set(x.columns), {"Fe", "B", "Si"})
        self.assertEqual(len(x), len(raw))
        self.assertAlmostEqual(float(y.iloc[0]), float(raw["Tx [K]"].iloc[0]))

    def test_prepare_dataset_parses_formula_column(self):
        raw = pd.DataFrame({"Composition": ["Fe72B20Si8", "Fe70B20Si10"] * 6, "Crystallization temperature (K)": range(700, 712)})
        x, y = prepare_dataset(raw)
        self.assertEqual(set(x.columns), {"Fe", "B", "Si"})
        self.assertEqual(len(y), 12)

    def test_clean_drops_bad_target_and_empty_columns(self):
        x = pd.DataFrame({"a": range(11), "empty": [np.nan] * 11})
        y = pd.Series([1,2,3,4,5,6,7,8,9,10,np.nan])
        clean_x, clean_y = clean_xy(x, y)
        self.assertEqual(len(clean_x), 10)
        self.assertEqual(list(clean_x.columns), ["a"])

    def test_builds_supported_models(self):
        self.assertIsNone(build_model("Median Baseline"))
        self.assertIsNotNone(build_model("Ridge Regression"))
        self.assertIsNotNone(build_model("Extra Trees"))

    def test_evaluation_finite_for_all_models(self):
        for model in ["Median Baseline", "Ridge Regression", "Extra Trees"]:
            result = evaluate_model(self.x, self.y, model, random_state=7)
            self.assertTrue(np.isfinite([result.mae, result.rmse, result.r2]).all())
            self.assertEqual(len(result.y_test), len(result.y_pred))

    def test_evaluation_reproducible(self):
        first = evaluate_model(self.x, self.y, "Ridge Regression", random_state=18)
        second = evaluate_model(self.x, self.y, "Ridge Regression", random_state=18)
        self.assertAlmostEqual(first.mae, second.mae, places=9)
        np.testing.assert_allclose(first.y_pred, second.y_pred)

    def test_predict_tx_returns_finite_prediction(self):
        result, elements = predict_tx(self.x, self.y, "Ridge Regression", "Fe72B20Si8")
        self.assertTrue(np.isfinite(result))
        self.assertEqual(elements, ["B", "Fe", "Si"])

    def test_predict_tx_rejects_unknown_element(self):
        with self.assertRaises(ValueError):
            predict_tx(self.x, self.y, "Ridge Regression", "Fe70B20Si8Al2")

    def test_invalid_test_fraction_raises(self):
        with self.assertRaises(ValueError):
            evaluate_model(self.x, self.y, "Median Baseline", test_size=0.9)

if __name__ == "__main__":
    unittest.main()
