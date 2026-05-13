import unittest
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))

from src.models.baseline_supply_demand import baseline_price_linearized


class BaselineSupplyDemandTests(unittest.TestCase):
    def test_linearized_price_uses_low_elasticity_counterfactual(self) -> None:
        price = baseline_price_linearized(base_price=80.0, shortage_ratio=0.10, elasticity=-0.05)

        self.assertAlmostEqual(price, 240.0)

    def test_linearized_price_requires_negative_elasticity(self) -> None:
        with self.assertRaises(ValueError):
            baseline_price_linearized(base_price=80.0, shortage_ratio=0.10, elasticity=0.05)


if __name__ == "__main__":
    unittest.main()
