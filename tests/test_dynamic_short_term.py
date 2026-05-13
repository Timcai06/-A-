import unittest
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))

import pandas as pd

from src.models.dynamic_short_term import BehavioralParameters, PhysicalAssumptions, simulate_dynamic_model


class DynamicShortTermTests(unittest.TestCase):
    def _flat_assumptions(self) -> PhysicalAssumptions:
        return PhysicalAssumptions(
            base_supply=100.0,
            base_demand=100.0,
            base_elasticity=-0.05,
            long_elasticity=-0.10,
            supply_interruption=0.0,
            spr_max_release=0.0,
            spr_delay_days=0,
            spr_ramp_days=1,
            route_start_day=0,
            route_max_capacity=0.0,
            route_ramp_days=1,
            commercial_inventory=10.0,
            inventory_daily_cap=5.0,
            observed_demand_decline=0.0,
            demand_decline_ramp_days=1,
            fear_initial=0.0,
            fear_decay=0.0,
        )

    def _flat_behavior(self) -> BehavioralParameters:
        return BehavioralParameters(
            pressure_scale=0.0,
            risk_weight=0.0,
            uncertainty_floor=0.0,
            inventory_response=0.5,
            adjustment_speed=0.30,
        )

    def _event_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2026-03-01", "2026-03-03", "2026-03-06"]),
                "pre_close": [80.0, 80.0, 80.0],
                "close_price": [80.0, 80.5, 81.0],
            }
        )

    def test_zero_gap_case_keeps_prices_finite_and_stable(self) -> None:
        result = simulate_dynamic_model(self._event_frame(), self._flat_assumptions(), self._flat_behavior())

        required_columns = {
            "day_index",
            "trade_date",
            "actual_price",
            "simulated_price",
            "effective_supply",
            "effective_demand",
            "supply_gap",
            "inventory_remaining",
        }
        self.assertTrue(required_columns.issubset(result.columns))
        self.assertEqual(len(result), 3)
        self.assertEqual(result["day_index"].tolist(), [0, 2, 5])
        self.assertTrue((result["supply_gap"] == 0.0).all())
        self.assertTrue((result["simulated_price"] == 80.0).all())

    def test_trading_day_axis_uses_row_index(self) -> None:
        result = simulate_dynamic_model(
            self._event_frame(),
            self._flat_assumptions(),
            self._flat_behavior(),
            time_axis="trading_day",
        )

        self.assertEqual(result["day_index"].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
