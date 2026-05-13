import unittest
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "tmp" / "matplotlib"))

from src.calibration.parameter_space import decode_continuous_parameters, encode_continuous_parameters
from src.models.dynamic_short_term import BehavioralParameters, PhysicalAssumptions


class ParameterSpaceTests(unittest.TestCase):
    def test_encode_decode_round_trip_preserves_representative_parameters(self) -> None:
        assumptions = PhysicalAssumptions(
            base_supply=10060.0,
            base_demand=10000.0,
            base_elasticity=-0.05,
            long_elasticity=-0.18,
            supply_interruption=1600.0,
            spr_max_release=450.0,
            spr_delay_days=7,
            spr_ramp_days=7,
            route_start_day=14,
            route_max_capacity=250.0,
            route_ramp_days=21,
            commercial_inventory=58000.0,
            inventory_daily_cap=450.0,
            observed_demand_decline=430.0,
            demand_decline_ramp_days=21,
            fear_initial=0.16,
            fear_decay=0.08,
        )
        behavior = BehavioralParameters(
            pressure_scale=0.045,
            risk_weight=2.20,
            uncertainty_floor=0.24,
            inventory_response=0.55,
            adjustment_speed=0.25,
            buffer_relief_strength=0.08,
            buffer_relief_decay_days=10,
            relief_discount_strength=0.12,
            relief_start_day=34,
            relief_peak_day=46,
            relief_decay_days=12,
        )

        decoded_assumptions, decoded_behavior = decode_continuous_parameters(
            encode_continuous_parameters(assumptions, behavior),
            assumptions,
        )

        self.assertAlmostEqual(decoded_assumptions.supply_interruption, assumptions.supply_interruption)
        self.assertAlmostEqual(decoded_assumptions.spr_max_release, assumptions.spr_max_release)
        self.assertEqual(decoded_assumptions.route_start_day, assumptions.route_start_day)
        self.assertAlmostEqual(decoded_assumptions.long_elasticity, assumptions.long_elasticity)
        self.assertAlmostEqual(decoded_behavior.pressure_scale, behavior.pressure_scale)
        self.assertAlmostEqual(decoded_behavior.risk_weight, behavior.risk_weight)
        self.assertEqual(decoded_behavior.relief_peak_day, behavior.relief_peak_day)
        self.assertEqual(decoded_behavior.relief_decay_days, behavior.relief_decay_days)


if __name__ == "__main__":
    unittest.main()
