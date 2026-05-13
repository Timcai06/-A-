import unittest

import pandas as pd

from src.experiments.run_model_suite import build_short_term_ranking


class ExperimentSuiteTests(unittest.TestCase):
    def test_short_term_ranking_keeps_ml_candidates_but_drops_reference_duplicates(self) -> None:
        summary = pd.DataFrame(
            [
                {"模型类别": "短期拟合/预测", "模型": "本文短期动态模型", "RMSE": 3.44, "MAE": 2.85},
                {"模型类别": "短期机器学习辅助", "模型": "机制递推主模型", "RMSE": 3.44, "MAE": 2.85},
                {"模型类别": "短期机器学习辅助", "模型": "机制+Ridge收益率修正", "RMSE": 3.35, "MAE": 2.81},
                {"模型类别": "长期概率模型", "模型": "蒙特卡洛情景树", "RMSE": None, "MAE": None},
            ]
        )

        ranking = build_short_term_ranking(summary)

        self.assertEqual(ranking["模型"].tolist(), ["机制+Ridge收益率修正", "本文短期动态模型"])
        self.assertEqual(ranking["排名"].tolist(), [1, 2])


if __name__ == "__main__":
    unittest.main()
