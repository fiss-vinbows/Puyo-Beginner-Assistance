"""相手の攻撃の記録(AttackEpisode)と平均手時間(TurnClock)。新方針の第1段階(記録のみ)。"""

from __future__ import annotations

import unittest

from src.engine.threat import AttackEpisode, TurnClock


class TurnClockTest(unittest.TestCase):
    def test_average_ignores_long_gaps(self) -> None:
        clock = TurnClock()
        self.assertIsNone(clock.average)
        for t in (0.0, 1.5, 3.0, 4.5):
            clock.spawn(t)
        self.assertAlmostEqual(clock.average, 1.5)
        clock.spawn(20.0)  # 演出・停止などの長い空白は数えない
        self.assertAlmostEqual(clock.average, 1.5)
        self.assertEqual(clock.samples, 3)
        self.assertEqual(clock.moves_within(5.0), 3)


class AttackEpisodeTest(unittest.TestCase):
    def test_summary_and_estimates(self) -> None:
        ep = AttackEpisode(attack_id=2, started_at=10.0, before=[], bonus=True, my_pending_at_start=0)
        ep.add_estimate(10.0, "前回解析の相手本線", 40)
        ep.add_estimate(10.5, "前回解析の相手本線", 40)  # 同じ根拠・同じ値は重複させない
        ep.add_estimate(11.0, "思考時の相手本線", 55)
        ep.steps, ep.dropped, ep.ended_at, ep.my_pending_at_end = 3, 20, 13.0, 6
        self.assertEqual(len(ep.estimates), 2)
        self.assertAlmostEqual(ep.duration, 3.0)
        text = ep.summary()
        self.assertIn("攻撃#2", text)
        self.assertIn("段数≥3", text)  # 段数は下限(点滅で段が合流する)
        self.assertIn("消去20個", text)
        self.assertIn("全消しボーナス付き", text)
        self.assertIn("自分の予告 0->6", text)
        self.assertIn("思考時の相手本線=55個", text)


if __name__ == "__main__":
    unittest.main()
