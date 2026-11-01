"""CPU 使用率と観測の FPS(2026-10-24)のテスト"""

from __future__ import annotations

import unittest

from src.engine.perf_meter import CpuMeter, FpsMeter, perf_text


class PerfMeterTest(unittest.TestCase):
    def test_cpu_from_system_times(self) -> None:
        # (待機, カーネル(待機を含む), ユーザー) の差から使用率を出す: 合計 200 のうち待機 50 → 75%
        seq = iter([(0, 0, 0), (50, 120, 80)])
        meter = CpuMeter(read=lambda: next(seq))
        self.assertAlmostEqual(meter.sample(), 75.0)

    def test_cpu_unavailable(self) -> None:
        self.assertIsNone(CpuMeter(read=lambda: None).sample())

    def test_fps_window(self) -> None:
        m = FpsMeter(window=1.0)
        for i in range(31):
            m.tick(i / 30.0)
        self.assertAlmostEqual(m.fps(), 30.0)
        m.tick(5.0)  # 間が空いたら古い観測は捨てる
        self.assertEqual(m.fps(), 0.0)

    def test_text(self) -> None:
        self.assertEqual(perf_text(45.4, 33.2), "CPU 45% / 観測 33fps")
        self.assertEqual(perf_text(None, 0.0), "CPU --% / 観測 0fps")


if __name__ == "__main__":
    unittest.main()
