"""シミュレーターの配ぷよモード(実機の配ぷよ65536通り)のテスト"""

from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.simulator import haipuyo  # noqa: E402
from src.simulator.rules import SimState  # noqa: E402


@unittest.skipUnless(haipuyo.available(), "assets/haipuyo.txt.gz が無い")
class HaipuyoTableTest(unittest.TestCase):
    def test_sequence_maps_colors_in_order_of_appearance(self) -> None:
        # 0番の配ぷよは "rrprpypyrbbb..." → 出てきた順に R/Y/G/B(1文字目を軸)
        seq = haipuyo.sequence(0)
        self.assertEqual(len(seq), haipuyo.PAIRS_PER_SEQUENCE)
        self.assertEqual(seq[:6], ["RR", "YR", "YG", "YG", "RB", "BB"])
        self.assertTrue(all(len(p) == 2 and set(p) <= set("RYGB") for p in seq))

    def test_first_two_pairs_use_at_most_three_colors(self) -> None:
        for i in range(0, haipuyo.SEQUENCE_COUNT, 997):
            seq = haipuyo.sequence(i)
            self.assertLessEqual(len(set(seq[0] + seq[1])), 3)

    def test_sim_state_deals_from_table_and_wraps(self) -> None:
        s = SimState(seed=0, haipuyo=12345)
        seq = haipuyo.sequence(12345)
        self.assertEqual(s.upcoming(3), seq[:3])
        s.ensure_pairs(130)
        self.assertEqual(s.pairs[128:130], seq[:2])  # 128組で最初に戻る
        same = s.reset(True)
        self.assertEqual((same.haipuyo, same.pairs[:3]), (12345, seq[:3]))
        new = s.reset(False)
        self.assertIsNotNone(new.haipuyo)  # 別配列でリセットしても配ぷよのまま
        self.assertIsNone(SimState.new().haipuyo)


class AdvisorHaipuyoTest(unittest.TestCase):
    def test_yamaneko_uses_haipuyo_reading_with_history(self) -> None:
        # 配ぷよモードでは YamaNEKO を配ぷよ読みの起動引数で起動し、置いた組ぷよの履歴を付けて対戦思考で呼ぶ
        from src.engine.ama_client import HAIPUYO_BRIDGE_ARGS
        from src.simulator import window

        calls: list = []

        class FakeClient:
            def __init__(self, bridge: str, extra_args=None) -> None:  # noqa: ANN001
                calls.append(("start", bridge, extra_args))

            def battle(self, me, enemy):  # noqa: ANN001, ANN201
                calls.append(("battle", me.history, me.placed))
                return "move"

            def solo(self, field, pairs):  # noqa: ANN001, ANN201
                calls.append(("solo",))
                return "move"

            def close(self) -> None:
                calls.append(("close",))

        import time

        results: list = []
        empty = ["......"] * 13
        with mock.patch.object(window, "AmaClient", FakeClient):
            th = window.AdvisorThread("yamaneko")
            th.advice_ready.connect(lambda *a: results.append(a), type=window.QtCore.Qt.ConnectionType.DirectConnection)
            th.start()
            for req in (window.AdviceRequest(1, empty, ["RY", "GB", "RR"], history=["RR", "YR"]),
                        window.AdviceRequest(2, empty, ["RY", "GB", "RR"])):
                n = len(results)
                th.request(req)
                end = time.time() + 2
                while len(results) == n and time.time() < end:
                    time.sleep(0.01)
            th.stop()
            th.wait(2000)
        self.assertIn(("start", "yamaneko", HAIPUYO_BRIDGE_ARGS), calls)
        self.assertIn(("battle", ["RR", "YR"], 2), calls)
        self.assertIn(("close",), calls)  # 配ぷよ読みのあり・なしが変わったら起動し直す
        self.assertIn(("start", "yamaneko", None), calls)  # 乱数の配列では配ぷよ読みなしで一人用
        self.assertIn(("solo",), calls)


if __name__ == "__main__":
    unittest.main()
