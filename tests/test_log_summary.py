"""実機ログの要約(tools/log_summary.py)の自由落下の集計のテスト(2026-10-21)"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import log_summary  # noqa: E402

LOG = """===== 支援モード開始 2026-10-21T10:00:00 モード=対戦 全自動=True
[  10.00s] self  spawn        turn=1 current=('red', 'red') next=('blue', 'blue') nextnext=('red', 'blue')
           自動操作: 先読みの手順を出現時点で送信(turn=1 x=0 UP)
           自動操作: turn=1 LEFT LEFT DROP (300ms, 落下 200ms)
[  11.00s] self  spawn        turn=2 current=('blue', 'blue') next=('red', 'blue') nextnext=('red', 'red')
           自動操作: 出現時点では送りません(turn=2): 先読みの盤面と実際の盤面が違う
           自動操作: 先読みの手順を出現の450ms後で送信(turn=2 x=3 UP)
[  12.00s] self  spawn        turn=3 current=('red', 'blue') next=('red', 'red') nextnext=('green', 'red')
           自動操作: 出現時点では送りません(turn=3): 先読みの手が小さい消去 [harass] なので3組目まで見た思考を待つ
[  14.00s] self  spawn        turn=4 current=('red', 'red') next=('green', 'red') nextnext=('blue', 'red')
===== 支援モード開始 2026-10-21T10:05:00 モード=対戦 全自動=False
[  20.00s] self  spawn        turn=5 current=('red', 'red') next=('green', 'red') nextnext=('blue', 'red')
[  21.00s] self  spawn        turn=6 current=('green', 'red') next=('blue', 'red') nextnext=('red', 'red')
"""


class FreeFallTest(unittest.TestCase):
    def test_turn_sends(self) -> None:
        all_recs = log_summary.turn_sends(LOG.splitlines())
        recs = {r["turn"]: r for r in all_recs if not r.get("unsent")}
        # 次の出現がある手番だけ(turn=4 は次が無い)。全自動でない間(turn=5・6)は数えない
        self.assertEqual(sorted(recs), [1, 2, 3])
        # 次の出現が無く一度も送らなかった手番は「送らなかった手番」として別に返す(2026-10-24: 負けの直前の手番が集計から漏れていた)
        self.assertEqual([r["turn"] for r in all_recs if r.get("unsent")], [4])
        self.assertFalse(recs[1]["free_fall"])
        self.assertAlmostEqual(recs[1]["delay"], 0.0)
        self.assertFalse(recs[2]["free_fall"])
        self.assertAlmostEqual(recs[2]["delay"], 0.45)
        self.assertEqual(recs[2]["reasons"], ["先読みの盤面と実際の盤面が違う"])
        # 送らないまま次の組ぷよが出た = 自由落下
        self.assertTrue(recs[3]["free_fall"])
        self.assertEqual(recs[3]["reasons"], ["先読みの手が小さい消去"])

    def test_report(self) -> None:
        text = "\n".join(log_summary.free_fall_report(log_summary.turn_sends(LOG.splitlines())))
        self.assertIn("自由落下(次の出現までに送れなかった)手番 1", text)
        self.assertIn("遅れ(>0.3s)で送った手番 1", text)


class CancelSummaryTest(unittest.TestCase):
    def test_cancel_section(self) -> None:
        import tempfile

        log = """===== 支援モード開始 2026-10-21T10:00:00 モード=対戦 全自動=True
[  10.00s] self  spawn        turn=1 current=('red', 'red') next=('blue', 'blue') nextnext=('red', 'blue')
           自動操作: turn=1 RIGHT DROP (300ms, 落下 200ms, 設置キャンセル: 最後のRIGHTを落下開始から330ms)
[  10.80s] self  spawn        turn=2 current=('blue', 'blue') next=('red', 'blue') nextnext=('red', 'red')
           自動操作: turn=2 LEFT DROP (300ms, 落下 200ms)
[  11.70s] self  spawn        turn=3 current=('red', 'blue') next=('red', 'red') nextnext=('green', 'red')
"""
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "debug_log_test.txt"
            p.write_text(log, encoding="utf-8")
            text = log_summary.summarize(p)
        self.assertIn("[設置キャンセル]", text)
        self.assertIn("狙った手 1", text)


if __name__ == "__main__":
    unittest.main()
