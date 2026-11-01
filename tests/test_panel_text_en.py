"""情報パネルの英語表記のテスト"""

from __future__ import annotations

import os
import re
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.engine.ama_client import FIRE_REASON_TEXT_JA, FORM_TEXT_JA, LABEL_TEXT_JA, BattleMove, action_text  # noqa: E402
from src.engine.perf_meter import perf_text  # noqa: E402
from src.overlay.panel_text_en import to_english  # noqa: E402
from src.overlay.renderer import PANEL_ENEMY_COLOR, PANEL_SELF_COLOR, OverlayDrawData, panel_lines  # noqa: E402
from tests.test_renderer import _analysis  # noqa: E402

JAPANESE = re.compile(r"[぀-ヿ一-鿿]")


class ToEnglishTest(unittest.TestCase):
    def assert_english(self, line: str) -> None:
        out = to_english(line)
        self.assertIsNone(JAPANESE.search(out), f"{line!r} → {out!r}")

    def test_known_lines_have_no_japanese(self) -> None:
        # app.py・ama_client.py がパネルに出す文言
        lines = [f"提案: {v}" for v in LABEL_TEXT_JA.values()]
        lines += [f"狙い: {v}" for v in FORM_TEXT_JA.values()] + ["狙い: (定型なし)", "定型を離脱: フキゲンGTR", "移行: GTR → フキゲンGTR"]
        lines += list(FIRE_REASON_TEXT_JA.values())
        lines += [
            "この手で 6連鎖 10,420点",
            "開始前: 第1手の提案(第1〜3手から)",
            "開始前: 仮の提案(第3手が見えたら変わります)",
            "先読み(確認中)",
            "次の手を計算中",
            "連鎖中: 次の手を待機",
            "発火 14連鎖 123,450点",
            "発火 3連鎖(点数は再現できず)",
            "自動操作は先読みの手を実行済み",
            "自動操作は送った手を実行中",
            "対応が必要: 提案を変更",
            "対応が必要: 提案を変更 (目安の連鎖に到達)",
            "相手の認識エラー: 相手を見ずに提案",
            "相手の消去開始を検出: 提案を更新",
            "相手の連鎖が終了: 提案を更新 (目安の連鎖に到達)",
            "予告おじゃまが増加: 提案を更新",
            "構想: 12連鎖 66,660点 @14",
            "7連鎖 17,700点", "  潰し 3連鎖 300点",
            perf_text(42.0, 30.0), perf_text(None, 30.0),
        ]
        for line in lines:
            self.assert_english(line)

    def test_action_text(self) -> None:
        # 判断の理由ごとの「何をすべきか」(数値入り)
        for reason, label, steps in [("EXTEND", "build", 0), ("ACCEPT", "accept", 0), ("COUNTER_PREP", "build", 3),
                                     ("COUNTER_PREP", "main", 1), ("ATTACKED_KILL", "build", 2), ("ATTACKED_KILL", "main", 1),
                                     ("COUNTER", "main", 0)]:
            move = BattleMove.__new__(BattleMove)
            move.reason, move.label, move.plan_steps = reason, label, steps
            for incoming in (0, 12):
                self.assert_english(f"提案: {action_text(move, incoming)}")

    def test_values(self) -> None:
        self.assertEqual(to_english("この手で 6連鎖 10,420点"), "This move: 6-chain 10,420 pts")
        self.assertEqual(to_english("7連鎖 17,700点"), "7-chain 17,700 pts")
        self.assertEqual(to_english("発火中 3連鎖(点数は再現できず)"), "Firing: 3-chain (score unknown)")
        self.assertEqual(to_english("点数を訂正: 1連鎖 40点 → 2連鎖 360点"), "Score corrected: 1-chain 40 pts → 2-chain 360 pts")
        self.assertEqual(to_english("--"), "--")
        self.assertEqual(to_english("構想: 12連鎖 66,660点 @14"), "Plan: 12-chain 66,660 pts @14")
        self.assertEqual(to_english("返しの準備 @3"), "Prepare counter @3")
        self.assertEqual(to_english("撃つ準備 @2(倒しきれる)"), "Prepare to fire @2 (it kills)")
        self.assertEqual(to_english("提案: All Green"), "Move: All Green")

    def test_unknown_line_kept(self) -> None:
        self.assertEqual(to_english("未知の文言"), "未知の文言")

    def test_panel_lines_keep_colors(self) -> None:
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(7, 17700), _analysis(6, 10560))
        lines = panel_lines(data, "en")
        self.assertEqual(lines[0][0], "Move: Build")
        self.assertIn(("7-chain 17,700 pts", PANEL_SELF_COLOR), lines)
        self.assertIn(("6-chain 10,560 pts", PANEL_ENEMY_COLOR), lines)


if __name__ == "__main__":
    unittest.main()
