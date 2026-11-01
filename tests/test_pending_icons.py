"""予告おじゃまのアイコンの種類の判定(src/vision/pending_icons.py)のテスト

素材は実機(Steam 版、2026-10-11〜12)の画像から盤面座標(1マス 32px)で切り出した予告の帯(幅 6 列 × 72px)。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from src.vision.pending_icons import KINDS, IconReading, default_reader

FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> IconReading:
    band = cv2.imread(str(FIXTURES / f"pending_band_{name}.png"))
    return default_reader().read(band)


class PendingIconTest(unittest.TestCase):
    def test_kinds_and_values(self) -> None:
        # 岩(30) + 大(6)×4 + 小(1) = 55
        r = _read("self_rock_large")
        self.assertEqual(r.kinds, ("rock", "large", "large", "large", "large", "small"))
        self.assertEqual(r.value(), 55)
        # 星(180) + 岩(30)×4 + 小 = 301(相手側の赤い背景)
        r = _read("enemy_star_rock")
        self.assertEqual(r.kinds, ("star", "rock", "rock", "rock", "rock", "small"))
        self.assertEqual(r.value(), 301)
        # 王冠(720) + 岩 + 大×3 + 小 = 769
        r = _read("enemy_crown")
        self.assertEqual(r.kinds, ("crown", "rock", "large", "large", "large", "small"))
        self.assertEqual(r.value(), 769)
        # 自分側の星と、6列目にずれて入ってくる途中の小
        r = _read("self_star_small")
        self.assertEqual(r.kinds, ("star", "large", "large", "large", "small", "small"))

    def test_empty_band_has_no_icons(self) -> None:
        # 枠の白い線と背景(自分側=青、相手側=赤)をアイコンと読まない
        for name in ("self_empty", "enemy_empty"):
            self.assertEqual(_read(name).kinds, (), name)
            self.assertEqual(_read(name).value(), 0)

    def test_values_follow_puyo_rules(self) -> None:
        self.assertEqual({k: v for k, (_, v) in KINDS.items()},
                         {"small_self": 1, "small_enemy": 1, "large": 6, "rock": 30, "star": 180, "moon": 360, "crown": 720})


if __name__ == "__main__":
    unittest.main()
