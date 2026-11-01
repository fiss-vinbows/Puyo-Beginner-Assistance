"""盤面4隅の座標変換・自動検出・キャリブレーション保存形式のテスト"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
from PyQt6 import QtCore

from src.capture.calibrate import CalibrationResult, adjust_rect, load_calibration, rect_to_quad, save_calibration
from src.capture.geometry import BoardGeometry, detect_board_quads
from src.engine.puyo_field import COLS, ROWS, VISIBLE_ROWS

FIXTURES = Path(__file__).parent / "fixtures"
RECT_QUAD = ((100.0, 200.0), (400.0, 200.0), (400.0, 800.0), (100.0, 800.0))  # 幅300, 高さ600


class BoardGeometryTest(unittest.TestCase):
    def test_rect_cell_center(self) -> None:
        g = BoardGeometry(RECT_QUAD)
        # 最下段左端(row=12, col=0): x = 100 + 25, y = 800 - 25
        x, y = g.cell_center(ROWS - 1, 0)
        self.assertAlmostEqual(x, 125, places=3)
        self.assertAlmostEqual(y, 775, places=3)
        # 隠し13段目(row=0)は可視領域の上に外挿される
        x, y = g.cell_center(0, 5)
        self.assertAlmostEqual(x, 375, places=3)
        self.assertAlmostEqual(y, 175, places=3)
        self.assertAlmostEqual(g.cell_size(), 50, places=3)

    def test_round_trip(self) -> None:
        g = BoardGeometry(((10, 20), (310, 5), (330, 620), (0, 600)))  # 歪んだ四角形
        for u, v in ((0, 0), (3.5, 6.5), (6, 12), (2, -1)):
            x, y = g.grid_to_screen(u, v)
            u2, v2 = g.screen_to_grid(x, y)
            self.assertAlmostEqual(u, u2, places=3)
            self.assertAlmostEqual(v, v2, places=3)

    @unittest.skipUnless((FIXTURES / "frame_00300.png").exists(), "ゲーム画面全体の画像は公開版に入れていない")
    def test_warp_size(self) -> None:
        img = cv2.imread(str(FIXTURES / "frame_00300.png"))
        g = BoardGeometry(RECT_QUAD)
        out = g.warp(img, cell_px=16, rows_above=1)
        self.assertEqual(out.shape[:2], (ROWS * 16, COLS * 16))
        out = g.warp(img, cell_px=16, rows_above=2)
        self.assertEqual(out.shape[:2], ((VISIBLE_ROWS + 2) * 16, COLS * 16))

    def test_bounding_box_includes_rows_above(self) -> None:
        g = BoardGeometry(RECT_QUAD)
        x, y, w, h = g.bounding_box(rows_above=2)
        self.assertEqual((x, y, w, h), (100, 100, 300, 700))


class DetectBoardQuadsTest(unittest.TestCase):
    @unittest.skipUnless((FIXTURES / "frame_00300.png").exists(), "ゲーム画面全体の画像は公開版に入れていない")
    def test_detects_both_boards_on_empty_field(self) -> None:
        img = cv2.imread(str(FIXTURES / "frame_00300.png"))
        quads = detect_board_quads(img)
        self.assertEqual(len(quads), 2)
        me, enemy = quads
        # 実測値(1080p): 自分 x=272..672, 相手 x=1248..1648, 暗い矩形(可視12段) y=159..882
        self.assertAlmostEqual(me[0][0], 272, delta=2)
        self.assertAlmostEqual(me[1][0], 672, delta=2)
        self.assertAlmostEqual(me[2][1], 882, delta=2)
        self.assertAlmostEqual(me[0][1], 159, delta=2)
        self.assertAlmostEqual(enemy[0][0], 1248, delta=2)
        self.assertAlmostEqual(enemy[1][0], 1648, delta=2)

    @unittest.skipUnless((FIXTURES / "frame_02400.png").exists(), "ゲーム画面全体の画像は公開版に入れていない")
    def test_detects_on_mid_game_frame(self) -> None:
        # ぷよが積まれた盤面でも最上部の暗い帯が連結していれば検出できることを確認
        img = cv2.imread(str(FIXTURES / "frame_02400.png"))
        quads = detect_board_quads(img)
        self.assertEqual(len(quads), 2)


class CalibrationIoTest(unittest.TestCase):
    def test_fixture_round_trip(self) -> None:
        result = load_calibration(FIXTURES / "calibration_1080p.json")
        self.assertEqual(len(result.self_player.next_rects), 2)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            save_calibration(result, path)
            again = CalibrationResult.from_dict(json.loads(path.read_text(encoding="utf-8")))
        self.assertEqual(again.to_dict(), result.to_dict())

    def test_capture_region_covers_everything(self) -> None:
        result = load_calibration(FIXTURES / "calibration_1080p.json")
        r = result.capture_region()
        self.assertLessEqual(r.left, 272)
        self.assertLessEqual(r.top, 160)
        self.assertGreaterEqual(r.left + r.width, 1648)
        self.assertGreaterEqual(r.top + r.height, 882)


class AdjustRectTest(unittest.TestCase):
    def test_move_whole(self) -> None:
        self.assertEqual(adjust_rect((100, 200, 300, 600), QtCore.Qt.Key.Key_Right, resize=False), (101, 200, 300, 600))

    def test_resize_bottom_right(self) -> None:
        self.assertEqual(adjust_rect((100, 200, 300, 600), QtCore.Qt.Key.Key_Up, resize=True), (100, 200, 300, 599))

    def test_non_arrow_returns_none(self) -> None:
        self.assertIsNone(adjust_rect((100, 200, 300, 600), QtCore.Qt.Key.Key_A, resize=False))

    def test_rect_to_quad(self) -> None:
        self.assertEqual(rect_to_quad((100, 200, 300, 600)), RECT_QUAD)


if __name__ == "__main__":
    unittest.main()
