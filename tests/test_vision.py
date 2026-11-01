"""画像認識のテスト(動画から抜いた1080pフレームで検証)"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np

from src.capture.calibrate import load_calibration
from src.vision.board_reader import cells_to_text, read_board, read_next
from src.vision.cell_classifier import (
    BLUE,
    EMPTY,
    GARBAGE,
    GREEN,
    PURPLE,
    RED,
    UNKNOWN,
    YELLOW,
    classify_cell_bgr,
    classify_cell_hsv,
    dominant_color,
    hue_to_color,
)
from src.vision.color_mapper import ColorMapper

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str):
    path = FIXTURES / f"frame_{name}.png"
    if not path.exists():
        raise unittest.SkipTest("ゲーム画面全体の画像は公開版に入れていない")
    return cv2.imread(str(path))


def _cal():
    return load_calibration(FIXTURES / "calibration_1080p.json")


class HueTest(unittest.TestCase):
    def test_measured_hues(self) -> None:
        # 実測値: 赤≈176, 黄≈25, 緑≈52, 紫≈139
        self.assertEqual(hue_to_color(176), RED)
        self.assertEqual(hue_to_color(2), RED)
        self.assertEqual(hue_to_color(25), YELLOW)
        self.assertEqual(hue_to_color(52), GREEN)
        self.assertEqual(hue_to_color(139), PURPLE)
        self.assertEqual(hue_to_color(110), BLUE)
        # ネクスト枠の水色(≈94)はどの色にも入らない
        self.assertEqual(hue_to_color(94), UNKNOWN)


class ClassifyCellTest(unittest.TestCase):
    def _solid(self, bgr: tuple[int, int, int]) -> np.ndarray:
        return np.full((32, 32, 3), bgr, dtype=np.uint8)

    def test_synthetic_cells(self) -> None:
        self.assertEqual(classify_cell_bgr(self._solid((40, 40, 230))), RED)
        self.assertEqual(classify_cell_bgr(self._solid((230, 230, 230))), GARBAGE)
        self.assertEqual(classify_cell_bgr(self._solid((50, 60, 40))), EMPTY)

    def test_cross_mark_without_eyes_is_empty(self) -> None:
        # 赤一色で白が無いセルは、×マーク位置なら空扱い
        hsv = cv2.cvtColor(self._solid((40, 40, 230)), cv2.COLOR_BGR2HSV)
        self.assertEqual(classify_cell_hsv(hsv, cross_mark=True), EMPTY)
        self.assertEqual(classify_cell_hsv(hsv, cross_mark=False), RED)


class ReadBoardTest(unittest.TestCase):
    def test_opening_frame(self) -> None:
        cal = _cal()
        img = _load("00300")
        me = read_board(img, cal.self_player.geometry())
        text = cells_to_text(me.cells).splitlines()
        self.assertEqual(text[-1], "YY....")
        self.assertEqual(text[-2], "PP....")
        self.assertTrue(all(row == "......" for row in text[:-2]))
        self.assertEqual(me.pending_garbage, 0)
        enemy = read_board(img, cal.enemy_player.geometry())
        self.assertEqual(cells_to_text(enemy.cells).splitlines()[-1], "....YY")
        # 落下中の組ぷよ(12段目の2列目に紫)がそのまま見える
        self.assertEqual(cells_to_text(enemy.cells).splitlines()[1], ".P....")

    def test_cross_mark_is_not_a_puyo(self) -> None:
        cal = _cal()
        for name in ("00300", "02400", "05310"):
            me = read_board(_load(name), cal.self_player.geometry())
            self.assertEqual(me.cells[1][2], EMPTY, name)

    def test_mid_game_bottom_rows(self) -> None:
        cal = _cal()
        me = read_board(_load("05310"), cal.self_player.geometry())
        text = cells_to_text(me.cells).splitlines()
        self.assertEqual(text[-1], "YYRGGG")
        self.assertEqual(text[-2], "PPYRRR")
        self.assertEqual(text[-3], "PYRYPG")
        self.assertEqual(text[-4], "RRYPY#")  # おじゃま
        enemy = read_board(_load("05310"), cal.enemy_player.geometry())
        self.assertEqual(cells_to_text(enemy.cells).splitlines()[-1], "RPYPPP")

    def test_pending_garbage(self) -> None:
        cal = _cal()
        self.assertEqual(read_board(_load("02400"), cal.enemy_player.geometry()).pending_garbage, 0)
        # 大(6個) 1つ + 光の下の小 4つ = 10個(2026-10-20 から種類ごとの個数で数える。以前はアイコンの数 5 だった)
        enemy = read_board(_load("05310"), cal.enemy_player.geometry())
        self.assertEqual(enemy.pending_garbage, 10)
        self.assertEqual(enemy.pending_icons[0], "large")
        self.assertEqual(read_board(_load("05310"), cal.self_player.geometry()).pending_garbage, 0)


class ReadNextTest(unittest.TestCase):
    def test_next_pairs(self) -> None:
        cal = _cal()
        img = _load("02400")
        self.assertEqual(read_next(img, cal.self_player.next_rects[0]), (RED, YELLOW))
        self.assertEqual(read_next(img, cal.self_player.next_rects[1]), (GREEN, GREEN))
        self.assertEqual(read_next(img, cal.enemy_player.next_rects[0]), (RED, YELLOW))
        img = _load("05310")
        self.assertEqual(read_next(img, cal.self_player.next_rects[0]), (YELLOW, GREEN))
        self.assertEqual(read_next(img, cal.self_player.next_rects[1]), (RED, GREEN))
        self.assertEqual(read_next(img, cal.enemy_player.next_rects[0]), (YELLOW, YELLOW))

    def test_enemy_next_in_red_frame_is_not_red(self) -> None:
        # 相手のネクスト枠は赤い水玉柄。枠の上半分に黄ぷよが小さく写ると楕円内の色画素の
        # 過半が枠の赤になり、色相の中央値では赤と誤判定されていた(実機 2026-09-15、
        # 5色目エラーで自分の思考が止まった)。目のまわりの色で判定する
        img = cv2.imread(str(FIXTURES / "enemy_next2_yellow_purple.png"))
        h, w = img.shape[:2]
        self.assertEqual(read_next(img, (0, 0, w, h)), (YELLOW, PURPLE))
        img = cv2.imread(str(FIXTURES / "enemy_next1_green_yellow.png"))
        h, w = img.shape[:2]
        self.assertEqual(read_next(img, (0, 0, w, h)), (GREEN, YELLOW))

    def test_yellow_with_orange_eyebrows(self) -> None:
        # 黄ぷよの目のまわりは茶色いまゆの陰影(色相10〜12)。黄の範囲(旧14〜)から外れて unknown になり、
        # ネクストが読めず手番が進まなかった(実機 2026-09-17)
        img = cv2.imread(str(FIXTURES / "self_next1_red_yellow.png"))
        h, w = img.shape[:2]
        self.assertEqual(read_next(img, (0, 0, w, h)), (RED, YELLOW))

    def test_dominant_color_handles_red_hue_wrap(self) -> None:
        # 赤は色相 0 付近と 179 付近に分かれるので、中央値ではなく範囲ごとの多数決で決める
        self.assertEqual(dominant_color(np.array([0, 1, 178, 179])), RED)
        self.assertEqual(dominant_color(np.array([24, 25, 26, 178])), YELLOW)
        self.assertEqual(dominant_color(np.array([0, 0, 52, 52])), UNKNOWN)  # 2色が同数なら不明
        self.assertEqual(dominant_color(np.array([], dtype=int)), UNKNOWN)

    def test_next_during_spawn_animation_is_unknown(self) -> None:
        # スポーン直後はネクスト枠のぷよが移動中で、枠の水色が混ざり unknown になる
        cal = _cal()
        child, axis = read_next(_load("04800"), cal.self_player.next_rects[0])
        self.assertIn(UNKNOWN, (child, axis))


class ColorMapperForkTest(unittest.TestCase):
    def test_fork_does_not_leak_failed_registration(self) -> None:
        # 変換を試してから採用する: 相手側の誤色を仮登録した後に失敗しても、元の割当は変わらない
        base = ColorMapper()
        base.slot("red")
        base.slot("yellow")
        base.slot("green")
        trial = base.fork()
        trial.slot("purple")  # 相手の誤認識で4枠目が埋まる
        with self.assertRaises(ValueError):
            trial.slot("blue")  # 5色目
        self.assertEqual(set(base.mapping), {"red", "yellow", "green"})
        base.slot("blue")  # 自分の本当の4色目は登録できる


class ColorMapperTest(unittest.TestCase):
    def test_preferred_then_free_slot(self) -> None:
        m = ColorMapper()
        self.assertEqual(m.slot(RED), "R")
        self.assertEqual(m.slot(PURPLE), "B")  # 青が未出現なら紫がB
        self.assertEqual(m.slot(GREEN), "G")
        self.assertEqual(m.slot(YELLOW), "Y")
        with self.assertRaises(ValueError):
            m.slot(BLUE)  # 5色目
        self.assertEqual(m.slot(GARBAGE), "#")
        self.assertEqual(m.slot(EMPTY), ".")
        self.assertEqual(m.pair(child=PURPLE, axis=RED), "RB")

    def test_blue_before_purple(self) -> None:
        m = ColorMapper()
        self.assertEqual(m.slot(BLUE), "B")
        self.assertEqual(m.slot(PURPLE), "R")  # 空いている先頭スロット


if __name__ == "__main__":
    unittest.main()


class AllClearBannerTest(unittest.TestCase):
    """「全消し」の文字を黄ぷよ・おじゃまとして読まない(実機 2026-09-20 13:56)。

    フィクスチャは実機フレームを盤面座標(1マス 32px、上に予告帯1段)に変換したもの。
    その画像自体を盤面としてジオメトリを作れば read_board がそのまま使える。
    """

    def _read(self, name: str):
        from src.capture.geometry import BoardGeometry
        from src.vision.board_reader import WARP_CELL_PX

        img = cv2.imread(str(FIXTURES / f"board_{name}.png"))
        h, w = img.shape[:2]
        geo = BoardGeometry(((0, WARP_CELL_PX), (w, WARP_CELL_PX), (w, h), (0, h)))
        return read_board(img, geo)

    def test_banner_on_empty_board_reads_empty(self) -> None:
        # 全消し直後の空盤面: 文字は 9〜11段目に出るが、盤面は空(以前は .YYYY. / .YY#Y. と読んだ)
        b = self._read("all_clear_banner_empty")
        self.assertGreater(b.banner_score, 0.9)
        rows = cells_to_text(b.cells).splitlines()
        self.assertEqual(rows[1:], ["..R..." if i == 0 else "......" for i in range(12)])  # 12段目3列目は出現直後の組ぷよ(R)
        self.assertEqual(b.eyeless_cells, frozenset())

    def test_banner_with_puyos_keeps_real_puyos(self) -> None:
        # 文字の手前に描かれたぷよ・落下中の組ぷよは読み、文字と着地予測(ゴースト)は空
        b = self._read("all_clear_banner_puyos")
        self.assertGreater(b.banner_score, 0.9)
        rows = cells_to_text(b.cells).splitlines()
        self.assertEqual(rows[1:6], ["..Y...", "..P...", "......", "......", "PRGP.."])
        self.assertEqual(rows[6:], ["GRGG..", "GGYPR.", "RRPPY.", "YYYRGG", "RRPRYG", "RGGPPY", "GPPRYY"])
        self.assertEqual(b.eyeless_cells, frozenset())

    def test_no_banner_is_not_detected(self) -> None:
        # 文字の無い盤面(コーパス中で最も文字に似た得点 0.49 の画像)では検出しない
        b = self._read("no_banner")
        self.assertEqual(b.banner_score, 0.0)

    def test_tracked_banner_survives_heavy_occlusion(self) -> None:
        # 一度検出した文字は、ぷよに大きく隠れて一致度が閾値を下回っても、インクの画素が一致する間は
        # 除き続ける(実機 2026-09-20 14:41: 積みが上がって一致度 0.47 になり、文字の残りが YYY と読まれた)
        from src.capture.geometry import BoardGeometry
        from src.vision.banner_detector import BANNER_LOST_FRAMES, BANNER_SCORE_MIN, AllClearBannerDetector
        from src.vision.board_reader import WARP_CELL_PX

        img = cv2.imread(str(FIXTURES / "board_all_clear_banner_empty.png"))
        h, w = img.shape[:2]
        geo = BoardGeometry(((0, WARP_CELL_PX), (w, WARP_CELL_PX), (w, h), (0, h)))
        det = AllClearBannerDetector()
        first = read_board(img, geo, banner_detector=det)
        self.assertGreaterEqual(first.banner_score, BANNER_SCORE_MIN)
        # 文字の左 2/3 を青ぷよ風の円で覆う(文字の一部だけが見える状態)
        occluded = img.copy()
        for cx in (36, 68, 100):
            for cy in (96, 128, 160):
                cv2.circle(occluded, (cx, cy), 16, (200, 80, 0), -1)
        stateless = read_board(occluded, geo)
        self.assertEqual(stateless.banner_score, 0.0)  # 状態が無いと見失う
        tracked = read_board(occluded, geo, banner_detector=det)
        self.assertGreater(tracked.banner_score, 0.0)
        # 見えている右側の文字は黄・おじゃまとして読まれない(以前は "し" が Y と読まれた)
        rows = cells_to_text(tracked.cells).splitlines()
        self.assertEqual([r[4:] for r in rows[2:5]], ["..", "..", ".."])
        # 文字が無い盤面が続けば追跡をやめる
        plain = cv2.imread(str(FIXTURES / "board_no_banner.png"))
        for _ in range(BANNER_LOST_FRAMES):
            last = read_board(plain, geo, banner_detector=det)
        self.assertEqual(last.banner_score, 0.0)
        self.assertIsNone(det.active)
