"""Steam 版のウィンドウ検出(盤面・ネクスト枠の位置の計算)のテスト"""

import json
import os
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from src.capture.calibrate import CalibrationResult  # noqa: E402
from src.capture.window_detect import (  # noqa: E402
    CLIENT_SIZE_PRESETS,
    GameWindow,
    resized_window_rect,
    SavedWindow,
    WindowEntry,
    find_game_window,
    load_saved_window,
    save_saved_window,
    content_rect,
    detect_calibration,
    is_game_title,
    layout_calibration,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TitleTests(unittest.TestCase):
    def test_game_titles(self) -> None:
        for title in ("Puyo Puyo Tetris 2", "ぷよぷよ™テトリス®２", "PuyoPuyoTetris2", "Puyo Puyo Champions", "ぷよぷよｅスポーツ"):
            self.assertTrue(is_game_title(title), title)
        for title in ("Tetris 99", "OBS 30.0 - ぷよぷよテトリス", "Puyo Puyo Tetris", ""):
            self.assertFalse(is_game_title(title), title)


class FindTests(unittest.TestCase):
    def test_find_by_title_exe_or_saved_choice(self) -> None:
        game = WindowEntry(GameWindow(1, "Game", 0, 0, 1280, 720), "unknown.exe")
        other = WindowEntry(GameWindow(2, "メモ帳", 0, 0, 800, 600), "notepad.exe")
        # 決め打ちに合わないウィンドウは見つからない(2026-10-28 実機)
        self.assertIsNone(find_game_window(SavedWindow(), [game, other]))
        # 一覧から選んだもの(実行ファイル名、読めなければタイトル)で見つかる。ゲームを起動し直しても同じ名前で探せる
        self.assertEqual(find_game_window(SavedWindow(exe="unknown.exe"), [other, game]), game.window)
        self.assertEqual(find_game_window(SavedWindow(title="Game"), [other, game]), game.window)
        titled = WindowEntry(GameWindow(3, "Puyo Puyo Tetris 2", 0, 0, 1920, 1080), "")
        self.assertEqual(find_game_window(SavedWindow(), [other, titled]), titled.window)

    def test_saved_window_round_trip(self) -> None:
        import tempfile

        path = Path(tempfile.mkdtemp()) / "game_window.json"
        self.assertIsNone(load_saved_window(path))
        save_saved_window(SavedWindow(exe="game.exe"), path)
        self.assertEqual(load_saved_window(path), SavedWindow(exe="game.exe"))


class ResizeTests(unittest.TestCase):
    def test_presets_are_16_9(self) -> None:
        for w, h in CLIENT_SIZE_PRESETS:
            self.assertEqual(w * 9, h * 16)

    def test_border_kept_and_fit_in_work_area(self) -> None:
        # 枠・タイトルバーの分(幅 +14、高さ +45)はそのまま足し、左上は動かさない
        window, client = (100, 50, 1014, 608), (107, 88, 1000, 563)
        self.assertEqual(resized_window_rect(window, client, (1600, 900)), (100, 50, 1614, 945))
        # 作業領域からはみ出すなら左・上へずらす
        self.assertEqual(resized_window_rect(window, client, (1920, 1080), (0, 0, 1920, 1040)), (0, 0, 1934, 1125))
        self.assertEqual(resized_window_rect((600, 300, 1014, 608), client, (1600, 900), (0, 0, 2560, 1400)), (600, 300, 1614, 945))
        self.assertEqual(resized_window_rect((1500, 800, 1014, 608), client, (1600, 900), (0, 0, 2560, 1400)), (946, 455, 1614, 945))


class PickDialogTests(unittest.TestCase):
    def test_selected_window_saved_by_exe_or_title(self) -> None:
        from PyQt6 import QtWidgets

        from src.window_dialog import WindowPickDialog

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        entries = [WindowEntry(GameWindow(1, "Game", 0, 0, 1280, 720), "game.exe"),
                   WindowEntry(GameWindow(2, "保護されたゲーム", 0, 0, 1280, 720), "")]
        d = WindowPickDialog(entries)
        self.assertEqual(d.selected(), SavedWindow(exe="game.exe"))
        d.list.setCurrentRow(1)
        self.assertEqual(d.selected(), SavedWindow(title="保護されたゲーム"))  # 実行ファイル名が読めなければタイトルで


class LayoutTests(unittest.TestCase):
    def test_1080p_window_matches_reference(self) -> None:
        # 1920×1080 のウィンドウでは、録画で検証済みのキャリブレーションと同じ(ウィンドウの位置だけずれる)
        ref = json.loads((FIXTURES / "calibration_1080p.json").read_text(encoding="utf-8"))
        cal = layout_calibration(GameWindow(1, "", 100, 50, 1920, 1080)).to_dict()
        for player in ("self", "enemy"):
            self.assertEqual(cal[player]["board_quad"], [[x + 100, y + 50] for x, y in ref[player]["board_quad"]])
            self.assertEqual(cal[player]["next_rects"], [[x + 100, y + 50, w, h] for x, y, w, h in ref[player]["next_rects"]])

    def test_real_window_2516x1415(self) -> None:
        # 利用者の実機の録画(2026-10-28、ぷよぷよテトリス2、タイトル PuyoPuyoTetris2): クライアント領域 (12, 51) 2516×1415 で、
        # 画像から見つけた盤面は 自分 (369, 259)-(892, 1208)、相手 (1647, 259)-(2172, 1208)。計算した位置が1px 以内で合う
        cal = layout_calibration(GameWindow(1, "PuyoPuyoTetris2", 12, 51, 2516, 1415))
        for got, want in ((cal.self_player.board_quad, (369, 259, 892, 1208)), (cal.enemy_player.board_quad, (1647, 259, 2172, 1208))):
            for a, b in zip((got[0][0], got[0][1], got[2][0], got[2][1]), want):
                self.assertLessEqual(abs(a - b), 1)

    def test_esports_same_layout(self) -> None:
        # ぷよぷよeスポーツ(タイトル Puyo Puyo Champions)の録画(2026-10-28): 配置はテトリス2と同じ。クライアント領域 (3, 49)
        # 2515×1414 で、画像から見つけた盤面は 自分 (360, 257)-(884, 1204)、相手 (1638, 257)-(2162, 1204)
        cal = layout_calibration(GameWindow(1, "Puyo Puyo Champions", 3, 49, 2515, 1414))
        for got, want in ((cal.self_player.board_quad, (360, 257, 884, 1204)), (cal.enemy_player.board_quad, (1638, 257, 2162, 1204))):
            for a, b in zip((got[0][0], got[0][1], got[2][0], got[2][1]), want):
                self.assertLessEqual(abs(a - b), 1)

    def test_scaled_window(self) -> None:
        # 1280×720 なら 2/3 倍
        cal = layout_calibration(GameWindow(1, "", 0, 0, 1280, 720))
        (x0, y0), _, (x1, y1), _ = cal.self_player.board_quad
        self.assertAlmostEqual(x0, 272 * 2 / 3, delta=1)
        self.assertAlmostEqual(x1 - x0, 400 * 2 / 3, delta=1)
        self.assertAlmostEqual(y1 - y0, 723 * 2 / 3, delta=1)
        CalibrationResult.from_dict(cal.to_dict())  # 読み込み時の検査を通る

    def test_letterbox(self) -> None:
        # 横長のウィンドウでは左右に黒帯(高さで倍率が決まり、中央に描かれる)
        ox, oy, scale = content_rect(2000, 1080)
        self.assertEqual((ox, oy, scale), (40, 0, 1.0))
        ox, oy, scale = content_rect(1920, 1200)
        self.assertEqual((ox, oy, scale), (0, 60, 1.0))

    def test_solo_copies_self(self) -> None:
        cal = layout_calibration(GameWindow(1, "", 0, 0, 1920, 1080), solo=True)
        self.assertEqual(cal.enemy_player, cal.self_player)


class DetectTests(unittest.TestCase):
    def test_refines_to_detected_board(self) -> None:
        # 画像の盤面の暗い矩形が計算した位置から少しずれていれば、その位置に合わせ直す(ネクスト枠も一緒に動く)
        window = GameWindow(1, "", 10, 20, 1920, 1080)
        image = np.full((1080, 1920, 3), 220, np.uint8)
        cv2.rectangle(image, (280, 165), (280 + 399, 165 + 722), (30, 30, 30), -1)  # 自分: 右下へ (8, 6) ずれ
        cv2.rectangle(image, (1248, 159), (1248 + 399, 159 + 722), (30, 30, 30), -1)
        grab = mock.Mock(return_value=image)
        cal = detect_calibration(window, grab=grab)
        grab.assert_called_once_with(window.region())
        self.assertEqual(cal.self_player.board_quad[0], (290.0, 185.0))
        self.assertEqual(cal.self_player.next_rects[0][:2], (720 + 8 + 10, 160 + 6 + 20))
        self.assertEqual(cal.enemy_player.board_quad[0], (1258.0, 179.0))

    def test_other_layout_uses_two_detected_boards(self) -> None:
        # どちらの盤面も計算した位置に無くても、同じ大きさの盤面が2つ見えればそれに合わせる(ぷよぷよeスポーツなど配置が違うゲーム)
        window = GameWindow(1, "Puyo Puyo Champions", 0, 0, 1920, 1080)
        image = np.full((1080, 1920, 3), 220, np.uint8)
        cv2.rectangle(image, (400, 200), (400 + 359, 200 + 649), (30, 30, 30), -1)
        cv2.rectangle(image, (1160, 200), (1160 + 359, 200 + 649), (30, 30, 30), -1)
        cal = detect_calibration(window, grab=lambda _r: image)
        self.assertEqual(cal.self_player.board_quad[0], (400.0, 200.0))
        self.assertEqual(cal.enemy_player.board_quad[0], (1160.0, 200.0))

    def test_far_or_missing_board_keeps_layout(self) -> None:
        window = GameWindow(1, "", 0, 0, 1920, 1080)
        image = np.full((1080, 1920, 3), 220, np.uint8)
        cv2.rectangle(image, (600, 100), (1000, 823), (30, 30, 30), -1)  # どちらの盤面とも離れている
        self.assertEqual(detect_calibration(window, grab=lambda _r: image), layout_calibration(window))

        def fail(_r):  # noqa: ANN001, ANN202
            raise OSError("capture")

        self.assertEqual(detect_calibration(window, grab=fail), layout_calibration(window))


class BadgeTests(unittest.TestCase):
    def test_badge_centered_on_title_bar(self) -> None:
        # ゲームのウィンドウのタイトルバーが分かれば、その中央にタイトルバーの高さで置き、検出状態を終了ボタンの隣に出す
        from PyQt6 import QtCore, QtWidgets

        from src.app import ControlBadge

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        anchor = QtCore.QRect(100, 50, 1600, 32)
        badge = ControlBadge(on_stop=lambda: None, detect_text="検出: Puyo Puyo Champions", anchor=anchor)
        badge.set_lines(["1P: 思考中", "2P: 思考中"])
        self.assertEqual(badge.height(), 32)
        self.assertLessEqual(abs(badge.geometry().center().x() - anchor.center().x()), 1)
        self.assertEqual(badge.y(), 50)
        self.assertIn("|", badge.label.text())  # 1行にまとめる
        self.assertIn("Champions", badge.detect_label.toolTip())
        # 終了ボタンは白いタイトルバーの上でも見えるよう黒地(2026-10-29)
        self.assertIn("background: black", badge.stop_button.styleSheet())
        free = ControlBadge(on_stop=lambda: None)  # 分からなければ今までどおり画面の右上
        free.set_lines(["1P: a", "2P: b"])
        self.assertEqual(free.height(), 26 + 18 * 2)


if __name__ == "__main__":
    unittest.main()
