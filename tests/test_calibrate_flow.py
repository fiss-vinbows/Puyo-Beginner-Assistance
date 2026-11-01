"""キャリブレーションUIの遷移テスト(オフスクリーンで、クリック・キー入力を合成して流す)"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from src.capture.calibrate import CalibrationOverlay, load_calibration  # noqa: E402


def _click(overlay: CalibrationOverlay, x: int, y: int) -> None:
    pos = QtCore.QPointF(x, y)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress, pos, pos, QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.KeyboardModifier.NoModifier,
    )
    overlay.mousePressEvent(event)


def _key(overlay: CalibrationOverlay, key: QtCore.Qt.Key, shift: bool = False) -> None:
    mod = QtCore.Qt.KeyboardModifier.ShiftModifier if shift else QtCore.Qt.KeyboardModifier.NoModifier
    overlay.keyPressEvent(QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, mod))


class CalibrationFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "c.json"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_solo_flow_two_points_fine_tune_and_next(self) -> None:
        o = CalibrationOverlay(solo=True, save_path=self.path)
        _click(o, 100, 100)  # 盤面 左上
        self.assertEqual(o._step_index, 1)
        _click(o, 400, 700)  # 盤面 右下 → 微調整へ
        self.assertIsNotNone(o._fine_tune)
        _key(o, QtCore.Qt.Key.Key_Right)  # 全体を右へ1px
        _key(o, QtCore.Qt.Key.Key_Down, shift=True)  # 右下を下へ1px
        _key(o, QtCore.Qt.Key.Key_Return)  # 確定
        self.assertIsNone(o._fine_tune)
        self.assertEqual(o._step_index, 2)
        for pt in ((500, 100), (560, 220), (560, 240), (600, 340)):
            _click(o, *pt)
        # 一人用: 相手側は複製されて保存される
        self.assertTrue(self.path.exists())
        result = load_calibration(self.path)
        self.assertEqual(result.self_player.board_quad[0], (101.0, 100.0))
        self.assertEqual(result.self_player.board_quad[2], (401.0, 701.0))
        self.assertEqual(result.enemy_player.board_quad, result.self_player.board_quad)

    def test_versus_flow_requires_enemy(self) -> None:
        o = CalibrationOverlay(solo=False, save_path=self.path)
        _click(o, 100, 100)
        _click(o, 400, 700)
        _key(o, QtCore.Qt.Key.Key_Return)
        for pt in ((500, 100), (560, 220), (560, 240), (600, 340)):
            _click(o, *pt)
        self.assertFalse(self.path.exists())  # まだ相手側が残っている
        self.assertEqual(o._step_index, 6)
        _click(o, 1100, 100)
        _click(o, 1400, 700)
        _key(o, QtCore.Qt.Key.Key_Return)
        for pt in ((1000, 100), (1060, 220), (1000, 240), (1040, 340)):
            _click(o, *pt)
        self.assertTrue(self.path.exists())


if __name__ == "__main__":
    unittest.main()
