"""シミュレーションモードの入力(src/simulator/controls.py)のテスト。"""

import tempfile
import unittest
from pathlib import Path

from PyQt6 import QtCore

from src.simulator import controls


class ControlsTest(unittest.TestCase):
    def test_key_name_with_modifier(self):
        self.assertEqual(controls.key_name(QtCore.Qt.Key.Key_R, QtCore.Qt.KeyboardModifier.ShiftModifier), "Shift+R")
        self.assertEqual(controls.key_name(QtCore.Qt.Key.Key_Left, QtCore.Qt.KeyboardModifier.NoModifier), "Left")
        self.assertIsNone(controls.key_name(QtCore.Qt.Key.Key_Shift, QtCore.Qt.KeyboardModifier.ShiftModifier))

    def test_bindings_roundtrip_and_fill_missing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "keys.json"
            path.write_text('{"drop": "Space", "unknown": "Q"}', encoding="utf-8")
            b = controls.load_bindings(path)
            self.assertEqual(b["drop"], "Space")
            self.assertEqual(b["move_left"], "Left")
            self.assertNotIn("unknown", b)
            controls.save_bindings(b, path)
            self.assertEqual(controls.load_bindings(path), b)
        self.assertEqual(controls.action_for(controls.DEFAULT_BINDINGS, "Shift+R"), "reset_new")

    def test_pad_inputs(self):
        inputs = controls.inputs_from_raw(0b11, 9000, 0, 32767)  # ボタン1・2、十字キー右、スティック左
        self.assertEqual(inputs, {"Button1", "Button2", "PovRight", "X-"})
        self.assertEqual(
            controls.actions_for_inputs(controls.DEFAULT_PAD_BINDINGS, inputs),
            {"rotate_ccw", "rotate_cw", "move_right", "move_left"},
        )

    def test_repeat_only_for_moves(self):
        done = []
        r = controls.RepeatTracker()
        r.update({"move_left", "drop"}, 0, done.append)
        r.update({"move_left", "drop"}, controls.REPEAT_DELAY_MS, done.append)
        r.update({"move_left", "drop"}, controls.REPEAT_DELAY_MS + controls.REPEAT_INTERVAL_MS, done.append)
        self.assertEqual(done.count("move_left"), 3)
        self.assertEqual(done.count("drop"), 1)  # 置く操作は押しっぱなしでも1回


    def test_step_texts(self):
        from src.simulator.window import step_texts

        lines = step_texts(["rotate_cw", "move_right", "move_right"], controls.DEFAULT_BINDINGS)
        self.assertEqual(lines, ["1. 右回転 (X)", "2. 右へ移動 (Right) ×2", "3. 置く (Down)"])


if __name__ == "__main__":
    unittest.main()
