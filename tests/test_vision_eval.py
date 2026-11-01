"""画像認識の評価用データ(src/vision/eval_dataset.py)のテスト"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2

from src.capture.calibrate import load_calibration
from src.vision.eval_dataset import compare, load_label, make_label, read_frame, save_label, summarize, validate_label

FIXTURES = Path(__file__).parent / "fixtures"
EMPTY_BOARD = ["......"] * 12


def _player(board=None, next1="RY", next2="GB", pending=0) -> dict:
    return {"board": list(board or EMPTY_BOARD), "next1": next1, "next2": next2, "pending": pending}


def _label(self_p: dict, enemy_p: dict | None = None, screen: str = "game") -> dict:
    return {"image": "x.png", "calibration": {}, "screen": screen, "verified": True, "note": "",
            "players": {"self": self_p, "enemy": enemy_p or _player()}}


class CompareTest(unittest.TestCase):
    def test_counts_matches_and_errors(self):
        want = _player(EMPTY_BOARD[:11] + ["RR...."])
        got = _player(EMPTY_BOARD[:11] + ["RG...."], next1="R?")
        res = compare("a", _label(want), {"self": got, "enemy": _player()})
        self.assertEqual(res.cells[("R", "R")], 1)
        self.assertEqual(res.cells[("R", "G")], 1)
        self.assertEqual(res.nexts[("Y", "?")], 1)  # 読めなかった(?)は誤り
        self.assertIn("self 1段目2列目: 正解R 読みG", res.errors)
        self.assertIn("self next1の軸: 正解Y 読み?", res.errors)

    def test_dont_care_is_skipped(self):
        want = _player(["*....."] + EMPTY_BOARD[1:], next1="*Y", pending="*")
        got = _player(["G....."] + EMPTY_BOARD[1:], next1="BY", pending=5)
        res = compare("a", _label(want), {"self": got, "enemy": _player()})
        self.assertEqual(res.errors, [])
        self.assertEqual(sum(res.cells.values()), 12 * 6 * 2 - 1)
        self.assertEqual(res.pending_total, 1)  # 相手の予告だけ

    def test_non_game_screen_is_not_scored(self):
        res = compare("a", _label(_player(), screen="result"), {"self": _player(["RRRRRR"] * 12), "enemy": _player()})
        self.assertEqual(sum(res.cells.values()), 0)
        self.assertEqual(res.errors, [])

    def test_summary_reports_filled_cell_accuracy(self):
        want = _player(EMPTY_BOARD[:11] + ["RR...."])
        got = _player(EMPTY_BOARD[:11] + ["RRG..."])
        text = summarize([compare("a", _label(want), {"self": got, "enemy": _player()})])
        self.assertIn("うちぷよのあるマス: 2/2 (100.00%)", text)
        self.assertIn("盤面のマス: 143/144", text)


class ValidateTest(unittest.TestCase):
    def test_rejects_bad_shapes_and_letters(self):
        with self.assertRaises(ValueError):
            validate_label(_label(_player(EMPTY_BOARD[:11])))  # 11行しかない
        with self.assertRaises(ValueError):
            validate_label(_label(_player(EMPTY_BOARD[:11] + ["?....."])))  # ? は正解に使えない
        with self.assertRaises(ValueError):
            validate_label(_label(_player(next1="R")))
        with self.assertRaises(ValueError):
            validate_label(_label(_player(), screen="title"))


class RoundTripTest(unittest.TestCase):
    @unittest.skipUnless((FIXTURES / "frame_02400.png").exists(), "ゲーム画面全体の画像は公開版に入れていない")
    def test_label_from_fixture_frame_scores_perfectly(self):
        """実機フレームの読みから作った初期ラベルは、同じ読みと比べて誤りなし(? は * になる)。"""
        cal = load_calibration(FIXTURES / "calibration_1080p.json")
        image = cv2.imread(str(FIXTURES / "frame_02400.png"))
        reading = read_frame(image, cal)
        label = make_label("frame_02400.png", cal, reading)
        self.assertNotIn("?", json.dumps(label["players"]))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.json"
            save_label(path, label)
            loaded = load_label(path)
        self.assertFalse(loaded["verified"])
        self.assertEqual(compare("a", loaded, reading).errors, [])


if __name__ == "__main__":
    unittest.main()
