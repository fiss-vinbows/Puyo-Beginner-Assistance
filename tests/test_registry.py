"""思考エンジン・定跡の登録と、定跡を AI の思考より優先する BookClient のテスト"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

from src.engine.ama_client import BattleMove, PlayerState  # noqa: E402
from src.engine.book_client import BookClient, book_text, is_book_move  # noqa: E402
from src.engine.puyo_field import PuyoField  # noqa: E402
from src.engine.registry import BUILTIN_BOOK, BookEntry, EngineEntry, Registry  # noqa: E402
from src.simulator.opening_book import BOOK_PATH, OpeningBook  # noqa: E402
from src.simulator.rules import SimState  # noqa: E402


class _FakeInner:
    def __init__(self) -> None:
        self.args = ["engine.exe"]
        self.sent: list = []

    def _move(self) -> BattleMove:
        return BattleMove(5, "UP", 0, (12, 5), (11, 5), "harass", None, None, 0, path=["RIGHT", "DROP"])

    def battle(self, me, enemy, **kw):  # noqa: ANN001, ANN003, ANN201
        self.sent.append((me.history, enemy.placed))
        return self._move()

    def solo(self, field, pairs):  # noqa: ANN001, ANN201
        return self._move()

    def close(self) -> None:
        self.sent.append("close")


def _book_start() -> tuple[OpeningBook, SimState]:
    book = OpeningBook.load()
    for seed in range(1, 200):
        s = SimState.new(seed=seed)
        if book.lookup(s.board, s.pairs[:3], 0) is not None:
            return book, s
    raise AssertionError("定跡に載っている配列が見つからない")


@unittest.skipUnless(BOOK_PATH.exists(), "組み込みの定跡が無い")
class BookClientTest(unittest.TestCase):
    def test_book_move_overrides_engine_placement(self) -> None:
        # 定跡どおりの間は定跡の手を優先する(置き場所・操作手順・置いた後の盤面を差し替え、ラベルは構築)
        book, s = _book_start()
        inner = _FakeInner()
        client = BookClient(inner, book)
        move = client.solo(PuyoField(), s.pairs[:3], history=[])
        self.assertTrue(is_book_move(move))
        self.assertEqual(move.label, "build")
        expected = book.lookup(s.board, s.pairs[:3], 0)
        self.assertEqual({(move.axis_cell, s.pairs[0][0]), (move.child_cell, s.pairs[0][1])}, set(expected.cells))
        self.assertEqual(move.path[-1], "DROP")
        self.assertIsNotNone(move.field_after)
        self.assertIn("の1手目", book_text(move))

    def test_without_history_or_off_book_uses_engine(self) -> None:
        book, s = _book_start()
        client = BookClient(_FakeInner(), book)
        self.assertFalse(is_book_move(client.solo(PuyoField(), s.pairs[:3])))  # 第1手からの履歴が無い
        odd = PuyoField.from_rows(["......"] * 12 + ["RRYY.."])
        self.assertFalse(is_book_move(client.solo(odd, s.pairs[:3], history=[])))  # 盤面が定跡と違う

    def test_history_not_sent_to_engine_without_haipuyo(self) -> None:
        # 定跡のために付けた履歴は、配ぷよ読みで起動していないエンジンには送らない
        book, s = _book_start()
        inner = _FakeInner()
        client = BookClient(inner, book)
        me = PlayerState(PuyoField(), s.pairs[:3], history=[], placed=0)
        en = PlayerState(PuyoField(), s.pairs[:3], placed=0)
        move = client.battle(me, en)
        self.assertEqual(inner.sent[-1], (None, None))
        self.assertTrue(is_book_move(move))
        inner.args = ["yamaneko-bridge.exe", "--haipuyo-threat"]
        client.battle(me, en)
        self.assertEqual(inner.sent[-1], ([], 0))
        client.close()  # ほかの呼び出しはそのままエンジンへ
        self.assertEqual(inner.sent[-1], "close")


class RegistryTest(unittest.TestCase):
    def test_save_and_load(self) -> None:
        path = Path(tempfile.mkdtemp()) / "registry.json"
        self.assertEqual(Registry.load(path).selected_book, BUILTIN_BOOK)  # 無ければ空の登録
        reg = Registry([EngineEntry("MyAI", "C:/ai/my.exe", ["--fast"])], [BookEntry("GTR定跡", "C:/books/gtr.json")], "GTR定跡")
        reg.save(path)
        got = Registry.load(path)
        self.assertEqual(got.engine("MyAI").args, ["--fast"])
        self.assertEqual(got.book_path(), Path("C:/books/gtr.json"))
        self.assertIsNone(got.book_path(BUILTIN_BOOK))

    def test_dialogs_validate(self) -> None:
        from src.registry_dialogs import BookRegistryDialog, EngineRegistryDialog

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        exe = Path(tempfile.mkdtemp()) / "my-ai.exe"
        exe.write_text("")
        d = EngineRegistryDialog(Registry())
        self.assertIsNone(d.add_engine("MyAI", str(exe), "--depth 3"))
        self.assertEqual(d.engines[0].args, ["--depth", "3"])
        self.assertIn("もう使われている", d.add_engine("MyAI", str(exe)))
        self.assertIn("もう使われている", d.add_engine("YamaNEKO", str(exe)))
        self.assertIn("見つかりません", d.add_engine("Other", str(exe) + ".none"))
        b = BookRegistryDialog(Registry())
        bad = exe.with_suffix(".json")
        bad.write_text('{"version": 2}')
        self.assertIn("形式が違います", b.add_book("壊れた定跡", str(bad)))
        if BOOK_PATH.exists():
            self.assertIsNone(b.add_book("コピー", str(BOOK_PATH)))


class MainWindowRegistryTest(unittest.TestCase):
    def test_registered_engine_in_list_and_started_with_args(self) -> None:
        from src import app

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with mock.patch.object(app.Registry, "load", return_value=Registry([EngineEntry("MyAI", "C:/ai/my.exe", ["--fast"])])):
            w = app.MainWindow()
        box = w.side_settings["2p"]
        idx = box.engine_combo.findData("custom:MyAI")
        self.assertGreaterEqual(idx, 0)
        box.engine_combo.setCurrentIndex(idx)
        self.assertEqual(w._sides(), ["1p", "2p"])
        self.assertFalse(box.haipuyo.isEnabled())  # YamaNEKO だけの設定は暗転
        started = {}
        with mock.patch.object(app, "AmaClient", lambda **kw: started.update(kw) or "client"):
            self.assertEqual(w._make_client(box, None), "client")
        self.assertEqual((str(started["exe_path"]), started["custom_args"]), (str(Path("C:/ai/my.exe")), ["--fast"]))
        w.close()

    def test_books_chosen_per_side(self) -> None:
        # 定跡は 1P・2P・一人用の枠ごとに選び、側ごとに保存する(2026-10-28)
        from src import app

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        reg = Registry(books=[BookEntry("GTR定跡", "C:/books/gtr.json")], side_books={"2p": "GTR定跡"})
        with mock.patch.object(app.Registry, "load", return_value=reg), mock.patch.object(Registry, "save") as save:
            w = app.MainWindow()
            self.assertEqual(w.side_settings["1p"].book_name(), "")
            self.assertEqual(w.side_settings["2p"].book_name(), "GTR定跡")
            box = w.side_settings["1p"]
            box.book_combo.setCurrentIndex(box.book_combo.findData(BUILTIN_BOOK))
            self.assertEqual(reg.side_books, {"1p": BUILTIN_BOOK, "2p": "GTR定跡", "solo": ""})
            self.assertEqual(reg.selected_book, BUILTIN_BOOK)  # シミュレーターは 1P の定跡
            save.assert_called()
            w.close()

    def test_side_books_saved(self) -> None:
        path = Path(tempfile.mkdtemp()) / "registry.json"
        Registry(side_books={"1p": BUILTIN_BOOK}).save(path)
        self.assertEqual(Registry.load(path).side_books, {"1p": BUILTIN_BOOK})


if __name__ == "__main__":
    unittest.main()
