"""メイン画面のテスト"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets  # noqa: E402

from src.autoplay_support import AVAILABLE as AUTOPLAY_AVAILABLE  # noqa: E402
from src.app import ENGINE_CHOICES, MainWindow  # noqa: E402
from src.toggle_switch import ToggleSwitch  # noqa: E402


def _select(window: MainWindow, side: str, engine: str) -> None:
    window.side_settings[side].engine_combo.setCurrentIndex(list(ENGINE_CHOICES).index(engine))


class MainWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.w = MainWindow()

    def tearDown(self) -> None:
        self.w.close()

    def test_two_panels_with_engine_choice(self) -> None:
        # 1P・2P の2つの枠で、思考エンジンは YamaNEKO / ama / 支援なし(利用者 2026-10-25)
        self.assertEqual(sorted(self.w.side_settings), ["1p", "2p"])
        combo = self.w.side_settings["2p"].engine_combo
        self.assertEqual([combo.itemText(i) for i in range(combo.count())], ["YamaNEKO", "ama", "支援なし"])
        self.assertEqual(self.w._sides(), ["1p"])  # 初期は 1P だけ YamaNEKO
        _select(self.w, "2p", "ama")
        self.assertEqual(self.w._sides(), ["1p", "2p"])
        _select(self.w, "1p", "none")
        self.assertEqual(self.w._sides(), ["2p"])

    def test_yamaneko_only_options_are_disabled_otherwise(self) -> None:
        # 設置キャンセル・段差キャンセル・配ぷよ読みは YamaNEKO の時だけ設定できる(他は暗転)
        box = self.w.side_settings["1p"]
        box.haipuyo.setChecked(True)
        self.assertTrue(box.haipuyo.isEnabled())
        self.assertTrue(box.yamaneko_option(box.haipuyo))
        _select(self.w, "1p", "ama")
        for sw in (box.place_cancel, box.step_cancel, box.haipuyo, box.peek):
            self.assertFalse(sw.isEnabled())
        self.assertFalse(box.yamaneko_option(box.haipuyo))  # ama には渡さない
        self.assertTrue(box.autoplay.isEnabled())
        _select(self.w, "1p", "none")
        self.assertFalse(box.autoplay.isEnabled())

    def test_all_switches_are_toggles(self) -> None:
        # チェックボックスはすべてトグルスイッチ
        boxes = self.w.findChildren(QtWidgets.QCheckBox)
        self.assertTrue(boxes)
        self.assertTrue(all(isinstance(b, ToggleSwitch) for b in boxes))

    def test_autoplay_per_side(self) -> None:
        self.w.side_settings["2p"].autoplay.setChecked(True)
        self.assertFalse(self.w.side_settings["1p"].autoplay.isChecked())

    def test_tabs_switch_versus_solo_simulator(self) -> None:
        # 対戦モード・一人用モード・シミュレーターをタブで切り替える(利用者 2026-10-25)
        tabs = self.w.tabs
        self.assertEqual([tabs.tabText(i) for i in range(tabs.count())], ["対戦モード", "一人用モード", "シミュレーター"])
        _select(self.w, "2p", "ama")
        self.assertFalse(self.w._solo())
        self.assertEqual(self.w._sides(), ["1p", "2p"])
        tabs.setCurrentIndex(1)  # 一人用: 自分の盤面1つだけ。キャリブレーションは一人用のファイル
        self.assertTrue(self.w._solo())
        self.assertEqual(self.w._sides(), ["1p"])
        self.assertIs(self.w._settings("1p"), self.w.solo_settings)
        self.assertEqual(self.w._calibration_path().name, "calibration_solo.json")

    def test_simulator_tab_hides_assist_button(self) -> None:
        # シミュレーターのタブでは支援モードの開始・ログは出さない(シミュレーター本体はタブを開いた時に作る)
        opened = []
        self.w._open_simulator = lambda: opened.append(True)  # type: ignore[method-assign]
        self.w.show()
        self.w.tabs.setCurrentIndex(2)
        self.assertEqual(opened, [True])
        self.assertFalse(self.w.assist_btn.isVisible())
        self.w.tabs.setCurrentIndex(0)
        self.assertTrue(self.w.assist_btn.isVisible())

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_tap_pad_releases_button(self) -> None:
        # 「A を押す」は押した後に必ず離す。離す処理が呼ばれず A が押しっぱなしになり、2回目以降が効かなかった(2026-10-25 実機)
        import sys
        import types
        from unittest import mock

        from src import app as app_module
        from pba_autoplay import pad_sender

        calls: list[tuple[str, str]] = []

        class FakePad:
            def press_button(self, button: str) -> None:
                calls.append(("press", button))

            def release_button(self, button: str) -> None:
                calls.append(("release", button))

            def update(self) -> None:
                pass

        fake = types.ModuleType("vgamepad")
        fake.VX360Gamepad = FakePad
        fake.XUSB_BUTTON = types.SimpleNamespace(XUSB_GAMEPAD_A="A")
        with mock.patch.dict(sys.modules, {"vgamepad": fake}), mock.patch.dict(pad_sender._SHARED_PADS, {}, clear=True), \
                mock.patch.object(app_module, "PAD_TAP_DELAY_MS", 10):
            box = self.w.side_settings["1p"]
            for _ in range(2):
                box.tap_pad("PAD_A")
                loop = QtCore.QEventLoop()
                QtCore.QTimer.singleShot(400, loop.quit)
                loop.exec()
        self.assertEqual(calls, [("press", "A"), ("release", "A"), ("press", "A"), ("release", "A")])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_pad_keyboard_per_side(self) -> None:
        # キーボードで仮想パッドを操作するのトグルは 1P・2P 別々(利用者 2026-10-25)
        self.assertEqual(sorted(self.w.pad_kb_switches), ["1p", "2p"])
        self.w._on_pad_kb_toggled("2p", True)  # 切り替えキーで 2P だけオンになった
        self.assertTrue(self.w.pad_kb_switches["2p"].isChecked())
        self.assertFalse(self.w.pad_kb_switches["1p"].isChecked())


if __name__ == "__main__":
    unittest.main()


class CpuControlTest(unittest.TestCase):
    """思考エンジンの負荷の設定と CPU 使用率のログ(2026-10-26、利用者「移動ミスは CPU の負荷によるものではないか」)。"""

    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.w = MainWindow()

    def tearDown(self) -> None:
        self.w.close()

    def test_threads_and_priority_passed_to_engine(self) -> None:
        from unittest import mock

        from src import app

        box = self.w.side_settings["1p"]
        started: dict = {}
        with mock.patch.object(app, "AmaClient", lambda **kw: started.update(kw) or "client"), \
                mock.patch.object(app.os, "cpu_count", return_value=16):
            self.w._make_client(box, None)
            self.assertIsNone(started["search_threads"])  # 自動・1つだけ: エンジンの既定(全コア)
            self.assertFalse(started["low_priority"])
            box.threads_combo.setCurrentIndex(box.threads_combo.findData("quarter"))
            box.low_priority.setChecked(True)
            self.w._make_client(box, 8)
            self.assertEqual(box.search_threads(8), 4)
        self.assertEqual(started["search_threads"], 4)
        self.assertTrue(started["low_priority"])
        box.threads_combo.setCurrentIndex(box.threads_combo.findData("auto"))
        self.assertEqual(box.search_threads(8), 8)  # 自動は両方の時の半分のまま

    def test_peek_and_haipuyo_args_passed_to_engine(self) -> None:
        # 相手のツモを覗く(2026-11-01)・配ぷよ読みは YamaNEKO の起動引数になる(両方なら両方つなぐ。どちらも無ければ None)
        from unittest import mock

        from src import app
        from src.engine.ama_client import HAIPUYO_BRIDGE_ARGS, PEEK_BRIDGE_ARGS

        box = self.w.side_settings["1p"]
        started: dict = {}
        with mock.patch.object(app, "AmaClient", lambda **kw: started.update(kw) or "client"):
            self.w._make_client(box, None)
            self.assertIsNone(started["extra_args"])
            box.peek.setChecked(True)
            self.w._make_client(box, None)
            self.assertEqual(started["extra_args"], PEEK_BRIDGE_ARGS)
            box.haipuyo.setChecked(True)
            self.w._make_client(box, None)
            self.assertEqual(started["extra_args"], HAIPUYO_BRIDGE_ARGS + PEEK_BRIDGE_ARGS)

    def test_thread_choice_only_for_yamaneko(self) -> None:
        box = self.w.side_settings["2p"]
        _select(self.w, "2p", "ama")
        self.assertFalse(box.threads_combo.isEnabled())  # ama のスレッド数は変えられない
        self.assertTrue(box.low_priority.isEnabled())  # 優先度はどのエンジンでも

    def test_tick_stats_report_cpu(self) -> None:
        from src.app import TICK_STATS_SECONDS, TickStats

        st = TickStats()
        st.add(0.0, 0.0, 0.01, 0.02)
        st.cpu += [30.0, 80.0, 50.0]
        report = st.add(TICK_STATS_SECONDS + 1, 1.0, 0.01, 0.02)
        self.assertIn("CPU 中央 50%・最大 80%", report)
