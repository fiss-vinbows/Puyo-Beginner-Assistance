"""プラグインの仕組み(src/plugin_loader.py)と、同梱のプラグイン(plugins/)のテスト(2026-10-28)"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

from src.autoplay_support import AVAILABLE as AUTOPLAY_AVAILABLE  # noqa: E402
from src.overlay.renderer import OverlayDrawData, panel_lines, register_language  # noqa: E402
from src.plugin_loader import PLUGIN_DIR, PluginManager, discover, load_disabled, save_disabled  # noqa: E402
from tests.test_renderer import _analysis  # noqa: E402


def _bundled() -> PluginManager:
    plugins, errors = discover(PLUGIN_DIR)
    assert not errors, errors
    return PluginManager(plugins)


class LoaderTest(unittest.TestCase):
    def test_bundled_plugins(self) -> None:
        # 公開する本体に同梱するプラグイン。全自動の2つ(全自動モード・ローカル対戦)は非公開の PT-plugins を
        # plugins/full_auto に置いた時だけ読まれる(2026-11-01)
        m = _bundled()
        ids = {p.id for p in m.plugins}
        core = {"switch_capture", "language_mokou", "debug_log", "window_size"}
        self.assertEqual(ids - core, {"full_auto", "local_versus"} if AUTOPLAY_AVAILABLE else set())
        self.assertLessEqual(core, ids)
        self.assertTrue(m.debug_log())
        self.assertTrue(m.window_resize())
        self.assertTrue(m.manual_calibration())
        self.assertIn("mokou", m.languages())

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_autoplay_plugins(self) -> None:
        m = _bundled()
        self.assertEqual(m.autoplay_sides(), {"1p", "2p"})
        self.assertEqual(m.guarded_autoplay_sides(), set())  # 全自動モードが条件なしで許可している
        # 全自動モードは配布しない
        self.assertFalse(next(p for p in m.plugins if p.id == "full_auto").distributable)

    def test_disabled_plugins_do_not_contribute(self) -> None:
        m = _bundled()
        m.disabled = {"full_auto", "switch_capture", "language_mokou", "local_versus", "debug_log", "window_size"}
        self.assertEqual((m.autoplay_sides(), m.manual_calibration(), m.languages(), m.debug_log()), (set(), False, {}, False))

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_local_versus_guards_2p(self) -> None:
        # ローカル対戦用プラグインだけなら、2P だけ・ネームプレートの確認つきで全自動を許可する
        m = _bundled()
        m.disabled = {"full_auto"}
        self.assertEqual(m.autoplay_sides(), {"2p"})
        self.assertEqual(m.guarded_autoplay_sides(), {"2p"})

    def test_broken_plugin_is_skipped_with_reason(self) -> None:
        d = Path(tempfile.mkdtemp())
        (d / "broken").mkdir()
        (d / "broken" / "plugin.py").write_text("raise RuntimeError('壊れている')", encoding="utf-8")
        (d / "notplugin").mkdir()
        (d / "notplugin" / "plugin.py").write_text("PLUGIN = 1", encoding="utf-8")
        plugins, errors = discover(d)
        self.assertEqual(plugins, [])
        self.assertIn("壊れている", errors["broken"])
        self.assertIn("Plugin ではありません", errors["notplugin"])

    def test_disabled_saved(self) -> None:
        path = Path(tempfile.mkdtemp()) / "plugins.json"
        self.assertEqual(load_disabled(path), set())  # 無ければ全部有効
        save_disabled({"full_auto"}, path)
        self.assertEqual(load_disabled(path), {"full_auto"})


class MokouTest(unittest.TestCase):
    """言語プラグイン「mokou」: 発火の行の 11〜19連鎖を「数の呼び名」にする(2026-10-25)。"""

    def setUp(self) -> None:
        lang = _bundled().languages()["mokou"]
        self.to_mokou = lang.transform
        register_language("mokou", lang.transform, lang.word_break)

    def test_fire_lines_use_chain_names(self) -> None:
        self.assertEqual(self.to_mokou("発火 13連鎖 81,060点"), "発火 13の雷 81,060点")
        self.assertEqual(self.to_mokou("発火中 19連鎖 300,000点"), "発火中 19の夜明け 300,000点")
        self.assertEqual(self.to_mokou("発火 11連鎖(点数は再現できず)"), "発火 11の刃(点数は再現できず)")
        self.assertEqual(self.to_mokou("点数を訂正: 12連鎖 60,000点 → 14連鎖 90,000点"), "点数を訂正: 12の破戒 60,000点 → 14の母 90,000点")

    def test_other_lines_and_other_counts_unchanged(self) -> None:
        self.assertEqual(self.to_mokou("発火 10連鎖 50,000点"), "発火 10連鎖 50,000点")
        self.assertEqual(self.to_mokou("発火 20連鎖 400,000点"), "発火 20連鎖 400,000点")
        self.assertEqual(self.to_mokou("13連鎖 81,060点"), "13連鎖 81,060点")
        self.assertEqual(self.to_mokou("提案: 構築"), "提案: 構築")

    def test_panel_uses_mokou(self) -> None:
        data = OverlayDrawData((12, 0), (11, 0), "red", "yellow", "build", _analysis(13, 81060), _analysis(6, 10560),
                               enemy_firing=(15, 120000, True))
        lines = [line for line, _ in panel_lines(data, "mokou")]
        self.assertIn("発火中 15の夜 120,000点", lines)
        self.assertIn("13連鎖 81,060点", lines)


class MainWindowPluginTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from unittest import mock

        from src.app import MainWindow

        # 手元の config/plugins.json で無効にしたプラグインに左右されないよう、全部有効で始める
        with mock.patch("src.plugin_loader.load_disabled", return_value=set()):
            self.w = MainWindow()

    def tearDown(self) -> None:
        self.w.close()

    def test_without_plugins_autoplay_and_calibration_hidden(self) -> None:
        # 全自動モード・Switch 対応が無効なら、全自動の設定と手動キャリブレーションは出さない
        self.w.plugins.disabled = {"full_auto", "switch_capture", "local_versus"}
        self.w._apply_plugins()
        for box in [*self.w.side_settings.values(), self.w.solo_settings]:
            self.assertFalse(box.autoplay_available())
        self.assertTrue(self.w.autoplay_common.isHidden())
        self.assertTrue(all(b.isHidden() for b in self.w.calibrate_buttons))
        if AUTOPLAY_AVAILABLE:
            self.w.plugins.disabled = set()
            self.w._apply_plugins()
            self.assertTrue(self.w.side_settings["2p"].autoplay_available())
            self.assertFalse(self.w.autoplay_common.isHidden())

    def test_autoplay_hidden_without_autoplay_package(self) -> None:
        # 全自動を許可するプラグインがあっても、操作の仕組み(PT-plugins)が無ければ全自動の設定は出さない(2026-11-01)
        from unittest import mock

        from src import app as app_module
        from src.plugin_api import Plugin

        self.w.plugins.plugins.append(Plugin(id="allow_all", name="許可だけ", autoplay_sides=frozenset({"1p", "2p"})))
        self.w.plugins.disabled = set()
        with mock.patch.object(app_module, "AUTOPLAY_AVAILABLE", False):
            self.w._apply_plugins()
            self.assertEqual(self.w._autoplay_sides(), set())
        for box in [*self.w.side_settings.values(), self.w.solo_settings]:
            self.assertFalse(box.autoplay_available())
        self.assertTrue(self.w.autoplay_common.isHidden())

    def test_game_window_used_without_manual_calibration(self) -> None:
        # Steam 版のウィンドウが見つかれば、手動のキャリブレーション無しでその位置から盤面を求める
        from unittest import mock

        from src.capture.window_detect import GameWindow

        self.w.plugins.disabled = {"switch_capture"}
        window = GameWindow(1, "Puyo Puyo Tetris 2", 0, 0, 1920, 1080)
        with mock.patch("src.app.find_game_window", return_value=window):
            self.w._refresh_calibration_status()
        self.assertEqual(self.w.game_window, window)
        self.assertEqual(self.w.calibration.self_player.board_quad[0], (272, 159))
        self.assertIn("ウィンドウを検出", self.w.status_label.text())
        with mock.patch("src.app.find_game_window", return_value=None):
            self.w._refresh_calibration_status()
        self.assertIsNone(self.w.calibration)
        self.assertIn("見つかりません", self.w.status_label.text())

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_nameplate_gates_2p_autoplay(self) -> None:
        # ローカル対戦用プラグインだけの時、2P の全自動は「Player 2」が見えている間だけ動き、見えなくなったら待機する
        from unittest import mock

        class FakeAutoplay:
            enabled = False

        class FakeWorker:
            def __init__(self) -> None:
                self.autoplay = FakeAutoplay()

            def set_autoplay(self, on: bool) -> None:
                self.autoplay.enabled = on

        self.w.plugins.disabled = {"full_auto"}
        worker = FakeWorker()
        self.w.sessions = {"2p": (worker, None, None)}
        try:
            with mock.patch.object(self.w, "_read_nameplate", return_value=True):
                self.w._check_nameplates()
                self.w._toggle_autoplay("2p")
            self.assertTrue(worker.autoplay.enabled)
            with mock.patch.object(self.w, "_read_nameplate", return_value=False):
                self.w._check_nameplates()
            self.assertFalse(worker.autoplay.enabled)
            with mock.patch.object(self.w, "_read_nameplate", return_value=True):
                self.w._check_nameplates()
            self.assertTrue(worker.autoplay.enabled)  # 利用者がオンにしたままなら戻る
            self.w._toggle_autoplay("1p")  # 許可の無い側は切り替えない
            self.assertNotIn("1p", self.w._autoplay_wanted)
        finally:
            self.w.sessions = {}

    def test_autoplay_turned_off_when_plugin_disabled(self) -> None:
        box = self.w.side_settings["1p"]
        box.autoplay.setChecked(True)
        self.w.plugins.disabled = {"full_auto"}
        self.w._apply_plugins()
        self.assertFalse(box.autoplay.isChecked())

    def test_debug_log_switch_follows_plugin(self) -> None:
        # ログの出力はログ出力プラグインが有効な時だけ出す
        self.w.plugins.disabled = set()
        self.w._apply_plugins()
        self.assertFalse(self.w.debug_log_checkbox.isHidden())
        self.w.debug_log_checkbox.setChecked(True)
        self.w.plugins.disabled = {"debug_log"}
        self.w._apply_plugins()
        self.assertTrue(self.w.debug_log_checkbox.isHidden())
        self.assertFalse(self.w.debug_log_checkbox.isChecked())

    def test_window_size_row(self) -> None:
        # ウィンドウの大きさプラグインが有効な時だけ出し、押すと選んだ大きさで合わせてから盤面の位置を求め直す
        from unittest import mock

        from src.capture.window_detect import GameWindow

        self.w.plugins.disabled = {"window_size"}
        self.w._apply_plugins()
        self.assertTrue(self.w.window_size_row.isHidden())
        self.w.plugins.disabled = set()
        self.w._apply_plugins()
        self.assertFalse(self.w.window_size_row.isHidden())
        self.w.window_size_combo.blockSignals(True)  # 選択を config に保存しない
        self.w.window_size_combo.setCurrentIndex(self.w.window_size_combo.findText("1600×900"))
        window = GameWindow(7, "PuyoPuyoTetris2", 0, 0, 1000, 563)
        with mock.patch("src.app.find_game_window", return_value=window), \
                mock.patch("src.app.resize_client", return_value=(1600, 900)) as resize, \
                mock.patch.object(self.w, "_refresh_calibration_status") as refresh:
            self.w._on_resize_window_clicked()
        resize.assert_called_once_with(7, (1600, 900))
        refresh.assert_called_once()

    def test_pad_keyboard_rows_in_side_boxes(self) -> None:
        # キーボードで仮想パッドを操作する切り替えは、1P・2P の枠の中にある
        for side, switch in self.w.pad_kb_switches.items():
            self.assertTrue(self.w.side_settings[side].autoplay_box.isAncestorOf(switch), side)

    def test_language_plugin_in_combo(self) -> None:
        combo = self.w.side_settings["1p"].language_combo
        self.assertGreaterEqual(combo.findData("mokou"), 0)


if __name__ == "__main__":
    unittest.main()
