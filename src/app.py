"""PBA (Puyo Beginner Assistance) アプリ本体

メインウィンドウ(キャリブレーション・支援モードの開始/終了)と、支援モード中の
「画面キャプチャ → 認識 → 状態追跡 → (手番ごとに) ama で思考 → オーバーレイ表示」
のループ(AssistWorker、別スレッド)を持つ。

Tetris版(TBA)の実機で得た知見を踏襲している:
  - オーバーレイ自身が画面キャプチャに写り込み盤面を汚染しないよう、
    SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) でキャプチャから除外する
  - 提案の描画は repaint()+DwmFlush() で同期させ、前の提案との二重表示を防ぐ
  - pythonw 起動では stdout/stderr が None になるのでダミーに差し替える
  - ワーカースレッドの未処理例外は crash_log.txt に残す
"""

from __future__ import annotations

import ctypes
import json
import threading
from collections import deque
import os
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402

from .capture.calibrate import CONFIG_PATH, SOLO_CONFIG_PATH, CalibrationResult, load_calibration, run_calibration  # noqa: E402
from .capture.screen_capture import CaptureRegion, PrefetchCapture, ScreenCapture  # noqa: E402
from .capture.window_detect import (  # noqa: E402
    GameWindow,
    client_rect,
    detect_calibration,
    find_game_window,
    layout_calibration,
    CLIENT_SIZE_PRESETS,
    list_windows,
    resize_client,
    save_saved_window,
    title_bar_rect,
)
from .window_dialog import WindowPickDialog  # noqa: E402
from .engine.ama_client import HAIPUYO_BRIDGE_ARGS, PEEK_BRIDGE_ARGS, resolve_bridge_exe  # noqa: E402
from .engine.book_client import BookClient, book_text, is_book_move  # noqa: E402
from .engine.ama_client import (  # noqa: E402
    NoValidPlacementError,
    FIRE_REASON_TEXT_JA,
    FORM_TEXT_JA,
    AmaClient,
    BattleMove,
    PlayerState,
    action_text,
    diag_text,
    is_build_like,
)
from .engine.puyo_field import COLS, ROWS, PuyoField  # noqa: E402
from .engine.think_input import build_think_input  # noqa: E402
from .engine.margin_time import margin_target_point  # noqa: E402
from .engine.background import ImageSaver, ThinkWorker  # noqa: E402
from .toggle_switch import ToggleSwitch  # noqa: E402
from .plugin_api import Language  # noqa: E402
from .plugin_loader import PluginManager, save_disabled  # noqa: E402
from .plugin_dialog import PluginDialog  # noqa: E402
from .overlay.renderer import register_language  # noqa: E402
from .engine.registry import BUILTIN_BOOK, Registry  # noqa: E402
from .registry_dialogs import BookRegistryDialog, EngineRegistryDialog  # noqa: E402
from .simulator.opening_book import OpeningBook  # noqa: E402
# 全自動の操作の仕組みは非公開のプラグイン(plugins/full_auto)にある。無ければ AUTOPLAY_AVAILABLE=False で、
# 何もしない代わりの部品が入る(2026-11-01)。プラグインにしか無い画面・キーフックは autoplay_ext.名前 で使う
from . import autoplay_support as autoplay_ext  # noqa: E402
from .autoplay_support import AVAILABLE as AUTOPLAY_AVAILABLE  # noqa: E402
from .autoplay_support import (  # noqa: E402
    AutoPlayer, KeyMap, PadSender, attack_involved, cancel_rows, early_drop_seconds, rotate_cancel, step_cancel, wall_prehold_action,
)
from .engine.threat import AttackEpisode, TurnClock  # noqa: E402
from .engine.tracker import Pair, PlayerTracker, Snapshot, clear_spawn_cells, count_puyos, has_poppable_group, is_covered, settle_cells  # noqa: E402
from .vision.cell_classifier import EMPTY, GAME_COLORS, GARBAGE  # noqa: E402
from .overlay.renderer import OverlayDrawData, OverlayLayout, plan_dots, render_overlay, render_perf  # noqa: E402
from .engine.perf_meter import CpuMeter, FpsMeter, perf_text  # noqa: E402
from .paths import APP_TITLE, app_root, resource_root  # noqa: E402
from .vision.banner_detector import AllClearBannerDetector  # noqa: E402
from .vision.board_reader import cells_to_text, read_board, read_next  # noqa: E402
from .vision.color_mapper import ColorMapper  # noqa: E402

# 認識ループの周期(秒)。キャプチャ約20ms+認識数msなので、これで約30fps相当
TICK_INTERVAL_SEC = 0.03
# 観測の間隔・処理時間をまとめてログに出す周期(秒)
TICK_STATS_SECONDS = 60.0
# オーバーレイウィンドウを盤面より少し大きく取る余白(物理px)
OVERLAY_MARGIN = 40
# 情報パネルを自分の盤面の外側(1P は左、2P は右)に置くため、その側だけ余白を広く取る(物理px)
OVERLAY_LEFT_SPACE = 320
# 操作する側ごとの全自動モードのキー割り当て(config/ 内)
KEYMAP_FILES = {"1p": "keymap.json", "2p": "keymap_2p.json"}
# メイン画面の「仮想パッド: A を押す」などで、実際に押すまでの待ち(その間にゲームの画面を前面にしてもらう)
PAD_TAP_DELAY_MS = 3000
# メイン画面の「操作する側」の選択肢と、それぞれで動かす側(both は1つの PBA で 1P と 2P を同時に動かす)
# デバッグ用に保存する切り出し画像の直近件数
FRAME_HISTORY_SIZE = 60
# 自分の連鎖を点滅から検出した時に保存する直前のフレーム数(誤検出の調査用。実機 2026-09-27 14:23:
# 予測どおりに置いて4連結が無いのに「1連鎖」と検出した。高速落下の残像を疑っている)
FLASH_EVIDENCE_FRAMES = 6
# 出現時点で送れなかった理由のうち、毎フレーム比べ直すもの(盤面の読みが一時的に違うだけのことが多い)
BOARD_MISMATCH = "先読みの盤面と実際の盤面が違う"
# 先読みの手が小さい消去なので、3組目まで見た思考を待つ(出現時点では送らない理由)
HARASS_WAIT = "先読みの手が小さい消去 [harass] なので3組目まで見た思考を待つ"
# その待ちの上限(2026-10-21): 出現からこの秒数たっても手を送っていなければ、先読みの手をそのまま送る。思考の反映は普段 0.1〜0.45秒。
# 上限が無く、思考の手が送られないまま2秒たって自由落下した(実機 2026-10-12 12:13 turn=34。思考の手は届いていたが送られなかった)
HARASS_WAIT_MAX_SEC = 0.6
# 保険(2026-10-24、利用者「最上段で自由落下になって窒息。人間ではありえない」): 思考の手が届いた時に盤面の確定待ちで送れなかった手番は、
# 出現からこの秒数たっても確定待ちのままなら、その手を送る(高い積みでは出現した組ぷよがすぐ積みに乗り、待つと出現位置で固定される)
BLOCKED_SEND_SEC = 0.2
# CPU 使用率と観測の FPS の表示を更新する間隔(秒)
PERF_INTERVAL_SEC = 1.0
# 「盤面が違う」時に出現の瞬間の画面を残す枚数の上限(1試合の記録で増えすぎないように)
MISMATCH_FRAME_LIMIT = 60
# 全自動モード: 提案の置き場所が分かっている時、観測個数の増加で下キーを離すのは落下開始からこの秒数後だけ
# (置き場所がずれた時の保険。早すぎると目標の手前で離してしまう)
COUNT_RELEASE_MIN_SEC = 0.25
# 置きミスの診断(2026-10-24): キーを送ってから着地までの組ぷよの位置とフレームを記録し、提案の2マス以外に着地した手だけ
# キーの時刻・組ぷよの動き・画面を残す(実機 2026-10-24 14:01 turn=17: 最後の横移動が効かず1列手前に着地して4連鎖が暴発した。
# 置きミスは全体の約1%で、横移動が抜ける理由がログから分からなかった)。記録するフレームの上限・保存する枚数・1回の起動で残す手数
MOVE_TRACE_FRAMES = 120
MOVE_MISS_FRAMES = 30
MOVE_MISS_LIMIT = 20
# 盤面の積みが(この個数以上あった状態から)0になり、それが NEW_GAME_EMPTY_SECONDS 以上
# 続いたら「新しい試合」とみなし、追跡状態と色の割当をリセットする(試合ごとに使われる
# 4色が変わるため、前の試合の割当のままだと5色目エラーになる)。対戦では両盤面、
# 一人用モードでは自分の盤面だけで判定する。画面効果で一瞬だけ盤面が空に見える
# ことがある(一人用モードで実機確認)ため、継続時間を要求する
NEW_GAME_MIN_PUYOS = 8
NEW_GAME_EMPTY_SECONDS = 1.0
# 全消し直後は次の組ぷよを置くまで盤面が空のままになる(実機 2026-09-20 13:56: 全消しの1秒後に
# 「新しい試合」と誤認して全消しボーナスと色割当を失った)。ボーナス持ちの間はこの秒数まで待つ
NEW_GAME_EMPTY_SECONDS_AFTER_ALL_CLEAR = 10.0
# 負けは3列目が12段目まで埋まった時なので、盤面が一気に消える時は必ず12個以上消える。これ以上消えて全消しにならなかった
# (all_clear_unconfirmed)ら試合終了とみなす(試合中に点滅を見逃した小さな全消しで相手のボーナスを消さないための下限)
GAME_OVER_MIN_DROPPED = 12
# 取り込み待ちがこの秒数を超えたら診断情報をログに残す
STUCK_WARN_SECONDS = 3.0


def _exclude_window_from_screen_capture(widget: QtWidgets.QWidget) -> bool:
    """このウィンドウを画面キャプチャ(mss等)の対象から除外する。成功したらTrueを返す。

    Windows 10 2004以降の SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) を使う。
    HWND はポインタサイズなので argtypes を明示しないと64bit環境で上位ビットが
    欠落して静かに失敗する(Tetris版で長期間の誤診断を招いた)。失敗は必ず呼び出し側へ返す。
    """
    if sys.platform != "win32":
        return False
    WDA_EXCLUDEFROMCAPTURE = 0x00000011
    try:
        import ctypes.wintypes

        set_affinity = ctypes.windll.user32.SetWindowDisplayAffinity
        set_affinity.argtypes = [ctypes.wintypes.HWND, ctypes.wintypes.DWORD]
        set_affinity.restype = ctypes.wintypes.BOOL
        hwnd = ctypes.wintypes.HWND(int(widget.winId()))
        if not set_affinity(hwnd, WDA_EXCLUDEFROMCAPTURE):
            return False
        get_affinity = ctypes.windll.user32.GetWindowDisplayAffinity
        get_affinity.argtypes = [ctypes.wintypes.HWND, ctypes.POINTER(ctypes.wintypes.DWORD)]
        get_affinity.restype = ctypes.wintypes.BOOL
        affinity = ctypes.wintypes.DWORD(0)
        if not get_affinity(hwnd, ctypes.byref(affinity)):
            return False
        return affinity.value == WDA_EXCLUDEFROMCAPTURE
    except (OSError, AttributeError):
        return False


class OverlayWindow(QtWidgets.QWidget):
    """盤面周辺だけをカバーする透過・クリックスルーの描画専用ウィンドウ"""

    def __init__(self, calibration: CalibrationResult, language: str = "ja") -> None:
        super().__init__()
        self.calibration = calibration
        self.language = language  # 情報パネルの表記("ja" / "en" / "mokou")
        self.draw_data: OverlayDrawData | None = None
        self.perf_text = ""  # CPU 使用率と観測の FPS(2026-10-24)。盤面の上端の枠外に出す

        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_NoSystemBackground)

        # 画面キャプチャ(mss)は物理ピクセル座標、Qtのウィンドウ配置は論理ピクセル座標
        self.dpr = QtWidgets.QApplication.primaryScreen().devicePixelRatio()
        region = calibration.capture_region()
        # 情報パネルは自分の盤面の外側に置く: 1P(左の盤面)は左、2P(右の盤面)は右に余白を広く取る
        self.panel_right = calibration.self_on_right()
        if self.panel_right:
            screen_right = round(QtWidgets.QApplication.primaryScreen().geometry().right() * self.dpr)
            left_space = min(OVERLAY_MARGIN, region.left)
            right_space = max(OVERLAY_MARGIN, min(OVERLAY_LEFT_SPACE, screen_right - (region.left + region.width)))
        else:
            left_space = min(OVERLAY_LEFT_SPACE, region.left)  # 画面左端からはみ出さない範囲で
            right_space = OVERLAY_MARGIN
        self.origin = (region.left - left_space, region.top - OVERLAY_MARGIN)
        w = region.width + left_space + right_space
        h = region.height + OVERLAY_MARGIN * 2
        self.setGeometry(
            round(self.origin[0] / self.dpr), round(self.origin[1] / self.dpr), round(w / self.dpr), round(h / self.dpr)
        )

    def update_draw_data(self, data: OverlayDrawData | None) -> None:
        self.draw_data = data
        self.repaint()
        if sys.platform == "win32":
            try:
                ctypes.windll.dwmapi.DwmFlush()
            except OSError:
                pass

    def update_perf(self, text: str) -> None:
        """CPU 使用率と観測の FPS の表示を更新する(1秒ごと。提案の表示とは別に描き直す)"""
        self.perf_text = text
        self.update()

    def paintEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 0))
        painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        layout = OverlayLayout(
            self_geometry=self.calibration.self_player.geometry(),
            enemy_geometry=self.calibration.enemy_player.geometry(),
            origin=self.origin,
            dpr=self.dpr,
            language=self.language,
        )
        if self.panel_right:
            board_right = layout.to_local(*layout.self_geometry.grid_to_screen(COLS, 0)).x()
            layout.panel_right_space = self.width() - board_right
        if self.draw_data is not None:
            render_overlay(painter, layout, self.draw_data)
        render_perf(painter, layout, self.perf_text)
        painter.end()


WINDOW_SIZE_PATH = app_root() / "config" / "window_size.json"


def _load_window_size(path: Path = WINDOW_SIZE_PATH) -> str:
    """前に選んだゲームのウィンドウの大きさ("1920×1080" の形)。無ければ 1920×1080。"""
    try:
        return str(json.loads(Path(path).read_text(encoding="utf-8")).get("size", "1920×1080"))
    except (OSError, ValueError, AttributeError):
        return "1920×1080"


def _save_window_size(text: str, path: Path = WINDOW_SIZE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"size": text}, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError:
        pass


STOP_BUTTON_STYLE = (
    "QPushButton { color: white; background: black; font-weight: bold; border: 1px solid #555; padding: 2px 6px; }"
    "QPushButton:hover { background: #333; }"
)


class ControlBadge(QtWidgets.QWidget):
    """支援モード中に常時表示する、クリックスルーなしの小さな終了バッジ。

    ゲームのウィンドウが分かっている時は、そのタイトルバーの中央に置く(2026-10-28、利用者)。その時は1行にまとめて
    タイトルバーの高さに収める。終了ボタンの隣にウィンドウの検出状態を出す。"""

    DETECT_WIDTH = 200

    def __init__(self, on_stop, detect_text: str = "", anchor: QtCore.QRect | None = None) -> None:  # noqa: ANN001
        super().__init__()
        self.setWindowFlags(
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.anchor = anchor  # タイトルバー(論理ピクセル)。None なら画面の右上
        layout = QtWidgets.QHBoxLayout(self)
        if anchor is not None:
            layout.setContentsMargins(2, 0, 2, 0)
        # 終了ボタンを先(左)に置き、ラベルは幅を固定して長い文言でもボタンが
        # 押し出されないようにする(実機で認識エラーの長文がボタンを画面外へ追いやった)
        stop_btn = QtWidgets.QPushButton("終了")
        stop_btn.setFixedWidth(60)
        # 標準の淡い灰色では白いタイトルバーに埋もれて見つからなかった(2026-10-29、利用者)ので黒地にする
        stop_btn.setStyleSheet(STOP_BUTTON_STYLE)
        stop_btn.clicked.connect(on_stop)
        self.stop_button = stop_btn
        layout.addWidget(stop_btn)
        style = "color: white; background: rgba(0,0,0,190); padding: 2px 8px;"
        self.detect_label = QtWidgets.QLabel()
        self.detect_label.setStyleSheet(style)
        self.detect_label.setFixedWidth(self.DETECT_WIDTH)
        self.detect_label.setToolTip(detect_text)
        self.detect_label.setText(
            self.detect_label.fontMetrics().elidedText(detect_text, QtCore.Qt.TextElideMode.ElideRight, self.DETECT_WIDTH - 16)
        )
        layout.addWidget(self.detect_label)
        self.label = QtWidgets.QLabel("支援モード中")
        self.label.setStyleSheet(style)
        self.label.setFixedWidth(260)
        layout.addWidget(self.label)
        self._layout = layout
        self._lines = 1
        self.autoplay_buttons: dict[str, QtWidgets.QPushButton] = {}
        self._resize()

    def set_status(self, text: str) -> None:
        self.set_lines([text])

    def add_autoplay_toggles(self, states: dict[str, bool], on_click) -> None:  # noqa: ANN001
        """支援モード中に全自動を切り替えるボタン(側ごと。2026-10-27)。"""
        for side, on in states.items():
            btn = QtWidgets.QPushButton()
            btn.setFixedWidth(120)
            btn.clicked.connect(lambda _c=False, s=side: on_click(s))
            self._layout.insertWidget(len(self.autoplay_buttons) + 2, btn)
            self.autoplay_buttons[side] = btn
            self.set_autoplay_state(side, on)
        self._resize()

    def set_autoplay_state(self, side: str, on: bool, waiting: bool = False) -> None:
        """waiting: オンにしたがネームプレートの確認(2P・Player 2)が通らず止めている。"""
        btn = self.autoplay_buttons.get(side)
        if btn is not None:
            btn.setText(f"{side.upper()} 全自動: {'待機' if waiting else 'オン' if on else 'オフ'}")

    def _resize(self) -> None:
        width = 360 + self.DETECT_WIDTH + 6 + 126 * len(self.autoplay_buttons)
        if self.anchor is not None:
            self.setFixedSize(width, max(24, self.anchor.height()))
            self.move(self.anchor.x() + (self.anchor.width() - width) // 2,
                      self.anchor.y() + (self.anchor.height() - self.height()) // 2)
            return
        self.setFixedSize(width, max(44, 26 + 18 * self._lines))
        screen = QtWidgets.QApplication.primaryScreen().geometry()
        self.move(screen.width() - self.width() - 20, 20)

    def set_lines(self, lines: list[str]) -> None:
        """1行ずつ幅に収めて表示する(1P と 2P を同時に動かす時は2行。タイトルバーに置く時は1行にまとめる)。"""
        if self.anchor is not None:
            lines = ["  |  ".join(lines)]
        metrics = self.label.fontMetrics()
        self._lines = len(lines)
        self._resize()
        self.label.setText(
            "\n".join(metrics.elidedText(t, QtCore.Qt.TextElideMode.ElideRight, self.label.width() - 16) for t in lines)
        )


# 手番の途中で再計算するのは「入力が変わった時」だけ: 相手の消去の開始/終了、自分への予告おじゃまが
# 前回の思考からこれ以上(1段=6個)増えた時。ラベルによる固定はしない(2026-09-20: ama の判断に任せる)
RETHINK_PENDING_INCREASE = 6
# 「全消し」の文字がこの一致度以上で見えたら、そのプレイヤーは全消しを持っているとする(2026-10-11。相手の文字は0.97、自分は0.65 だった)
ALL_CLEAR_BANNER_MIN_SCORE = 0.9
# 相手の発火を想定した先読み(2026-10-10、利用者「相手がこの場面で発火してきたらどうなるかをあらかじめ想定できないか」):
# 攻撃されていない時、相手が今すぐ撃てる攻撃がこの個数以上なら、それを受けた状態での次の手も先に計算しておく。
# 相手の連鎖が始まった直後の手番で、見積もった着弾量が想定に近ければその手順を出現時点で送る
ANTICIPATE_MIN_GARBAGE = 12
ANTICIPATE_TOLERANCE_MIN = 6  # 想定と見積もりの差の許容(個)。想定の 1/4 と大きい方
# 観測の空白: 1tick(取得+認識+思考)がこれ以上かかったらログに残す(分析文書 2026-09-19 P1-2:
# 思考中は画面取得が止まる。非同期化の前にまず空白の長さを計測する)
OBSERVATION_GAP_WARN_SEC = 0.25
# 相手の連鎖を再現した着弾量を、連鎖終了後に予告が無いままこの秒数続いたら解除する
INCOMING_CARRY_CLEAR_SECONDS = 2.0
# 再計算の結果がこれら(=対応不要の構築)で、入力(着弾量)も変わっていなければ表示中の提案を変えない
# (ビームサーチの乱数で構築の手は毎回変わるため、そのまま出すと提案が振動する)
BUILD_LABELS = ("build", "accept")

class AssistWorker(QtCore.QThread):
    """支援モードのメインループ(別スレッド)。

    毎tick: 1回のキャプチャで自分・相手の盤面とネクストを読み、各 PlayerTracker に
    渡す。自分側の turn_ready が出たら、その時点の両者の状態で ama(battle) を呼び、
    提案を draw_data_ready で通知する。提案は次の turn_ready まで固定する
    (ネクスト・盤面が変わらない限り表示を切り替えない、という方針)。
    """

    draw_data_ready = QtCore.pyqtSignal(object)
    status_changed = QtCore.pyqtSignal(str)
    fatal_error = QtCore.pyqtSignal(str)
    perf_ready = QtCore.pyqtSignal(str)  # CPU 使用率と観測の FPS(1秒ごと)

    def __init__(
        self,
        calibration: CalibrationResult,
        client: AmaClient,
        debug_log_path: Path | None,
        solo: bool = False,
        autoplay: bool = False,
        side: str = "1p",
        place_cancel: bool | None = None,
        haipuyo: bool = False,
        step_cancel: bool | None = None,
        prehold: bool | None = None,
        peek: bool = False,
    ) -> None:
        super().__init__()
        self.calibration = calibration
        # 操作する側。1P と 2P を別々の PBA で同時に動かせるよう、キー割り当てと画像の保存先を分ける(2026-10-12)
        self.side = side
        self.history_dir = app_root() / ("debug_frames_history" if side == "1p" else f"debug_frames_history_{side}")
        self.client = client
        self.debug_log_path = debug_log_path
        self.solo = solo  # 一人用モード: 相手を見ず、連鎖構築だけを提案する
        # 配ぷよ読み: 要求に試合の最初から置いた組ぷよと置いた数を入れる(ブリッジは HAIPUYO_BRIDGE_ARGS で起動する)
        self.haipuyo = haipuyo
        # 相手のツモを覗く(2026-11-01): 要求に自分と相手の置いた数を入れる(ブリッジは PEEK_BRIDGE_ARGS で起動する)
        self.peek = peek
        # 全自動モード(画像認識の試験用。CPU戦で使う): 提案の操作手順をキーボード入力としてゲームへ送る
        keymap = KeyMap.load(app_root() / "config" / KEYMAP_FILES[side])
        sender = None
        if keymap.device == "pad":
            # 仮想ゲームパッドで送る(2026-10-25)。作れなければ送らず、ログに理由を残す
            # 側ごとの仮想パッドを使い回す(支援モードを止めても外さない。ゲームの参加の画面で割り当てたパッドのまま送る)
            sender = PadSender(slot=side) if autoplay else PadSender(send=False)
            if autoplay and not sender.available:
                self._log(f"自動操作: 仮想パッドを作れません({sender.error})。ViGEmBus ドライバーと vgamepad を入れてください")
        self.autoplay = AutoPlayer(keymap.resolved(), sender=sender, log=self._log)
        self.autoplay.enabled = autoplay
        # 設置キャンセルのあり・なし(画面の切り替え、2026-10-23)。None なら keymap の place_cancel のまま。
        # ありで keymap が "off" なら "move"(keymap が "move_rotate" ならそのまま)
        if place_cancel is not None:
            km = self.autoplay.keymap
            if not place_cancel:
                km.place_cancel = "off"
            elif km.place_cancel == "off":
                km.place_cancel = "move"
        # 段差キャンセルのあり・なし(画面の切り替え、2026-10-25)。None なら keymap の step_cancel のまま
        if step_cancel is not None:
            self.autoplay.keymap.step_cancel = step_cancel
        # 先行入力のあり・なし(画面の切り替え、2026-10-25 試験)。None なら keymap の prehold のまま
        if prehold is not None:
            self.autoplay.keymap.prehold = prehold
        self.autoplay.prehold_provider = self._prehold_action
        self._last_data: OverlayDrawData | None = None  # 直近の描画内容
        self.last_cpu: float | None = None  # 直近に測った PC 全体の CPU 使用率(%、1秒ごと。ログ用)
        self._enemy_fire: tuple[int, int, bool] | None = None  # 相手が撃っている(撃った)連鎖。パネルの相手の欄に出す
        self._enemy_fire_end = 0.0  # 相手の連鎖が終わった時刻(perf_counter)
        # 自分が撃っている(撃った)連鎖 (連鎖数, 点数, 連鎖中か)。パネルの自分の欄に出す(利用者 2026-10-25)
        self._self_fire: tuple[int, int, bool] | None = None
        self._self_fire_end = 0.0
        self._self_fire_first: tuple[int, int, bool] | None = None
        self._think_submitted = 0.0  # 最後に反映した思考を依頼した時刻(perf_counter)
        self._pending_at_think = 0  # 前回の思考時の自分への予告おじゃま(再計算の判断用)
        self._fifth_color_logged: tuple | None = None  # 5色目の診断を残した (手番, 盤面の由来)
        self._incoming_at_think = 0  # 前回の思考時の着弾量(構築のままの再計算を維持するかの判断用)
        self._think_pending = False  # turn_ready で思考できなかった(ネクスト未認識等)ので、認識が揃い次第やり直す
        self._last_form: str | None = None  # 前の手番で提案が最も近かった形(移行の表示用)
        self._running = True
        self.trackers = {"self": PlayerTracker(), "enemy": PlayerTracker()}
        self.mapper = ColorMapper()
        self._history_index = 0
        # 先読み: (期待する設置後の盤面, その盤面に対する提案)。提案を出した直後に
        # 「提案通りに置いた後の盤面」で次の手を計算しておき、次の手番で観測が一致したら即表示する
        # (期待する設置後の盤面, その時の(現在の組ぷよ, ネクスト), 提案)。組ぷよも一致した時だけ使う
        self._precomputed: tuple[list[list[str]], tuple[Pair | None, Pair | None], BattleMove] | None = None
        self._precomputed_attack: tuple | None = None  # 先読みを計算した時の攻撃の状況(_attack_snapshot)
        # 相手の発火を想定した先読み(盤面, (組ぷよ, ネクスト), 手, 想定した着弾量)。相手の連鎖が始まった直後の出現で使う
        self._precomputed_alt: tuple | None = None
        self._last_move: BattleMove | None = None  # 最後に表示した(全自動なら送った)現在の手
        # 全自動で実際にキーを送った手 (手番, 手)。キーは取り消せないので、送った後の思考の手が違っても表示と次の手の先読みはこちらで行う
        self._sent_move: tuple[int, BattleMove] | None = None
        # 計算中の先読みの依頼(結果が届くか、実行前に取り消すと None)と、最後に依頼した先読みの前提(重複の依頼を避ける)
        self._pre_inflight: dict | None = None
        self._pre_basis: tuple | None = None
        # 着地した手番(全自動の下キーを離した時)と、その後に観測盤面で先読みをやり直した時の盤面(同じ盤面で二度やらない)
        self._landed_turn: int | None = None
        self._replanned: tuple[int, str] | None = None
        self._last_obs_text: str = ""
        self._max_seen = {"self": 0, "enemy": 0}  # 現在の試合で観測した積みの最大個数
        self._banner_seen = {"self": False, "enemy": False}  # 「全消し」の文字が見えているか(ログ用)
        # 開始前の初手: 組ぷよの並び(2組または3組) → 計算した手。開始の出現で、出現した組ぷよ・ネクストに合うものを先読みとして使う
        self._opening_moves: dict[tuple[Pair, ...], BattleMove] = {}
        self._opening_requested: set[tuple[Pair, ...]] = set()  # 思考スレッドに依頼済みの並び
        self._banner_detectors = {"self": AllClearBannerDetector(), "enemy": AllClearBannerDetector()}
        self._empty_since: float | None = None  # 判定対象の盤面がすべて空になった時刻
        self._stuck_logged: set[tuple[str, float]] = set()  # (誰, 最後に取り込めた時刻)
        self._recent_frames: deque[tuple[float, np.ndarray]] = deque(maxlen=FLASH_EVIDENCE_FRAMES)
        self._now: float | None = None  # この tick の時刻(思考入力の期限の見積もりに使う)
        self._game_start_t = 0.0  # 現在の試合が始まった時刻(診断用)
        # マージンタイム(2026-10-21): 自分の最初の組ぷよが出た時刻と、最後に観測した時刻。経過秒数からおじゃま1個に必要な点を出す
        self._first_spawn_t: float | None = None
        self._frame_t = 0.0
        # 新方針の第1段階(記録のみ): 相手の攻撃エピソードと自分の平均手時間
        self._attack: AttackEpisode | None = None  # 進行中の相手の連鎖
        self._attack_count = 0
        self._turn_clock = TurnClock()
        # 実行できなかった再計算の理由(自分が取り込み待ちの間に相手の攻撃を検出した等)を次の tick へ持ち越す
        self._deferred_rethink = ""
        # 自分に降るおじゃま(個)。相手の連鎖を盤面から再現できた時だけブリッジが返し、予告が消える
        # (落下・相殺)まで保持する。予告のアイコン数(0〜6)では大きな攻撃を過小評価するため
        self._incoming_carry = 0
        self._carry_zero_since: float | None = None
        # 先読み(提案通りに置いた後の盤面で次の手を事前計算)。観測の空白を減らす検証のために切れる
        # 常に有効。分析文書第3弾の提案で一度無効にしたが、2手先の表示が無いと置くのが遅くなり相手の
        # 攻撃に間に合わなくなった(利用者 2026-09-19)。観測の空白の対策は先読みを止めるのではなく、
        # 思考を別スレッドに分ける(非同期化)方向で行う
        self.enable_precompute = True
        # 観測を止めない(2026-09-27): 思考は思考スレッド、画像保存は保存スレッドで行う。
        # 思考スレッドは run() で作る(テストでは作らず、その場で実行して結果を反映する)
        self._think_worker: ThinkWorker | None = None
        self._client_lock = threading.Lock()  # ブリッジの呼び出しを直列にする(思考スレッドと共有)
        self._saver = ImageSaver()
        # 出現時点で「盤面が違う」で送れなかった手番と、その出現の時刻。取り込みを待たずに毎フレーム比べ直す
        # (実機 2026-10-11 21:30: 相手の連鎖中に「盤面が違う」で送れなかった40手のうち27手は、取り込み時点の
        # 確定した盤面では先読みどおりだった。待つ0.04〜0.8秒+思考の間に自由落下した)
        self._mismatch_retry: tuple[int, float] | None = None
        # 盤面の確定待ちで送れなかった思考の手(手番, 手)と、自分の組ぷよが出現した時刻(BLOCKED_SEND_SEC の保険に使う)
        self._blocked_move: tuple[int, BattleMove] | None = None
        self._self_spawn_t = 0.0
        self._mismatch_frames_saved = 0
        # 置きミスの診断: 送信中の手の記録(_autoplay_send で作り、着地で判定する)、残した手数、観測ループの開始時刻(キーの時刻の換算用)
        self._move_trace: dict | None = None
        self._move_miss_saved = 0
        self._t0_perf = time.perf_counter()
        self._trace_saver = ImageSaver(maxsize=MOVE_MISS_FRAMES * 2)
        # 出現の瞬間の自分の盤面の右の枠の横位置(校正からのずれ px)。先読みどおりに送れた出現の分を普段の値として残す
        # (利用者 2026-10-11「横揺れが認識に悪影響を及ぼしていた?」の確認用)
        self._edge_normal: deque[int] = deque(maxlen=50)

    def _reset_game(self, t: float = 0.0) -> None:
        self._game_start_t = t
        self._first_spawn_t = None
        if self._think_worker is not None:
            self._think_worker.cancel("main")
            self._think_worker.cancel("pre")
            self._think_worker.cancel("pre_atk")
        self._stuck_logged.clear()
        self._last_form = None
        self._last_data = None
        self._enemy_fire = None
        self._enemy_fire_end = 0.0
        self._self_fire = None
        self._self_fire_end = 0.0
        self._think_submitted = 0.0
        self.trackers = {"self": PlayerTracker(), "enemy": PlayerTracker()}
        self.mapper = ColorMapper()
        self._precomputed = None
        self._precomputed_alt = None
        self._sent_move = None
        self._move_trace = None
        self._pre_inflight = None
        self._pre_basis = None
        self._max_seen = {"self": 0, "enemy": 0}
        self._banner_detectors = {"self": AllClearBannerDetector(), "enemy": AllClearBannerDetector()}
        self._opening_moves = {}
        self._opening_requested = set()
        self.autoplay.reset()
        self._empty_since = None
        self._attack = None
        self._attack_count = 0
        self._turn_clock = TurnClock()
        self._deferred_rethink = ""
        self._incoming_carry = 0
        self._think_pending = False
        self._pending_at_think = 0
        self.draw_data_ready.emit(None)

    def _on_game_over(self, t: float, who: str, tr: PlayerTracker) -> None:
        """盤面が一気に空になり全消しにならなかった(負け)時、両者の全消しボーナスの記録を消す。

        ボーナス持ちの間は新しい試合の検出を10秒待つ(NEW_GAME_EMPTY_SECONDS_AFTER_ALL_CLEAR)ので、全消しの直後に
        試合が終わると次の試合が始まってもリセットされず、前の試合の組ぷよが残って5色目で約2秒思考できず、次の試合を
        全消しボーナス持ちとして戦った(実機 2026-10-20 22:28 の 108.5s 全消し → 115.0s 相手の負け)。試合が終われば
        ボーナスは無意味なので消して、通常の1秒待ちでリセットさせる。"""
        if tr.chain_dropped < GAME_OVER_MIN_DROPPED:
            return
        for w, other in self.trackers.items():
            if other.all_clear:
                other.all_clear = False
                self._log(f"[{t:7.2f}s] {who}の盤面が一気に消えた({tr.chain_dropped}個、負け): 試合終了として{w}の全消しボーナスを消します")

    def _log_fifth_color(self, t: float, me: PlayerTracker, en: PlayerTracker, my_pairs: list, frame_bgr: np.ndarray) -> None:
        """試合の途中(盤面が大きい)で5色目が出て思考できなかった時、どこに5色目があるかと画面を残す(手番ごとに1回)。
        原因(盤面の誤認・ネクストの誤認・試合の切り替わり)を画像で調べるための診断(実機 2026-10-20 22:28 の 336s。
        それまでは画像が無く原因を確かめられなかった)。"""
        if self._fifth_color_logged == (me.turn, me.last_take_in_reason):
            return
        self._fifth_color_logged = (me.turn, me.last_take_in_reason)
        known = set(self.mapper.mapping)
        board_colors = {c for row in me.settled for c in row if c in GAME_COLORS}
        pair_colors = {c for p in my_pairs if p for c in p}
        enemy_colors = {c for row in en.settled for c in row if c in GAME_COLORS}
        path = self.history_dir / f"fifth_color_self_{t:07.2f}s_turn{me.turn}.png"
        self._saver.save(path, frame_bgr)
        self._log(
            f"           5色目の診断: 色の割当={self.mapper.mapping} 自分の盤面の色={sorted(board_colors)}"
            f"(未割当 {sorted(board_colors - known)}) 組ぷよ={my_pairs}(未割当 {sorted(pair_colors - known)})"
            f" 相手の盤面の色={sorted(enemy_colors)} 盤面の由来={me.last_take_in_reason} 画像={path.name}\n"
            f"{_indent(cells_to_text(me.settled))}"
        )

    def _check_new_game(self, t: float, snaps: dict[str, Snapshot]) -> None:
        targets = ["self"] if self.solo else ["self", "enemy"]
        if any(is_covered(snaps[who].cells, snaps[who].eyeless_cells) for who in targets):
            return  # 演出に覆われている間は判定しない
        counts = {who: count_puyos(settle_cells(snaps[who].cells)) for who in targets}
        for who, c in counts.items():
            self._max_seen[who] = max(self._max_seen[who], c)
        if all(c == 0 for c in counts.values()) and all(self._max_seen[w] >= NEW_GAME_MIN_PUYOS for w in targets):
            # 全消しボーナスを持っている間は、全消し直後の空盤面(次の手を置くまで続く)と区別できないので長く待つ
            wait = NEW_GAME_EMPTY_SECONDS_AFTER_ALL_CLEAR if any(self.trackers[w].all_clear for w in targets) else NEW_GAME_EMPTY_SECONDS
            if self._empty_since is None:
                self._empty_since = t
            elif t - self._empty_since >= wait:
                self._log(f"[{t:7.2f}s] 盤面の積みが消えたまま{wait:.0f}秒: 新しい試合として追跡状態と色の割当をリセット")
                self._reset_game(t)
                self.status_changed.emit("新しい試合を検出")
        else:
            self._empty_since = None

    def stop(self) -> None:
        self._running = False
        self.autoplay.stop()
        if self._think_worker is not None:
            self._think_worker.stop()
        self._saver.stop()
        self._trace_saver.stop()

    def run(self) -> None:  # noqa: D102
        self._think_worker = ThinkWorker()
        self._client_lock = self._think_worker.lock
        try:
            # 画面の取得を別スレッドで1枚先に進め、読み取りと並行させる(観測1回 約40ms → TICK_INTERVAL_SEC の30ms。2026-10-21)
            with PrefetchCapture() as capture:
                self._loop(capture)
        except Exception as exc:  # noqa: BLE001
            crash_log = app_root() / "crash_log.txt"
            with open(crash_log, "a", encoding="utf-8") as f:
                f.write(f"\n===== クラッシュ(AssistWorker) {datetime.now().isoformat()} =====\n")
                traceback.print_exc(file=f)
            self.fatal_error.emit(str(exc))

    def _release_carry_if_pending_cleared(self, t: float, me: PlayerTracker) -> None:
        """自分への予告が消えたら、保持している着弾見込みを解除する。

        相手の連鎖中は解除しない: 自分の連鎖の相殺で予告の表示が一時的に0になっても、残りの段の着弾は続く
        (実機 2026-09-27 14:23: 相手の4連鎖中に48個を解除し、次の手番の ama が着弾2個と見て判断した)。
        """
        if me.pending_garbage == 0 and self._incoming_carry > 0 and not self.trackers["enemy"].chain_active:
            self._log(f"[{t:7.2f}s] 予告が消えたので着弾量 {self._incoming_carry}個 を解除")
            self._incoming_carry = 0

    def _save_flash_evidence(self, t: float) -> None:
        """自分の連鎖を点滅から検出した時、直前のフレームを保存してログに残す(誤検出の調査用)。"""
        names = []
        for ft, frame in self._recent_frames:
            path = self.history_dir / f"flash_self_{t:07.2f}s_{ft:07.2f}.png"
            self._saver.save(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            names.append(path.name)
        self._log(f"[{t:7.2f}s] 点滅検出の直前フレームを保存: {', '.join(names)}")

    def _loop(self, capture: PrefetchCapture) -> None:
        region = self.calibration.capture_region()
        origin = (region.left, region.top)
        players = {"self": self.calibration.self_player, "enemy": self.calibration.enemy_player}
        # ログの先頭に動作モードを残す(一人用モードでは相手を見ないので、相手の攻撃への対応は動かない。
        # 実機 2026-09-19 18:12: 一人用で起動した試合を対戦のつもりで評価していた)
        self._log(
            f"===== 支援モード開始 {datetime.now().isoformat(timespec='seconds')} "
            f"モード={'一人用(相手を見ない)' if self.solo else '対戦'} "
            f"先読み={self.enable_precompute} "
            f"全自動={self.autoplay.enabled} 設置キャンセル={self.autoplay.keymap.place_cancel} 段差キャンセル={self.autoplay.keymap.step_cancel} 配ぷよ読み={self.haipuyo} 相手のツモを覗く={self.peek} エンジン={getattr(getattr(self, 'client', None), 'exe_path', '?')} 起動引数={getattr(getattr(self, 'client', None), 'args', '?')[1:]} 優先度を下げる={getattr(getattr(self, 'client', None), 'low_priority', '?')} ====="
        )
        t0 = time.perf_counter()
        self._t0_perf = t0
        stats = TickStats()
        cpu_meter, fps_meter = CpuMeter(), FpsMeter()
        perf_at = 0.0
        while self._running:
            tick_start = time.perf_counter()
            frame_bgr = capture.grab_bgr(region)
            grab_end = time.perf_counter()
            t = time.perf_counter() - t0
            self._now = t
            self._recent_frames.append((t, frame_bgr))

            snaps: dict[str, Snapshot] = {}
            banner_all_clear: list[str] = []
            for who, pc in players.items():
                board = read_board(frame_bgr, pc.geometry(), origin, banner_detector=self._banner_detectors[who])
                # 「全消し」の文字の有無が変わったらログに残す(文字の画素は read_board が分類から除いている)
                if (board.banner_score > 0) != self._banner_seen[who]:
                    self._banner_seen[who] = board.banner_score > 0
                    self._log(f"[{t:7.2f}s] {who:5s} 全消しの文字を{'検出(一致度 %.2f)' % board.banner_score if self._banner_seen[who] else '見なくなった'}")
                if board.banner_score >= ALL_CLEAR_BANNER_MIN_SCORE:
                    banner_all_clear.append(who)
                n1 = read_next(frame_bgr, pc.next_rects[0], origin)
                n2 = read_next(frame_bgr, pc.next_rects[1], origin)
                snaps[who] = Snapshot(t=t, cells=board.cells, next1=n1, next2=n2, pending_garbage=board.pending_garbage, eyeless_cells=board.eyeless_cells)
            self._check_new_game(t, snaps)
            self._frame_t = t
            self.trackers["self"].target_point = self._target_point()
            # 自分・相手の両方を更新し終えてから思考する(相手の更新前に思考すると
            # 同じフレームで起きた相手の発火・予告変化を見落とす)
            think = False
            # 再計算の理由(入力が変わった時だけ)。前の tick で実行できなかった理由があれば引き継ぐ
            rethink = self._deferred_rethink
            for who, snap in snaps.items():
                tr = self.trackers[who]
                if who in banner_all_clear:
                    self._all_clear_from_banner(t, who)
                for ev in tr.update(snap):
                    self._log(f"[{t:7.2f}s] {who:5s} {ev.kind:12s} {ev.detail}")
                    if ev.kind == "spawn_wait" and ev.detail.split(" ", 1)[0] in ("#1", "#10", "#20"):
                        # 診断(2026-10-10): 着地後に次の spawn を検出できない時の画面を残す
                        no = ev.detail.split(" ", 1)[0][1:]
                        self._saver.save(self.history_dir / f"spawn_wait_{who}_{t:07.2f}s_{no}.png", frame_bgr)
                    if who == "self" and ev.kind == "next_prestart" and tr.turn == 0:
                        self.autoplay.hold_down_for_start()  # 先行入力の初手(keymap.prehold の時だけ)
                    if who == "self" and ev.kind == "spawn":
                        if self._first_spawn_t is None:
                            self._first_spawn_t = t
                        self._turn_clock.spawn(t)
                        self._self_spawn_t = t
                        self._on_self_spawn(tr)
                    if who == "self" and ev.kind == "turn_ready":
                        think = True
                    if who == "self" and ev.kind == "chain_start":
                        self._on_self_chain_start()
                    if who == "self" and ev.kind == "chain_step":
                        self._on_self_chain_step(tr)
                        if "点滅から" in ev.detail:
                            self._save_flash_evidence(t)
                    if who == "self" and ev.kind == "chain_end":
                        self._on_self_chain_end(tr)
                    if ev.kind == "all_clear_unconfirmed":
                        self._on_game_over(t, who, tr)
                    if not self.solo and who == "enemy":
                        self._record_attack(t, tr, ev.kind)
                    if not self.solo and who == "enemy" and ev.kind == "spawn" and "initial" in ev.detail:
                        # 相手(CPU)の初手 = 対戦が始まった。自分の初手を動くのを待たずに確定できるようにする
                        me_tr = self.trackers["self"]
                        # 両者のツモは同じなので、相手の初手(と枠)は自分の初手の候補になる(自分の開始の判定に使う)
                        me_tr.start_hint = (tr.current_pair, tr.next1, tr.next2)
                        if me_tr.current_pair is None and not me_tr.start_confirmed:
                            me_tr.start_confirmed = True
                            self._log(f"[{t:7.2f}s] 相手の初手を検出: 自分の初手を動くのを待たずに確定します")
                    # 相手の仕掛けは自分の手番の途中でも提案を更新する(思考は手番開始時にしか走らない
                    # ため、操作中の仕掛けに次の手番まで対応できなかった)。再計算するのは入力が変わる
                    # 「相手の消去開始(点滅から復元した盤面で着弾量が確定する)」「連鎖終了」「予告が1段以上増加」の時
                    if not self.solo and who == "enemy" and ev.kind in ("chain_start", "chain_end"):
                        self._on_enemy_chain(tr, ev.kind)
                    if not self.solo and who == "enemy" and ev.kind == "chain_start":
                        rethink = "相手の消去開始を検出"
                        self._precomputed = None  # 相手の状態が変わったので古い先読みは使わない
                    if not self.solo and who == "enemy" and ev.kind == "chain_end":
                        rethink = "相手の連鎖が終了"
                    if not self.solo and who == "enemy" and ev.kind in ("chain_start", "chain_end"):
                        self._refresh_precompute_for_sent()
                    if who == "self" and ev.kind == "pending":
                        self._release_carry_if_pending_cleared(t, tr)
                    if who == "self" and ev.kind == "pending" and tr.pending_garbage - self._pending_at_think >= RETHINK_PENDING_INCREASE:
                        rethink = "予告おじゃまが増加"
                # 診断: 手番が進んでいるのに取り込めない状態が続いている場合、原因の手がかりと
                # その瞬間の画像を一度だけ記録する(spawn ごとに settling_since はリセットされるので、
                # 最後に取り込めた時刻から数える。全消し後に17秒止まった 2026-09-16 の原因調査用)
                stuck_for = t - max(tr.last_take_in_t, self._game_start_t)
                if tr.settling and not tr.chain_active and tr.turn > 0 and stuck_for > STUCK_WARN_SECONDS and (who, tr.last_take_in_t) not in self._stuck_logged:
                    self._stuck_logged.add((who, tr.last_take_in_t))
                    path = self.history_dir / f"stuck_{who}_{int(t):04d}s.png"
                    self._saver.save(path, frame_bgr)
                    self._log(
                        f"[{t:7.2f}s] {who:5s} 取り込めない状態が{stuck_for:.1f}秒続いています "
                        f"unknown={tr.last_unknown} covered={is_covered(snap.cells, snap.eyeless_cells)} poppable={tr.last_poppable} chain_active={tr.chain_active} "
                        f"pending={tr.pending_garbage} 画像={path.name}\n"
                        f"           生の認識結果:\n{_indent(tr.last_raw_text)}\n"
                        f"           確定盤面:\n{_indent(cells_to_text(tr.settled))}"
                    )
            me = self.trackers["self"]
            # 着弾量の保持は「予告が見えている間」だけ。相手の連鎖が終わって予告も無い状態が続いたら
            # (自分の攻撃で相殺し切った等)解除する
            en_tr = self.trackers["enemy"]
            if self._incoming_carry > 0 and not en_tr.chain_active and me.pending_garbage == 0:
                if self._carry_zero_since is None:
                    self._carry_zero_since = t
                elif t - self._carry_zero_since >= INCOMING_CARRY_CLEAR_SECONDS:
                    self._log(f"[{t:7.2f}s] 相手の連鎖が終わり予告も無いので着弾量 {self._incoming_carry}個 を解除")
                    self._incoming_carry = 0
                    self._carry_zero_since = None
            else:
                self._carry_zero_since = None
            self._deferred_rethink = rethink
            self._record_move_trace(t, frame_bgr)
            self._retry_mismatched_precompute(me)
            self._send_blocked_move(me)
            self._poll_think_results()
            self._replan_after_landing()
            self._opening_preview(t, me)
            ready = not me.settling and not me._await_spawn_after_chain and me.current_pair is not None
            retry = self._think_pending and me.next1 is not None and (me.next2 is not None or self._last_data is None)
            if ready and (think or retry or rethink):
                if retry and not think:
                    self._log(f"[{t:7.2f}s] 認識が揃ったので同じ手番で思考をやり直します")
                if rethink and not think:
                    self._log(f"[{t:7.2f}s] {rethink}: 操作中の組ぷよの提案を再計算")
                # 同じ tick の turn_ready/認識回復と重なっても再計算の理由を失わない。
                # 思考できなかった(入力不足)なら理由を次の tick へ持ち越す
                # 新しい手番(turn_ready)では前の手番の表示を「維持」してはいけないので new_turn を渡す
                if self._think(t, frame_bgr, note=f"{rethink}: 提案を更新" if rethink else "", new_turn=think):
                    self._deferred_rethink = ""

            elapsed = time.perf_counter() - tick_start
            if elapsed >= OBSERVATION_GAP_WARN_SEC:
                self._log(f"[{t:7.2f}s] 観測の空白 {elapsed * 1000:.0f}ms (この間の画面変化は見えていない)")
            report = stats.add(t, tick_start, grab_end - tick_start, elapsed)
            if report:
                self._log(f"[{t:7.2f}s] {report}")
            fps_meter.tick(t)
            if t - perf_at >= PERF_INTERVAL_SEC:
                perf_at = t
                cpu = cpu_meter.sample()
                # CPU 使用率をログにも残す(2026-10-26、利用者「移動ミスは CPU の負荷によるものではないか」): 1分ごとの統計と置きミスの診断に出す
                self.last_cpu = cpu
                if cpu is not None:
                    stats.cpu.append(cpu)
                self.perf_ready.emit(perf_text(cpu, fps_meter.fps()))
            if elapsed < TICK_INTERVAL_SEC:
                time.sleep(TICK_INTERVAL_SEC - elapsed)

    def _all_clear_from_banner(self, t: float, who: str) -> None:
        """「全消し」の文字がはっきり見えたら、そのプレイヤーは全消しを持っているとする。連鎖の終わりの盤面を読み違えて(1個残った等)
        全消しを記録できず、双方が全消しの時の構えが働かなかった(実機 2026-10-11: 相手の文字は一致度0.97)"""
        tr = self.trackers[who]
        if not tr.all_clear:
            tr.all_clear = True
            self._log(f"[{t:7.2f}s] {who:5s} 全消しの文字から全消しを持っているとします")

    def _opening_preview(self, t: float, me: PlayerTracker) -> None:
        """開始前(操作中の組ぷよが無く、盤面が空、ネクスト枠に第1手・第2手が見えている)に初手を先に計算して
        表示する(実機ではカウントダウンの約4秒間、枠に第1手・第2手が出る 2026-09-20 15:50)。
        第3手は見えないので2組で solo し、第2手は開始後の通常思考に任せる。"""
        if me.current_pair is not None or me.turn != 0 or me.next1 is None or me.next2 is None or count_puyos(me.settled) != 0:
            return
        # 開始の瞬間にすぐ送れるよう、2通りの解釈の初手を思考スレッドで先に計算しておく(観測ループで計算すると、開始の直前に
        # 画面を見る処理が約0.2秒止まった。実機 2026-10-11):
        #  - 枠がまだ入れ替わっていない: 枠の第1・2手(第3手は未読)
        #  - 直前の入れ替わりが開始だった: 枠から消えた組(tracker._prestart_first)と枠の第1・2手
        # 開始の直前は枠に第1・2手が出ているので、開始の入れ替わりの後も照合キー(出現した組ぷよ, ネクスト)は同じになる
        seqs: list[tuple[Pair, ...]] = [(me.next1, me.next2)]
        if me._prestart_first is not None:
            seqs.append((me._prestart_first, me.next1, me.next2))
        for seq in seqs:
            if seq in self._opening_moves or seq in self._opening_requested:
                continue
            self._opening_requested.add(seq)
            trial = self.mapper.fork()
            try:
                pairs = [trial.pair(*p) for p in seq]
            except ValueError as exc:
                self._log(f"[{t:7.2f}s] 開始前の初手計算をスキップ: {exc}")
                continue
            kind = "open3" if len(seq) > 2 else "open2"
            self._dispatch(kind, {"t": t, "seq": seq},
                           lambda pairs=pairs: self.client.solo(PuyoField(), pairs, history=[]))

    def _apply_opening(self, ctx: dict, move: BattleMove | None, exc: BaseException | None) -> None:
        """開始前の初手の計算結果を残す。開始の後に届き、出現した組ぷよと合えば、その場で送る。"""
        t, seq = ctx["t"], ctx["seq"]
        if exc is not None:
            if isinstance(exc, (ValueError, RuntimeError, NoValidPlacementError)):
                self._log(f"[{t:7.2f}s] 開始前の初手計算をスキップ: {exc}")
                return
            raise exc
        assert move is not None
        self._opening_moves[seq] = move
        self._log(f"[{t:7.2f}s] 開始前の初手: 第1手={seq[0]} 第2手={seq[1]} 第3手={seq[2] if len(seq) > 2 else None} → x={move.x} {move.rotation} form={move.form} ({move.think_ms}ms)")
        me = self.trackers["self"]
        if me.current_pair is not None:
            if me.turn == 0 and (seq[0], seq[1]) == (me.current_pair, me.next1) and self._select_opening(me):
                self._log("           開始前の初手が開始の後に届いた: 送れるか確かめます")
                self._autoplay_precomputed(me)
            return
        # 表示は今の枠の解釈に合うもの(入れ替わりの後なら3組の方)
        preferred = (me._prestart_first, me.next1, me.next2) if me._prestart_first is not None else (me.next1, me.next2)
        if seq != preferred:
            return
        child_color, axis_color = seq[0]
        data = OverlayDrawData(
            axis_cell=move.axis_cell, child_cell=move.child_cell, axis_color=axis_color, child_color=child_color,
            # 第3手が見えるまで(開始の約0.3秒前に枠が入れ替わるまで)は2組だけの計算なので、開始時に変わりうる
            # 「仮」の提案と明示する(実機 2026-10-10 14:08: 利用者「開始前のガイドが正しくない。2手先を認識していない?」)
            label=move.label, self_analysis=move.self_analysis,
            note="開始前: 第1手の提案(第1〜3手から)" if len(seq) > 2 else "開始前: 仮の提案(第3手が見えたら変わります)",
            form=self._form_text(move),
        )
        self._last_data = data
        self.draw_data_ready.emit(data)

    def _select_opening(self, me: PlayerTracker) -> bool:
        """試合開始の出現で、計算済みの初手から出現した組ぷよ・ネクストに合うものを先読みとして置く(3組で計算した方を優先)。"""
        best = None
        for seq, move in self._opening_moves.items():
            if (seq[0], seq[1]) != (me.current_pair, me.next1):
                continue
            if best is None or (len(seq) > len(best[0]) and (me.next2 is None or seq[2] == me.next2)):
                best = (seq, move)
        if best is None:
            return False
        empty = [[EMPTY] * COLS for _ in range(ROWS)]
        self._precomputed = (empty, (me.current_pair, me.next1), best[1])
        self._precomputed_attack = None
        return True

    def _attack_involved(self, me: PlayerTracker) -> bool:
        """相手の攻撃が絡む局面か(自分への予告・着弾見込み・相手の連鎖中・相手への予告)。"""
        en = self.trackers["enemy"]
        return attack_involved(me.pending_garbage, self._incoming_carry, self.solo, en.chain_active, en.pending_garbage,
                               getattr(self.client, "engine", "ama"))

    def set_autoplay(self, on: bool) -> None:
        """支援モードの途中で全自動を切り替える(2026-10-27、利用者)。オフでは押しているキーを離して送信中の手順も止め、
        オンでは次の手番から送る(操作中の組ぷよの途中からは送らない)。仮想パッドの側は、オンにした時につなぐ。
        画面のスレッドから呼ばれる(全自動の値の読み書きは1つの bool で、観測のスレッドは毎手番その値を見る)。"""
        if on == self.autoplay.enabled:
            return
        me = self.trackers["self"]
        if on:
            sender = self.autoplay.sender
            if isinstance(sender, PadSender) and not sender.send:
                sender = PadSender(slot=self.side)
                if not sender.available:
                    self._log(f"自動操作: 仮想パッドを作れません({sender.error})。ViGEmBus ドライバーと vgamepad を入れてください")
                self.autoplay.sender = sender
            self.autoplay.resume_after_turn = me.turn
            self.autoplay.enabled = True
            self._log(f"[{self._now or 0.0:7.2f}s] 全自動をオンにしました(turn={me.turn} の次の手番から送ります)")
        else:
            self.autoplay.enabled = False
            self.autoplay.stop()
            self._log(f"[{self._now or 0.0:7.2f}s] 全自動をオフにしました(押していたキーを離しました)")

    def _prehold_action(self, turn: int) -> str | None:
        """先行入力(keymap.prehold、試験): 手番 turn の着地の後に、次の組ぷよが出る前から押しっぱなしにしてよい操作。
        次の組ぷよの先読みが回転なしで壁まで動かすだけの手で(wall_prehold_action)、出現の時刻が読める(置いた手で消えない・
        おじゃまが降らない・相手の攻撃が絡まない)時だけ返す。全自動のスレッドから呼ばれる。"""
        me = self.trackers["self"]
        pre = self._precomputed
        if pre is None or not pre[2].path or me.turn != turn:
            return None
        if pre[1] != (me.next1, me.next2):  # 次に出る組ぷよ(今のネクスト)用の先読みでない
            return None
        if has_poppable_group(pre[0]) or me.pending_garbage > 0 or self._attack_involved(me) or pre[2].label == "harass":
            return None
        return wall_prehold_action(pre[2].path, pre[2].x)

    def _autoplay_precomputed(self, me: PlayerTracker) -> None:
        """組ぷよが出た時点(spawn)で、先読みしてある手順を送る。

        先読みは「提案どおりに置いた後の盤面」で計算してあるので、その盤面どおりになっていれば
        思考を待つ必要がない。取り込み・思考を待つと 0.15〜0.25 秒の硬直になる
        (実機 2026-09-27: 利用者「盤面に変化がないのならノータイムで置けるはず」)。
        盤面が予想と違う(連鎖・おじゃま・置き間違い)時は送らず、通常の思考に任せる。
        """
        if not self.autoplay.enabled or self.autoplay.committed_turn == me.turn:
            return
        reason = self._send_precomputed(me)
        if reason:
            # 送れなかった理由は必ず残す(出現時点で送れないと、取り込み・思考を待つ間に組ぷよが自由落下する)
            self._log(f"           自動操作: 出現時点では送りません(turn={me.turn}): {reason}")
        edge = self._self_edge_offset()
        if reason is None and edge is not None:
            self._edge_normal.append(edge)
        if reason == BOARD_MISMATCH:
            self._log_board_mismatch(me, edge)
        if reason in (BOARD_MISMATCH, HARASS_WAIT):
            self._mismatch_retry = (me.turn, self._now or 0.0)

    def _self_edge_offset(self) -> int | None:
        """最新のフレームで、自分の盤面の右の枠の横位置を校正からのずれ(px)で測る。校正・フレームが無ければ None。"""
        if self.calibration is None or not self._recent_frames:
            return None
        region = self.calibration.capture_region()
        return field_edge_offset(self._recent_frames[-1][1], self.calibration.self_player.geometry(), (region.left, region.top))

    def _edge_text(self, edge: int | None) -> str:
        if edge is None:
            return ""
        text = f" 枠の横位置: {edge:+d}px"
        if self._edge_normal:
            vals = sorted(self._edge_normal)
            text += f"(普段 {vals[len(vals) // 2]:+d}px、送れた出現{len(vals)}回の中央値)"
        return text

    def _log_board_mismatch(self, me: PlayerTracker, edge: int | None = None) -> None:
        """「盤面が違う」時に、違ったマス(期待した色と読んだ色)と出現の瞬間の画面を残す(何を読み違えたかの調査用)。
        盤面の右の枠の横位置も添える(横揺れなら普段の値からずれる)。"""
        pre = self._precomputed
        if pre is None:
            return
        diff = board_diff(me.observed_cells, pre[0])
        name = ""
        if self._recent_frames and self._mismatch_frames_saved < MISMATCH_FRAME_LIMIT:
            ft, frame = self._recent_frames[-1]
            name = f"mismatch_self_{ft:07.2f}s_turn{me.turn}.png"
            self._saver.save(self.history_dir / name, frame)
            self._mismatch_frames_saved += 1
        self._log(f"           違うマス: {'、'.join(diff) if diff else '(なし)'}" + (f" 画像={name}" if name else "")
                  + self._edge_text(edge))

    def _retry_mismatched_precompute(self, me: PlayerTracker) -> None:
        """出現時点で「盤面が違う」で送れなかった先読みを、毎フレーム観測盤面と比べ直し、一致した時点で送る。
        盤面以外の理由になった・キーを送った・手番が進んだら比べ直しをやめる(本当に盤面が違えば今までどおり思考を待つ)。
        「小さい消去なので思考を待つ」の手番も、出現から HARASS_WAIT_MAX_SEC たっても送っていなければ先読みの手を送る。"""
        if self._mismatch_retry is None:
            return
        turn, t_spawn = self._mismatch_retry
        if turn != me.turn or not self.autoplay.enabled or self.autoplay.committed_turn == me.turn:
            self._mismatch_retry = None
            return
        ms = ((self._now or t_spawn) - t_spawn) * 1000
        give_up_wait = ms >= HARASS_WAIT_MAX_SEC * 1000
        reason = self._send_precomputed(me, when=f"出現の{ms:.0f}ms後", allow_harass=give_up_wait)
        if reason is None:
            if give_up_wait and self._sent_move is not None and self._sent_move[1].label == "harass":
                self._log(f"           自動操作: 思考の手が出現から{HARASS_WAIT_MAX_SEC:.1f}秒たっても送られないので、先読みの小さい消去をそのまま送りました(turn={me.turn})")
            edge = self._self_edge_offset()
            if edge is not None:
                self._log(f"           送った時の{self._edge_text(edge).strip()}")
        if reason not in (BOARD_MISMATCH, HARASS_WAIT):
            self._mismatch_retry = None

    def _send_precomputed(self, me: PlayerTracker, when: str = "出現時点", board: list[list[str]] | None = None,
                          allow_harass: bool = False) -> str | None:
        """_autoplay_precomputed の本体。送ったら None、送れなければ理由を返す。when はログの「いつ送ったか」。
        board は比べる盤面(省略時は画面の観測盤面。取り込み時点では確定した盤面を渡す)。
        allow_harass=True なら小さい消去の先読みも送る(思考を待つ上限を過ぎた時)。"""
        pre = self._precomputed
        if me.chain_active:
            return "自分の連鎖中"
        if pre is None or not pre[2].path or pre[1] != (me.current_pair, me.next1):
            # 相手の消去開始で先読みを捨てた後: 相手の発火を想定した先読みがあれば使う
            alt_reason = ""
            if self._attack_involved(me):
                alt_reason = self._autoplay_anticipated(me)
                if alt_reason is None:
                    return None
            if pre is None:
                base = "先読みを計算中(届いたら送る)" if self._pre_inflight is not None else "先読みが無い"
            elif not pre[2].path:
                base = "先読みの手順が無い"
            else:
                base = "組ぷよが先読みと違う"
            return base + (f"。相手の発火の想定: {alt_reason}" if alt_reason else "")
        # 予想した盤面になっているか: 個数ではなく盤面全体で比べる(13段目と出現位置は除く)。個数だけだと、前の手が
        # 置き場所からずれても一致とみなし、別の盤面用の手順を送って次の手もずれた(実機 2026-10-10 15:28: 提案と
        # 違う設置17件のうち9件。高い積みでは列の高さが違うだけで行き先が変わる)
        if not (observed_match(me, pre[0]) if board is None else boards_match(board, pre[0])):
            return "先読みの盤面と実際の盤面が違う"
        if pre[2].label == "harass" and not allow_harass:
            # 小さい消去(単発・小連鎖)は、組ぷよが2組しか見えない先読みでは判断が粗いので、3組目まで見た思考を待つ
            # (実機 2026-10-11 07:23: 先読みが緑4個の単発を選んで送り、思考は積む手だった)。本線の発火(main)は送る
            return HARASS_WAIT
        # 相手の攻撃が絡む局面では思考が別の手(受け・潰し)に変えることがあり、キーは取り消せないので
        # 先読みで先走らない(実機 2026-09-27 12:01: 送った手順と表示が食い違う設置が 49/428件)
        same_attack = False
        if self._attack_involved(me):
            # 相手の攻撃中でも、先読みを計算した時と攻撃の状況が同じなら、その手は今の状況で考えた手と同じ前提
            # (相手の連鎖中に硬直しない。実機 2026-10-10 15:28)
            now = self._attack_snapshot(me)
            if self._precomputed_attack is None or self._precomputed_attack != now:
                alt_reason = self._autoplay_anticipated(me)
                if alt_reason is None:
                    return None
                return f"攻撃の状況が先読みと違う(先読み {self._precomputed_attack} / 今 {now})。相手の発火の想定: {alt_reason}"
            same_attack = True
        if self._autoplay_send(me, pre[2].path, pre[2], board=pre[0]):
            note = "、攻撃の状況は先読みと同じ" if same_attack else ""
            self._log(f"           自動操作: 先読みの手順を{when}で送信(turn={me.turn} x={pre[2].x} {pre[2].rotation}{note})")
            return None
        return "キーを送れなかった"

    def _autoplay_anticipated(self, me: PlayerTracker) -> str | None:
        """相手の発火を想定した先読み(_dispatch_precompute の pre_atk)を、出現時点で送る。相手が連鎖中で、
        組ぷよ・盤面が先読みどおり、見積もった着弾量が想定に近い時だけ(違えば通常の思考に任せる)。
        送ったら None、送れなければ理由を返す。"""
        alt = self._precomputed_alt
        if alt is None:
            return "無い"
        if not self.trackers["enemy"].chain_active:
            return "相手が連鎖中でない"
        expected, pairs, move, assumed = alt
        if pairs != (me.current_pair, me.next1):
            return "組ぷよが違う"
        if not observed_match(me, expected):
            return "盤面が違う"
        est = self._incoming_estimate()
        if not est or abs(est - assumed) > max(ANTICIPATE_TOLERANCE_MIN, assumed // 4):
            return f"想定({assumed}個)と見積もり({est}個)が違う"
        if self._autoplay_send(me, move.path, move, board=expected):
            self._precomputed_alt = None
            self._log(f"           自動操作: 相手の発火を想定した先読みの手順を出現時点で送信(turn={me.turn} x={move.x} {move.rotation}、"
                      f"想定{assumed}個・見積もり{est}個)")
            return None
        return "キーを送れなかった"

    def _on_self_spawn(self, me: PlayerTracker) -> None:
        """手番が切り替わった: 前の手番のドット(置いたツモの色)を出したままにしない(実機 2026-09-19:
        「すでに置いたツモの色を提示することがある」)。2手先の提案があればそれを新しい組ぷよの提案として
        先に出し(色はネクストから読んだ新しい組ぷよ)、無ければドットを消してパネルだけ残す。
        turn_ready で思考が終われば正式な提案で上書きされる。"""
        if me.turn == 0 and count_puyos(me.settled) == 0:
            self._select_opening(me)  # 試合開始: 計算済みの初手から、出現した組ぷよに合うものを使う
        self._autoplay_precomputed(me)
        data = self._last_data
        if data is None:
            return
        if (
            data.next_axis_cell is not None and data.next_child_cell is not None
            and me.current_pair is not None and (data.next_child_color, data.next_axis_color) == me.current_pair
        ):
            promoted = replace(
                data,
                axis_cell=data.next_axis_cell, child_cell=data.next_child_cell,
                axis_color=data.next_axis_color, child_color=data.next_child_color,
                next_axis_cell=None, next_child_cell=None, next_axis_color="", next_child_color="",
                note="先読み(確認中)", fire_chain=(0, 0), waiting=False,  # 連鎖中に隠していたら再表示
                chaining=me.chain_active,
            )
        else:
            promoted = replace(data, waiting=True, note="次の手を計算中", next_axis_cell=None, next_child_cell=None, chaining=me.chain_active)
        self._last_data = promoted
        self.draw_data_ready.emit(promoted)

    def _on_self_chain_start(self) -> None:
        """自分の発火を検出した: 置き終わった組ぷよのドットを連鎖の間(約8秒)出したままにしない
        (実機 2026-09-20 10:20: 自分のネクスト枠は連鎖が終わるまで入れ替わらず spawn が出ないため、
        旧ドットが次の手番まで残って干渉した)。2手先の提案は残しておき、次の組ぷよが出た spawn で
        _on_self_spawn が色の一致を見て差し替える。"""
        me = self.trackers["self"]
        exp = me.chain_expected
        # 撃っている連鎖(点滅から再現した連鎖数・点数)を自分の欄に出す。再現できなければ段数を観測で数えて出す(_on_self_chain_step)
        self._self_fire = (exp.steps, exp.score, True) if exp is not None else (0, 0, True)
        self._self_fire_first = self._self_fire  # 連鎖の始めに出した値(終わりに訂正したかを比べる)
        data = self._last_data
        if data is None:
            return
        if data.waiting:
            shown = replace(data, chaining=True, self_firing=self._self_fire)
        else:
            shown = replace(data, waiting=True, note="連鎖中: 次の手を待機", chaining=True, self_firing=self._self_fire)
        self._last_data = shown
        self.draw_data_ready.emit(shown)

    def _on_self_chain_step(self, me: PlayerTracker) -> None:
        """自分の連鎖の段が消えた: 再現が外れていたら(再現を打ち切った・観測の段数が再現より多い)、出している連鎖を訂正する。"""
        fire = self._self_fire
        if fire is None or not fire[2]:
            return
        exp = me.chain_expected
        if exp is not None and me.chain_steps <= exp.steps:
            return  # 再現どおりに進んでいる
        corrected = (me.chain_steps, 0, True)  # 観測で数えた段数(点数は分からない)
        if corrected == fire:
            return
        self._self_fire = corrected
        data = self._last_data
        if data is not None:
            self._last_data = replace(data, self_firing=corrected)
            self.draw_data_ready.emit(self._last_data)

    def _on_self_chain_end(self, me: PlayerTracker) -> None:
        """自分の連鎖が終わった: 放った連鎖数・点数(発火直前盤面から再現できた時)をパネルに出す。"""
        shown_fire = self._self_fire_first
        self._self_fire_first = None
        self._self_fire_end = time.perf_counter()
        self._self_fire = (me.chain_steps, me.chain_score, False) if me.chain_steps > 0 else None
        data = self._last_data
        if data is None:
            return
        # 連鎖中に出していた値と違えば訂正したことをメモに出す(点滅から再現した盤面が外れていた時など)
        note = data.note
        if (shown_fire is not None and shown_fire[0] > 0 and self._self_fire is not None
                and (shown_fire[0], shown_fire[1]) != self._self_fire[:2] and shown_fire[1] > 0):
            after = (f"{me.chain_steps}連鎖 {me.chain_score:,}点" if me.chain_score > 0 else f"{me.chain_steps}連鎖(点数は再現できず)")
            note = f"点数を訂正: {shown_fire[0]}連鎖 {shown_fire[1]:,}点 → {after}"
            self._log(f"           {note}")
        elif note == "連鎖中: 次の手を待機":
            note = ""
        shown = replace(data, note=note, chaining=False, self_firing=self._self_fire)
        self._last_data = shown
        self.draw_data_ready.emit(shown)

    def _on_enemy_chain(self, en: PlayerTracker, kind: str) -> None:
        """相手の連鎖が始まった・終わった: 撃っている連鎖(点滅から再現した連鎖数・点数)をパネルの相手の欄に出す。
        連鎖中に届く思考の相手の解析は撃つ前の盤面のもので、先の手で撃てる最大(撃っている連鎖とは別)を出してしまう
        (実機 2026-10-25 12:23: 6連鎖を撃っている間「9連鎖」)。連鎖が終わった後の思考が届いたら解析に戻す(_emit_move)"""
        if kind == "chain_start":
            exp = en.chain_expected
            self._enemy_fire = (exp.steps, exp.score, True) if exp is not None else (0, 0, True)
        else:
            self._enemy_fire = (en.chain_steps, en.chain_score, False)
            self._enemy_fire_end = time.perf_counter()
        data = self._last_data
        if data is None:
            return
        shown = replace(data, enemy_firing=self._enemy_fire)
        self._last_data = shown
        self.draw_data_ready.emit(shown)

    def _record_attack(self, t: float, en: PlayerTracker, kind: str) -> None:
        """相手の連鎖を攻撃エピソードとして記録する(新方針の第1段階: 提案は変えない)。"""
        me = self.trackers["self"]
        if kind == "chain_start":
            self._attack_count += 1
            self._attack = AttackEpisode(
                attack_id=self._attack_count,
                started_at=t,
                before=[row[:] for row in (en.chain_before or en.settled)],
                bonus=en.chain_bonus,
                my_pending_at_start=me.pending_garbage,
            )
            self._attack.add_estimate(t, "前回解析の相手本線", self._last_enemy_main_garbage())
            self._log(
                f"[{t:7.2f}s] 攻撃#{self._attack_count} 開始: 発火直前の相手盤面(由来={en.chain_before_source})\n"
                f"{_indent(cells_to_text(self._attack.before))}"
            )
        elif self._attack is not None and kind == "chain_step":
            self._attack.steps = en.chain_steps
            self._attack.dropped = en.chain_dropped
        elif self._attack is not None and kind == "chain_end":
            ep = self._attack
            ep.ended_at = t
            ep.steps = max(ep.steps, en.chain_steps)
            ep.dropped = max(ep.dropped, en.chain_dropped)
            ep.my_pending_at_end = me.pending_garbage
            if self._incoming_carry > 0:
                self._log(f"[{t:7.2f}s] 自分への着弾見込み {self._incoming_carry}個 を予告が消えるまで保持")
            avg = self._turn_clock.average
            clock = "平均手時間 未計測" if avg is None else f"平均手時間 {avg:.2f}s(n={self._turn_clock.samples}) → この連鎖の間に約{self._turn_clock.moves_within(ep.duration or 0.0)}手"
            self._log(f"[{t:7.2f}s] {ep.summary()} / {clock}")
            self._attack = None

    def _build_states(self, me: PlayerTracker, en: PlayerTracker, my_settled: list[list[str]], my_pairs_src: list[tuple[str, str]],
                      ahead: bool = False) -> tuple[PlayerState, PlayerState | None, list[str], list[str]]:
        """思考エンジンに渡す自分・相手の状態を作る。

        自分側の変換失敗(5色目など)は ValueError のまま呼び出し側に伝える。
        相手側の変換失敗は自分の提案を止めない: 相手状態を None にして相手を見ない思考に
        切り替える(実機 2026-09-15: 相手ネクストの誤認識で自分の思考が5手番止まった)。
        """
        # 状態の生成は録画検証(tools/replay_video.py)と共通(src/engine/think_input.py)。
        # 色の割当は変換を試してから採用する(途中で失敗しても仮の登録を残さない)
        inp = build_think_input(self.mapper, me, en, my_settled, my_pairs_src, solo=self.solo, incoming_hint=self._incoming_carry,
                                now=self._now, haipuyo=self.haipuyo or isinstance(self.client, BookClient), ahead=ahead, peek=self.peek)
        self.mapper = inp.mapper
        if inp.enemy_error:
            self._log(f"           相手の認識エラー: {inp.enemy_error} → 相手を仮の状態にして思考します")
        return inp.my_state, inp.en_state, inp.my_pairs, inp.en_pairs

    def _last_enemy_main_garbage(self) -> int:
        """直近の提案時に解析した相手の本線のおじゃま個数(解析が無ければ 0)。"""
        data = self._last_data
        if data is None or data.enemy_analysis is None:
            return 0
        return data.enemy_analysis.max.garbage

    def _emit_move(self, me: PlayerTracker, move: BattleMove, next_move: BattleMove | None = None, note: str = "") -> None:
        assert me.current_pair is not None
        child_color, axis_color = me.current_pair
        data = OverlayDrawData(
            axis_cell=move.axis_cell,
            child_cell=move.child_cell,
            axis_color=axis_color,
            child_color=child_color,
            label=move.label,
            action=action_text(move, move.incoming or self._incoming_carry or me.pending_garbage),
            fire_chain=move.chain_after,
            self_analysis=move.self_analysis,
            enemy_analysis=move.enemy_analysis,
            note=note or (book_text(move) if is_book_move(move) else FIRE_REASON_TEXT_JA.get(move.fire_reason or "", "")),
            form=self._form_text(move),
            chaining=me.chain_active,
        )
        # 相手の連鎖中は撃っている連鎖を出したままにする。連鎖が終わった後に依頼した思考なら解析(撃てる最大)に戻す
        # (連鎖中に依頼した思考の相手の解析は撃つ前の盤面のもの)
        if (self._enemy_fire is not None and not self._enemy_fire[2] and not self.trackers["enemy"].chain_active
                and self._think_submitted >= self._enemy_fire_end):
            self._enemy_fire = None
        data.enemy_firing = self._enemy_fire
        # 自分も同じ: 連鎖中は撃っている連鎖、終わった後は、その後に依頼した思考が届くまで撃った連鎖を出す
        if (self._self_fire is not None and not self._self_fire[2] and not me.chain_active
                and self._think_submitted >= self._self_fire_end):
            self._self_fire = None
        data.self_firing = self._self_fire
        # 最終的な構想(2026-10-24、利用者): 配ぷよ読み+全自動の時だけ、撃つ直前の盤面をドットで出す
        if self.haipuyo and self.autoplay.enabled and move.plan_field is not None:
            data.plan_dots = plan_dots(self.mapper.rows_to_cells(move.plan_field), me.settled, move.plan_trigger)
            c, s, k = move.plan_chain
            data.plan_text = f"構想: {c}連鎖 {s:,}点 @{k}"
        self._last_data = data
        self._autoplay_move(me, move)
        if next_move is not None and me.next1 is not None:
            data.next_axis_cell = next_move.axis_cell
            data.next_child_cell = next_move.child_cell
            data.next_child_color, data.next_axis_color = me.next1
        self.draw_data_ready.emit(data)

    def _autoplay_move(self, me: PlayerTracker, move: BattleMove) -> None:
        """全自動モード: この手番の操作手順をゲームへ送る(画像認識の試験用。CPU戦で使う)。

        手順は ama が生成したもの(回し・床蹴り込み)。手番ごとに1回だけ送り、連鎖中・取り込み待ちには送らない。
        """
        if not self.autoplay.enabled:
            return
        if not move.path:
            self._log(f"           自動操作: 手順が無いので送りません(x={move.x} {move.rotation})")
            return
        if me.chain_active or me.settling:
            # 理由を残す(2026-10-21: 思考の手が届いたのに送られず自由落下した手番の原因がログから分からなかった)
            self._log(f"           自動操作: 思考の手を送りません(turn={me.turn}): {'自分の連鎖中' if me.chain_active else '盤面の確定待ち'}")
            if not me.chain_active:
                self._blocked_move = (me.turn, move)
            return
        if not self._autoplay_send(me, move.path, move) and self.autoplay.committed_turn != me.turn:
            self._log(f"           自動操作: 思考の手のキーを送り始められませんでした(turn={me.turn})")

    def _send_blocked_move(self, me: PlayerTracker) -> None:
        """保険(2026-10-24): 盤面の確定待ちで送れなかった思考の手を、出現から BLOCKED_SEND_SEC たっても確定待ちのままなら送る。
        手番が進んだ・キーを送った・自分の連鎖が始まった時はやめる。"""
        blocked = self._blocked_move
        if blocked is None:
            return
        turn, move = blocked
        if turn != me.turn or not self.autoplay.enabled or self.autoplay.committed_turn == me.turn or me.chain_active:
            self._blocked_move = None
            return
        if not me.settling:
            return  # 確定待ちが解けたら通常どおり思考し直す
        waited = (self._now or self._self_spawn_t) - self._self_spawn_t
        if waited < BLOCKED_SEND_SEC:
            return
        self._blocked_move = None
        if self._autoplay_send(me, move.path, move):
            self._log(f"           自動操作: 盤面の確定待ちが出現から{waited * 1000:.0f}ms続いたので、思考の手を送りました(turn={me.turn} x={move.x} {move.rotation})")
        else:
            self._log(f"           自動操作: 盤面の確定待ちの保険で思考の手を送れませんでした(turn={me.turn})")

    def _autoplay_send(self, me: PlayerTracker, path: list[str], move: BattleMove | None = None,
                       board: list[list[str]] | None = None) -> bool:
        """操作手順を送る。落下の押しっぱなしは、観測盤面にぷよが2個増えた(=着地した)ら離す。

        手番の進行(spawn)を待つと着地の0.7秒後になり、その間に出た次の組ぷよまで下移動させてしまう
        (実機 2026-09-27 11:47)。
        """
        turn, base = me.turn, me.observed_count
        base_text = me.last_observed_text
        state = {"first_poll": None, "ghost_logged": False}
        pair_colors = set(me.current_pair) if me.current_pair is not None else set()
        # 着地の判定は「提案した2マスが組ぷよの2色で埋まった」を主にする。観測個数が2個増えただけでは、
        # 高い列の真上を通る組ぷよや演出が積みとつながって見えた瞬間に着地とみなし、目標の手前で下キーを
        # 離して自然落下させていた(実機 2026-09-27 14:57: 落下42〜63msで離した例)。個数の条件は、
        # 置き場所がずれた時の保険として落下開始から COUNT_RELEASE_MIN_SEC 経ってからだけ使う
        # 13段目(row=0)は見えないので、置き場所が13段目にかかる時は見える方の1マスで判定する(2マスとも見なければ、
        # 個数も1個しか増えず、下キーを次の組ぷよの出現まで押し続けた。実機 2026-10-11 11:19 turn=24)
        target = None
        if move is not None and me.current_pair is not None:
            child_color, axis_color = me.current_pair
            cells = [(cell, color) for cell, color in ((move.axis_cell, axis_color), (move.child_cell, child_color)) if cell[0] > 0]
            target = tuple(cells) if cells else None
        # 置きミスの診断の記録(送り始めたら self._move_trace にする)。下キーを離した理由は思考とは別のスレッドで書き込む
        trace = {"turn": turn, "t_send": time.perf_counter() - self._t0_perf, "target": target, "colors": pair_colors,
                 "path": list(path), "move": move, "released": None, "track": [], "frames": []}

        def placed() -> bool:
            now = time.perf_counter()
            if state["first_poll"] is None:
                state["first_poll"] = now
            elapsed = now - state["first_poll"]
            reasons = []
            cells = me.observed_cells
            if target is not None and cells is not None and all(cells[r][c] == color for (r, c), color in target):
                # 落下中の組ぷよ(組ぷよの色の浮いているぷよ)がまだ見えている間は着地とみなさない。提案の2マスにちょうど出る
                # 着地予測の点を組ぷよと読み、列に着いた直後に下キーを離して自然落下させていた(実機 2026-10-12 07:18 turn=22:
                # 5段落とす位置で落下開始から24msで離し、次の出現まで1.93秒。相手の連鎖中は演出・横揺れで点を除く判定が外れやすい)
                falling = [(r, c) for r, c, color in me.floating_now if color in pair_colors]
                if not falling:
                    reasons.append("提案の2マスに着地")
                elif not state["ghost_logged"]:
                    state["ghost_logged"] = True
                    self._log(f"           自動操作: turn={turn} 提案の2マスが埋まって見えるが、落下中の組ぷよが浮いているので下キーを離しません"
                              f"(浮いている {falling}、落下開始から{elapsed * 1000:.0f}ms。着地予測の点の読み違いの疑い)")
            if me.observed_count >= base + 2 and (target is None or elapsed >= COUNT_RELEASE_MIN_SEC):
                reasons.append(f"観測個数 {base}→{me.observed_count}")
            if me.turn != turn:
                reasons.append(f"手番 {turn}→{me.turn}")
            # 自分の連鎖中かどうかでは離さない: キーは連鎖中には送らないので、落下中に始まった「連鎖」は相手の攻撃の演出などの
            # 誤検出で、離すと組ぷよが自由落下した(実機 2026-10-11 11:19 turn=25: 落下開始から0msで離した)。
            # 置いた組ぷよが連鎖を起こした時は、先に着地(提案のマス・個数)で離す。外れても上限時間で離す
            if not reasons:
                return False
            # 下キーを離した理由を残す。落下の直後(150ms未満)に離した時は、着地前に条件が成立した疑いが
            # あるので送信時と今の観測盤面も残す(実機 2026-09-27 14:52: 落下11ms・21msで離し、
            # 組ぷよが自然落下した。個数が2個増えた理由が分からなかった)
            held_ms = elapsed * 1000
            text = f"           自動操作: turn={turn} 下キーを離した理由: {'、'.join(reasons)}(落下開始から{held_ms:.0f}ms)"
            if held_ms < 150:
                text += (f"\n           送信時の観測盤面:\n{_indent(base_text)}"
                         f"\n           離した時の観測盤面:\n{_indent(me.last_observed_text)}")
            self._log(text)
            trace["released"] = reasons
            self._landed_turn = turn  # 着地した: 盤面が先読みと違えば次の手を計算し直す(_replan_after_landing)
            return True

        # 移動しながら落とす: 通り道(3列目〜置き場所の列)の高さから、入力の最後と下キーを重ねる時間を決める
        early = 0.0
        cells = me.observed_cells
        if move is not None and cells is not None:
            heights = [sum(1 for r in range(ROWS) if cells[r][c] != EMPTY) for c in range(COLS)]
            early = early_drop_seconds(path, heights, [move.axis_cell[1], move.child_cell[1]], self.autoplay.keymap)

        def piece_cols() -> tuple[int, int] | None:
            # 押しっぱなし移動用: 画面で浮いている操作中の組ぷよの列(左端, 右端)。雑音を避けるため組ぷよの色だけ見る
            cols = [c for r, c, color in me.floating_now if color in pair_colors]
            return (min(cols), max(cols)) if cols else None

        # 設置キャンセル(2026-10-21 仮): 狙える置き方なら落ちる段数を渡す(keymap の place_cancel が "off" なら使われない)
        # 高さは確定した盤面で数える(画面の観測盤面には出現した組ぷよが3列目の上に見えている)。出現時点で送る先読みの手は
        # 先読みの盤面(board、画面と一致を確かめ済み)で数える: 確定した盤面の更新(turn_ready)は出現の後に届くことがあり、
        # 1手前の盤面で段差キャンセルの落ちる段数を見積もって回転と横移動が着地の後に入った(実機 2026-10-25 08:46 turn=49:
        # 2列目を1段と見たが実際は3段)
        cancel_fall = None
        extra: dict = {}
        settled_heights = [sum(1 for r in range(ROWS) if (me.settled if board is None else board)[r][c] != EMPTY)
                           for c in range(COLS)]
        if move is not None and self.autoplay.keymap.place_cancel != "off":
            if self.autoplay.keymap.place_cancel == "rotate_move":
                # 回転つき(2026-10-23、利用者の動画): 縦のまま落とし、着地の瞬間に回転 + 最後の横移動
                plan = rotate_cancel(path, settled_heights, move.axis_cell, move.child_cell)
                if plan is not None:
                    extra = {"cancel_fall_rows": plan[0], "cancel_pre": plan[1], "cancel_deferred": plan[2]}
            else:
                cancel_fall = cancel_rows(path, settled_heights, move.axis_cell, move.child_cell)
        if not extra and cancel_fall:
            extra = {"cancel_fall_rows": cancel_fall}  # 狙わない時は従来どおりの呼び方
        # 段差キャンセル(2026-10-25): 縦置きを1段低い隣の列へ置く手(keymap の step_cancel が false なら使わない)。
        # 平地の設置キャンセルとは対象が重ならない(段差がある時だけ)
        if move is not None and self.autoplay.keymap.step_cancel:
            step = step_cancel(path, settled_heights, move.axis_cell, move.child_cell,
                               max_height=self.autoplay.keymap.step_cancel_max_height)
            if step is not None:
                extra = {"step_plan": step}
        started =self.autoplay.play(turn, path, placed=placed, early_drop_sec=early, piece_cols=piece_cols, **extra)
        if started and move is not None:
            self._sent_move = (turn, move)
        if started and target is not None:
            self._finish_move_trace()  # 前の手の記録が判定前に残っていれば、先に判定する
            self._move_trace = trace
        if started and self.autoplay.sender.failures and isinstance(self.autoplay.sender, PadSender):
            self._log(f"           自動操作: 仮想パッドへの入力が {self.autoplay.sender.failures} 回失敗しています"
                      f"({self.autoplay.sender.error or 'ボタンの送信に失敗'}。ViGEmBus ドライバーと vgamepad を確かめてください)")
        elif started and self.autoplay.sender.failures:
            self._log(f"           自動操作: キー送信が {self.autoplay.sender.failures} 回失敗しています"
                      f"(GetLastError={self.autoplay.sender.last_error}。5=権限不足: ゲームが管理者権限で"
                      "動いているなら PBA も管理者として実行してください)")
        return started

    def _record_move_trace(self, t: float, frame_bgr: np.ndarray) -> None:
        """置きミスの診断: 送信中の手の組ぷよの位置(変わった時だけ)とフレームを毎 tick 記録し、着地したら判定する。"""
        trace = self._move_trace
        if trace is None:
            return
        me = self.trackers["self"]
        pos = tuple(sorted((r, c) for r, c, color in me.floating_now if color in trace["colors"]))
        if not trace["track"] or trace["track"][-1][1] != pos:
            trace["track"].append((t, pos))
        if len(trace["frames"]) < MOVE_TRACE_FRAMES:
            trace["frames"].append((t, self._self_crop(frame_bgr)))
        if trace["released"] is None and me.turn == trace["turn"]:
            return
        self._finish_move_trace()

    def _self_crop(self, frame_bgr: np.ndarray) -> np.ndarray:
        """自分の盤面とネクスト枠の範囲を切り出した複製(フレーム全体は両方の盤面を含み大きいので、記録用に小さくする)。"""
        if self.calibration is None:
            return frame_bgr
        region = self.calibration.capture_region()
        pc = self.calibration.self_player
        boxes = [pc.geometry().bounding_box(), *pc.next_rects]
        x0 = max(0, min(b[0] for b in boxes) - region.left)
        y0 = max(0, min(b[1] for b in boxes) - region.top)
        x1 = max(b[0] + b[2] for b in boxes) - region.left
        y1 = max(b[1] + b[3] for b in boxes) - region.top
        return frame_bgr[y0:y1, x0:x1].copy()

    def _finish_move_trace(self) -> None:
        """記録中の手を判定する。提案の2マスに組ぷよの色が無ければ(置きミス)、キーの時刻・組ぷよの動き・着地した盤面・画面を残す。"""
        trace, self._move_trace = self._move_trace, None
        if trace is None:
            return
        released = trace["released"] or []
        if "提案の2マスに着地" in released:
            return
        me = self.trackers["self"]
        cells = me.observed_cells
        if cells is not None and all(cells[r][c] == color for (r, c), color in trace["target"]):
            return
        if self._move_miss_saved >= MOVE_MISS_LIMIT:
            return
        self._move_miss_saved += 1
        t_send = trace["t_send"]
        move = trace["move"]
        target = "、".join(f"{c + 1}列{ROWS - r}段{color}" for (r, c), color in trace["target"])
        keys = [(at - self._t0_perf, name, down) for at, name, down in list(getattr(self.autoplay.sender, "timed", ()))
                if at - self._t0_perf >= t_send - 0.001]
        key_text = " ".join(f"+{(at - t_send) * 1000:.0f}{name}{'↓' if down else '↑'}" for at, name, down in keys)
        track_text = " ".join(
            f"+{(ft - t_send) * 1000:.0f}[{'・'.join(f'{c + 1}列{ROWS - r}段' for r, c in pos) or '見えない'}]"
            for ft, pos in trace["track"]
        )
        frames = trace["frames"]
        if len(frames) > MOVE_MISS_FRAMES:
            step = len(frames) / MOVE_MISS_FRAMES
            frames = [frames[int(i * step)] for i in range(MOVE_MISS_FRAMES)]
        names = []
        for i, (ft, frame) in enumerate(frames):
            name = f"movemiss_self_{t_send:07.2f}s_turn{trace['turn']}_{i:02d}_{(ft - t_send) * 1000:+05.0f}ms.png"
            if self._trace_saver.save(self.history_dir / name, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                names.append(name)
        where = f" x={move.x} {move.rotation}" if move is not None else ""
        self._log(
            f"           置きミスの診断(turn={trace['turn']}{where}): 提案の2マス {target} に組ぷよが無い"
            f"(CPU {'--' if self.last_cpu is None else f'{self.last_cpu:.0f}'}%)。"
            f"手順 {' '.join(trace['path'])} / 下キーを離した理由: {'、'.join(released) or '(離していない)'}"
            f"\n           キー(送信から、↓押す ↑離す): {key_text or '(記録なし)'}"
            f"\n           組ぷよの位置(送信から、変わった時): {track_text or '(記録なし)'}"
            f"\n           着地した時の観測盤面:\n{_indent(me.last_observed_text)}"
            f"\n           画面 {len(names)}枚: {names[0] if names else '(保存できず)'} ほか(送信からの時刻つき)"
        )

    def _form_text(self, move: BattleMove) -> str:
        """パネルに出す「狙い: 弥生時代」「移行: 弥生時代 → フキゲンGTR」の文字列。"""
        if move.form is None:
            return "狙い: (定型なし)" if self._last_form is None else f"定型を離脱: {FORM_TEXT_JA.get(self._last_form, self._last_form)}"
        name = FORM_TEXT_JA.get(move.form, move.form)
        if self._last_form is not None and self._last_form != move.form:
            return f"移行: {FORM_TEXT_JA.get(self._last_form, self._last_form)} → {name}"
        return f"狙い: {name}"

    def _battle(self, my_state: PlayerState, en_state: PlayerState | None) -> BattleMove:
        """対戦モードなら battle で思考する。一人用モード、または相手の状態が作れなかった
        (en_state=None)ときは build(相手を考慮しない)で思考する。"""
        if self.solo:
            return self.client.solo(my_state.field, my_state.pairs, history=my_state.history)
        # 相手の認識に失敗した時は空の相手を仮に渡す(記録用に enemy_unknown を添える)
        enemy_unknown = en_state is None
        if en_state is None:
            en_state = PlayerState(PuyoField(), list(my_state.pairs))
        return self.client.battle(
            my_state,
            en_state,
            enemy_unknown=enemy_unknown,
            target_point=self._target_point(),
        )

    def _target_point(self) -> int:
        """今のおじゃま1個に必要な点(マージンタイム)。自分の最初の組ぷよが出る前は 70点。"""
        if self._first_spawn_t is None:
            return margin_target_point(0)
        return margin_target_point(self._frame_t - self._first_spawn_t)

    def _dispatch(self, kind: str, ctx: dict, fn, urgent: bool = False) -> None:
        """思考を依頼する。思考スレッドがあれば渡して観測を続け、無ければ(テスト等)その場で実行して結果を反映する。"""
        if self._think_worker is None:
            try:
                result, exc = fn(), None
            except Exception as e:  # noqa: BLE001 結果の反映側で種類ごとに扱う
                result, exc = None, e
            self._handle_think_result(kind, ctx, result, exc)
        else:
            self._think_worker.submit(kind, ctx, fn, urgent=urgent)

    def _poll_think_results(self) -> None:
        """思考スレッドの結果を観測ループで反映する(毎 tick)。"""
        if self._think_worker is None:
            return
        for kind, ctx, result, exc in self._think_worker.poll():
            self._handle_think_result(kind, ctx, result, exc)

    def _handle_think_result(self, kind: str, ctx: dict, result, exc: BaseException | None) -> None:
        if kind == "main":
            self._apply_think(ctx, result, exc)
        elif kind == "pre":
            self._apply_precompute(ctx, result, exc)
        elif kind == "pre_atk":
            self._apply_precompute_alt(ctx, result, exc)
        elif kind in ("open2", "open3"):
            self._apply_opening(ctx, result, exc)

    def _think_key(self, me: PlayerTracker) -> tuple:
        """思考結果がまだ有効かを確かめる鍵(手番・操作中の組ぷよ・ネクスト・確定盤面)。"""
        return (me.turn, me.current_pair, me.next1, me.next2, cells_to_text(me.settled))

    def _think(self, t: float, frame_bgr: np.ndarray, note: str = "", new_turn: bool = False) -> bool:
        """現在の組ぷよの提案を計算して表示する。note は再計算の理由(相手の仕掛け等)。
        new_turn=True は手番の開始(turn_ready)で、前の手番の表示を維持する分岐を使わない。
        思考を依頼できた(または提案を出した)ら True、入力不足で思考できなかったら False。

        入力が揃っていない(ネクスト未認識、自分側の色変換失敗)時は _think_pending を立てて戻り、
        同じ手番で認識が揃ったら _loop がやり直す(分析文書 2026-09-17 P1-A)。
        ama の呼び出しは思考スレッドで行い、結果は _apply_think で反映する(観測を止めない。2026-09-27)。
        """
        me, en = self.trackers["self"], self.trackers["enemy"]
        self._think_pending = True
        if me.current_pair is None or me.next1 is None:
            self.status_changed.emit("ネクスト認識待ち")
            return False
        if new_turn and self.autoplay.enabled and self.autoplay.committed_turn != me.turn and self._precomputed is not None:
            # 出現時点で送れなかった先読みを、盤面を取り込んだ時点でもう一度試す。出現の瞬間は置いたぷよがまだ読めていない、
            # 自分の発火の直後に出現を早く判定した(本当の出現は連鎖の後)などで、取り込みの時点では先読みどおりのことがある
            # (実機 2026-10-11 11:29: 4手。思考を待つより約0.2秒早い)
            # 比べるのは確定した盤面(画面の観測盤面は相手の攻撃の演出などでまだ揺れていることがある。実機 2026-10-11: 取り込みは
            # 先読みどおりだったのに観測盤面と比べて送れなかった手番が17手)
            reason = self._send_precomputed(me, when="取り込み時点", board=me.settled)
            if reason != HARASS_WAIT:
                self._mismatch_retry = None  # 小さい消去で思考を待つ手番は、待ちの上限まで比べ直しを続ける
            elif self._mismatch_retry is None or self._mismatch_retry[0] != me.turn:
                self._mismatch_retry = (me.turn, self._now or 0.0)
            if reason:
                self._log(f"           自動操作: 取り込み時点でも送りません(turn={me.turn}): {reason}")
        # ネクネクが読めていない(演出でネクスト枠が隠れ、盤面から設置を検出した直後)なら
        # 2組で思考する(ama は2組以上あれば探索できる)。ネクネクが読めたら _think_pending でやり直す
        my_pairs_src = [me.current_pair, me.next1] + ([me.next2] if me.next2 is not None else [])

        try:
            my_state, en_state, my_pairs, en_pairs = self._build_states(me, en, me.settled, my_pairs_src)
        except ValueError as exc:
            self._log(f"           思考スキップ: {exc}")
            self.status_changed.emit(f"認識エラー: {exc}")
            if "5色目" in str(exc) and count_puyos(me.settled) <= NEW_GAME_MIN_PUYOS:
                # 新しい試合の検出を取りこぼした可能性が高い。色の割当だけ作り直して次の手番から復帰する
                self._log("           盤面が小さいので色の割当を作り直します(新しい試合の検出漏れ)")
                self.mapper = ColorMapper()
            elif "5色目" in str(exc):
                self._log_fifth_color(t, me, en, my_pairs_src, frame_bgr)
            return False

        # 先読みが当たっていれば、事前計算の提案(前の手番で「2」として見せていた手)を即表示する。
        # ただし事前計算は相手の状態が古い。相手の潰し・予告おじゃまなど「対応」が必要な局面
        # (最新の相手状態で計算した手が構築以外)では、一貫性より対応を優先して上書きする
        # (実機で相手の潰しにやられた指摘 2026-09-14)
        pre = self._precomputed
        self._precomputed = None
        if self._think_worker is not None and self._think_worker.cancel("pre"):
            self._pre_inflight = None  # 実行前の先読みを取り消した(実行中のものは結果が届く)
        if new_turn:
            # 相手の発火を想定した先読みは出現時点で使い終わっている(再計算では残す: 相手の消去開始の再計算の後の出現で使う)
            self._precomputed_alt = None
            if self._think_worker is not None:
                self._think_worker.cancel("pre_atk")
        # 再計算(note あり)では古い先読みを一度表示する経路を使わない(相手の状態が変わっている)
        # 全自動モードでは即表示で先読みの手順も送ってしまうので、相手の攻撃が絡む局面では使わない
        # (実機 2026-09-27 12:40: 相手の5連鎖中に先読みの手を送り、ama の受けの手と違う単発になった)
        instant_ok = not (self.autoplay.enabled and self._attack_involved(me))
        # 全自動では、小さい消去の先読みも即表示(=即送信)しない(_autoplay_precomputed と同じ理由: 3組目まで見た思考を待つ)
        if pre is not None and self.autoplay.enabled and pre[2].label == "harass":
            instant_ok = False
        instant = None
        if not note and instant_ok and pre is not None and pre[0] == me.settled and pre[1] == (me.current_pair, me.next1):
            instant = pre[2]
            self._emit_move(me, instant)
            self._log(f"           先読み一致: 事前計算の提案を即表示 x={instant.x} {instant.rotation}")
        ctx = {
            "t": t, "frame": frame_bgr, "note": note, "new_turn": new_turn, "instant": instant,
            "en_state": en_state, "my_pairs": my_pairs, "en_pairs": en_pairs,
            "key": self._think_key(me), "submitted": time.perf_counter(),
        }
        self._think_pending = False  # 結果の反映でやり直しの要否を決め直す
        self._dispatch("main", ctx, lambda: self._battle(my_state, en_state), urgent=True)
        sent = self._committed_move(me) if (note and not new_turn) else None
        if sent is not None:
            # キーを送り終えた手の再計算(相手の連鎖の開始・終了など): 今の手は変えられないので、次の手の先読みを再計算より
            # 先に計算する。再計算の後に先読みすると、次の組ぷよの出現に間に合わず思考を待って自由落下した
            # (実機 2026-10-11 11:07: 相手の連鎖の開始・終了の0.3〜0.7秒後に出た手番で、出現時点で送れなかった)
            self._dispatch_precompute(me, en, sent, urgent=True)
        return True

    def _apply_think(self, ctx: dict, move: BattleMove | None, exc: BaseException | None) -> None:
        """思考スレッドの結果を反映する(観測ループで呼ぶ)。依頼した後に状況が変わっていれば捨てる。"""
        me, en = self.trackers["self"], self.trackers["enemy"]
        t, note, new_turn, instant = ctx["t"], ctx["note"], ctx["new_turn"], ctx["instant"]
        wait_ms = (time.perf_counter() - ctx["submitted"]) * 1000
        if ctx["key"] != self._think_key(me):
            self._log(f"           思考結果を破棄: 依頼後に手番・組ぷよ・盤面が変わった(依頼 {t:.2f}s、待ち {wait_ms:.0f}ms)")
            if ctx["key"][0] == me.turn and me.current_pair is not None:
                self._think_pending = True  # 同じ手番で盤面が変わった: 今の状態で思考し直す
            return
        if self._think_worker is not None and wait_ms >= 200:
            self._log(f"           思考結果の反映まで {wait_ms:.0f}ms(この間も観測は続いている)")
        if isinstance(exc, NoValidPlacementError):
            # 置ける手が無い(盤面の誤認・ゲームオーバー等)。ループを止めず提案を消して観測を続け、
            # 次の手番・再計算の理由が来た時に再探索する(同じ状態で毎フレーム再試行はしない)
            self._log(f"           思考スキップ: {exc} → 提案を消して観測を続けます")
            self.status_changed.emit("置ける手がありません(盤面の認識を確認)")
            self._last_data = None
            self._precomputed = None
            self.draw_data_ready.emit(None)
            return
        if exc is not None:
            raise exc
        assert move is not None
        self._think_submitted = ctx["submitted"]  # 相手の連鎖の後の思考か(_emit_move で相手の撃った連鎖の表示を外す判断)
        if instant is not None:
            fresh = move
            if instant.label == "build" and is_build_like(fresh):
                # 思考も構築(伸ばす・受けて組み続けるを含む): 乱数・ネクネクの違いで置き場所が変わるだけなので先読みの手を
                # 表示したまま、判断の内容(ラベル・理由・着弾量)だけ思考のものにする
                move = instant
                move.self_analysis = fresh.self_analysis
                move.enemy_analysis = fresh.enemy_analysis
                if fresh.label != "build":
                    move.label, move.reason, move.plan_steps, move.request = fresh.label, fresh.reason, fresh.plan_steps, fresh.request
                    move.incoming, move.incoming_source, move.incoming_chain = fresh.incoming, fresh.incoming_source, fresh.incoming_chain
                    self._log(f"           先読みの手を表示したまま(思考は x={fresh.x} {fresh.rotation} [{fresh.label}] 理由={fresh.reason}。中身は構築)")
                self._emit_move(me, move)
            elif self.autoplay.committed_turn == me.turn:
                # 全自動モードで先読みの手順を既に送ってしまっている: キーは取り消せないので、表示も
                # 送った手に合わせる(食い違うと人間が追えず、記録も実際と違うものになる)
                move = instant
                move.self_analysis = fresh.self_analysis
                move.enemy_analysis = fresh.enemy_analysis
                self._emit_move(me, move, note="自動操作は先読みの手を実行済み")
                self._log(f"           自動操作: 先読みの手を実行済みのため提案を変えません"
                          f"(思考は x={fresh.x} {fresh.rotation} [{fresh.label}])")
            else:
                reason = FIRE_REASON_TEXT_JA.get(move.fire_reason or "", "")
                self._emit_move(me, move, note="対応が必要: 提案を変更" + (f" ({reason})" if reason else ""))
                self._log(f"           対応のため提案を変更: x={move.x} {move.rotation} [{move.label}]")
        elif (sent := self._committed_move(me)) is not None and (sent.axis_cell, sent.child_cell) != (move.axis_cell, move.child_cell):
            # この手番のキーを送り終えた(先読み・相手の発火の想定を出現時点で送った、または再計算の前に送った)後で、思考が
            # 別の手を選んだ: キーは取り消せないので、表示と次の手の先読みは送った手で行う。思考の手の盤面で先読みすると
            # 次の手番で盤面が合わず、安定待ちと思考の間に組ぷよが自由落下した(実機 2026-10-11 07:07: 相手の6連鎖中、
            # 先読みの x=5 LEFT を送った後の思考が x=5 DOWN → 次の手番の送信が約0.6秒遅れ、その次も盤面がずれて底まで落ちた)
            fresh = move
            move = sent
            move.self_analysis = fresh.self_analysis
            move.enemy_analysis = fresh.enemy_analysis
            move.incoming, move.incoming_source, move.incoming_chain = fresh.incoming, fresh.incoming_source, fresh.incoming_chain
            self._emit_move(me, move, note="自動操作は送った手を実行中")
            self._log(f"           自動操作: 送った手を実行中のため提案を変えません(思考は x={fresh.x} {fresh.rotation} [{fresh.label}]、"
                      f"送った手 x={move.x} {move.rotation}。次の手は送った手の盤面で先読み)")
        else:
            last = self._last_data
            same_placement = last is not None and (move.axis_cell, move.child_cell) == (last.axis_cell, last.child_cell)
            incoming_changed = abs((move.incoming or 0) - self._incoming_at_think) >= RETHINK_PENDING_INCREASE
            if (
                note and not new_turn and last is not None
                and move.label in BUILD_LABELS and last.label in BUILD_LABELS and not incoming_changed
            ):
                # 構築の再計算で構築のまま、かつ着弾量も変わっていない(入力が同じ): ビームサーチの乱数で手が
                # 変わるだけなので表示中の提案を維持し、解析値だけ更新する。着弾量が変わった再計算は採用する
                self._log(f"           再計算の結果は構築のまま(x={move.x} {move.rotation} [{move.label}])で着弾量も同程度: 表示中の提案を維持")
                last.self_analysis = move.self_analysis
                last.enemy_analysis = move.enemy_analysis
                last.chaining = me.chain_active
                self.draw_data_ready.emit(last)
                self._pending_at_think = me.pending_garbage
                # 相手の消去開始などで先読みを捨てた後は、表示中の手で次の手を先読みし直す(しないと次の手番が
                # 思考待ちで硬直する。実機 2026-10-10 15:28: 相手の連鎖中に出た手番は送信まで中央0.25秒・最大1.25秒)
                if self._precomputed is None and self._last_move is not None:
                    self._dispatch_precompute(me, en, self._last_move)
                return
            if note and not new_turn and same_placement:
                # 置き場所は同じでも判断の意味(ラベル・理由・連鎖表示)は変わりうるので表示内容は更新する
                self._log(f"           再評価: 置き場所は同じ(x={move.x} {move.rotation} [{move.label}])、判断内容を更新")
            if note:
                reason = FIRE_REASON_TEXT_JA.get(move.fire_reason or "", "")
                note = note + (f" ({reason})" if reason else "")
            elif ctx["en_state"] is None and not self.solo:
                note = "相手の認識エラー: 相手を見ずに提案"
            self._emit_move(me, move, note=note)
        self._last_form = move.form
        self._pending_at_think = me.pending_garbage
        self._incoming_at_think = move.incoming or 0
        if move.incoming is not None:
            # 相手の連鎖を再現できた時だけ返る。予告が消えるまで保持し、自分の attack として ama に渡す
            if self._incoming_carry and move.incoming != self._incoming_carry:
                self._log(f"           着弾量を更新 {self._incoming_carry}個 → {move.incoming}個 ({move.incoming_source})")
            self._incoming_carry = move.incoming
            if self._attack is not None:
                self._attack.add_estimate(t, f"再現({move.incoming_source})", move.incoming)
        self._think_pending = me.next2 is None  # ネクネク未読なら、読めた時に3組でやり直す
        self.status_changed.emit(f"手番 {me.turn}  思考 {move.think_ms}ms")
        self._log_think(t, me, en, ctx["my_pairs"], ctx["en_pairs"], move, ctx["frame"])

        self._last_move = move
        self._dispatch_precompute(me, en, move)

    def _replan_after_landing(self) -> None:
        """全自動で置いた組ぷよが着地した後、観測盤面が先読みの期待盤面と違えば(置き場所のずれ・おじゃまの落下など)、その盤面で
        次の手を計算し直す。次の組ぷよの出現(着地から約0.38秒)までに間に合わせ、出現時点で送れるようにする
        (実機 2026-10-11: 盤面が本当に違った手番が41手あり、出現の後に考え直す間に組ぷよが自由落下した)。
        比べる時も計算し直す時も、白い目の無い色(攻撃が当たる演出・着地予測の点)を空にした盤面を使う(実機 2026-10-12 07:33:
        着地の瞬間に攻撃が当たり、演出の赤を4列目12段目のぷよと読んだ盤面で計算し直したため、次の出現で「盤面が違う」となって
        送信が0.36秒遅れ、3列目10段の上に組ぷよが自然落下して窒息した)"""
        me = self.trackers["self"]
        obs = me.observed_cells
        text = me.last_observed_text
        stable = text == self._last_obs_text
        self._last_obs_text = text
        if (
            self._landed_turn != me.turn or obs is None or not stable or not self.autoplay.enabled
            or me.chain_active or me.floating_now or me.next1 is None or me.next2 is None
            or self._committed_move(me) is None or has_poppable_group(obs)
        ):
            return
        eyed = me.observed_cells_eyed if me.observed_cells_eyed is not None else obs
        pre = self._precomputed
        if pre is not None and pre[1] == (me.next1, me.next2) and (boards_match(obs, pre[0]) or boards_match(eyed, pre[0])):
            return  # 先読みどおり(違いが目の無い色だけなら、それは演出)
        inflight = self._pre_inflight
        if (
            pre is None and inflight is not None and inflight["pairs"] == (me.next1, me.next2)
            and (boards_match(obs, inflight["expected"]) or boards_match(eyed, inflight["expected"]))
        ):
            # 計算中の先読みが想定している盤面どおり: 届くのを待つ。今の手の思考の後に先読みを計算するので、着地の時点では
            # まだ届いていないことが多く、それを「盤面が違う」として同じ盤面で依頼し直していた(実機 2026-10-20 22:28:
            # 着地した盤面が先読みと違うとした手番 183/323。思考のスレッドは1本なので、無駄な計算が次の手の思考を遅らせた)
            return
        if self._replanned == (me.turn, text):
            return
        self._replanned = (me.turn, text)
        note = "(目の無い色を空にした盤面で)" if eyed is not obs and eyed != obs else ""
        self._log(f"           自動操作: 着地した盤面が先読みと違うので、次の手を観測盤面{note}で計算し直します(turn={me.turn})"
                  + self._landing_diff_text(me, eyed, pre))
        self._precomputed = None
        self._dispatch_precompute(me, self.trackers["enemy"], self._committed_move(me), urgent=True, expected_cells=eyed)

    def _landing_diff_text(self, me: PlayerTracker, eyed: list[list[str]], pre) -> str:  # noqa: ANN001
        """着地した盤面が先読みと違った理由の手がかり(2026-10-21): 送った手の2マスに組ぷよがあるか(置き場所のずれ)と、
        先読みの盤面と違うマス(おじゃまの落下・読み違い)。全ログで「着地した盤面が先読みと違う」は2,495回あったが、
        原因(置き場所・おじゃま・認識)を切り分ける記録が無かった。"""
        move = self._committed_move(me)
        placed = "?"
        if move is not None and me.current_pair is not None:
            child_color, axis_color = me.current_pair
            cells = [(cell, color) for cell, color in ((move.axis_cell, axis_color), (move.child_cell, child_color)) if cell[0] > 0]
            placed = "提案どおり" if cells and all(eyed[r][c] == color for (r, c), color in cells) else "ずれ"
        expected = pre[0] if pre is not None else (self._pre_inflight or {}).get("expected")
        if expected is None:
            return f" [置いた2マス: {placed} / 先読みなし]"
        diff = board_diff(eyed, expected)
        garbage = sum(1 for d in diff if d.endswith("読みgarbage"))
        shown = "、".join(diff[:6]) + (f" ほか{len(diff) - 6}マス" if len(diff) > 6 else "")
        return f" [置いた2マス: {placed} / 違うマス{len(diff)}(うちおじゃま{garbage}): {shown or '(なし。組ぷよが先読みと違う)'}]"

    def _refresh_precompute_for_sent(self) -> None:
        """相手の連鎖の開始・終了で攻撃の状況が変わった: この手番のキーを送り終えていれば、送った手の盤面で次の手の先読みを
        すぐやり直す。再計算は自分の連鎖中などで保留されることがあり、それを待つと次の組ぷよの出現に間に合わなかった
        (実機 2026-10-11 11:29: 自分の連鎖中に相手が連鎖を始め、先読みが無いまま出現した手番が4手)"""
        me = self.trackers["self"]
        sent = self._committed_move(me)
        if sent is not None and me.next1 is not None and me.next2 is not None:
            self._dispatch_precompute(me, self.trackers["enemy"], sent, urgent=True)

    def _committed_move(self, me: PlayerTracker) -> BattleMove | None:
        """全自動でこの手番のキーを送った手(送っていなければ None)。"""
        sent = self._sent_move
        if not self.autoplay.enabled or self.autoplay.committed_turn != me.turn or sent is None or sent[0] != me.turn:
            return None
        return sent[1]

    def _attack_snapshot(self, me: PlayerTracker) -> tuple:
        """先読みを計算した時の攻撃の状況(相手の連鎖中か、思考が見積もった着弾量)。出現時に同じなら、相手の攻撃中でも
        先読みの手は今の状況で考えた手と同じ前提で計算されている。着弾量は予告の数ではなく見積もり(保持している着弾見込み、
        無ければ直前の思考の着弾量)で比べる: 予告の数は相手の連鎖が進むと増えるので、同じ攻撃でも食い違った
        (実機 2026-10-10 15:50: 相手の連鎖中に出た手番26件のうち16件で送れなかった)。見積もりが無い時だけ予告の有無を使う"""
        en = self.trackers["enemy"]
        if en.chain_active:
            # 相手の連鎖中は「どの連鎖か」で比べる。着弾量は思考も先読みもブリッジが同じ連鎖を点滅から再現して見積もるので、
            # PBA 側の見積もりが後から更新されても前提は同じ(キーを送った手の再計算の前に先読みを計算するため。2026-10-11)
            return (True, self._attack_count)
        est = self._incoming_carry or self._incoming_at_think
        return (False, est if est else bool(me.pending_garbage))

    def _dispatch_precompute(self, me: PlayerTracker, en: PlayerTracker, move: BattleMove, urgent: bool = False,
                             expected_cells: list[list[str]] | None = None) -> None:
        """次の手を先読みする: 提案通りに置いた後の盤面で、ネクストを現在の組ぷよとして計算する
        (現在の手を表示・送信した後に、思考スレッドで計算する)。同じ盤面・組ぷよ・手・攻撃の状況の先読みを
        計算中か計算済みなら、もう一度は依頼しない(キーを送った手の再計算で先に依頼した分と重ならないように)。"""
        if not (self.enable_precompute and (move.field_after is not None or expected_cells is not None) and me.next2 is not None):
            return
        place_key = (move.axis_cell, move.child_cell) if expected_cells is None else ("observed", cells_to_text(expected_cells))
        basis = (self._think_key(me), place_key, self._attack_snapshot(me))
        if basis == self._pre_basis and (self._pre_inflight is not None or self._precomputed is not None):
            return
        expected = self.mapper.rows_to_cells(move.field_after) if expected_cells is None else [row[:] for row in expected_cells]
        try:
            pre_my, pre_en, _, _ = self._build_states(me, en, expected, [me.next1, me.next2], ahead=True)
        except ValueError as exc2:
            self._log(f"           先読みスキップ: {exc2}")
            return
        pre_ctx = {"key": self._think_key(me), "move": move, "expected": expected, "pairs": (me.next1, me.next2),
                   "attack": self._attack_snapshot(me)}
        self._pre_basis = basis
        self._pre_inflight = pre_ctx
        self._dispatch("pre", pre_ctx, lambda: self._battle(pre_my, pre_en), urgent=urgent)
        # 相手がこの場面で撃ってきた場合の想定: 相手が今すぐ撃てる最大の攻撃を受けた状態で、次の手も計算しておく
        imm = move.enemy_analysis.immediate if move.enemy_analysis is not None else None
        if (
            self.autoplay.enabled and not self.solo and pre_en is not None and imm is not None
            and imm.garbage >= ANTICIPATE_MIN_GARBAGE and not self._attack_involved(me)
        ):
            atk_my = replace(pre_my, attack=pre_my.attack + imm.garbage)
            atk_en = replace(pre_en, attack_frame=max(1, imm.count // 2))
            alt_ctx = {"key": self._think_key(me), "expected": expected, "pairs": (me.next1, me.next2), "assumed": imm.garbage}
            self._dispatch("pre_atk", alt_ctx, lambda: self._battle(atk_my, atk_en))

    def _apply_precompute_alt(self, ctx: dict, pre_move: BattleMove | None, exc: BaseException | None) -> None:
        """相手の発火を想定した先読みの結果を残す(依頼した後に手番・盤面が変わっていれば捨てる)。"""
        if exc is not None:
            if isinstance(exc, (ValueError, RuntimeError)):
                return
            raise exc
        me = self.trackers["self"]
        if ctx["key"] != self._think_key(me) or pre_move is None or not pre_move.path:
            return
        self._precomputed_alt = (ctx["expected"], ctx["pairs"], pre_move, ctx["assumed"])

    def _incoming_estimate(self) -> int:
        """今の着弾量の見積もり(保持している着弾見込み → 直前の思考の着弾量 → 相手の連鎖を点滅から再現した量)。"""
        est = self._incoming_carry or self._incoming_at_think
        if est:
            return est
        en = self.trackers["enemy"]
        if en.chain_active and en.chain_expected is not None and en.chain_expected.steps > 0:
            bonus = 30 if en.chain_bonus else 0
            return en.chain_expected.score // 70 + bonus
        return 0

    def _apply_precompute(self, ctx: dict, pre_move: BattleMove | None, exc: BaseException | None) -> None:
        """先読み(2手先)の結果を反映する。依頼した後に手番・盤面が変わっていれば捨てる。"""
        me = self.trackers["self"]
        if ctx is self._pre_inflight:
            self._pre_inflight = None
        if exc is not None:
            if isinstance(exc, (ValueError, RuntimeError)):
                self._log(f"           先読みスキップ: {exc}")
                return
            raise exc
        if ctx["key"] != self._think_key(me):
            # 出現の後に届いた先読み: ちょうど次の手番(組ぷよが先読みどおり)で、まだキーを送っていなければ使う。
            # 捨てて考え直すと思考を待つ間に自由落下した(実機 2026-10-11 11:07: 出現の直前に依頼した先読みが間に合わなかった)
            if (
                pre_move is not None and pre_move.path and me.turn == ctx["key"][0] + 1
                and ctx["pairs"] == (me.current_pair, me.next1)
                and self.autoplay.enabled and self.autoplay.committed_turn != me.turn
            ):
                self._precomputed = (ctx["expected"], ctx["pairs"], pre_move)
                self._precomputed_attack = ctx.get("attack")
                if me.settling:
                    me.set_expected(ctx["expected"])
                self._log(f"           先読みが出現の後に届いた(turn={me.turn}): 送れるか確かめます")
                self._autoplay_precomputed(me)
            elif pre_move is not None and pre_move.path and me.turn == ctx["key"][0] and ctx["pairs"] == (me.next1, me.next2):
                # 同じ手番のうちに盤面の取り込みが進んだ(自分の連鎖中など)だけ: 次の手番の出現で盤面と組ぷよを確かめてから
                # 使うので残す(捨てると、自分の連鎖中に相手が連鎖を始めた後の出現で先読みが無かった)
                self._precomputed = (ctx["expected"], ctx["pairs"], pre_move)
                self._precomputed_attack = ctx.get("attack")
            return  # 次の手番が始まった等: 古い先読みは使わない
        if self._last_data is None:
            return
        assert pre_move is not None
        expected = ctx["expected"]
        self._precomputed = (expected, ctx["pairs"], pre_move)
        self._precomputed_attack = ctx.get("attack")
        me.set_expected(expected)
        # 2手先として小さく表示する(提案通りに置いた後の盤面での着地マス)
        self._emit_move(me, ctx["move"], next_move=pre_move)

    # ---- デバッグ出力 ----

    def _log(self, text: str) -> None:
        if self.debug_log_path is None:
            return
        with open(self.debug_log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def _log_think(
        self,
        t: float,
        me: PlayerTracker,
        en: PlayerTracker,
        my_pairs: list[str],
        en_pairs: list[str],
        move: BattleMove,
        frame_bgr: np.ndarray,
    ) -> None:
        if self.debug_log_path is None:
            return
        self._log(
            f"           提案: x={move.x} {move.rotation} [{move.label}] eval={move.eval} form={move.form} fire={move.fire_reason} "
            + (f"理由={move.reason} " if move.reason else "")
            + (f"跳考{move.choukou}回 " if move.choukou else "")
            + (f"着弾{move.incoming}個({move.incoming_source},{move.incoming_chain}連鎖) " if move.incoming is not None else "")
            + (f"保持{self._incoming_carry}個 " if self._incoming_carry > 0 else "")
            + (f"自分:最大{move.self_analysis.max.count}連鎖{move.self_analysis.max.score}点 " if move.self_analysis else "")
            + (f"相手:最大{move.enemy_analysis.max.count}連鎖{move.enemy_analysis.max.score}点 " if move.enemy_analysis else "")
            + f"({move.think_ms}ms)\n"
            + (f"           {diag_text(move.diag)}\n" if move.diag else "")
            + (f"           要求: {move.request}\n" if move.request else "")
            + f"           自分 {my_pairs} 予告{me.pending_garbage}{' 全消し持ち' if me.all_clear else ''}"
            f" 盤面の由来={me.last_take_in_reason}\n{_indent(cells_to_text(me.settled))}\n"
            f"           相手 {en_pairs} 予告{en.pending_garbage} 連鎖中={en.chain_active}"
            f"{' 全消し持ち' if en.all_clear else ''}{' 連鎖にボーナス付き' if en.chain_bonus else ''}\n{_indent(cells_to_text(en.settled))}"
        )
        if self._attack is not None and move.enemy_analysis is not None:
            self._attack.add_estimate(t, "思考時の相手本線", move.enemy_analysis.max.garbage)
        path = self.history_dir / f"{self._history_index % FRAME_HISTORY_SIZE:03d}_turn{me.turn}.png"
        self._saver.save(path, frame_bgr)
        self._history_index += 1


def observed_match(me: PlayerTracker, expected: list[list[str]]) -> bool:
    """出現時点の観測盤面が期待盤面どおりか。観測盤面そのものか、白い目の無い色のマス(着地予測の点・攻撃が当たる演出)を
    空にした盤面のどちらかが一致すればよい(本物のぷよには目がある。目が見えない瞬間の本物のぷよは、そのままの観測盤面で一致する)。
    どちらも、操作中の組ぷよの着地予測の点(3列目の積みのすぐ上の2マスにある組ぷよの色)は空として比べる。"""
    pair = me.current_pair
    obs = me.observed_cells
    if obs is not None and boards_match(strip_spawn_ghost(obs, expected, pair), expected):
        return True
    eyed = me.observed_cells_eyed
    return eyed is not None and eyed is not obs and boards_match(strip_spawn_ghost(eyed, expected, pair), expected)


def strip_spawn_ghost(observed: list[list[str]], expected: list[list[str]], pair: Pair | None) -> list[list[str]]:
    """出現の直後に3列目の積み(期待盤面)のすぐ上の2マスに出る、操作中の組ぷよの着地予測の点を空にした盤面を返す。
    組ぷよは3列目に縦向きで出るので、点の位置は期待盤面の3列目の高さで決まる。色が組ぷよのどちらかの色の時だけ空にする
    (実機 2026-10-12 07:00: 点の光沢を目と読み、目の有無では除けなかった。turn25 紫/赤・turn26 黄/赤・turn21 黄/青)。"""
    if pair is None:
        return observed
    col = 2
    top = next((r for r in range(len(expected)) if expected[r][col] != EMPTY), len(expected))
    rows = [r for r in (top - 1, top - 2) if r >= 0]
    colors = set(pair)
    hits = [r for r in rows if observed[r][col] != EMPTY and observed[r][col] in colors]
    if not hits:
        return observed
    out = [row[:] for row in observed]
    for r in hits:
        out[r][col] = EMPTY
    return out


def boards_match(observed: list[list[str]], expected: list[list[str]]) -> bool:
    """出現時点で、観測盤面が先読みの期待盤面どおりか(13段目と出現位置は除く)。期待盤面がおじゃまのマスは、観測が何色でも
    埋まっていればよい: 相手の攻撃の演出中は、おじゃまが色ぷよに読まれることがあり、先読みを送れずに自由落下した
    (実機 2026-10-11: 相手の連鎖中に「盤面が違う」で送れなかった手番が約30手。おじゃまの色は変わらない)"""
    obs, exp = clear_spawn_cells(observed), clear_spawn_cells(expected)
    for r in range(1, len(exp)):
        for c in range(len(exp[r])):
            e, o = exp[r][c], obs[r][c]
            if e == o:
                continue
            if e == GARBAGE and o != EMPTY:
                continue
            return False
    return True


def field_edge_offset(frame_bgr: np.ndarray, geometry, origin: tuple[int, int] = (0, 0)) -> int:  # noqa: ANN001
    """盤面の右の枠(盤面の内側と枠の境目)の横位置を、校正した位置からのずれ(px、右が正)で返す。
    校正した境目から外側(枠の側)へ0.7マス、内側へ0.1マスの範囲で、横方向の明るさの差を縦に足して最も強い位置を境目と
    みなす(内側まで探すと壁際のぷよの輪郭を取り違える。縦は見える盤面の2〜11段目。上の方は攻撃の演出がかかりやすい)。
    左の枠は、自分の盤面がキャプチャの範囲の左端に接していて測れないので見ない。値そのものより、普段の値との差を見る
    (実機の画面で、画面を横に7pxずらすと値も7px変わることを確かめた)。"""
    h, w = frame_bgr.shape[:2]
    cell = geometry.cell_size()
    outer, inner = max(4, int(cell * 0.7)), max(2, int(cell * 0.1))
    x_top, y0 = geometry.grid_to_screen(float(COLS), 1.0)
    x_bot, y1 = geometry.grid_to_screen(float(COLS), 10.0)
    x = int(round((x_top + x_bot) / 2 - origin[0]))
    ya, yb = max(0, int(min(y0, y1) - origin[1])), min(h, int(max(y0, y1) - origin[1]))
    xa, xb = max(0, x - inner), min(w, x + outer + 1)
    if xb - xa < 3 or yb - ya < 3:
        return 0
    gray = cv2.cvtColor(np.ascontiguousarray(frame_bgr[ya:yb, xa:xb]), cv2.COLOR_BGR2GRAY).astype(np.int32)
    gx = np.abs(np.diff(gray, axis=1)).sum(axis=0)
    return int(xa + int(np.argmax(gx)) + 1 - x)


def board_diff(observed: list[list[str]] | None, expected: list[list[str]]) -> list[str]:
    """boards_match で違うとされたマスを「12段目3列目 期待red 読みgreen」の形で返す(13段目と出現位置は除く)。"""
    if observed is None:
        return ["観測盤面なし"]
    obs, exp = clear_spawn_cells(observed), clear_spawn_cells(expected)
    out = []
    for r in range(1, len(exp)):
        for c in range(len(exp[r])):
            e, o = exp[r][c], obs[r][c]
            if e == o or (e == GARBAGE and o != EMPTY):
                continue
            out.append(f"{len(exp) - r}段目{c + 1}列目 期待{e} 読み{o}")
    return out


class TickStats:
    """観測ループの診断: 観測の間隔・画面取得の待ち・1回の処理時間を TICK_STATS_SECONDS ごとにまとめてログに出す
    (2026-10-21。速さの改善(画面取得の並行化など)が実機で効いたかを確かめる)。"""

    def __init__(self) -> None:
        self.since: float | None = None
        self.last_start: float | None = None
        self.intervals: list[float] = []
        self.grabs: list[float] = []
        self.works: list[float] = []
        self.cpu: list[float] = []  # その間の PC 全体の CPU 使用率(%、1秒ごと)

    def add(self, t: float, tick_start: float, grab: float, elapsed: float) -> str:
        if self.last_start is not None:
            self.intervals.append(tick_start - self.last_start)
        self.last_start = tick_start
        self.grabs.append(grab)
        self.works.append(elapsed)
        if self.since is None:
            self.since = t
        if t - self.since < TICK_STATS_SECONDS or not self.intervals:
            return ""

        def ms(v: list[float]) -> str:
            v = sorted(v)
            return f"中央 {v[len(v) // 2] * 1000:.0f}ms・90% {v[len(v) * 9 // 10] * 1000:.0f}ms"

        text = (f"観測の統計({len(self.works)}回): 間隔 {ms(self.intervals)} / 画面取得の待ち {ms(self.grabs)}"
                f" / 1回の処理(取得込み) {ms(self.works)}")
        if self.cpu:
            cpu = sorted(self.cpu)
            text += f" / CPU 中央 {cpu[len(cpu) // 2]:.0f}%・最大 {cpu[-1]:.0f}%"
        self.__init__()
        self.last_start = tick_start
        return text


def _indent(text: str) -> str:
    return "\n".join("               " + line for line in text.splitlines())


# メイン画面の「CPU の使用」(YamaNEKO の探索スレッド数)の選択肢 (表示, キー)。「自動」は今までどおり
CPU_USE_CHOICES = [("自動", "auto"), ("全コア", "all"), ("半分", "half"), ("4分の1", "quarter"), ("2スレッド", "two")]
# メイン画面の思考エンジンの選択肢(2026-10-25、利用者)。"none" はその側を支援しない
ENGINE_CHOICES = {"yamaneko": "YamaNEKO", "ama": "ama", "none": "支援なし"}


class SideSettings(QtWidgets.QGroupBox):
    """メイン画面の 1P / 2P の設定の枠。設置キャンセル・段差キャンセル・配ぷよ読み・相手のツモを覗くは YamaNEKO の時だけ設定できる。"""

    changed = QtCore.pyqtSignal()

    def __init__(self, side: str, engine: str, open_keys, title: str | None = None,  # noqa: ANN001
                 languages: dict[str, Language] | None = None) -> None:
        super().__init__(title or ("1P(左の盤面)" if side == "1p" else "2P(右の盤面)"))
        self.side = side
        form = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("思考エンジン:"))
        self.engine_combo = QtWidgets.QComboBox()
        self.set_custom_engines([], engine)
        self.engine_combo.currentIndexChanged.connect(lambda _i: self._on_engine_changed())
        row.addWidget(self.engine_combo, 1)
        form.addLayout(row)

        # 情報パネルの表記(2026-10-25、利用者)。ログは日本語のまま
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("パネルの表記:"))
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItem("日本語", "ja")
        self.language_combo.addItem("English", "en")
        for key, lang in (languages or {}).items():
            self.language_combo.addItem(lang.label, key)  # 言語プラグインの表記(mokou など)
        row.addWidget(self.language_combo, 1)
        form.addLayout(row)

        # 定跡(2026-10-28、利用者「1P と 2P で使い分け」)。使うと AI の思考より定跡の手を優先し、定跡に無い手順になったら AI に戻る
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("定跡:"))
        self.book_combo = QtWidgets.QComboBox()
        self.book_combo.setToolTip("選んだ定跡の手を AI の思考より優先します。定跡に無い手順になったら AI の思考に戻ります")
        self.set_books([], "")
        self.book_combo.currentIndexChanged.connect(lambda _i: self.changed.emit())
        row.addWidget(self.book_combo, 1)
        form.addLayout(row)

        # 全自動に関わる設定(2026-10-28、プラグイン化): 全自動を許可するプラグイン(全自動モードなど)が有効な時だけ出す
        self.autoplay_box = QtWidgets.QWidget()
        outer_form, form = form, QtWidgets.QVBoxLayout(self.autoplay_box)
        form.setContentsMargins(0, 0, 0, 0)

        row = QtWidgets.QHBoxLayout()
        self.autoplay = ToggleSwitch("全自動モード")
        self.autoplay.setToolTip(
            "提案の操作手順を、ゲーム画面をアクティブにしている間キー入力(または仮想パッド)として送ります。\n"
            "Steam版・CPU戦での画像認識テスト用です。対人戦では使わないでください。"
        )
        row.addWidget(self.autoplay)
        row.addStretch(1)
        # 1P・2P のキーを画面で割り当てる(2026-10-25、利用者「ファイルのテキストを書き換えるのは手間」)
        keys_btn = QtWidgets.QPushButton("キー割り当て")
        keys_btn.clicked.connect(open_keys)
        row.addWidget(keys_btn)
        form.addLayout(row)

        # 仮想パッド(2026-10-25): ゲームの参加の画面で、この側の仮想パッドのボタンを押して割り当てる。パッドは最初に押した時
        # (送り先が仮想パッドなら PBA の起動時)につなぎ、PBA を閉じるまで外さない
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("仮想パッド:"))
        self.pad_label = QtWidgets.QLabel("つないでいない")
        row.addWidget(self.pad_label, 1)
        for text, button in (("A を押す", "PAD_A"), ("START を押す", "PAD_START")):
            btn = QtWidgets.QPushButton(text)
            btn.setToolTip("この側の仮想パッドのボタンを押します。ゲームの参加の画面で押して、この側に割り当ててください")
            btn.clicked.connect(lambda _c=False, b=button: self.tap_pad(b))
            row.addWidget(btn)
        form.addLayout(row)

        try:
            keymap = KeyMap.load(app_root() / "config" / KEYMAP_FILES[side])
        except Exception:
            keymap = KeyMap()
        # 設置キャンセルのあり・なし(2026-10-23)。最初の状態は keymap の place_cancel("off" 以外ならあり)
        self.place_cancel = ToggleSwitch("設置キャンセル")
        self.place_cancel.setToolTip(
            "全自動の時、最後の1列の横移動を着地の直前に入れて設置を速くします。\n"
            "config/keymap.json の place_cancel を支援モードの間だけ上書きします(ありで off なら move)。置き場所がずれるなら外してください。"
        )
        self.place_cancel.setChecked(keymap.place_cancel != "off")
        form.addWidget(self.place_cancel)
        # 段差キャンセルのあり・なし(2026-10-25)。最初の状態は keymap の step_cancel
        self.step_cancel = ToggleSwitch("段差キャンセル(試験中)")
        self.step_cancel.setToolTip(
            "全自動の時、1段低い隣の列への縦置きを、高い列で回して戻してから滑らせます。\n"
            "config/keymap.json の step_cancel を支援モードの間だけ上書きします。置き場所がずれるなら外してください。"
        )
        self.step_cancel.setChecked(keymap.step_cancel)
        form.addWidget(self.step_cancel)
        # 先行入力(2026-10-25 試験)。最初の状態は keymap の prehold。思考エンジンを問わず全自動で使える
        self.prehold = ToggleSwitch("先行入力(試験中)")
        self.prehold.setToolTip(
            "全自動の時、次の手が壁(1列目・6列目)まで動かすだけなら、その向きを組ぷよが出る前から押しっぱなしにします。\n"
            "初手は開始の合図から下を押しっぱなしにします。連鎖・おじゃまが無い時だけ。ログの「先行入力」の行で確かめてください。"
        )
        self.prehold.setChecked(keymap.prehold)
        form.addWidget(self.prehold)
        outer_form.addWidget(self.autoplay_box)
        form = outer_form
        # 思考エンジンの負荷(2026-10-26、利用者「移動ミスは CPU の負荷によるものではないか」)。探索スレッド数は YamaNEKO だけ
        # (YamaNEKO は起動時に思考時間を測って探索の幅を決めるので、スレッドを減らすと幅が狭まる)。優先度はどのエンジンでも
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("CPU の使用:"))
        self.threads_combo = QtWidgets.QComboBox()
        for text, key in CPU_USE_CHOICES:
            self.threads_combo.addItem(text, key)
        self.threads_combo.setToolTip("YamaNEKO の探索スレッド数。減らすと CPU の使用率が下がるが、探索の幅が狭まり手の質が少し落ちる。\n"
                                      "「自動」は今までどおり(1つの時は全コア、1P・2P の両方の時は半分ずつ)")
        row.addWidget(self.threads_combo, 1)
        form.addLayout(row)
        self.low_priority = ToggleSwitch("思考エンジンの優先度を下げる")
        self.low_priority.setToolTip("思考エンジンのプロセスの優先度を「通常以下」にします。使用率は下がりませんが、\n"
                                     "ゲームが CPU を必要とする時はゲームが先に動きます")
        form.addWidget(self.low_priority)
        # 配ぷよ読み(2026-10-23)。試合の最初からの組ぷよで配ぷよ(65536通り)を照合し、先のツモで読む
        self.haipuyo = ToggleSwitch("配ぷよ読み")
        self.haipuyo.setToolTip(
            "試合の最初からの組ぷよで配ぷよを照合し、先のツモで組みます。\n"
            "他の方の AI との対戦で使うかどうかは、相手との取り決めで決めてください。"
        )
        form.addWidget(self.haipuyo)
        # 相手のツモを覗く(2026-11-01)。相手が先に進んでいる時の相手のネクストを自分の先のツモとして使う
        self.peek = ToggleSwitch("相手のツモを覗く")
        self.peek.setToolTip(
            "相手も同じ配ぷよなら、相手が先に進んでいる時に相手のネクストが自分の先のツモになるので、それを使って組みます。\n"
            "画面に見えている相手のネクストを使うだけです。配ぷよが相手と違うと分かったら、その試合では使いません。"
        )
        form.addWidget(self.peek)
        form.addStretch(1)
        self._on_engine_changed(emit=False)

    def engine(self) -> str:
        """"yamaneko" / "ama" / "none" / "custom:<登録した名前>"。"""
        return self.engine_combo.currentData()

    def set_custom_engines(self, names: list[str], select: str | None = None) -> None:
        """思考エンジンの一覧を、組み込み(YamaNEKO・ama・支援なし)と登録したエンジンで作り直す。選んでいたものは残す。"""
        current = select or self.engine_combo.currentData()
        self.engine_combo.blockSignals(True)
        self.engine_combo.clear()
        for key, text in ENGINE_CHOICES.items():
            self.engine_combo.addItem(text, key)
        for name in names:
            self.engine_combo.addItem(f"{name}(登録)", f"custom:{name}")
        idx = self.engine_combo.findData(current)
        self.engine_combo.setCurrentIndex(idx if idx >= 0 else self.engine_combo.findData("none"))
        self.engine_combo.blockSignals(False)
        if select is None and idx < 0:
            self._on_engine_changed()

    def set_books(self, names: list[str], selected: str) -> None:
        """定跡の一覧(使わない・標準の定跡・登録した定跡)。selected が無くなっていれば「使わない」。"""
        self.book_combo.blockSignals(True)
        self.book_combo.clear()
        self.book_combo.addItem("使わない", "")
        self.book_combo.addItem(BUILTIN_BOOK, BUILTIN_BOOK)
        for name in names:
            self.book_combo.addItem(name, name)
        self.book_combo.setCurrentIndex(max(0, self.book_combo.findData(selected)))
        self.book_combo.blockSignals(False)

    def book_name(self) -> str:
        """使う定跡の名前(使わないなら "")。"""
        return self.book_combo.currentData() or ""

    def add_autoplay_row(self, row: QtWidgets.QLayout) -> None:
        """全自動の設定の枠に行を足す(全自動を許可するプラグインが無い時は枠ごと隠れる)。"""
        self.autoplay_box.layout().addLayout(row)

    def set_autoplay_available(self, available: bool) -> None:
        """全自動を許可するプラグインが有効な側だけ、全自動の設定を出す。許可が無ければ全自動は切る。"""
        self.autoplay_box.setVisible(available)
        if not available:
            self.autoplay.setChecked(False)

    def autoplay_available(self) -> bool:
        return not self.autoplay_box.isHidden()

    def search_threads(self, auto: int | None) -> int | None:
        """YamaNEKO に渡す探索スレッド数(auto は「自動」の時の値。None ならエンジンの既定 = 全コア)。"""
        key = self.threads_combo.currentData()
        cores = os.cpu_count() or 2
        if key == "all":
            return cores
        if key == "half":
            return max(1, cores // 2)
        if key == "quarter":
            return max(1, cores // 4)
        if key == "two":
            return 2
        return auto

    def yamaneko_option(self, switch: QtWidgets.QCheckBox) -> bool:
        """YamaNEKO だけの設定(設置キャンセル・段差キャンセル・配ぷよ読み)が有効か。他のエンジンでは常に無効。"""
        return self.engine() == "yamaneko" and switch.isChecked()

    def connect_pad(self) -> bool:
        """この側の仮想パッドをつなぐ(つないであれば何もしない)。つなげたら True。"""
        try:
            autoplay_ext.shared_pad(self.side)
        except Exception as exc:  # パッケージ・ドライバーが無い
            self.pad_label.setText(f"つなげません({type(exc).__name__})。キー割り当ての画面でドライバーを入れてください")
            return False
        self.pad_label.setText("つないでいます(Xbox 360 コントローラー)")
        return True

    def tap_pad(self, button: str) -> None:
        """この側の仮想パッドのボタンを、PAD_TAP_DELAY_MS 後に押して離す(ゲームの参加の画面でこの側に割り当てる用)。
        すぐ押すと PBA の画面が前面のままで、裏に回ったゲーム(と Steam)がコントローラーの入力を無視する(2026-10-25 実機:
        「A を押す」に反応しなかった)。待つ間にゲームの画面をクリックして前面にしてもらう"""
        if not self.connect_pad():
            return
        sender = PadSender(slot=self.side)
        self.pad_label.setText(f"{PAD_TAP_DELAY_MS // 1000}秒後に押します。ゲームの画面をクリックして前面にしてください")

        def tap() -> None:
            sender.press(button)
            # lambda で sender を持っておく。束縛メソッドを渡すと PyQt は sender を弱参照で持ち、tap() を抜けた時点で sender が消えて
            # 離す処理が呼ばれず、A が押しっぱなしになって2回目以降が効かなかった(2026-10-25 実機、joy.cpl でボタン1が点灯したまま)
            QtCore.QTimer.singleShot(150, lambda: sender.release_all())
            self.pad_label.setText(f"{autoplay_ext.PAD_BUTTON_TEXT.get(button, button)} を押しました(仮想パッドをつないでいます)")

        QtCore.QTimer.singleShot(PAD_TAP_DELAY_MS, tap)

    def _on_engine_changed(self, emit: bool = True) -> None:
        engine = self.engine()
        for w in (self.place_cancel, self.step_cancel, self.haipuyo, self.peek, self.threads_combo):
            w.setEnabled(engine == "yamaneko")  # YamaNEKO 以外では暗転させて設定できなくする
        for w in (self.language_combo, self.book_combo, self.autoplay, self.prehold, self.low_priority):
            w.setEnabled(engine != "none")
        if emit:
            self.changed.emit()


def default_engine() -> str:
    """メイン画面の思考エンジンの初期値。YamaNEKO が見つからなければ同梱の ama(2026-11-01、配布する exe には YamaNEKO が入らない)。"""
    return "yamaneko" if resolve_bridge_exe().exists() else "ama"


class MainWindow(QtWidgets.QWidget):
    autoplay_toggle_requested = QtCore.pyqtSignal(str)  # 切り替えキーが押された側(フックのスレッドから)
    pad_kb_toggled = QtCore.pyqtSignal(str, bool)  # キーボードで仮想パッドを操作するの(側, オン・オフ)(フックのスレッドから)

    def __init__(self, side: str = "1p") -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(380, 180)
        self.calibration: CalibrationResult | None = None  # 読み込んだまま(対戦用は 自分=1P=左)
        self.game_window: GameWindow | None = None  # 見つけた Steam 版のウィンドウ(無ければ手動のキャリブレーション)
        # 支援モード中にゲームのウィンドウが動いた・大きさが変わったら止める(盤面の位置が変わるため)
        self._window_watch_timer = QtCore.QTimer(self)
        self._window_watch_timer.timeout.connect(self._check_game_window)
        # ローカル対戦用プラグイン(2026-10-28): 2P の全自動はネームプレートが「2P」「Player 2」の間だけ動かす
        self._autoplay_wanted: dict[str, bool] = {}  # 利用者が全自動をオンにしている側
        self._nameplate_ok: dict[str, bool] = {}
        self._nameplate_cap: ScreenCapture | None = None
        self._nameplate_timer = QtCore.QTimer(self)
        self._nameplate_timer.timeout.connect(self._check_nameplates)
        self.badge: ControlBadge | None = None
        # 支援モード中の側ごとの部品(1P と 2P を同時に動かす時は2つ)。側 -> (ワーカー, 思考エンジン, オーバーレイ)
        self.sessions: dict[str, tuple[AssistWorker, AmaClient, OverlayWindow]] = {}
        self._badge_lines: dict[str, str] = {}
        self.simulator: QtWidgets.QWidget | None = None
        # プラグイン(2026-10-28): plugins/ のプラグインを読み、言語プラグインの表記を情報パネルに登録する
        self.plugins = PluginManager.load()
        languages = self.plugins.languages()
        for key, lang in languages.items():
            register_language(key, lang.transform, lang.word_break)

        layout = QtWidgets.QVBoxLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel()
        row.addWidget(self.status_label, 1)
        # PC(Steam)版のウィンドウを探し直す(2026-10-28、利用者「ゲームを起動し直した時にも検出できるように」)
        self.detect_window_btn = QtWidgets.QPushButton("ゲームのウィンドウを検出")
        self.detect_window_btn.setToolTip("PC 版のゲームのウィンドウを探します。見つからなければ一覧から選べます(選んだものは覚えます)")
        self.detect_window_btn.clicked.connect(self._on_detect_window_clicked)
        row.addWidget(self.detect_window_btn)
        layout.addLayout(row)
        # ゲームのウィンドウを決まった大きさに合わせる(ウィンドウの大きさプラグインが有効な時だけ出す。2026-10-28、利用者)
        self.window_size_row = QtWidgets.QWidget()
        row = QtWidgets.QHBoxLayout(self.window_size_row)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(QtWidgets.QLabel("ゲームのウィンドウの大きさ:"))
        self.window_size_combo = QtWidgets.QComboBox()
        for w, h in CLIENT_SIZE_PRESETS:
            self.window_size_combo.addItem(f"{w}×{h}")
        self.window_size_combo.setCurrentIndex(max(0, self.window_size_combo.findText(_load_window_size())))
        self.window_size_combo.currentTextChanged.connect(_save_window_size)
        row.addWidget(self.window_size_combo)
        resize_btn = QtWidgets.QPushButton("この大きさに合わせる")
        resize_btn.setToolTip("ゲームの描画領域(タイトルバー・枠を除く)をこの大きさ(画面の物理ピクセル)にします")
        resize_btn.clicked.connect(self._on_resize_window_clicked)
        row.addWidget(resize_btn)
        row.addStretch(1)
        layout.addWidget(self.window_size_row)

        # 対戦モード・一人用モード・シミュレーターをタブで切り替える(2026-10-25、利用者)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        # 対戦: 1P・2P を左右に分け、側ごとに設定する(2026-10-25、利用者)
        versus = QtWidgets.QWidget()
        v_layout = QtWidgets.QVBoxLayout(versus)
        panels = QtWidgets.QHBoxLayout()
        self.side_settings: dict[str, SideSettings] = {}
        for key in ("1p", "2p"):
            box = SideSettings(key, default_engine() if key == side else "none", self._open_autoplay_keys, languages=languages)
            box.changed.connect(self._refresh_calibration_status)
            self.side_settings[key] = box
            panels.addWidget(box, 1)
        v_layout.addLayout(panels)
        calibrate_btn = QtWidgets.QPushButton("対戦用キャリブレーション")
        calibrate_btn.clicked.connect(lambda: self._on_calibrate_clicked(solo=False))
        v_layout.addWidget(calibrate_btn)
        self.calibrate_buttons = [calibrate_btn]  # 手動キャリブレーション(Switch 対応プラグインが有効な時だけ出す)
        self.tabs.addTab(versus, "対戦モード")

        # 一人用: 相手を見ず、連鎖構築だけを提案する(画面レイアウトが対戦と異なるため、キャリブレーションは別)
        solo = QtWidgets.QWidget()
        s_layout = QtWidgets.QVBoxLayout(solo)
        self.solo_settings = SideSettings("1p", default_engine(), self._open_autoplay_keys, title="自分の盤面", languages=languages)
        self.solo_settings.changed.connect(self._refresh_calibration_status)
        s_layout.addWidget(self.solo_settings, 1)
        calibrate_solo_btn = QtWidgets.QPushButton("一人用キャリブレーション")
        calibrate_solo_btn.clicked.connect(lambda: self._on_calibrate_clicked(solo=True))
        s_layout.addWidget(calibrate_solo_btn)
        self.calibrate_buttons.append(calibrate_solo_btn)
        self.tabs.addTab(solo, "一人用モード")

        # シミュレーター: 画像認識を使わず、自前の盤面で AI の提案を見ながら練習する。思考エンジンを2つ起動するので、
        # タブを初めて開いた時に作る
        self._simulator_page = QtWidgets.QWidget()
        QtWidgets.QVBoxLayout(self._simulator_page).setContentsMargins(0, 0, 0, 0)
        self.tabs.addTab(self._simulator_page, "シミュレーター")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # 思考エンジン・定跡の登録と、定跡を使うか(2026-10-26、利用者)。定跡を使うと AI の思考より定跡の手を優先し、
        # 定跡に無い手順になったら AI の思考に戻る(engine/book_client.py)。支援モードとシミュレーターで使う
        self.registry = Registry.load()
        # 定跡は 1P・2P・一人用の枠ごとに選ぶ(2026-10-28)。ここは登録だけ
        row = QtWidgets.QHBoxLayout()
        book_btn = QtWidgets.QPushButton("定跡の登録")
        book_btn.clicked.connect(self._open_book_registry)
        row.addWidget(book_btn)
        engine_btn = QtWidgets.QPushButton("エンジンの登録")
        engine_btn.clicked.connect(self._open_engine_registry)
        row.addWidget(engine_btn)
        plugin_btn = QtWidgets.QPushButton("プラグイン")
        plugin_btn.setToolTip("入っているプラグインの一覧と、有効・無効の切り替え")
        plugin_btn.clicked.connect(self._open_plugins)
        row.addWidget(plugin_btn)
        layout.addLayout(row)
        self._refresh_registry()
        for box in self._book_boxes().values():
            box.book_combo.currentIndexChanged.connect(lambda _i: self._on_book_selected())

        # 全自動に関わる共通の設定(2026-10-28、プラグイン化): 全自動を許可するプラグインが有効な時だけ出す
        self.autoplay_common = QtWidgets.QWidget()
        outer_layout, layout = layout, QtWidgets.QVBoxLayout(self.autoplay_common)
        layout.setContentsMargins(0, 0, 0, 0)
        # 全自動の部品(キーボードフック・仮想パッド・切り替えキー)は、全自動のプラグインがある時だけ作る(2026-11-01)
        self._pad_kb: dict = {}
        self._pad_kb_hook = None
        self._key_toggle = None
        self._pad_toggle = None
        self._toggle_bindings: dict = {}
        self.pad_kb_switches: dict[str, QtWidgets.QCheckBox] = {}
        self.pad_kb_labels: dict[str, QtWidgets.QLabel] = {}
        self._pad_toggle_timer = QtCore.QTimer(self)
        if AUTOPLAY_AVAILABLE:
            self._build_autoplay_controls(layout)
        outer_layout.addWidget(self.autoplay_common)
        layout = outer_layout

        # 共通: ログの出力と、支援モードの開始(対戦・一人用のタブの時だけ)
        # ログの出力はログ出力プラグインが有効な時だけ出す(2026-10-28、利用者)
        self.debug_log_checkbox = ToggleSwitch("認識結果と提案をログに出力する(debug_logs/)")
        layout.addWidget(self.debug_log_checkbox)

        self.assist_btn = QtWidgets.QPushButton("支援モード開始")
        self.assist_btn.clicked.connect(self._on_toggle_assist)
        layout.addWidget(self.assist_btn)

        self._apply_plugins()
        self._refresh_calibration_status()

    def _build_autoplay_controls(self, layout: QtWidgets.QVBoxLayout) -> None:
        """全自動の切り替えキー・キーボードで仮想パッドを操作する設定(全自動のプラグインにある部品を使う)。"""
        row = QtWidgets.QHBoxLayout()
        toggle_btn = QtWidgets.QPushButton("全自動の切り替えキー")
        toggle_btn.setToolTip("支援モード中に全自動のオン・オフを切り替えるキーボードのキー・ゲームパッドのボタン")
        toggle_btn.clicked.connect(self._open_toggle_keys)
        row.addWidget(toggle_btn)
        row.addStretch(1)
        layout.addLayout(row)

        # キーボードで仮想パッドを操作する(2026-10-25、利用者): キャラクター選択・ルール設定で、仮想パッドの側を
        # キーボードで動かす。オンの間は割り当てたキーをゲームに届けない。切り替えキー(既定 ScrollLock)でも切り替わる
        # 1P・2P それぞれ別にオン・オフ・キーの割り当てを持つ(2026-10-25、利用者)
        self._pad_kb_config_path = app_root() / "config" / autoplay_ext.CONFIG_NAME
        self._pad_kb_configs = autoplay_ext.load_side_configs(self._pad_kb_config_path)
        self._pad_kb: dict[str, "autoplay_ext.KeyToPad"] = {}
        self.pad_kb_switches: dict[str, QtWidgets.QCheckBox] = {}
        self.pad_kb_labels: dict[str, QtWidgets.QLabel] = {}
        for pad_side in ("1p", "2p"):
            self._pad_kb[pad_side] = autoplay_ext.KeyToPad(self._pad_kb_configs[pad_side],
                                              on_toggle=lambda on, s=pad_side: self.pad_kb_toggled.emit(s, on))
            row = QtWidgets.QHBoxLayout()
            switch = ToggleSwitch("キーボードで仮想パッドを操作")
            switch.setToolTip("キャラクター選択やルール設定で、この側の仮想パッドをキーボードで動かします。\n"
                              "オンの間は割り当てたキーがゲームに届きません。試合の前にオフにしてください。")
            switch.toggled.connect(lambda on, s=pad_side: self._on_pad_kb_switch(s, on))
            self.pad_kb_switches[pad_side] = switch
            row.addWidget(switch)
            label = QtWidgets.QLabel()
            self.pad_kb_labels[pad_side] = label
            row.addWidget(label, 1)
            btn = QtWidgets.QPushButton("キー設定")
            btn.clicked.connect(lambda _c=False, s=pad_side: self._open_pad_kb_dialog(s))
            row.addWidget(btn)
            # 1P・2P の枠の中に置く(2026-10-28、利用者「仮想パッドの操作は 1P と 2P の区画ごとに」)
            self.side_settings[pad_side].add_autoplay_row(row)
        # 支援モード中に全自動を切り替えるキー・ゲームパッドのボタン(2026-10-27)。キーは同じキーボードフックで見る
        self._toggle_bindings = autoplay_ext.load_toggle_bindings()
        self._key_toggle = autoplay_ext.KeyToggle(self._toggle_bindings, on_toggle=lambda side: self.autoplay_toggle_requested.emit(side))
        self._pad_toggle = autoplay_ext.PadToggleWatcher(self._toggle_bindings)
        self._pad_toggle_timer = QtCore.QTimer(self)
        self._pad_toggle_timer.timeout.connect(self._poll_pad_toggle)
        self.autoplay_toggle_requested.connect(self._toggle_autoplay, QtCore.Qt.ConnectionType.QueuedConnection)
        self._pad_kb_hook = autoplay_ext.KeyboardHook(list(self._pad_kb.values()) + [self._key_toggle])
        self.pad_kb_toggled.connect(self._on_pad_kb_toggled, QtCore.Qt.ConnectionType.QueuedConnection)
        self._update_pad_kb_labels()

    def _autoplay_sides(self) -> set[str]:
        """全自動で動かしてよい側。全自動のプラグイン(操作の仕組み)が無ければ、許可するプラグインがあっても空。"""
        return self.plugins.autoplay_sides() if AUTOPLAY_AVAILABLE else set()

    def _guarded_autoplay_sides(self) -> set[str]:
        return self.plugins.guarded_autoplay_sides() if AUTOPLAY_AVAILABLE else set()

    def _apply_plugins(self) -> None:
        """有効なプラグインに合わせて、メイン画面の項目を出し入れする。"""
        allowed = self._autoplay_sides()
        for side, box in self.side_settings.items():
            box.set_autoplay_available(side in allowed)
        self.solo_settings.set_autoplay_available("1p" in allowed)
        self.autoplay_common.setVisible(bool(allowed))
        for btn in self.calibrate_buttons:
            btn.setVisible(self.plugins.manual_calibration())
        on_simulator = self.tabs.currentWidget() is self._simulator_page
        self.debug_log_checkbox.setVisible(self.plugins.debug_log() and not on_simulator)
        if not self.plugins.debug_log():
            self.debug_log_checkbox.setChecked(False)
        self.window_size_row.setVisible(self.plugins.window_resize())

    def _open_plugins(self) -> None:
        dialog = PluginDialog(self.plugins, self)
        if dialog.exec():
            self.plugins.disabled = dialog.disabled
            save_disabled(self.plugins.disabled)
            self._apply_plugins()

    def connect_virtual_pads(self) -> None:
        """送り先が仮想パッドの側は、PBA の起動時にパッドをつなぐ(ゲームの参加の画面で割り当てられるように。2026-10-25)。
        テストで本物のパッドをつながないよう、__init__ ではなく main から呼ぶ。"""
        for side, box in self.side_settings.items():
            try:
                device = KeyMap.load(app_root() / "config" / KEYMAP_FILES[side]).device
            except Exception:
                continue
            if device == "pad":
                box.connect_pad()

    def _open_autoplay_keys(self) -> None:
        """全自動モードのキー割り当て画面。保存は次に支援モードを開始した時から使われる。"""
        paths = {side: app_root() / "config" / name for side, name in KEYMAP_FILES.items()}
        autoplay_ext.AutoplayKeyDialog(paths, self).exec()

    def _refresh_registry(self) -> None:
        """登録した思考エンジン・定跡を画面の一覧に反映する。"""
        names = [e.name for e in self.registry.engines]
        for box in [*self.side_settings.values(), self.solo_settings]:
            box.set_custom_engines(names)
        books = [b.name for b in self.registry.books]
        for key, box in self._book_boxes().items():
            box.set_books(books, self.registry.side_books.get(key, ""))

    def _book_boxes(self) -> dict[str, "SideSettings"]:
        """定跡の選択を覚える名前 → 設定の枠。"""
        return {**self.side_settings, "solo": self.solo_settings}

    def _on_book_selected(self) -> None:
        """側ごとの定跡の選択を保存する。シミュレーターの定跡は 1P で選んだもの(使わないなら今までのまま)。"""
        side_books = {key: box.book_name() for key, box in self._book_boxes().items()}
        if side_books == self.registry.side_books:
            return
        self.registry.side_books = side_books
        if side_books.get("1p"):
            self.registry.selected_book = side_books["1p"]
        self.registry.save()

    def _open_engine_registry(self) -> None:
        dialog = EngineRegistryDialog(self.registry, self)
        if dialog.exec():
            self.registry.engines = dialog.engines
            self.registry.save()
            self._refresh_registry()

    def _open_book_registry(self) -> None:
        dialog = BookRegistryDialog(self.registry, self)
        if dialog.exec():
            self.registry.books = dialog.books
            if self.registry.selected_book not in [BUILTIN_BOOK, *[b.name for b in dialog.books]]:
                self.registry.selected_book = BUILTIN_BOOK
            self.registry.save()
            self._refresh_registry()

    def _selected_book(self, name: str) -> OpeningBook | None:
        """その名前の定跡(読めなければ None)。"""
        path = self.registry.book_path(name)
        try:
            return OpeningBook.load() if path is None else OpeningBook.load(path)
        except (ValueError, KeyError, TypeError):
            return None

    def _make_client(self, box: "SideSettings", threads: int | None) -> AmaClient:
        """その側の思考エンジンを起動する(登録したエンジンは実行ファイルと起動引数で)。"""
        engine = box.engine()
        if engine.startswith("custom:"):
            entry = self.registry.engine(engine.split(":", 1)[1])
            if entry is None:
                raise FileNotFoundError(f"登録した思考エンジンが見つかりません: {engine}")
            return AmaClient(exe_path=Path(entry.exe), custom_args=entry.args, low_priority=box.low_priority.isChecked())
        extra = (HAIPUYO_BRIDGE_ARGS if box.yamaneko_option(box.haipuyo) else []) + (PEEK_BRIDGE_ARGS if box.yamaneko_option(box.peek) else [])
        return AmaClient(bridge=engine, search_threads=box.search_threads(threads),
                         extra_args=extra or None,
                         low_priority=box.low_priority.isChecked())

    def _open_toggle_keys(self) -> None:
        dialog = autoplay_ext.AutoplayToggleDialog(self._toggle_bindings, self)
        if dialog.exec():
            self._toggle_bindings = dialog.bindings
            autoplay_ext.save_toggle_bindings(self._toggle_bindings)
            self._key_toggle.configure(self._toggle_bindings)
            self._pad_toggle.configure(self._toggle_bindings)

    def _toggle_autoplay(self, side: str) -> None:
        """支援モード中に、その側の全自動を切り替える(バッジのボタン・切り替えキー・ゲームパッドのボタン)。"""
        if side not in self.sessions or side not in self._autoplay_sides():
            return  # 全自動を許可するプラグインが無い側は切り替えない
        self._autoplay_wanted[side] = not self._autoplay_wanted.get(side, False)
        self._apply_autoplay(side)

    def _guarded_sides(self) -> set[str]:
        """支援中で、全自動にネームプレートの確認(2P・Player 2)が要る側(対戦モードだけ。一人用は 2P が無い)。"""
        if self._solo():
            return set()
        return self._guarded_autoplay_sides() & self.sessions.keys()

    def _nameplate_allows(self, side: str) -> bool:
        """その側の全自動を動かしてよいか(ネームプレートの確認が要らない側は常に True)。確認が要る側は画面を見て判断する。"""
        if self._solo() or side not in self._guarded_autoplay_sides():
            return True
        ok = self._read_nameplate(side)
        self._nameplate_ok[side] = ok
        return ok

    def _read_nameplate(self, side: str) -> bool:
        """その側のネームプレートに「2P」「Player 2」が見えるか。読めない時は False(動かさない方に倒す)。"""
        if self.calibration is None:
            return False
        # 対戦用のキャリブレーションは 自分=1P=左 なので、2P の盤面は相手側
        player = self.calibration.enemy_player if side == "2p" else self.calibration.self_player
        board = player.geometry()
        x, y, w, h = autoplay_ext.search_box(board)
        try:
            if self._nameplate_cap is None:
                self._nameplate_cap = ScreenCapture()
            crop = self._nameplate_cap.grab_bgr(CaptureRegion(left=x, top=y, width=w, height=h))
        except Exception:
            return False
        return autoplay_ext.is_player2(crop, board.bounding_box()[2])

    def _apply_autoplay(self, side: str) -> None:
        """利用者の選択とネームプレートの確認から、その側の全自動を切り替えてバッジに出す。"""
        if side not in self.sessions:
            return
        worker = self.sessions[side][0]
        on = self._autoplay_wanted.get(side, False) and self._nameplate_ok.get(side, True)
        worker.set_autoplay(on)
        self._update_autoplay_badges()

    def _update_autoplay_badges(self) -> None:
        if self.badge is None:
            return
        for side, (worker, _c, _o) in self.sessions.items():
            waiting = self._autoplay_wanted.get(side, False) and not worker.autoplay.enabled
            self.badge.set_autoplay_state(side, worker.autoplay.enabled, waiting=waiting)

    def _check_nameplates(self) -> None:
        """支援中、1秒ごとにネームプレートを確かめ、「2P」「Player 2」が見えない間は全自動を止める(試合の合間・オンライン対戦)。"""
        for side in self._guarded_sides():
            ok = self._read_nameplate(side)
            if ok != self._nameplate_ok.get(side):
                self._nameplate_ok[side] = ok
                self._apply_autoplay(side)

    def _poll_pad_toggle(self) -> None:
        for side in self._pad_toggle.poll():
            self._toggle_autoplay(side)

    def _update_pad_kb_labels(self) -> None:
        overlap = self._pad_kb_configs["1p"].keys() & self._pad_kb_configs["2p"].keys()
        for pad_side, label in self.pad_kb_labels.items():
            c = self._pad_kb_configs[pad_side]
            text = f"{autoplay_ext.PRESET_TEXT[c.preset]}・切り替え {c.toggle_key}"
            if overlap:
                text += f"(1P と重なるキー: {'・'.join(sorted(overlap))}。両方オンの時は 1P が優先)" if pad_side == "2p" else ""
            label.setText(text)

    def _open_pad_kb_dialog(self, pad_side: str) -> None:
        other = "2p" if pad_side == "1p" else "1p"
        dialog = autoplay_ext.PadKeyboardDialog(self._pad_kb_configs[pad_side], self, taken=self._pad_kb_configs[other].keys())
        if dialog.exec():
            self._pad_kb_configs[pad_side] = dialog.config
            autoplay_ext.save_side_configs(self._pad_kb_config_path, self._pad_kb_configs)
            handler = self._pad_kb[pad_side]
            enabled = handler.enabled
            handler.configure(dialog.config)
            handler.enabled = enabled
            self._update_pad_kb_labels()

    def _on_pad_kb_switch(self, pad_side: str, on: bool) -> None:
        """画面のトグル。最初にオンにした時にキーボードフックを付け、PBA を閉じるまで付けたままにする(切り替えキーを効かせるため)。"""
        if on and not self._pad_kb_hook.start():
            self.pad_kb_labels[pad_side].setText(f"使えません: {self._pad_kb_hook.error}")
            self._set_pad_kb_switch(pad_side, False)
            return
        if self._pad_kb[pad_side].enabled != on:
            self._pad_kb[pad_side].set_enabled(on)

    def _set_pad_kb_switch(self, pad_side: str, on: bool) -> None:
        switch = self.pad_kb_switches[pad_side]
        switch.blockSignals(True)
        switch.setChecked(on)
        switch.blockSignals(False)

    def _on_pad_kb_toggled(self, pad_side: str, on: bool) -> None:
        """切り替えキーなどでオン・オフが変わった(フックのスレッドから届く)。画面のトグルを合わせる。"""
        if self.pad_kb_switches[pad_side].isChecked() != on:
            self._set_pad_kb_switch(pad_side, on)

    def _on_tab_changed(self, index: int) -> None:
        on_simulator = self.tabs.widget(index) is self._simulator_page
        self.assist_btn.setVisible(not on_simulator)
        self.debug_log_checkbox.setVisible(not on_simulator and self.plugins.debug_log())
        if on_simulator:
            self._open_simulator()
        else:
            self._refresh_calibration_status()

    def _open_simulator(self) -> None:
        """シミュレーターのタブの中身を作る(最初の1回だけ)。キー操作を受け取れるよう焦点を移す。"""
        from .simulator.window import SimulatorWindow

        if self.simulator is None:
            self.simulator = SimulatorWindow()
            self._simulator_page.layout().addWidget(self.simulator)
        self.status_label.setText("シミュレーター")
        self.simulator.setFocus()

    def _solo(self) -> bool:
        """一人用モードのタブか(一人用はキャリブレーションのファイルが別)。"""
        return self.tabs.currentIndex() == 1

    def _settings(self, side: str) -> SideSettings:
        """支援する側の設定の枠(一人用のタブは1つだけ)。"""
        return self.solo_settings if self._solo() else self.side_settings[side]

    def _calibration_path(self) -> Path:
        # 一人用モードは画面レイアウトが対戦と異なるため別ファイル
        return SOLO_CONFIG_PATH if self._solo() else CONFIG_PATH

    def _sides(self) -> list[str]:
        """支援する側(思考エンジンが「支援なし」でない側)。一人用は自分(1P として扱う)だけ。"""
        if self._solo():
            return ["1p"] if self.solo_settings.engine() != "none" else []
        return [s for s, box in self.side_settings.items() if box.engine() != "none"]

    def _calibration_for(self, side: str) -> CalibrationResult:
        """その側を自分としたキャリブレーション。対戦用は 1P(左)を自分として保存しているので 2P は入れ替える。"""
        assert self.calibration is not None
        if side == "2p" and not self._solo():
            return self.calibration.swapped()
        return self.calibration

    def _refresh_calibration_status(self) -> None:
        path = self._calibration_path()
        mode = "一人用" if self._solo() else "対戦"
        sides = self._sides()
        label = "・".join(s.upper() for s in sides) or "支援なし"
        self.setWindowTitle(APP_TITLE if sides == ["1p"] else f"{APP_TITLE} - {label}")
        # Steam 版のウィンドウが見つかればその位置から盤面を求める(2026-10-28、標準機能)。見つからなければ
        # 手動のキャリブレーション(Switch 対応プラグイン)のファイルを使う
        self.game_window = find_game_window()
        if self.game_window is not None:
            w = self.game_window
            self.calibration = layout_calibration(w, solo=self._solo())
            self.status_label.setText(f"[{mode}・{label}] ゲームのウィンドウを検出: {w.width}×{w.height}")
            return
        if not self.plugins.manual_calibration():
            self.calibration = None
            self.status_label.setText(f"[{mode}] ゲームのウィンドウが見つかりません(Steam 版をウィンドウ表示で起動してください)")
            return
        try:
            self.calibration = load_calibration(path)
            self.status_label.setText(f"[{mode}・{label}] キャリブレーション済み: {path.name}")
        except FileNotFoundError:
            self.calibration = None
            self.status_label.setText(f"[{mode}] 未キャリブレーション(先にキャリブレーションを実行してください)")
        except (KeyError, ValueError) as exc:
            # 壊れた設定(幅0のネクスト枠など)で支援モードを起動すると認識が落ちるため、理由を出してやり直しを促す
            self.calibration = None
            self.status_label.setText(f"[{mode}] キャリブレーションが不正です。やり直してください:\n{exc}")

    def _on_detect_window_clicked(self) -> None:
        """ゲームのウィンドウを探す。見つからなければ表示中のウィンドウの一覧から選んでもらい、次からはそれも探す。"""
        if find_game_window() is None:
            entries = list_windows()
            if not entries:
                QtWidgets.QMessageBox.warning(self, APP_TITLE, "表示中のウィンドウが見つかりません。ゲームをウィンドウ表示で起動してください。")
                return
            dialog = WindowPickDialog(entries, self)
            if not dialog.exec() or dialog.selected() is None:
                return
            save_saved_window(dialog.selected())
        self._refresh_calibration_status()
        if self.game_window is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "ゲームのウィンドウが見つかりません(最小化していないか確かめてください)。")

    def _on_resize_window_clicked(self) -> None:
        """ゲームのウィンドウの描画領域を選んだ大きさにする。"""
        window = find_game_window()
        if window is None:
            QtWidgets.QMessageBox.warning(self, APP_TITLE, "ゲームのウィンドウが見つかりません。先に「ゲームのウィンドウを検出」を押してください。")
            return
        size = CLIENT_SIZE_PRESETS[max(0, self.window_size_combo.currentIndex())]
        got = resize_client(window.hwnd, size)
        self._refresh_calibration_status()
        if got is None:
            QtWidgets.QMessageBox.warning(
                self, APP_TITLE, "ウィンドウの大きさを変えられませんでした(全画面表示か、ゲームが管理者として動いている可能性があります)。"
            )
        elif tuple(got) != tuple(size):
            QtWidgets.QMessageBox.information(
                self, APP_TITLE, f"{size[0]}×{size[1]} にできず、{got[0]}×{got[1]} になりました(画面に収まらないか、ゲームが大きさを決めています)。"
            )

    def _on_calibrate_clicked(self, solo: bool) -> None:
        run_calibration(solo=solo, save_path=SOLO_CONFIG_PATH if solo else CONFIG_PATH)
        self._refresh_calibration_status()

    def _on_toggle_assist(self) -> None:
        if not self.sessions:
            self._start_assist_mode()
        else:
            self._stop_assist_mode()

    def _start_assist_mode(self) -> None:
        sides = self._sides()
        if not sides:
            QtWidgets.QMessageBox.warning(self, "エラー", "思考エンジンを選んでください(「支援なし」になっています)")
            return
        self._refresh_calibration_status()
        if self.calibration is None:
            message = ("先にキャリブレーションを実行してください" if self.plugins.manual_calibration()
                       else "ゲームのウィンドウが見つかりません。Steam 版をウィンドウ表示で起動してください")
            QtWidgets.QMessageBox.warning(self, "エラー", message)
            return
        # 思考エンジンは側ごとに1つ(同時に考えるため)。1つでも起動できなければ全て閉じる
        # 両方の時は探索スレッドをコア数の半分ずつにして、2つの YamaNEKO が CPU を取り合わないようにする。
        # 起動は1つずつ(AmaClient は ready まで待つ)なので、YamaNEKO の起動時の思考時間の計測は重ならない
        threads = max(1, (os.cpu_count() or 2) // 2) if len(sides) > 1 else None
        clients: dict[str, AmaClient] = {}
        self.status_label.setText("思考エンジンを起動しています(YamaNEKO は思考時間の計測で数十秒かかることがあります)…")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            for side in sides:
                clients[side] = self._make_client(self._settings(side), threads)
            for side in sides:
                name = self._settings(side).book_name()
                if name:
                    book = self._selected_book(name)
                    if book is None:
                        raise ValueError(f"定跡ファイルを読めません: {name}")
                    clients[side] = BookClient(clients[side], book)
        except (FileNotFoundError, RuntimeError, TimeoutError, OSError, ValueError) as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._refresh_calibration_status()
            for c in clients.values():
                c.close()
            QtWidgets.QMessageBox.critical(self, "エラー", f"思考エンジンのブリッジの起動に失敗しました:\n{exc}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        self._refresh_calibration_status()

        self.hide()
        if self.game_window is not None:
            # メイン画面を隠してからウィンドウの画像を取り、盤面の暗い矩形が見えればその位置に合わせ直す
            QtWidgets.QApplication.processEvents()
            with ScreenCapture() as cap:
                self.calibration = detect_calibration(self.game_window, solo=self._solo(), grab=cap.grab_bgr)
            self._window_watch_timer.start(1000)
        self.badge = ControlBadge(on_stop=self._stop_assist_mode, detect_text=self._detect_text(), anchor=self._badge_anchor())
        self.badge.show()
        excluded = _exclude_window_from_screen_capture(self.badge)
        stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
        self._badge_lines = {}
        for side in sides:
            box = self._settings(side)
            calibration = self._calibration_for(side)
            overlay = OverlayWindow(calibration, language=box.language_combo.currentData())
            overlay.show()
            excluded = _exclude_window_from_screen_capture(overlay) and excluded

            debug_log_path = None
            if self.plugins.debug_log() and self.debug_log_checkbox.isChecked():
                log_dir = app_root() / "debug_logs"
                log_dir.mkdir(exist_ok=True)
                suffix = "" if side == "1p" else f"_{side}"
                debug_log_path = log_dir / f"debug_log_{stamp}{suffix}.txt"

            self._autoplay_wanted[side] = box.autoplay.isChecked() and side in self._autoplay_sides()
            worker = AssistWorker(
                calibration,
                clients[side],
                debug_log_path,
                solo=self._solo(),
                autoplay=self._autoplay_wanted[side] and self._nameplate_allows(side),
                side=side,
                # 設置キャンセル・段差キャンセル・配ぷよ読み・相手のツモを覗くは YamaNEKO の時だけ(ama では使わない)
                place_cancel=box.yamaneko_option(box.place_cancel),
                haipuyo=box.yamaneko_option(box.haipuyo),
                peek=box.yamaneko_option(box.peek),
                step_cancel=box.yamaneko_option(box.step_cancel),
                prehold=box.prehold.isChecked(),
            )
            worker.draw_data_ready.connect(overlay.update_draw_data, QtCore.Qt.ConnectionType.QueuedConnection)
            worker.perf_ready.connect(overlay.update_perf, QtCore.Qt.ConnectionType.QueuedConnection)
            worker.status_changed.connect(
                lambda text, side=side: self._on_status_changed(side, text), QtCore.Qt.ConnectionType.QueuedConnection
            )
            worker.fatal_error.connect(self._on_fatal_error, QtCore.Qt.ConnectionType.QueuedConnection)
            self.sessions[side] = (worker, clients[side], overlay)
        if not excluded:
            QtWidgets.QMessageBox.warning(
                self,
                "警告",
                "オーバーレイを画面キャプチャから除外できませんでした。\n表示中のドットをぷよとして誤認識する可能性があります。",
            )
        for worker, _client, _overlay in self.sessions.values():
            worker.start()
        # 全自動の切り替え(2026-10-27): バッジのボタン、キーボードのキー(フック)、ゲームパッドのボタン(30msごとに読む)
        allowed = self._autoplay_sides()
        self.badge.add_autoplay_toggles({s: w.autoplay.enabled for s, (w, _c, _o) in self.sessions.items() if s in allowed},
                                        self._toggle_autoplay)
        self._update_autoplay_badges()
        if self._guarded_sides():
            self._nameplate_timer.start(1000)
        if self._key_toggle is not None:
            self._key_toggle.active = True
        if self._pad_kb_hook is not None and any(self._toggle_bindings[s].key for s in self.sessions):
            self._pad_kb_hook.start()
        if self._pad_toggle is not None and self._pad_toggle.needed:
            self._pad_toggle.poll()  # 押しっぱなしのボタンで開始直後に切り替わらないよう、今の状態を覚えておく
            self._pad_toggle_timer.start(30)

    def _detect_text(self) -> str:
        """バッジに出すウィンドウの検出状態。"""
        if self.game_window is not None:
            return f"検出: {self.game_window.title or 'ゲームのウィンドウ'}"
        return "未検出(キャリブレーションのファイル)"

    def _badge_anchor(self) -> QtCore.QRect | None:
        """バッジを置くゲームのウィンドウのタイトルバー(論理ピクセル)。分からなければ None(画面の右上に置く)。"""
        if self.game_window is None:
            return None
        rect = title_bar_rect(self.game_window.hwnd)
        if rect is None:
            return None
        dpr = QtWidgets.QApplication.primaryScreen().devicePixelRatio()
        return QtCore.QRect(*(round(v / dpr) for v in rect))

    def _check_game_window(self) -> None:
        """支援モード中にゲームのウィンドウが動いた・大きさが変わった・閉じたら止める(盤面の位置が変わるため)。"""
        w = self.game_window
        if not self.sessions or w is None:
            return
        if client_rect(w.hwnd) == (w.left, w.top, w.width, w.height):
            return
        self._stop_assist_mode()
        QtWidgets.QMessageBox.information(
            self, APP_TITLE, "ゲームのウィンドウが動いたか大きさが変わったので、支援モードを止めました。\nもう一度開始してください。"
        )

    def _stop_assist_mode(self) -> None:
        self._window_watch_timer.stop()
        self._nameplate_timer.stop()
        self._autoplay_wanted.clear()
        self._nameplate_ok.clear()
        if self._key_toggle is not None:
            self._key_toggle.active = False
        self._pad_toggle_timer.stop()
        sessions, self.sessions = self.sessions, {}
        for worker, _client, _overlay in sessions.values():
            worker.stop()
        for worker, client, overlay in sessions.values():
            if not worker.wait(3000):
                worker.terminate()
                worker.wait()
            client.close()
            overlay.close()
        if self.badge is not None:
            self.badge.close()
            self.badge = None
        self.show()

    def _on_status_changed(self, side: str, text: str) -> None:
        if self.badge is None or side not in self.sessions:
            return
        if len(self.sessions) == 1:
            prefix = "" if side == "1p" else f"[{side.upper()}] "
            self.badge.set_status(f"{prefix}支援モード中  {text}")
            return
        self._badge_lines[side] = f"{side.upper()}: {text}"
        self.badge.set_lines([self._badge_lines.get(s, f"{s.upper()}: 開始中") for s in self.sessions])

    def _on_fatal_error(self, message: str) -> None:
        if not self.sessions:
            return  # 両方動かしている時に、止めた後でもう片方のエラーが届いた
        self._stop_assist_mode()
        QtWidgets.QMessageBox.critical(self, "エラー", f"支援モードでエラーが発生しました。\n{message}\n(crash_log.txt 参照)")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        if self._pad_kb_hook is not None:
            self._pad_kb_hook.stop()
        self._stop_assist_mode()
        if self.simulator is not None:
            self.simulator.close()
        super().closeEvent(event)


def _log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:  # noqa: ANN001
    with open(app_root() / "crash_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\n===== クラッシュ(メインスレッド) {datetime.now().isoformat()} =====\n")
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)


_instance_mutex = None  # 起動中ずっと持ち続ける(閉じると次に PBA を起動できる)


def _claim_first_instance(name: str = "Local\\PBA_PuyoBeginnerAssistance") -> bool:
    """この PBA が1つ目の起動か(Windows の名前付きミューテックスで判定)。判定できない環境では True。
    多重起動の防止に使う(2つ起動するとキー入力・画面の取り込みが競合する。1P と 2P は「両方」で1つの PBA から動かす)。"""
    global _instance_mutex
    if sys.platform != "win32":
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(None, False, name)
    if not handle:
        return True
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        return False
    _instance_mutex = handle
    return True


def main() -> None:
    sys.excepthook = _log_uncaught_exception
    app = QtWidgets.QApplication(sys.argv)
    icon_path = resource_root() / "assets" / "icon.ico"
    if icon_path.exists():
        app.setWindowIcon(QtGui.QIcon(str(icon_path)))
    if not _claim_first_instance():
        QtWidgets.QMessageBox.warning(
            None, APP_TITLE, "PBA はすでに起動しています。\n1P と 2P を同時に動かす時は、メイン画面で両方の思考エンジンを選んでください。"
        )
        return
    window = MainWindow()
    window.show()
    window.connect_virtual_pads()
    app.exec()


if __name__ == "__main__":
    main()
