"""キャリブレーションツール

対象ゲーム画面上で、自分と相手それぞれの「盤面の可視12段の左上・右下」と
「ネクスト枠2つ」を指定してもらい、config/calibration.json に保存する。

盤面は左上・右下の2点(矩形)で指定し、緑の格子を重ねて矢印キーで微調整する
(Tetris版と同じ操作。以前は4隅指定だったが不便との指摘で変更)。内部では矩形を
4隅の透視変換(BoardGeometry)として扱うので、将来歪んだ画面にも対応できる。
実機画面では盤面内部の暗い矩形が明度で綺麗に分離できるので、「自動検出」(Aキー)で
盤面を一括取得し、微調整で合わせ込む運用を基本にする。

一度キャリブレーションすれば、同じウィンドウ配置・解像度である限り再利用できる。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..engine.puyo_field import COLS, VISIBLE_ROWS
from ..paths import app_root
from .geometry import BoardGeometry, Quad, detect_board_quads
from .screen_capture import CaptureRegion, ScreenCapture

CONFIG_PATH = app_root() / "config" / "calibration.json"
# 一人用(とことん等)モードは画面レイアウトが対戦と異なる(ネクスト枠の位置など)ため、
# 別ファイルにキャリブレーションを持つ
SOLO_CONFIG_PATH = app_root() / "config" / "calibration_solo.json"

NEXT_SLOT_COUNT = 2
Rect = tuple[int, int, int, int]  # (x, y, w, h)


@dataclass
class PlayerCalibration:
    board_quad: Quad  # 可視12段の4隅(左上, 右上, 右下, 左下)。物理ピクセル座標
    next_rects: list[Rect]  # [ネクスト, ネクネク]。各枠は縦2ぷよ(上が子、下が軸)

    def geometry(self) -> BoardGeometry:
        return BoardGeometry(self.board_quad)


@dataclass
class CalibrationResult:
    self_player: PlayerCalibration
    enemy_player: PlayerCalibration

    def to_dict(self) -> dict:
        return {
            "self": {
                "board_quad": [list(p) for p in self.self_player.board_quad],
                "next_rects": [list(r) for r in self.self_player.next_rects],
            },
            "enemy": {
                "board_quad": [list(p) for p in self.enemy_player.board_quad],
                "next_rects": [list(r) for r in self.enemy_player.next_rects],
            },
        }

    @staticmethod
    def from_dict(data: dict) -> "CalibrationResult":
        def player(name: str, d: dict) -> PlayerCalibration:
            quad = tuple((float(x), float(y)) for x, y in d["board_quad"])
            if len(quad) != 4:
                raise ValueError(f"{name}: board_quad は4点必要です")
            (x1, y1), (x2, _), (_, y3), _ = quad
            if x2 - x1 < COLS * 4 or y3 - y1 < VISIBLE_ROWS * 4:
                raise ValueError(f"{name}: 盤面の指定が小さすぎるか順序が不正です(左上→右下の順にクリックしてください)")
            rects = [tuple(int(v) for v in r) for r in d["next_rects"]]
            for i, (_, _, w, h) in enumerate(rects):
                if w <= 0 or h <= 0:
                    raise ValueError(f"{name}: ネクスト枠{i + 1}の幅または高さが0以下です(左上→右下の順にクリックしてください)")
            return PlayerCalibration(board_quad=quad, next_rects=rects)  # type: ignore[arg-type]

        return CalibrationResult(self_player=player("自分", data["self"]), enemy_player=player("相手", data["enemy"]))

    def swapped(self) -> "CalibrationResult":
        """自分と相手を入れ替えたもの。2P(右の盤面)を操作する時は、対戦用キャリブレーション
        (自分=1P=左)をそのまま入れ替えて使う(画像認識・思考は自分/相手の割り当てだけで動くため)。"""
        return CalibrationResult(self_player=self.enemy_player, enemy_player=self.self_player)

    def self_on_right(self) -> bool:
        """自分の盤面が相手の盤面より右にあるか(2P を操作する時)。情報パネルを盤面の右に置く判断に使う。"""
        return self.self_player.geometry().bounding_box()[0] > self.enemy_player.geometry().bounding_box()[0]

    def capture_region(self) -> CaptureRegion:
        """自分・相手の盤面(予告行含む)とネクスト枠をすべて含む1つのキャプチャ範囲。"""
        boxes = [
            self.self_player.geometry().bounding_box(),
            self.enemy_player.geometry().bounding_box(),
            *self.self_player.next_rects,
            *self.enemy_player.next_rects,
        ]
        x0 = min(b[0] for b in boxes)
        y0 = min(b[1] for b in boxes)
        x1 = max(b[0] + b[2] for b in boxes)
        y1 = max(b[1] + b[3] for b in boxes)
        return CaptureRegion(left=x0, top=y0, width=x1 - x0, height=y1 - y0)


def load_calibration(path: Path = CONFIG_PATH) -> CalibrationResult:
    return CalibrationResult.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save_calibration(result: CalibrationResult, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


# 順にクリックしてもらうステップ。盤面の2点は自動検出で飛ばせる。
_STEPS: list[tuple[str, str]] = [
    ("self_board_tl", "【自分】盤面の左上(12段目の左上マスの角)をクリック  ※Aキーで盤面を自動検出"),
    ("self_board_br", "【自分】盤面の右下(1段目の右下マスの角)をクリック"),
    ("self_next1_tl", "【自分】ネクスト枠(大きい方)の左上をクリック"),
    ("self_next1_br", "【自分】ネクスト枠(大きい方)の右下をクリック"),
    ("self_next2_tl", "【自分】ネクネク枠(小さい方)の左上をクリック"),
    ("self_next2_br", "【自分】ネクネク枠(小さい方)の右下をクリック"),
    ("enemy_board_tl", "【相手】盤面の左上をクリック"),
    ("enemy_board_br", "【相手】盤面の右下をクリック"),
    ("enemy_next1_tl", "【相手】ネクスト枠(大きい方)の左上をクリック"),
    ("enemy_next1_br", "【相手】ネクスト枠(大きい方)の右下をクリック"),
    ("enemy_next2_tl", "【相手】ネクネク枠(小さい方)の左上をクリック"),
    ("enemy_next2_br", "【相手】ネクネク枠(小さい方)の右下をクリック"),
]

_ARROW_KEY_DELTAS: dict[QtCore.Qt.Key, tuple[int, int]] = {
    QtCore.Qt.Key.Key_Left: (-1, 0),
    QtCore.Qt.Key.Key_Right: (1, 0),
    QtCore.Qt.Key.Key_Up: (0, -1),
    QtCore.Qt.Key.Key_Down: (0, 1),
}


def adjust_rect(rect: tuple[int, int, int, int], key: QtCore.Qt.Key, resize: bool) -> tuple[int, int, int, int] | None:
    """矢印キーで矩形(x, y, w, h)を1px調整する。resize=False なら全体を移動、True なら右下の角だけを動かす。純粋関数。"""
    delta = _ARROW_KEY_DELTAS.get(key)
    if delta is None:
        return None
    dx, dy = delta
    x, y, w, h = rect
    if resize:
        return (x, y, max(1, w + dx), max(1, h + dy))
    return (x + dx, y + dy, w, h)


def rect_to_quad(rect: tuple[int, int, int, int]) -> Quad:
    x, y, w, h = rect
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))


class CalibrationOverlay(QtWidgets.QWidget):
    """全画面を覆う半透明ウィンドウ上で、指示に従ってクリックしてもらうためのUI

    solo=True(一人用モード)では相手側の指定を求めず、自分側の設定を相手側にも複製して保存する。
    """

    def __init__(self, solo: bool = False, save_path: Path = CONFIG_PATH) -> None:
        super().__init__()
        self._solo = solo
        self._save_path = save_path
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)

        screen = QtWidgets.QApplication.primaryScreen()
        self.setGeometry(screen.geometry())

        self._step_index = 0
        self._points: dict[str, tuple[int, int]] = {}
        # 盤面矩形の微調整フェーズ。(プレイヤー名, (x, y, w, h))。None なら通常フェーズ
        self._fine_tune: tuple[str, tuple[int, int, int, int]] | None = None

        self._label = QtWidgets.QLabel(self)
        self._label.setStyleSheet("background-color: rgba(0,0,0,190); color: white; font-size: 18px; padding: 12px;")
        self._label.move(40, 40)
        self._update_label()

    # ---- 表示 ----

    def _update_label(self) -> None:
        if self._fine_tune is not None:
            player, _ = self._fine_tune
            who = "自分" if player == "self" else "相手"
            self._label.setText(
                f"【{who}】緑の格子が盤面の6×12マスとぴったり重なるよう微調整してください。\n"
                "矢印キー: 全体を1px移動 / Shift+矢印キー: 右下の角を1px移動(大きさの調整)\n"
                "Enterで確定 (Escでキャンセル)"
            )
            self._label.adjustSize()
            return
        if self._step_index >= len(_STEPS):
            return
        _, message = _STEPS[self._step_index]
        total = 6 if self._solo else len(_STEPS)  # 一人用は自分側の6手順だけ
        mode = "一人用" if self._solo else "対戦用"
        self._label.setText(f"【{mode}】[{self._step_index + 1}/{total}] {message}\n(Escでキャンセル)")
        self._label.adjustSize()

    def _dpr(self) -> float:
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        return screen.devicePixelRatio()

    def _to_local(self, x: float, y: float) -> QtCore.QPointF:
        dpr = self._dpr()
        p = self.mapFromGlobal(QtCore.QPoint(round(x / dpr), round(y / dpr)))
        return QtCore.QPointF(p)

    # ---- 入力 ----

    def _start_fine_tune(self, player: str) -> None:
        x1, y1 = self._points[f"{player}_board_tl"]
        x2, y2 = self._points[f"{player}_board_br"]
        self._fine_tune = (player, (x1, y1, max(1, x2 - x1), max(1, y2 - y1)))
        self._update_label()
        self.update()

    def _advance(self) -> None:
        """現在のステップを終えた後の遷移。2点が揃った盤面は微調整へ、全て終われば保存。

        自動検出で相手盤面の2点も既に埋まっている場合はそのクリックを飛ばして直接微調整に入る。
        一人用モードでは相手側の手順を丸ごと飛ばし、自分側を複製する。
        """
        while self._step_index < len(_STEPS):
            key, _ = _STEPS[self._step_index]
            if self._solo and key.startswith("enemy_"):
                for name in ("board_tl", "board_br", "next1_tl", "next1_br", "next2_tl", "next2_br"):
                    self._points[f"enemy_{name}"] = self._points[f"self_{name}"]
                self._step_index = len(_STEPS)
                break
            if key.endswith("_board_tl"):
                player = key.split("_")[0]
                if all(f"{player}_board_{n}" in self._points for n in ("tl", "br")):
                    self._step_index += 2
                    self._start_fine_tune(player)
                    return
            break
        if self._step_index >= len(_STEPS):
            self._finish()
        else:
            self._update_label()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        if self._fine_tune is not None or self._step_index >= len(_STEPS):
            return
        key, _ = _STEPS[self._step_index]
        pos = event.globalPosition().toPoint()
        # Qtの論理座標 → 物理ピクセル座標(mssと同じ座標系)
        dpr = self._dpr()
        self._points[key] = (round(pos.x() * dpr), round(pos.y() * dpr))
        self._step_index += 1
        if key.endswith("_board_br"):
            self._start_fine_tune(key.split("_")[0])
            return
        self._advance()

    def _auto_detect(self) -> None:
        """画面全体をキャプチャし、盤面の暗い矩形から両盤面の4隅を検出して微調整へ進む。"""
        screen = self.screen() or QtWidgets.QApplication.primaryScreen()
        geo = screen.geometry()
        dpr = self._dpr()
        region = CaptureRegion(
            left=round(geo.x() * dpr), top=round(geo.y() * dpr), width=round(geo.width() * dpr), height=round(geo.height() * dpr)
        )
        # 自分自身(半透明オーバーレイ)が映り込まないよう一旦隠す
        self.hide()
        QtWidgets.QApplication.processEvents()
        try:
            with ScreenCapture() as cap:
                frame_rgb = cap.grab(region)
        finally:
            self.showFullScreen()
        frame_bgr = np.ascontiguousarray(frame_rgb[:, :, ::-1])
        quads = detect_board_quads(frame_bgr)
        need = 1 if self._solo else 2
        if len(quads) < need:
            self._label.setText(f"自動検出に失敗しました(見つかった盤面: {len(quads)})。盤面が空に近い状態で試すか、手動でクリックしてください。")
            self._label.adjustSize()
            return
        if self._solo:
            quads = [quads[0], quads[0]]  # 相手側は自分側の複製(使わない)
        for player, quad in (("self", quads[0]), ("enemy", quads[1])):
            (x1, y1), _, (x2, y2), _ = quad
            self._points[f"{player}_board_tl"] = (round(x1 + region.left), round(y1 + region.top))
            self._points[f"{player}_board_br"] = (round(x2 + region.left), round(y2 + region.top))
        # 自分の盤面の微調整から再開する(相手盤面の微調整は _advance が自動で挟む)
        self._step_index = 0
        self._advance()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.close()
            return
        if self._fine_tune is None:
            if event.key() == QtCore.Qt.Key.Key_A and self._step_index == 0:
                self._auto_detect()
            return
        player, rect = self._fine_tune
        if event.key() in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            x, y, w, h = rect
            self._points[f"{player}_board_tl"] = (x, y)
            self._points[f"{player}_board_br"] = (x + w, y + h)
            self._fine_tune = None
            self._advance()
            return
        resize = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        adjusted = adjust_rect(rect, event.key(), resize)
        if adjusted is not None:
            self._fine_tune = (player, adjusted)
            self.update()

    # ---- 描画 ----

    def paintEvent(self, event: QtCore.QEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        # 完全に透明な領域はクリックを拾えないことがあるため、ごく薄く塗る
        painter.fillRect(self.rect(), QtGui.QColor(0, 0, 0, 1))
        if self._fine_tune is not None:
            self._draw_fine_tune_grid(painter)
            painter.end()
            return
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 120, 255), 3))
        for x, y in self._points.values():
            painter.drawEllipse(self._to_local(x, y), 6, 6)
        painter.end()

    def _draw_fine_tune_grid(self, painter: QtGui.QPainter) -> None:
        assert self._fine_tune is not None
        _, rect = self._fine_tune
        geom = BoardGeometry(rect_to_quad(rect))
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 255, 120, 220), 1))
        for c in range(COLS + 1):
            p0 = geom.grid_to_screen(c, -1)
            p1 = geom.grid_to_screen(c, VISIBLE_ROWS)
            painter.drawLine(self._to_local(*p0), self._to_local(*p1))
        for r in range(-1, VISIBLE_ROWS + 1):
            p0 = geom.grid_to_screen(0, r)
            p1 = geom.grid_to_screen(COLS, r)
            painter.drawLine(self._to_local(*p0), self._to_local(*p1))
        # 左上と右下の角を強調
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80, 255), 3))
        x, y, w, h = rect
        painter.drawEllipse(self._to_local(x, y), 6, 6)
        painter.drawEllipse(self._to_local(x + w, y + h), 6, 6)

    # ---- 保存 ----

    def _finish(self) -> None:
        def normalized(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
            # クリック順が逆(右下→左上)でも矩形として成立させる
            x1, x2 = sorted((a[0], b[0]))
            y1, y2 = sorted((a[1], b[1]))
            return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

        def player(prefix: str) -> PlayerCalibration:
            quad = rect_to_quad(normalized(self._points[f"{prefix}_board_tl"], self._points[f"{prefix}_board_br"]))
            rects = [normalized(self._points[f"{prefix}_next{i}_tl"], self._points[f"{prefix}_next{i}_br"]) for i in (1, 2)]
            return PlayerCalibration(board_quad=quad, next_rects=rects)  # type: ignore[arg-type]

        result = CalibrationResult(self_player=player("self"), enemy_player=player("enemy"))
        save_calibration(result, self._save_path)
        print(f"キャリブレーション結果を保存しました: {self._save_path}")
        self.close()


def run_calibration(solo: bool = False, save_path: Path | None = None) -> None:
    """キャリブレーションを実行する(app.pyから呼ばれる場合はネストしたイベントループで待つ)。"""
    existing_app = QtWidgets.QApplication.instance()
    app = existing_app if existing_app is not None else QtWidgets.QApplication(sys.argv)

    overlay = CalibrationOverlay(solo=solo, save_path=save_path or (SOLO_CONFIG_PATH if solo else CONFIG_PATH))
    overlay.showFullScreen()

    if existing_app is None:
        app.exec()
    else:
        loop = QtCore.QEventLoop()
        overlay.destroyed.connect(loop.quit)
        loop.exec()


if __name__ == "__main__":
    run_calibration()
