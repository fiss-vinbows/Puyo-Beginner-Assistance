"""シミュレーションモードの練習ウィンドウ(PyQt6)。

支援モード(画像認識+オーバーレイ)とは別に、自前の盤面でぷよを置きながら ama の提案を見て練習する。
見た目は TBA(../Tetris)の練習画面(src/education/window.py)に合わせている:
- 上部に状態(手数・点数・連鎖)とキー割り当て
- 中央に盤面(13段目を薄く表示)、右にネクスト・ネクネク・提示の切替・推奨と操作手順
- 下部に「おじゃまを降らせる」「盤面編集」「ツモ順設定」「画像を保存」「一手戻す」「同一配列/別配列でリセット」

ゲームの規則と状態は rules.SimState が持ち、ここは描画と入力と ama の呼び出しだけを担当する。
ama は別スレッド(AdvisorThread)で呼び、盤面が変わった後に届いた古い結果は捨てる。
"""

from __future__ import annotations

import queue
import uuid
from dataclasses import dataclass
from datetime import datetime

from PyQt6 import QtCore, QtGui, QtWidgets

from ..engine.ama_client import HAIPUYO_BRIDGE_ARGS, FORM_TEXT_JA, AmaClient, BattleMove, NoValidPlacementError, PlayerState
from ..engine.puyo_field import COLS, EMPTY, GARBAGE, ROWS, PuyoField
from ..paths import app_root
from . import controls
from .opening_book import BookMove, OpeningBook
from .haipuyo import available as haipuyo_data_available
from .rules import SimState, parse_pairs

SCREENSHOT_DIR = app_root() / "screenshots"
CELL = 40  # 1マスの描画サイズ(px)
HIDDEN_ROWS = 1  # 13段目
CHAIN_STEP_MS = 380  # 連鎖の演出: 1段の表示時間
VISIBLE_PAIRS = 3  # 操作中+ネクスト+ネクネク(実機と同じく ama にはこの3組だけ見せる)

PUYO_RGB = {
    "R": (232, 64, 64),
    "Y": (240, 200, 40),
    "G": (64, 190, 90),
    "B": (64, 120, 235),
    GARBAGE: (175, 175, 185),
}
LABEL_TEXT_JA = {"build": "構築", "main": "本線発火", "harass": "潰し", "all_clear": "全消し", "accept": "受け", "prepare": "発火準備"}


def puyo_color(cell: str, alpha: int = 255) -> QtGui.QColor:
    r, g, b = PUYO_RGB[cell]
    return QtGui.QColor(r, g, b, alpha)


def draw_puyo(painter: QtGui.QPainter, rect: QtCore.QRectF, cell: str, alpha: int = 255) -> None:
    """ぷよを1個描く(色の丸だけ。目は描かない: 利用者の指示 2026-10-03)。"""
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(puyo_color(cell, alpha))
    painter.drawEllipse(rect.adjusted(2, 2, -2, -2))


@dataclass
class AdviceRequest:
    request_id: int
    rows: list[str]
    pairs: list[str]
    # 配ぷよモード(2026-10-26): 第1手からこれまでに置いた組ぷよ。None でなければ YamaNEKO に配ぷよ読みで考えさせる
    history: list[str] | None = None


# 推奨を出す思考エンジン: (名前, 表示名)。どちらも自身の既定で考える
ENGINES: list[tuple[str, str]] = [("ama", "ama"), ("yamaneko", "YamaNEKO")]


def _selected_book() -> OpeningBook | None:
    """メイン画面の「定跡の登録」で選んでいる定跡(2026-10-26)。無ければ・読めなければ組み込みの定跡。"""
    from ..engine.registry import Registry

    path = Registry.load().book_path()
    if path is not None:
        try:
            book = OpeningBook.load(path)
            if book is not None:
                return book
        except (ValueError, KeyError, TypeError):
            pass
    return OpeningBook.load()


class AdvisorThread(QtCore.QThread):
    """思考エンジン1つを別スレッドで呼ぶ。依頼は最新の1件だけ処理する。"""

    advice_ready = QtCore.pyqtSignal(str, int, object, str)  # (エンジン, 依頼番号, BattleMove または None, エラー文)

    def __init__(self, engine: str) -> None:
        super().__init__()
        self.engine = engine
        self._requests: queue.Queue[AdviceRequest | None] = queue.Queue()
        self._client: AmaClient | None = None
        self._haipuyo_client = False  # 今の _client を配ぷよ読みの起動引数で起動したか

    def request(self, req: AdviceRequest) -> None:
        self._requests.put(req)

    def stop(self) -> None:
        self._requests.put(None)

    def run(self) -> None:
        try:
            while True:
                req = self._requests.get()
                # 溜まっている依頼は最新だけ残す
                while not self._requests.empty():
                    nxt = self._requests.get()
                    if nxt is None:
                        return
                    req = nxt
                if req is None:
                    return
                try:
                    haipuyo = req.history is not None and self.engine == "yamaneko"
                    if self._client is not None and self._haipuyo_client != haipuyo:
                        self._client.close()  # 配ぷよ読みのあり・なしは起動引数で決まるので起動し直す
                        self._client = None
                    if self._client is None:
                        self._client = AmaClient(bridge=self.engine, extra_args=HAIPUYO_BRIDGE_ARGS if haipuyo else None)
                        self._haipuyo_client = haipuyo
                    field = PuyoField.from_rows(req.rows)
                    if haipuyo:
                        # 配ぷよ読みは対戦思考(battle)で使うので、相手は空の盤面にして呼ぶ(相手の攻撃は無い)
                        me = PlayerState(field, req.pairs, history=list(req.history), placed=len(req.history))
                        enemy = PlayerState(PuyoField(), req.pairs, placed=len(req.history))
                        move = self._client.battle(me, enemy)
                    else:
                        move = self._client.solo(field, req.pairs)
                    self.advice_ready.emit(self.engine, req.request_id, move, "")
                except NoValidPlacementError:
                    self.advice_ready.emit(self.engine, req.request_id, None, "置ける手がありません")
                except (FileNotFoundError, RuntimeError, TimeoutError, OSError) as exc:
                    # ブリッジが落ちたら次の依頼で起動し直す
                    if self._client is not None:
                        self._client.close()
                        self._client = None
                    self.advice_ready.emit(self.engine, req.request_id, None, f"呼び出しに失敗しました: {exc}")
        finally:
            if self._client is not None:
                self._client.close()


LINK_WIDTH_RATIO = 0.5  # 連結の線の太さ(マスの幅に対する割合)


def draw_links(painter: QtGui.QPainter, board: list[list[str]], cell_rect) -> None:  # noqa: ANN001
    """同じ色で上下左右に隣り合うぷよの中心どうしを、その色の太い線でつなぐ(ぷよの丸より先に描く)。"""
    for r in range(HIDDEN_ROWS, ROWS):
        for c in range(COLS):
            color = board[r][c]
            if color == EMPTY or color == GARBAGE:
                continue
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr < ROWS and nc < COLS and board[nr][nc] == color:
                    a, b = cell_rect(r, c), cell_rect(nr, nc)
                    pen = QtGui.QPen(puyo_color(color), a.width() * LINK_WIDTH_RATIO)
                    pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
                    painter.setPen(pen)
                    painter.drawLine(a.center(), b.center())


class BoardView(QtWidgets.QWidget):
    """盤面・操作中の組ぷよ・落下位置・推奨配置・連鎖の演出を描く。"""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.board: list[list[str]] = [[EMPTY] * COLS for _ in range(ROWS)]
        self.current: list[tuple[tuple[int, int], str]] = []  # 操作中の組ぷよ
        self.landing: list[tuple[tuple[int, int], str]] = []  # 落下位置
        self.suggestion: list[tuple[tuple[int, int], str]] = []  # 推奨配置
        self.popping: set[tuple[int, int]] = set()  # 連鎖の演出で消えるマス
        self.message = ""
        self.setFixedSize(COLS * CELL + 2, ROWS * CELL + 2)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def cell_rect(self, r: int, c: int) -> QtCore.QRectF:
        return QtCore.QRectF(1 + c * CELL, 1 + r * CELL, CELL - 1, CELL - 1)

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(20, 20, 24))
        for r in range(ROWS):
            for c in range(COLS):
                painter.fillRect(self.cell_rect(r, c), QtGui.QColor(34, 34, 40) if r >= HIDDEN_ROWS else QtGui.QColor(26, 26, 30))
        # 窒息マーク(3列目の12段目)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 80, 80, 90), 2))
        x_rect = self.cell_rect(1, 2).adjusted(10, 10, -10, -10)
        painter.drawLine(x_rect.topLeft(), x_rect.bottomRight())
        painter.drawLine(x_rect.topRight(), x_rect.bottomLeft())
        # 13段目(隠し段)との境界線
        painter.setPen(QtGui.QPen(QtGui.QColor(200, 200, 200, 120), 1, QtCore.Qt.PenStyle.DashLine))
        y = 1 + HIDDEN_ROWS * CELL
        painter.drawLine(0, y, self.width(), y)
        # 同じ色で隣り合うぷよを太い線でつなぐ(13段目は連結に数えないのでつながない。おじゃまもつながない)
        draw_links(painter, self.board, self.cell_rect)
        # 固定されたぷよ(13段目は薄く)
        for r in range(ROWS):
            for c in range(COLS):
                cell = self.board[r][c]
                if cell != EMPTY:
                    draw_puyo(painter, self.cell_rect(r, c), cell, 150 if r < HIDDEN_ROWS else 255)
                    if (r, c) in self.popping:
                        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255), 3))
                        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                        painter.drawEllipse(self.cell_rect(r, c).adjusted(1, 1, -1, -1))
        # 推奨配置: 薄い色に斜線の模様(白い破線の輪は利用者の指示で消した 2026-10-03)
        for (r, c), color in self.suggestion:
            if r < 0:
                continue
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(puyo_color(color, 90))
            painter.drawEllipse(self.cell_rect(r, c).adjusted(4, 4, -4, -4))
            painter.setBrush(QtGui.QBrush(puyo_color(color, 230), QtCore.Qt.BrushStyle.BDiagPattern))
            painter.drawEllipse(self.cell_rect(r, c).adjusted(4, 4, -4, -4))
        # 落下位置(輪郭だけ)
        for (r, c), color in self.landing:
            if r < 0:
                continue
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            painter.setPen(QtGui.QPen(puyo_color(color, 220), 3))
            painter.drawEllipse(self.cell_rect(r, c).adjusted(8, 8, -8, -8))
        # 操作中の組ぷよ(14段目以上は描かない)
        for (r, c), color in self.current:
            if r >= 0:
                draw_puyo(painter, self.cell_rect(r, c), color)
        if self.message:
            painter.setPen(QtGui.QColor(255, 90, 90))
            painter.setFont(QtGui.QFont("", 15, QtGui.QFont.Weight.Bold))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, self.message)


class PairView(QtWidgets.QWidget):
    """ネクスト・ネクネクの1枠に組ぷよ(子が上、軸が下)を描く。"""

    def __init__(self, scale: float = 1.0, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.pair = ""
        self.size_px = int(CELL * scale)
        self.setFixedSize(self.size_px + 24, self.size_px * 2 + 16)

    def set_pair(self, pair: str) -> None:
        self.pair = pair
        self.update()

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(30, 30, 36))
        if len(self.pair) != 2:
            return
        x = (self.width() - self.size_px) / 2
        top = (self.height() - self.size_px * 2) / 2
        draw_puyo(painter, QtCore.QRectF(x, top, self.size_px, self.size_px), self.pair[1])
        draw_puyo(painter, QtCore.QRectF(x, top + self.size_px, self.size_px, self.size_px), self.pair[0])


class ToggleSwitch(QtWidgets.QAbstractButton):
    """左右にスライドするトグルスイッチ(TBA の練習画面と同じ見た目)。"""

    TRACK_W = 40
    TRACK_H = 20

    def __init__(self, title: str, off_text: str, on_text: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.title = title
        self.off_text = off_text
        self.on_text = on_text
        self.setCheckable(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 盤面のキー操作を奪わない
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(200, 46)
        self.toggled.connect(lambda _v: self.update())

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        text_color = self.palette().color(QtGui.QPalette.ColorRole.WindowText)
        dim = QtGui.QColor(text_color)
        dim.setAlpha(110)
        painter.setPen(text_color)
        painter.drawText(QtCore.QRect(0, 0, self.width(), 20), QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, self.title)
        on = self.isChecked()
        bold = QtGui.QFont(self.font())
        bold.setBold(True)
        normal = QtGui.QFont(self.font())
        off_w = QtGui.QFontMetrics(bold).horizontalAdvance(self.off_text)
        y = 22
        painter.setFont(normal if on else bold)
        painter.setPen(dim if on else text_color)
        painter.drawText(QtCore.QRect(0, y, off_w, self.TRACK_H), QtCore.Qt.AlignmentFlag.AlignVCenter, self.off_text)
        track = QtCore.QRectF(off_w + 6, y, self.TRACK_W, self.TRACK_H)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(70, 170, 255) if on else QtGui.QColor(110, 110, 120))
        painter.drawRoundedRect(track, self.TRACK_H / 2, self.TRACK_H / 2)
        knob_x = track.right() - self.TRACK_H + 2 if on else track.left() + 2
        painter.setBrush(QtGui.QColor(255, 255, 255))
        painter.drawEllipse(QtCore.QRectF(knob_x, y + 2, self.TRACK_H - 4, self.TRACK_H - 4))
        painter.setFont(bold if on else normal)
        painter.setPen(text_color if on else dim)
        left = int(track.right()) + 6
        painter.drawText(QtCore.QRect(left, y, self.width() - left, self.TRACK_H), QtCore.Qt.AlignmentFlag.AlignVCenter, self.on_text)


class BoardEditor(QtWidgets.QWidget):
    """盤面編集用の盤面。クリック・ドラッグで選んだ色を塗る(右クリックで消す)。"""

    def __init__(self, board: list[list[str]], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.board = [row[:] for row in board]
        self.paint_value = "R"
        self._drag_value: str | None = None
        self.setFixedSize(COLS * CELL + 2, ROWS * CELL + 2)

    def _cell_at(self, pos: QtCore.QPointF) -> tuple[int, int] | None:
        r, c = int((pos.y() - 1) // CELL), int((pos.x() - 1) // CELL)
        return (r, c) if 0 <= r < ROWS and 0 <= c < COLS else None

    def _paint(self, pos: QtCore.QPointF) -> None:
        cell = self._cell_at(pos)
        if cell is not None and self._drag_value is not None:
            self.board[cell[0]][cell[1]] = self._drag_value
            self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drag_value = EMPTY if event.button() == QtCore.Qt.MouseButton.RightButton else self.paint_value
        self._paint(event.position())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._paint(event.position())

    def mouseReleaseEvent(self, _event: QtGui.QMouseEvent) -> None:  # noqa: N802
        self._drag_value = None

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(20, 20, 24))
        for r in range(ROWS):
            for c in range(COLS):
                rect = QtCore.QRectF(1 + c * CELL, 1 + r * CELL, CELL - 1, CELL - 1)
                painter.fillRect(rect, QtGui.QColor(60, 34, 34) if r < HIDDEN_ROWS else QtGui.QColor(34, 34, 40))
        draw_links(painter, self.board, lambda r, c: QtCore.QRectF(1 + c * CELL, 1 + r * CELL, CELL - 1, CELL - 1))
        for r in range(ROWS):
            for c in range(COLS):
                rect = QtCore.QRectF(1 + c * CELL, 1 + r * CELL, CELL - 1, CELL - 1)
                if self.board[r][c] != EMPTY:
                    draw_puyo(painter, rect, self.board[r][c])


class BoardEditDialog(QtWidgets.QDialog):
    """盤面編集。決定すると浮いているぷよは下に落ちる(連鎖は次に置いた時に起きる)。"""

    def __init__(self, board: list[list[str]], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("盤面編集")
        self.editor = BoardEditor(board)
        palette = QtWidgets.QVBoxLayout()
        palette.addWidget(QtWidgets.QLabel("塗る色\n(右クリックで消す)"))
        group = QtWidgets.QButtonGroup(self)
        for value, text in (("R", "赤"), ("Y", "黄"), ("G", "緑"), ("B", "青"), (GARBAGE, "おじゃま"), (EMPTY, "消す")):
            radio = QtWidgets.QRadioButton(text)
            if value in PUYO_RGB:
                radio.setStyleSheet(f"color: {puyo_color(value).name()}; font-weight: bold;")
            radio.toggled.connect(lambda on, v=value: on and setattr(self.editor, "paint_value", v))
            radio.setChecked(value == "R")
            group.addButton(radio)
            palette.addWidget(radio)
        clear_btn = QtWidgets.QPushButton("全部消す")
        clear_btn.clicked.connect(self._clear)
        palette.addWidget(clear_btn)
        palette.addStretch(1)
        body = QtWidgets.QHBoxLayout()
        body.addWidget(self.editor)
        body.addLayout(palette)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(body)
        layout.addWidget(buttons)

    def _clear(self) -> None:
        self.editor.board = [[EMPTY] * COLS for _ in range(ROWS)]
        self.editor.update()


class PairsEditDialog(QtWidgets.QDialog):
    """ツモ順設定。操作中の組ぷよから先を「軸→子」の2文字で並べる(その先は乱数で続く)。"""

    def __init__(self, pairs: list[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ツモ順設定")
        self.pairs: list[str] = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "操作中の組ぷよから順に、1組を「軸(下)→子(上)」の2文字で書きます。\n"
                "R=赤 Y=黄 G=緑 B=青。空白・改行・カンマ区切り。書いた分より先は乱数で続きます。"
            )
        )
        self.edit = QtWidgets.QPlainTextEdit(" ".join(pairs))
        self.edit.setFont(QtGui.QFont("Consolas", 12))
        layout.addWidget(self.edit)
        self.error = QtWidgets.QLabel()
        self.error.setStyleSheet("color: #ff6060;")
        layout.addWidget(self.error)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(460, 260)

    def _on_ok(self) -> None:
        try:
            self.pairs = parse_pairs(self.edit.toPlainText())
        except ValueError as exc:
            self.error.setText(str(exc))
            return
        if not self.pairs:
            self.error.setText("1組以上書いてください")
            return
        self.accept()


class SimulatorWindow(QtWidgets.QWidget):
    """シミュレーションモードの本体。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PBA シミュレーター")
        self.state = SimState.new()
        self.advices: dict[str, BattleMove | None] = {name: None for name, _ in ENGINES}
        self.errors: dict[str, str] = {name: "" for name, _ in ENGINES}
        self._request_id = 0
        self._chain_frames: list = []  # 連鎖の演出の残り (盤面, 消えるマス)
        self._chain_timer = QtCore.QTimer(self)
        self._chain_timer.timeout.connect(self._next_chain_frame)
        self.advisors: dict[str, AdvisorThread] = {}
        for name, _label in ENGINES:
            thread = AdvisorThread(name)
            thread.advice_ready.connect(self._on_advice, QtCore.Qt.ConnectionType.QueuedConnection)
            thread.start()
            self.advisors[name] = thread

        # 入力: キーボード・コントローラーの割り当てと、押しっぱなしの連続入力
        self.key_bindings = controls.load_bindings()
        self.pad_bindings = controls.load_pad_bindings()
        self.gamepad = controls.Gamepad()
        self._held_keys: dict[int, str] = {}  # 押しているキー(キーコード) → 操作名
        self._repeat = controls.RepeatTracker()
        self._pad_dialog: controls.PadMappingDialog | None = None
        self._pad_retry_ms = 0.0
        self._clock = QtCore.QElapsedTimer()
        self._clock.start()

        # 上部: 状態とキー割り当て
        self.status_label = QtWidgets.QLabel()
        self.status_label.setStyleSheet("font-size: 13px;")
        self.keys_label = QtWidgets.QLabel()
        self.keys_label.setWordWrap(True)
        self.keys_label.setStyleSheet("color: gray;")

        # 左: 連鎖・点数
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("連鎖"))
        self.chain_label = QtWidgets.QLabel()
        self.chain_label.setStyleSheet("font-weight: bold; font-size: 15px;")
        self.chain_label.setWordWrap(True)
        self.chain_label.setFixedWidth(170)
        self.chain_label.setMinimumHeight(24)
        left.addWidget(self.chain_label)
        left.addSpacing(16)
        left.addWidget(QtWidgets.QLabel("発火した場合"))
        self.analysis_label = QtWidgets.QLabel()
        self.analysis_label.setWordWrap(True)
        self.analysis_label.setFixedWidth(170)
        self.analysis_label.setMinimumHeight(60)
        self.analysis_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        left.addWidget(self.analysis_label)
        left.addStretch(1)

        self.board_view = BoardView()

        # 右: ネクスト・ネクネク・切替・推奨
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("NEXT"))
        self.next_views = [PairView(1.0), PairView(0.75)]
        for view in self.next_views:
            right.addWidget(view)
        right.addSpacing(12)
        self.show_toggle = ToggleSwitch("提示 (F3)", "非表示", "表示")
        self.show_toggle.setChecked(True)
        self.show_toggle.toggled.connect(lambda _v: self._refresh())
        right.addWidget(self.show_toggle)
        self.steps_toggle = ToggleSwitch("操作手順", "非表示", "表示")
        self.steps_toggle.setChecked(True)
        self.steps_toggle.toggled.connect(lambda _v: self._refresh())
        right.addWidget(self.steps_toggle)
        # 6手目までの定跡(Puyo-YamaNEKO の opening_tree.exe --export。assets/opening_6moves.json)
        self.opening_book = _selected_book()
        self.book_toggle = ToggleSwitch("定跡 (1〜6手目)", "使わない", "使う")
        self.book_toggle.setChecked(False)  # 中身の良し悪しが未検証なので既定は使わない(2026-10-15)
        self.book_toggle.setEnabled(self.opening_book is not None)
        self.book_toggle.toggled.connect(lambda _v: self._refresh())
        right.addWidget(self.book_toggle)
        # 実機の配ぷよ(65536通り)で配る(2026-10-26)。オンにすると YamaNEKO は配ぷよ読みで考える
        self.haipuyo_toggle = ToggleSwitch("配ぷよ", "乱数", "実機の配ぷよ")
        self.haipuyo_toggle.setChecked(False)
        self.haipuyo_toggle.setEnabled(haipuyo_data_available())
        self.haipuyo_toggle.toggled.connect(self._on_haipuyo_toggled)
        right.addWidget(self.haipuyo_toggle)
        # 盤面のドットと操作手順に使う AI(もう一方は推奨の欄に並べて出す)
        self.engine_toggle = ToggleSwitch("ドットで示す AI", "ama", "YamaNEKO")
        self.engine_toggle.setChecked(True)  # 初期値は YamaNEKO(PBA の思考エンジン)
        self.engine_toggle.toggled.connect(lambda _v: self._refresh())
        right.addWidget(self.engine_toggle)
        right.addSpacing(8)
        self.advice_label = QtWidgets.QLabel()
        self.advice_label.setWordWrap(True)
        self.advice_label.setFixedWidth(220)
        self.advice_label.setMinimumHeight(260)
        self.advice_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        right.addWidget(self.advice_label)
        right.addStretch(1)

        body = QtWidgets.QHBoxLayout()
        body.addLayout(left)
        body.addWidget(self.board_view, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        body.addSpacing(16)
        body.addLayout(right)

        # 下部: おじゃま・編集・保存、一手戻す・リセット
        tools = QtWidgets.QHBoxLayout()
        self.garbage_spin = QtWidgets.QSpinBox()
        self.garbage_spin.setRange(1, 72)
        self.garbage_spin.setValue(6)
        self.garbage_spin.setSuffix(" 個")
        tools.addWidget(self.garbage_spin)
        for text, slot in (
            ("おじゃまを降らせる", self._drop_garbage),
            ("盤面編集", self._edit_board),
            ("ツモ順設定", self._edit_pairs),
            ("画像を保存", self._save_screenshot),
            ("キーボードの割り当て", self._edit_keys),
            ("コントローラーの割り当て", self._edit_pad),
        ):
            tools.addWidget(self._button(text, slot))
        tools.addStretch(1)
        resets = QtWidgets.QHBoxLayout()
        resets.addWidget(self._button("一手戻す", self._undo))
        resets.addWidget(self._button("同一配列でリセット", lambda: self._reset(True)))
        resets.addStretch(1)
        resets.addWidget(self._button("別配列でリセット", lambda: self._reset(False)))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.keys_label)
        layout.addLayout(body)
        layout.addLayout(tools)
        layout.addLayout(resets)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.timeout.connect(self._poll_inputs)
        self._poll_timer.start(controls.POLL_MS)
        self._refresh()
        self._request_advice()

    def _button(self, text: str, slot) -> QtWidgets.QPushButton:  # noqa: ANN001
        btn = QtWidgets.QPushButton(text)
        btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # 盤面のキー操作を奪わない
        btn.clicked.connect(slot)
        return btn

    # ---- 思考エンジン ----
    @property
    def selected_engine(self) -> str:
        return "yamaneko" if self.engine_toggle.isChecked() else "ama"

    @property
    def advice(self) -> BattleMove | None:
        """ドットで示す AI の推奨。"""
        return self.advices[self.selected_engine]

    def _request_advice(self) -> None:
        self._request_id += 1
        for name, _label in ENGINES:
            self.advices[name] = None
            self.errors[name] = ""
            if not self.state.game_over:
                self.advisors[name].request(
                    AdviceRequest(
                        self._request_id,
                        self.state.field().to_rows(),
                        self.state.upcoming(VISIBLE_PAIRS),
                        history=list(self.state.pairs[: self.state.turn]) if self.state.haipuyo is not None else None,
                    )
                )
        self._refresh()

    def _on_advice(self, engine: str, request_id: int, move: BattleMove | None, error: str) -> None:
        if request_id != self._request_id:
            return  # 盤面が変わった後に届いた古い結果
        self.advices[engine] = move
        self.errors[engine] = error
        self._refresh()

    def _advice_lines(self) -> list[str]:
        """両方の AI の推奨を1行ずつ(ドットで示している方に ●)。"""
        lines = []
        for name, label in ENGINES:
            mark = "● " if name == self.selected_engine else "○ "
            a = self.advices[name]
            if a is not None:
                text = f"{a.x + 1}列目・子が{_rotation_ja(a.rotation)}({LABEL_TEXT_JA.get(a.label, a.label)})"
            elif self.errors[name]:
                text = f"<span style='color:#ff6060'>{self.errors[name]}</span>"
            else:
                text = "<span style='color:gray'>思考中…</span>"
            lines.append(f"{mark}<b>{label}</b>: {text}")
        return lines

    # ---- 表示 ----
    def _refresh(self) -> None:
        s = self.state
        animating = bool(self._chain_frames) or self._chain_timer.isActive()
        self.status_label.setText(
            f"手数: {s.turn}   点数: {s.score:,}   最大連鎖: {s.max_chain}   "
            + (f"配ぷよ: #{s.haipuyo}" if s.haipuyo is not None else f"配列: #{s.seed}")
            + ("   全消しボーナスあり" if s.all_clear else "")
            + f"   コントローラー: {'接続中' if self.gamepad.connected() else '未接続'}"
        )
        self.keys_label.setText(controls.bindings_help(self.key_bindings) + "  (割り当ては下のボタンで変更できます)")
        last = s.last_chain
        self.chain_label.setText(f"{last.count}連鎖  {last.score:,}点" if last and last.count else "-")
        upcoming = s.upcoming(VISIBLE_PAIRS)
        for view, pair in zip(self.next_views, upcoming[1:]):
            view.set_pair(pair)
        view = self.board_view
        if not animating:
            view.board = s.board
            view.popping = set()
        view.message = "ばたんきゅ〜\n(一手戻す / リセット)" if s.game_over and not animating else ""
        playing = not s.game_over and not animating
        view.current = s.current_cells() if playing else []
        view.landing = s.landing_cells() if playing else []
        view.suggestion = []
        lines: list[str] = []
        book = self._book_move() if playing else None
        if self.show_toggle.isChecked() and playing:
            if book is not None:
                # 定跡(YamaNEKO の6手目までの解析)どおりに進んでいる間は定跡の手を出す
                view.suggestion = sorted(book.cells)
                lines.append(f"<b>推奨: 定跡 {book.pattern} の{book.turn}手目</b>")
                lines.append("判断: 定跡(YamaNEKO・6手目までの全通りの解析)")
                if self.steps_toggle.isChecked():
                    lines.append("操作手順(今の位置から):")
                    lines.extend(self._step_lines(set(book.cells)))
                lines.append("参考:")
                lines.extend(self._advice_lines())
            else:
                lines.extend(self._advice_lines())
                a = self.advice
                if a is not None:
                    pair = s.current
                    view.suggestion = [(a.axis_cell, pair[0]), (a.child_cell, pair[1])]
                    if a.form:
                        lines.append(f"形: {FORM_TEXT_JA.get(a.form, a.form)}")
                    if self.steps_toggle.isChecked():
                        lines.append("操作手順(今の位置から):")
                        lines.extend(self._step_lines({(a.axis_cell, pair[0]), (a.child_cell, pair[1])}))
        self.advice_label.setText("<br>".join(lines))
        analysis = self.advice.self_analysis if self.advice is not None else None
        if analysis is not None and self.show_toggle.isChecked():
            self.analysis_label.setText(
                f"今の組ぷよで: {_attack_text(analysis.immediate)}<br>"
                f"ネクストまでで: {_attack_text(analysis.max)}"
            )
        else:
            self.analysis_label.setText("")
        view.update()

    def _book_move(self) -> BookMove | None:
        """定跡を使う設定で、定跡どおりに進んでいれば次の手。見えているツモ(操作中+2組)だけで引く。"""
        if self.opening_book is None or not self.book_toggle.isChecked():
            return None
        s = self.state
        return self.opening_book.lookup(s.board, s.pairs[: s.turn + VISIBLE_PAIRS], s.turn)

    def _step_lines(self, target: set[tuple[tuple[int, int], str]]) -> list[str]:
        """提案の2マスまでの手順(このシミュレーターの規則で探索。回し・クイックターンを含む)。"""
        actions = self.state.find_actions(target)
        if actions is None:
            return ["<span style='color:#ff6060'>今の位置からは届きません(一手戻すかやり直してください)</span>"]
        return step_texts(actions, self.key_bindings)

    # ---- 連鎖の演出 ----
    def _start_chain_animation(self) -> None:
        last = self.state.last_chain
        if last is None or not last.count:
            return
        frames = []
        for step in last.steps:
            frames.append((step.board, set()))
            frames.append((step.board, step.popped))
        self._chain_frames = frames
        self._next_chain_frame()
        self._chain_timer.start(CHAIN_STEP_MS // 2)

    def _next_chain_frame(self) -> None:
        if not self._chain_frames:
            self._chain_timer.stop()
            self._refresh()
            return
        board, popped = self._chain_frames.pop(0)
        self.board_view.board = board
        self.board_view.popping = popped
        self._refresh()

    def _stop_animation(self) -> None:
        self._chain_frames = []
        self._chain_timer.stop()

    # ---- 操作 ----
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if event.isAutoRepeat():
            return  # OS のキーリピートは使わない(連続入力は RepeatTracker で行う)
        name = controls.key_event_name(event)
        action = controls.action_for(self.key_bindings, name) if name else None
        if action is None:
            super().keyPressEvent(event)
            return
        self._held_keys[event.key()] = action
        self._poll_inputs()  # 押した瞬間に反応させる

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if not event.isAutoRepeat():
            self._held_keys.pop(event.key(), None)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:  # noqa: N802
        self._held_keys.clear()  # 離したことを受け取れないので押しっぱなしを残さない
        super().focusOutEvent(event)

    def _poll_inputs(self) -> None:
        """キーボードとコントローラーの入力をまとめて操作にする(POLL_MS ごと+キーを押した瞬間)。"""
        now = self._clock.elapsed()
        if not self.gamepad.connected() and now >= self._pad_retry_ms:
            self._pad_retry_ms = now + 2000  # 後から挿したコントローラーを2秒ごとに探す
            if self.gamepad.find():
                self._refresh()
        was_connected = self.gamepad.connected()
        pad_inputs = self.gamepad.poll()
        if was_connected and not self.gamepad.connected():
            self._refresh()
        if self._pad_dialog is not None:
            self._pad_dialog.feed(pad_inputs)
            self._repeat.clear()
            return
        if QtWidgets.QApplication.activeModalWidget() is not None or not self.isActiveWindow():
            self._repeat.clear()
            return
        actions = set(self._held_keys.values()) | controls.actions_for_inputs(self.pad_bindings, pad_inputs)
        self._repeat.update(actions, now, self._perform)

    def _perform(self, action: str) -> None:
        animating = bool(self._chain_frames) or self._chain_timer.isActive()
        if action == "toggle_hints":
            self.show_toggle.toggle()
        elif action == "screenshot":
            self._save_screenshot()
        elif action == "undo":
            self._undo()
        elif action in ("reset_same", "reset_new"):
            self._reset(action == "reset_same")
        elif animating:
            # 連鎖の演出中は置く操作だけ受け付けて演出を飛ばす
            if action == "drop":
                self._stop_animation()
                self._refresh()
            return
        elif action == "move_left":
            self.state.move(-1)
        elif action == "move_right":
            self.state.move(1)
        elif action == "rotate_ccw":
            self.state.rotate(-1)
        elif action == "rotate_cw":
            self.state.rotate(1)
        elif action == "drop":
            if self.state.drop() is not None:
                self._request_advice()
                self._start_chain_animation()
        self._refresh()

    def _edit_keys(self) -> None:
        dialog = controls.KeyMappingDialog(self.key_bindings, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.key_bindings = dialog.bindings
            controls.save_bindings(self.key_bindings)
        self._held_keys.clear()
        self._refresh()

    def _edit_pad(self) -> None:
        if not self.gamepad.connected():
            self.gamepad.find()
        self._pad_dialog = controls.PadMappingDialog(self.pad_bindings, self)
        if not self.gamepad.connected():
            self._pad_dialog.message.setText("コントローラーが見つかりません。接続してから「設定」を押してください。")
        try:
            if self._pad_dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
                self.pad_bindings = self._pad_dialog.bindings
                controls.save_pad_bindings(self.pad_bindings)
        finally:
            self._pad_dialog = None
        self._refresh()

    def _undo(self) -> None:
        self._stop_animation()
        if self.state.undo():
            self._request_advice()

    def _on_haipuyo_toggled(self, on: bool) -> None:
        """配ぷよのあり・なしを切り替えたら、新しい配列で最初から(実機の配ぷよなら65536通りから1つ)。"""
        self._stop_animation()
        self.state = SimState.new(haipuyo=on)
        self._request_advice()

    def _reset(self, same_sequence: bool) -> None:
        self._stop_animation()
        self.state = self.state.reset(same_sequence)
        self._request_advice()

    def _drop_garbage(self) -> None:
        self._stop_animation()
        self.state.drop_garbage(self.garbage_spin.value())
        self._request_advice()

    def _edit_board(self) -> None:
        self._stop_animation()
        dialog = BoardEditDialog(self.state.board, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.state.set_board(dialog.editor.board)
            self._request_advice()

    def _edit_pairs(self) -> None:
        self._stop_animation()
        dialog = PairsEditDialog(self.state.upcoming(16), self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.state.set_upcoming(dialog.pairs)
            self._request_advice()

    def _save_screenshot(self) -> None:
        SCREENSHOT_DIR.mkdir(exist_ok=True)
        path = SCREENSHOT_DIR / f"sim_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}.png"
        self.grab().save(str(path))
        self.status_label.setText(self.status_label.text() + f"   保存しました: {path.name}")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._stop_animation()
        self._poll_timer.stop()
        for thread in self.advisors.values():
            thread.stop()
        for thread in self.advisors.values():
            thread.wait(3000)
        super().closeEvent(event)


def step_texts(actions: list[str], bindings: dict[str, str]) -> list[str]:
    """操作列を「1. 右へ移動 (Right) ×2」のような番号付きの手順にする(同じ操作が続いたらまとめる)。
    最後に「置く」を足す。キー名は利用者の割り当てを使う。"""
    out: list[str] = []
    i = 0
    actions = list(actions) + ["drop"]
    while i < len(actions):
        j = i
        while j < len(actions) and actions[j] == actions[i]:
            j += 1
        key = bindings.get(actions[i]) or "未割り当て"
        count = f" ×{j - i}" if j - i > 1 else ""
        out.append(f"{len(out) + 1}. {controls.ACTIONS[actions[i]]} ({key}){count}")
        i = j
    return out


def _rotation_ja(rotation: str) -> str:
    return {"UP": "上", "RIGHT": "右", "DOWN": "下", "LEFT": "左"}.get(rotation, rotation)


def _attack_text(attack) -> str:  # noqa: ANN001
    return f"{attack.count}連鎖 {attack.score:,}点" if attack.count else "なし"

