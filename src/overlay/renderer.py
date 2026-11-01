"""最善手オーバーレイの描画ロジック

QPainterに対する純粋な描画処理のみを担当し、実ウィンドウへの描画・
オフスクリーン画像への描画のどちらからも共用できる。

描くもの:
  - 軸ぷよ・子ぷよの着地マス: それぞれのぷよの色で塗ったドット。軸は太い白縁、
    子は細い縁で区別し、2点を線で結ぶ(どちらが軸か一目で分かるように)
  - 2手先(ネクストの組ぷよ)の着地マス: 小さめのドットに「2」を添えて表示。人間が
    提案を見てから操作に移る時間を稼ぐため(実機フィードバック 2026-09-14)
  - 情報パネル: 自分の盤面の左枠外(下端に揃える)に置く。盤面に重ねる案は
    ぷよが見づらいため却下された(2026-09-14)
  - 最終的な構想(2026-10-24、配ぷよ読み+全自動の時だけ): これから置く予定のマスに小さな薄いドット、
    発火の組ぷよのマスは白い二重の縁で示す
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PyQt6 import QtCore, QtGui

from ..capture.geometry import BoardGeometry
from ..engine.ama_client import LABEL_TEXT_JA, FieldAnalysis
from ..engine.puyo_field import COLS, VISIBLE_ROWS
from ..vision.cell_classifier import EMPTY, GARBAGE
from .panel_text_en import to_english

# 情報パネルの表記: キー → (1行の置き換え, 空白で折り返すか)。日本語("ja")はそのまま。英語は標準、ほかは言語プラグインが
# register_language で足す(2026-10-28、mokou を言語プラグインへ)
_LANGUAGES: dict[str, tuple] = {"en": (to_english, True)}


def register_language(key: str, transform, word_break: bool = False) -> None:  # noqa: ANN001
    """情報パネルの表記を足す(言語プラグイン用)。"""
    _LANGUAGES[key] = (transform, word_break)


def _translate(text: str, language: str) -> str:
    lang = _LANGUAGES.get(language)
    return lang[0](text) if lang is not None and text else text


def _word_break(language: str) -> bool:
    lang = _LANGUAGES.get(language)
    return bool(lang and lang[1])

# ゲーム上の色名 → 描画色
PUYO_DRAW_COLORS: dict[str, QtGui.QColor] = {
    "red": QtGui.QColor(235, 60, 60),
    "yellow": QtGui.QColor(245, 210, 40),
    "green": QtGui.QColor(70, 210, 80),
    "blue": QtGui.QColor(60, 130, 245),
    "purple": QtGui.QColor(170, 80, 220),
}
UNKNOWN_DRAW_COLOR = QtGui.QColor(200, 200, 200)

AXIS_DOT_RADIUS_RATIO = 0.26  # セルサイズに対する軸ぷよドットの半径
CHILD_DOT_RADIUS_RATIO = 0.20
AXIS_OUTLINE_WIDTH = 4
CHILD_OUTLINE_WIDTH = 2
LINK_LINE_WIDTH = 3
# 2手先は一回り小さく、薄めに描く
NEXT_AXIS_DOT_RADIUS_RATIO = 0.17
NEXT_CHILD_DOT_RADIUS_RATIO = 0.13
NEXT_DOT_ALPHA = 170
# 最終的な構想: 今の盤面に無く、構想の盤面にあるマスに描く小さなドット
PLAN_DOT_RADIUS_RATIO = 0.12
PLAN_DOT_ALPHA = 150
PLAN_TRIGGER_RADIUS_RATIO = 0.17

# 発火点の枠(相手の仕掛けに対応するとき、ここに何色を置けば本線が発火するかを示す)

# 形勢判断: 「自分の本線のおじゃま − 自分への予告」と「相手の本線のおじゃま − 相手への予告」の差(個)。
# この個数以上なら有利(青)、以下なら不利(赤)、間は互角(白)。1段=6個なので2段ぶんを目安にする

PANEL_FONT_PX = 14
PANEL_PADDING = 6
PANEL_ALPHA = 170
PANEL_MIN_WIDTH = 120  # 盤面の左に余裕が無くてもこれ以上には縮めない(論理px)
# パネルの大きさは文言によらず固定(利用者の指定 2026-10-25。行数・文字数で枠が伸び縮みしないように)。
# 幅は文字の部分の幅(盤面の横の余白が狭ければそれに合わせて縮める)、高さは PANEL_SLOTS の行数の合計
PANEL_TEXT_WIDTH = 240
# 情報の種類ごとの行の位置と行数(上から順)。情報が無い種類は空白にして、他の行の位置を動かさない
# (情報が増えると行が間に入って見づらい 2026-10-25)。行数を超える分は最後の行の末尾を「…」で省く
PANEL_SLOTS: list[tuple[str, int]] = [
    ("move", 2),  # 提案
    ("fire", 1),  # この手で N連鎖
    ("note", 2),  # メモ・再計算の理由
    ("form", 1),  # 開幕の形
    ("plan", 1),  # 構想
    ("self", 1),  # 自分の最大連鎖(青)
    ("self_sub1", 1),  # 自分の補足(今撃つと・組んでいる・潰しの順に、入る分だけ)
    ("enemy", 1),  # 相手の最大連鎖・発火中(濃いピンク)
    ("enemy_sub1", 1),  # 相手の補足(同上)
    ("enemy_sub2", 1),
]
PANEL_LINES = sum(rows for _, rows in PANEL_SLOTS)
# 自分の連鎖の消化中に空白にする種類(組んでいる連鎖の情報。連鎖中は盤面が崩れていて意味が無い)
CHAINING_HIDDEN_SLOTS = ("fire", "plan", "self", "self_sub1")
# パネルと盤面左端の間隔(論理px)
PANEL_GAP = 12
# パネルの文字色。青・ピンクは半透明の黒の背景で読める明るさにする
PANEL_TEXT_COLOR = QtGui.QColor(255, 255, 255)
PANEL_SELF_COLOR = QtGui.QColor(80, 160, 255)
PANEL_ENEMY_COLOR = QtGui.QColor(255, 40, 150)


@dataclass
class OverlayDrawData:
    """1フレームぶんの描画内容。座標はすべて PuyoField の (row, col)。"""

    axis_cell: tuple[int, int]
    child_cell: tuple[int, int]
    axis_color: str  # ゲーム上の色名
    child_color: str
    label: str  # ama_client.BattleMove.label
    self_analysis: FieldAnalysis | None = None
    enemy_analysis: FieldAnalysis | None = None
    note: str = ""  # 追加の短いメッセージ(認識中・エラー等)
    form: str = ""  # 開幕の積み方の狙い・移行(「狙い: 弥生時代」等)
    fire_chain: tuple[int, int] = (0, 0)  # 提案通りに置くと起きる連鎖 (連鎖数, 点数)。0なら無し
    # 2手先(ネクストの組ぷよ)の提案。提案通りに置いた後の盤面での着地マス
    next_axis_cell: tuple[int, int] | None = None
    next_child_cell: tuple[int, int] | None = None
    next_axis_color: str = ""
    next_child_color: str = ""
    # 手番が切り替わって新しい提案を計算中(前の手番のドットは置いたツモの色なので出さない)。パネルは残す
    waiting: bool = False
    action: str = ""  # 何をすべきか(ama_client.action_text。空ならラベルの名前を出す)
    # 最終的な構想(plan_dots の結果)と、パネルに出す説明(「構想: 12連鎖 66,660点(14手目に撃つ)」)
    plan_dots: list[tuple[int, int, str, bool]] = field(default_factory=list)
    plan_text: str = ""
    # 相手が撃っている(撃った)連鎖 (連鎖数, 点数, 連鎖中か)。点滅から再現した値で、再現できなければ連鎖数 0。
    # None なら相手の解析(撃てる最大)を出す。相手の連鎖中も撃つ前の盤面の解析(先の手で撃てる最大)が残り、
    # 撃っている連鎖と違う数を出していた(実機 2026-10-25 12:23: 6連鎖を撃っている間「9連鎖」のまま)
    enemy_firing: tuple[int, int, bool] | None = None
    # 自分が撃っている(撃った)連鎖。形は enemy_firing と同じ。自分の連鎖中も自分の行に出す(利用者 2026-10-25「現在発火している
    # 点数は表示しておいて。誤った点数の場合は訂正」)
    self_firing: tuple[int, int, bool] | None = None
    # 自分の連鎖を消化中。組んでいる連鎖の情報(この手で・構想・自分の最大連鎖と潰し)は空白にする(利用者 2026-10-25)
    chaining: bool = False


def plan_dots(plan_cells: list[list[str]], settled: list[list[str]], trigger: list[tuple[int, int]]) -> list[tuple[int, int, str, bool]]:
    """構想の盤面(色名の13行)のうち、今の盤面(settled)で空のマスを (row, col, 色名, 発火の組ぷよか) で返す。
    おじゃまは描かない。今の盤面にぷよがあるマスは描かない(本物のぷよが見えている)"""
    trig = set(trigger)
    out = []
    for r, row in enumerate(plan_cells):
        for c, color in enumerate(row):
            if color in (EMPTY, GARBAGE) or r >= len(settled) or settled[r][c] != EMPTY:
                continue
            out.append((r, c, color, (r, c) in trig))
    return out


@dataclass
class OverlayLayout:
    """オーバーレイウィンドウ内のローカル(論理ピクセル)座標系への変換。

    origin は物理ピクセルでのウィンドウ左上、dpr は devicePixelRatio。
    BoardGeometry は物理ピクセル座標で作られているので、ここで変換する。
    """

    self_geometry: BoardGeometry
    enemy_geometry: BoardGeometry
    origin: tuple[float, float]
    dpr: float
    # 情報パネルを盤面の右に置く時の、盤面右端からウィンドウ右端までの幅(論理px)。None なら盤面の左に置く。
    # 2P(右の盤面)を操作する時、盤面の左は画面中央のネクスト枠に重なるので外側(右)に置く(2026-10-12)
    panel_right_space: float | None = None
    # 情報パネル・CPU/FPS の行の表記。"ja"(日本語)か "en"(英語、panel_text_en で置き換える)
    language: str = "ja"

    def to_local(self, x: float, y: float) -> QtCore.QPointF:
        return QtCore.QPointF((x - self.origin[0]) / self.dpr, (y - self.origin[1]) / self.dpr)


def _draw_dot(
    painter: QtGui.QPainter, center: QtCore.QPointF, radius: float, color: QtGui.QColor, outline: int, alpha: int = 255
) -> None:
    fill = QtGui.QColor(color)
    fill.setAlpha(alpha)
    painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, min(230, alpha)), outline))
    painter.setBrush(QtGui.QBrush(fill))
    painter.drawEllipse(center, radius, radius)


def _format_analysis(a: FieldAnalysis | None, firing: tuple[int, int, bool] | None = None, show_now: bool = False) -> list[str]:
    # 自分・相手は文字色で分かるので「自分」「相手」の見出しとおじゃまの個数は出さない(文字数を減らす 2026-10-25)
    if firing is not None:
        steps, score, active = firing
        head = "発火中" if active else "発火"
        if steps <= 0:
            return [f"{head}(段数は再現できず)"]
        return [f"{head} {steps}連鎖 {score:,}点" if score > 0 else f"{head} {steps}連鎖(点数は再現できず)"]
    if a is None:
        return ["--"]
    # 撃てる最大が先の手(ネクスト・ネクネクを置いてから)で撃つ連鎖なら「@何手目」を添え、show_now なら今撃てる連鎖も出す
    # (実機 2026-10-25 12:23: 相手の「9連鎖」は3手目に撃つ場合で、すぐ撃てるのは6連鎖。相手は6連鎖を撃った)
    later = a.max.count > 0 and a.max.frame > 0
    lines = [f"{a.max.count}連鎖 {a.max.score:,}点" + (f" @{a.max.frame + 1}" if later else "")]
    if show_now and later and 0 < a.immediate.count < a.max.count:
        lines.append(f"  今撃つと {a.immediate.count}連鎖 {a.immediate.score:,}点")
    # 「N連鎖」は見えている組ぷよ(ネクネクまで)で撃てる最大なので、発火の色が来ていなければ組んでいる本線より小さく出る
    # (2026-10-25 利用者「相手の連鎖数が組んでいる連鎖より小さい」)。その時は組んでいる連鎖を別の行に出す
    if a.potential_chain > a.max.count:
        lines.append(f"  組んでいる {a.potential_chain}連鎖 {a.potential:,}点")
    if 0 < a.harass.count < a.max.count:
        lines.append(f"  潰し {a.harass.count}連鎖 {a.harass.score:,}点")
    return lines


def panel_slots(data: OverlayDrawData, language: str = "ja") -> list[tuple[str, QtGui.QColor, int]]:
    """PANEL_SLOTS の順に (文言, 文字色, 行数)。情報の無い種類は空文字(その行は空白にする)。
    自分の連鎖・点数は青、相手は濃いピンク、他は白(利用者の指定 2026-10-25)。"""
    label = data.action or LABEL_TEXT_JA.get(data.label, data.label)
    # 自分・相手の行: 先頭が最大連鎖(相手の連鎖中は発火中)、残りが補足。補足は割り当ての行数ぶんだけ上から出す
    self_rows = _format_analysis(data.self_analysis, data.self_firing) + ["", ""]
    enemy_rows = _format_analysis(data.enemy_analysis, data.enemy_firing, show_now=True) + ["", "", ""]
    texts = {
        "move": (f"提案: {label}", PANEL_TEXT_COLOR),
        # 自分の「N連鎖」(青の行)は手持ちで組める最大であり、実際に発火する連鎖とは異なることがある
        # (実機で「発火中の連鎖が表示と違う」指摘 2026-09-16)ので、置くと起きる連鎖を別に出す
        "fire": (f"この手で {data.fire_chain[0]}連鎖 {data.fire_chain[1]:,}点" if data.fire_chain[0] > 0 else "", PANEL_TEXT_COLOR),
        "note": (data.note, PANEL_TEXT_COLOR),
        "form": (data.form, PANEL_TEXT_COLOR),
        "plan": (data.plan_text, PANEL_TEXT_COLOR),
        "self": (self_rows[0], PANEL_SELF_COLOR),
        "self_sub1": (self_rows[1], PANEL_SELF_COLOR),
        "enemy": (enemy_rows[0], PANEL_ENEMY_COLOR),
        "enemy_sub1": (enemy_rows[1], PANEL_ENEMY_COLOR),
        "enemy_sub2": (enemy_rows[2], PANEL_ENEMY_COLOR),
    }
    out = []
    for key, rows in PANEL_SLOTS:
        text, color = texts[key]
        if data.chaining and key in CHAINING_HIDDEN_SLOTS and not (key == "self" and data.self_firing is not None):
            text = ""
        text = _translate(text, language)
        out.append((text, color, rows))
    return out


def panel_lines(data: OverlayDrawData, language: str = "ja") -> list[tuple[str, QtGui.QColor]]:
    """情報パネルの空でない行と文字色(上から順)。"""
    return [(text, color) for text, color, _ in panel_slots(data, language) if text]


def format_panel_lines(data: OverlayDrawData) -> list[str]:
    """情報パネルの行(テスト可能なよう分離)。"""
    return [line for line, _ in panel_lines(data)]


def wrap_line(text: str, width_of, max_width: float, word_break: bool = False) -> list[str]:
    """文字幅の関数 width_of で測って max_width に収まるよう文字単位で折り返す(日本語は空白が無い)。
    word_break なら行の途中の空白で折り返す(英語の単語を切らない。空白が無ければ文字単位)。"""
    if width_of(text) <= max_width or not text:
        return [text]
    out: list[str] = []
    cur = ""
    for ch in text:
        if cur and width_of(cur + ch) > max_width:
            if word_break and ch == " ":
                out.append(cur)
                cur = ""
                continue
            cut = cur.rstrip().rfind(" ") if word_break else -1
            if cut > 0 and cur[:cut].strip():
                out.append(cur[:cut])
                cur = cur[cut + 1:].lstrip() + ch
                continue
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def layout_panel_rows(slots: list[tuple[str, QtGui.QColor, int]], width_of, max_width: float, word_break: bool = False) -> list[tuple[str, QtGui.QColor]]:
    """各種類を決まった行数に割り付けた、PANEL_LINES 行ぶんの (文言, 文字色)。空の種類・余った行は空文字。
    折り返して行数を超える種類は、最後の行の末尾を「…」にして収める。"""
    out: list[tuple[str, QtGui.QColor]] = []
    for text, color, rows in slots:
        wrapped = wrap_line(text, width_of, max_width, word_break) if text else []
        if len(wrapped) > rows:
            last = wrapped[rows - 1]
            while last and width_of(last + "…") > max_width:
                last = last[:-1]
            wrapped = wrapped[: rows - 1] + [last + "…"]
        out += [(line, color) for line in wrapped] + [("", color)] * (rows - len(wrapped))
    return out


def render_perf(painter: QtGui.QPainter, layout: OverlayLayout, text: str) -> None:
    """CPU 使用率と観測の FPS(2026-10-24): 自分の盤面の上端の枠外(1P は左、2P は右。情報パネルは下端なので重ならない)に1行で描く"""
    if not text:
        return
    text = _translate(text, layout.language)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    geom = layout.self_geometry
    font = painter.font()
    font.setPixelSize(PANEL_FONT_PX)
    font.setBold(True)
    painter.setFont(font)
    metrics = QtGui.QFontMetrics(font)
    width = metrics.horizontalAdvance(text) + PANEL_PADDING * 2
    height = metrics.height() + PANEL_PADDING * 2
    if layout.panel_right_space is None:
        anchor = layout.to_local(*geom.grid_to_screen(0, 0))
        left = anchor.x() - PANEL_GAP - width
    else:
        anchor = layout.to_local(*geom.grid_to_screen(COLS, 0))
        left = anchor.x() + PANEL_GAP
    rect = QtCore.QRectF(left, anchor.y(), width, height)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, PANEL_ALPHA))
    painter.drawRoundedRect(rect, 6, 6)
    painter.setPen(QtGui.QColor(255, 255, 255))
    painter.drawText(QtCore.QPointF(rect.x() + PANEL_PADDING, rect.y() + PANEL_PADDING + metrics.ascent()), text)


def render_overlay(painter: QtGui.QPainter, layout: OverlayLayout, data: OverlayDrawData) -> None:
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    geom = layout.self_geometry
    cell = geom.cell_size() / layout.dpr

    # 最終的な構想: 提案のドットより下に、小さく薄く描く
    if not data.waiting:
        for r, c, color, is_trigger in data.plan_dots:
            center = layout.to_local(*geom.cell_center(r, c))
            base = PUYO_DRAW_COLORS.get(color, UNKNOWN_DRAW_COLOR)
            if is_trigger:
                _draw_dot(painter, center, cell * PLAN_TRIGGER_RADIUS_RATIO, base, 3, 220)
            else:
                _draw_dot(painter, center, cell * PLAN_DOT_RADIUS_RATIO, base, 1, PLAN_DOT_ALPHA)

    if not data.waiting:
        axis = layout.to_local(*geom.cell_center(*data.axis_cell))
        child = layout.to_local(*geom.cell_center(*data.child_cell))

        # 軸と子を結ぶ線
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 200), LINK_LINE_WIDTH))
        painter.drawLine(axis, child)

        _draw_dot(painter, child, cell * CHILD_DOT_RADIUS_RATIO, PUYO_DRAW_COLORS.get(data.child_color, UNKNOWN_DRAW_COLOR), CHILD_OUTLINE_WIDTH)
        _draw_dot(painter, axis, cell * AXIS_DOT_RADIUS_RATIO, PUYO_DRAW_COLORS.get(data.axis_color, UNKNOWN_DRAW_COLOR), AXIS_OUTLINE_WIDTH)

    # 2手先: 小さめ・薄めのドットと「2」の印
    if not data.waiting and data.next_axis_cell is not None and data.next_child_cell is not None:
        n_axis = layout.to_local(*geom.cell_center(*data.next_axis_cell))
        n_child = layout.to_local(*geom.cell_center(*data.next_child_cell))
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 120), 2, QtCore.Qt.PenStyle.DashLine))
        painter.drawLine(n_axis, n_child)
        _draw_dot(painter, n_child, cell * NEXT_CHILD_DOT_RADIUS_RATIO, PUYO_DRAW_COLORS.get(data.next_child_color, UNKNOWN_DRAW_COLOR), 1, NEXT_DOT_ALPHA)
        _draw_dot(painter, n_axis, cell * NEXT_AXIS_DOT_RADIUS_RATIO, PUYO_DRAW_COLORS.get(data.next_axis_color, UNKNOWN_DRAW_COLOR), 2, NEXT_DOT_ALPHA)
        font = painter.font()
        font.setPixelSize(max(10, int(cell * 0.3)))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(255, 255, 255, 230))
        painter.drawText(QtCore.QPointF(n_axis.x() + cell * 0.18, n_axis.y() - cell * 0.18), "2")

    # 情報パネル: 自分の盤面の左枠外に、下端を盤面の下端に揃えて置く。
    # 盤面の左の余白に収まるよう、長い行は折り返す(文字数が多くて画面外にはみ出た 2026-09-16)
    font = painter.font()
    font.setPixelSize(PANEL_FONT_PX)
    font.setBold(True)
    painter.setFont(font)
    metrics = QtGui.QFontMetrics(font)
    if layout.panel_right_space is None:
        anchor = layout.to_local(*geom.grid_to_screen(0, VISIBLE_ROWS))
        space = anchor.x() - PANEL_GAP - PANEL_PADDING * 2
    else:
        anchor = layout.to_local(*geom.grid_to_screen(COLS, VISIBLE_ROWS))
        space = layout.panel_right_space - PANEL_GAP - PANEL_PADDING * 2
    max_text_width = max(PANEL_MIN_WIDTH, min(PANEL_TEXT_WIDTH, space))
    colored = layout_panel_rows(panel_slots(data, layout.language), metrics.horizontalAdvance, max_text_width, _word_break(layout.language))
    width = max_text_width + PANEL_PADDING * 2
    height = metrics.height() * PANEL_LINES + PANEL_PADDING * 2
    left = anchor.x() - PANEL_GAP - width if layout.panel_right_space is None else anchor.x() + PANEL_GAP
    rect = QtCore.QRectF(left, anchor.y() - height, width, height)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(0, 0, 0, PANEL_ALPHA))
    painter.drawRoundedRect(rect, 6, 6)
    for i, (line, color) in enumerate(colored):
        if not line:
            continue
        painter.setPen(color)
        painter.drawText(
            QtCore.QPointF(rect.x() + PANEL_PADDING, rect.y() + PANEL_PADDING + metrics.ascent() + metrics.height() * i),
            line,
        )
