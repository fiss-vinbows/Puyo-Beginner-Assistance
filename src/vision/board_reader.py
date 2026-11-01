"""盤面・予告おじゃま・ネクストの読み取り

キャプチャ画像と BoardGeometry から、
  - 13段×6列の各マスの内容(ゲーム上の色名 / garbage / empty)
  - 盤面上部の帯に表示される予告おじゃまの個数
  - ネクスト枠の組ぷよ(子, 軸)
を読み取る。ここで返すのは「画面に見えているもの」そのままで、落下中の
組ぷよの除外や連鎖中の判定は呼び出し側(app)が時系列を見て行う。

13段目(row=0)について: ゲーム画面では隠し段でぷよは描かれず、その位置には
予告おじゃまが表示される。この段は盤面としては常に空として扱う。
12段目3列目には赤い「×」の窒息マークが常に描かれており赤ぷよと誤認しうる
ため、そのマスだけ cell_classifier の×マーク判定(白い目が無ければ空)を使う。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from ..capture.geometry import BoardGeometry
from ..engine.puyo_field import COLS, ROWS
from .banner_detector import AllClearBannerDetector, default_detector
from .cell_classifier import EMPTY, EYE_WHITE_FRAC_MIN, GAME_COLORS, UNKNOWN, classify_cell_hsv, eye_white_fraction, white_fraction
from .pending_icons import BAND_H as ICON_BAND_H
from .pending_icons import default_reader as default_icon_reader

# 窒息点(12段目3列目)。row=1 が12段目
CROSS_MARK_CELL = (1, 2)

WARP_CELL_PX = 32  # pending_icons.CELL_PX と同じ(予告アイコンの帯を盤面と同じ切り出しから取る)
# 予告おじゃま帯(=13段目の位置)を含めて上に1段ぶん切り出す(row 0 = 予告帯/13段目, row 1.. = 12段目..)
ROWS_ABOVE = 1
# 予告帯の上端には盤面枠の白い縁が入るため、この比率ぶん上を捨てる
BAND_TOP_TRIM_RATIO = 0.15

# 予告おじゃまは13段目の位置の帯に、列にそろえて横一列に並ぶ(半透明のものもある)。
# アイコンの種類(小=1・大=6・岩=30・星=180・王冠=720)は pending_icons が見本との一致で見分けて個数に換算する
# (2026-10-20。以前はどのアイコンも1個と数えていた)。見分けられなかった列(月・彗星など見本の無いもの、強い演出の下)は
# 従来どおり白画素の割合がこの閾値以上なら1個とみなす。
PENDING_GARBAGE_WHITE_FRAC_MIN = 0.30

# 盤面を覆う演出(全消し直後のキャラクターのカットイン等)は、顔や服が色ぷよとして読まれる。
# 本物のぷよには必ず白い目があるので、「色と判定されたが白い目が無いセル」の位置を返し、
# tracker 側で「確定盤面に入りうる位置」にあるものを数える(実機 2026-09-17: カットインで
# 37セル中26セルが目なし。通常の盤面は0。全消し後の背景の緑の光や全消しバナーは空マスの上に
# 浮いているので、settle と同じ基準で除外する)


@dataclass
class BoardReading:
    cells: list[list[str]]  # [row][col]。row=0 が13段目(常に empty)
    pending_garbage: int  # 予告おじゃまの個数(小ぷよ換算。アイコンの種類ごとの個数の合計)
    garbage_white_fracs: list[float] = field(default_factory=list)  # 予告帯の列ごとの白割合
    pending_icons: tuple[str, ...] = ()  # 予告帯で見分けたアイコンの種類(左から順)
    eyeless_cells: frozenset[tuple[int, int]] = frozenset()  # 色と判定されたが白い目が無いセル (row, col)
    banner_score: float = 0.0  # 「全消し」の文字の一致度(0 なら文字なし)。文字の画素は分類から除いてある


def _count_pending_garbage(band_hsv: np.ndarray, cell_px: int) -> tuple[int, list[float]]:
    """予告帯を列ごとに見て、白いぷよがある列を数える。列ごとの白割合も返す(デバッグ用)。"""
    fracs = [white_fraction(band_hsv[:, col * cell_px : (col + 1) * cell_px]) for col in range(COLS)]
    return sum(1 for f in fracs if f >= PENDING_GARBAGE_WHITE_FRAC_MIN), fracs


def read_board(
    frame_bgr: np.ndarray, geometry: BoardGeometry, origin: tuple[int, int] = (0, 0),
    banner_detector: AllClearBannerDetector | None = None,
) -> BoardReading:
    """キャプチャ画像から盤面を読む。origin は frame の左上の画面座標。

    banner_detector を渡すと「全消し」の文字を盤面ごとに追跡する(隠れても位置を覚える)。
    省略時は状態を持たない共有の検出器を使う。"""
    # 予告アイコンは13段目の帯の上にはみ出すので、もう1段上まで1回で切り出し、盤面はその下を使う
    # (段の整数倍の平行移動なので、rows_above=ROWS_ABOVE で切り出した画像と画素まで同じ)
    warped_tall = geometry.warp(frame_bgr, cell_px=WARP_CELL_PX, rows_above=ROWS_ABOVE + 1, origin=origin)
    warped = warped_tall[WARP_CELL_PX:]
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    # 「全消し」の文字が見えていれば、その画素を背景(彩度0・明度0)にして分類から除く。
    # 文字はぷよより奥に描かれるので、残った画素だけで手前のぷよを通常どおり読める
    if banner_detector is not None:
        banner_mask, banner_score = banner_detector.observe(warped)
    else:
        found = default_detector().find(warped)
        banner_mask = default_detector().pixel_mask(warped, found) if found is not None else None
        banner_score = found.score if found is not None else 0.0
    hsv_raw = hsv
    if banner_mask is not None:
        hsv = hsv.copy()
        hsv[banner_mask] = 0

    cells: list[list[str]] = []
    eyeless: set[tuple[int, int]] = set()
    for row in range(ROWS):
        line: list[str] = []
        for col in range(COLS):
            if row == 0:
                line.append(EMPTY)
                continue
            r0 = row * WARP_CELL_PX
            c0 = col * WARP_CELL_PX
            cell = hsv[r0 : r0 + WARP_CELL_PX, c0 : c0 + WARP_CELL_PX]
            label = classify_cell_hsv(cell, cross_mark=(row, col) == CROSS_MARK_CELL)
            # 目の有無は文字を除く前の画素で見る(文字の白と重なった目を除くと本物のぷよが目なしになる)
            if label in GAME_COLORS and eye_white_fraction(hsv_raw[r0 : r0 + WARP_CELL_PX, c0 : c0 + WARP_CELL_PX]) < EYE_WHITE_FRAC_MIN:
                eyeless.add((row, col))
            line.append(label)
        cells.append(line)

    band = hsv[int(WARP_CELL_PX * BAND_TOP_TRIM_RATIO) : WARP_CELL_PX, :]
    _, fracs = _count_pending_garbage(band, WARP_CELL_PX)
    icon_band = warped_tall[:ICON_BAND_H]
    icons = default_icon_reader().read(icon_band)
    pending = icons.value() + sum(
        1 for col, frac in enumerate(fracs) if frac >= PENDING_GARBAGE_WHITE_FRAC_MIN and not icons.column_has_icon(col)
    )
    return BoardReading(
        cells=cells, pending_garbage=pending, garbage_white_fracs=fracs, eyeless_cells=frozenset(eyeless),
        banner_score=banner_score, pending_icons=icons.kinds,
    )


def read_next(frame_bgr: np.ndarray, rect: tuple[int, int, int, int], origin: tuple[int, int] = (0, 0)) -> tuple[str, str]:
    """ネクスト枠(縦2ぷよ)から (子ぷよの色, 軸ぷよの色) を読む。上が子、下が軸。"""
    x, y, w, h = rect
    x -= origin[0]
    y -= origin[1]
    crop = frame_bgr[y : y + h, x : x + w]
    if crop.size == 0 or h < 2:
        # 枠がキャプチャ範囲の外か大きさが不正(キャリブレーション不良)。落とさず unknown を返す
        return UNKNOWN, UNKNOWN
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    half = h // 2
    child = classify_cell_hsv(hsv[:half], require_eyes=True, reject_ghost=False)
    axis = classify_cell_hsv(hsv[half:], require_eyes=True, reject_ghost=False)
    return child, axis


def cells_to_text(cells: list[list[str]]) -> str:
    """デバッグ表示用: 1文字表記で行ごとに並べる。"""
    short = {"red": "R", "yellow": "Y", "green": "G", "blue": "B", "purple": "P", "garbage": "#", "empty": ".", "unknown": "?"}
    return "\n".join("".join(short.get(c, "?") for c in row) for row in cells)
