"""1マスぶんの画像から「何が置かれているか」を判定する

ぷよぷよテトリス2のぷよは、色ごとに色相(OpenCVのH: 0〜179)が明確に分かれる
一方、目の部分が白く、連結すると形も変わる。そのためセルの中心1点ではなく
セル全体の画素統計で判定する:

  色ぷよ   : 彩度が高く明るい画素(colored)がある程度あり、その色相の中央値で色を決める
             (ネクスト枠(require_eyes=True)では枠の柄が混ざるので、白い目のまわりの色で決める)
  おじゃま : 白〜灰色(低彩度・高明度)の画素が多く、colored がほぼ無い
  空       : どちらも少ない(盤面の暗い背景。キャラ絵が透けていても彩度は低い)

実測(1080p、tests/fixtures): 赤≈176, 黄≈25, 緑≈52, 紫≈139。
青は動画に登場していないため推定値(≈105)で、実機で要確認。

ここではゲーム上の色名(red/yellow/green/blue/purple)を返す。思考エンジン(ama)の
4スロット(R/Y/G/B)への写像は color_mapper.py が行う。
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

# ゲーム上の色名
RED, YELLOW, GREEN, BLUE, PURPLE = "red", "yellow", "green", "blue", "purple"
GARBAGE = "garbage"
EMPTY = "empty"
UNKNOWN = "unknown"  # 色ぷよらしいが色相がどの範囲にも入らない

GAME_COLORS = (RED, YELLOW, GREEN, BLUE, PURPLE)

# 色相の範囲 (下限, 上限)。赤は0付近で循環するので2区間
_HUE_RANGES: dict[str, list[tuple[int, int]]] = {
    RED: [(165, 179), (0, 8)],
    YELLOW: [(9, 36)],  # 黄ぷよの目のまわり(まゆの陰影)は色相 10〜12 なので、9 まで含める(2026-09-17)
    GREEN: [(40, 78)],
    BLUE: [(100, 125)],  # ネクスト枠の水色(≈94)を含めないよう下限を上げている
    PURPLE: [(128, 160)],
}

# 画素分類の閾値
COLORED_S_MIN = 110
COLORED_V_MIN = 120
WHITE_S_MAX = 60
WHITE_V_MIN = 170

# セル判定の閾値(セル内画素に占める割合)
COLORED_FRAC_MIN = 0.10  # これ以上なら色ぷよ
GARBAGE_WHITE_FRAC_MIN = 0.30  # colored が無く、これ以上白ければおじゃま

# ぷよは丸いので、セルの四隅には背景や枠(ネクスト枠の水色など)が見える。
# セル中心の楕円の内側だけを見ることで、それらの混入を防ぐ。
CELL_ELLIPSE_RATIO = 0.72  # セル幅・高さに対する楕円の直径の比


# 操作中の組ぷよの着地予測(ゴースト)は、本物のぷよより一回り小さい色付きの玉として
# 描かれる(目は無い)。中心の楕円だけ見ると色ぷよに見えてしまうので、楕円の外側の
# リング部分にも色があるか(=セルいっぱいの大きさか)で本物と区別する。
# 実測: 本物は ≥0.05、ゴーストは ≤0.01
CELL_RING_OUTER_RATIO = 0.98
GHOST_RING_COLORED_MAX = 0.04


@lru_cache(maxsize=64)
def _ellipse(h: int, w: int, ratio: float) -> np.ndarray:
    # マスの大きさは毎回同じなので作り置きする(毎回作ると観測1回で約7ms かかっていた 2026-10-21)。
    # 呼び出し側は読むだけ(書き換えない)
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    ry, rx = h * ratio / 2, w * ratio / 2
    mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0
    mask.flags.writeable = False
    return mask


def _ellipse_mask(h: int, w: int) -> np.ndarray:
    return _ellipse(h, w, CELL_ELLIPSE_RATIO)


@lru_cache(maxsize=64)
def _ring_mask(h: int, w: int) -> np.ndarray:
    mask = _ellipse(h, w, CELL_RING_OUTER_RATIO) & ~_ellipse(h, w, CELL_ELLIPSE_RATIO)
    mask.flags.writeable = False
    return mask


def hue_to_color(hue: int) -> str:
    for name, ranges in _HUE_RANGES.items():
        for lo, hi in ranges:
            if lo <= hue <= hi:
                return name
    return UNKNOWN


# 12段目3列目の「×」窒息マークは赤一色で、ぷよのような白い目が無い。
# 赤と判定されたが白画素がこれ未満なら×マークとみなす(cross_mark=True で有効)
CROSS_MARK_WHITE_FRAC_MAX = 0.02

# ネクスト枠の読み取りでは、ぷよが移動中のフレームで枠自体の色(相手側は赤、
# 自分側は水色)が楕円内を占めることがある。ぷよには必ず白い目があるので、
# require_eyes=True のときは白画素がこれ未満なら色ぷよとみなさない。
# (盤面には適用しない: 連鎖演出のテキストに隠れたぷよ等を空と誤判定しないため)
EYE_WHITE_FRAC_MIN = 0.05

# ネクスト枠の背景は枠色の柄(相手側は赤い水玉)で、ぷよが枠の半分に小さく・ずれて
# 写ると楕円内の色画素の過半が枠の色になり、色相の中央値では枠の色に化ける
# (実機 2026-09-15: 相手の黄ぷよが赤と誤判定され、5色目エラーで思考が止まった)。
# そこで require_eyes=True のときは、白い目(と口・ハイライト)のすぐ外側にある色画素だけで
# 色相範囲ごとの多数決をとる。目のまわりは必ずぷよ本体なので、枠の色や位置ずれに影響されない。
EYE_BORDER_RATIO = 0.06  # 目の外側に何画素見るか(セル幅に対する比)
EYE_BORDER_COLOR_SHARE_MIN = 0.5  # 最多色がこれを超える割合を占めること。超えなければ複数のぷよが混ざっている


def dominant_color(hue: np.ndarray, share_min: float = EYE_BORDER_COLOR_SHARE_MIN) -> str:
    """色相の配列を色相範囲ごとに数え、最多の色を返す(赤の 0/179 分断に強い)。"""
    if hue.size == 0:
        return UNKNOWN
    best, best_hits = UNKNOWN, 0
    for name, ranges in _HUE_RANGES.items():
        hits = 0
        for lo, hi in ranges:
            hits += int(((hue >= lo) & (hue <= hi)).sum())
        if hits > best_hits:
            best, best_hits = name, hits
    if best_hits <= hue.size * share_min:  # 過半を要求(同数は不明)
        return UNKNOWN
    return best


def _color_around_eyes(cell_hsv: np.ndarray, white: np.ndarray, colored: np.ndarray) -> str:
    """白い目のすぐ外側にある色画素で色を決める。white/colored はセル全体のマスク。"""
    k = max(3, int(round(cell_hsv.shape[1] * EYE_BORDER_RATIO)) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(white.astype(np.uint8), kernel).astype(bool)
    border = dilated & ~white & colored
    return dominant_color(cell_hsv[:, :, 0][border])


def classify_cell_hsv(
    cell_hsv: np.ndarray, cross_mark: bool = False, require_eyes: bool = False, reject_ghost: bool = True
) -> str:
    """HSV画像(セル1個ぶん)を判定する。

    cross_mark=True は×マークがあるマス用。reject_ghost=True なら着地予測の
    小さな玉を空とみなす(ネクスト枠のように周囲が枠色で囲まれる場合は False にする)。
    """
    h, w = cell_hsv.shape[:2]
    mask = _ellipse_mask(h, w)
    sat_all = cell_hsv[:, :, 1]
    val_all = cell_hsv[:, :, 2]
    colored_all = (sat_all > COLORED_S_MIN) & (val_all > COLORED_V_MIN)
    hue = cell_hsv[:, :, 0][mask]
    colored = colored_all[mask]
    white = ((sat_all < WHITE_S_MAX) & (val_all > WHITE_V_MIN))[mask]
    colored_frac = float(colored.mean())
    white_frac = float(white.mean())
    if colored_frac >= COLORED_FRAC_MIN:
        if require_eyes:
            if white_frac < EYE_WHITE_FRAC_MIN:
                return UNKNOWN
            white_all = (sat_all < WHITE_S_MAX) & (val_all > WHITE_V_MIN)
            return _color_around_eyes(cell_hsv, white_all & mask, colored_all)
        color = hue_to_color(int(np.median(hue[colored])))
        if cross_mark and color == RED and white_frac < CROSS_MARK_WHITE_FRAC_MAX:
            return EMPTY
        if reject_ghost and float(colored_all[_ring_mask(h, w)].mean()) <= GHOST_RING_COLORED_MAX:
            return EMPTY
        return color
    if white_frac >= GARBAGE_WHITE_FRAC_MIN:
        return GARBAGE
    return EMPTY


def classify_cell_bgr(cell_bgr: np.ndarray) -> str:
    return classify_cell_hsv(cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2HSV))


def eye_white_fraction(cell_hsv: np.ndarray) -> float:
    """セル中心の楕円内の白画素の割合(ぷよの白い目の有無の判定に使う)。"""
    h, w = cell_hsv.shape[:2]
    mask = _ellipse_mask(h, w)
    return float(((cell_hsv[:, :, 1] < WHITE_S_MAX) & (cell_hsv[:, :, 2] > WHITE_V_MIN))[mask].mean())


def white_fraction(cell_hsv: np.ndarray) -> float:
    """セル内の白画素の割合(予告おじゃま行の判定などに使う)。"""
    sat = cell_hsv[:, :, 1]
    val = cell_hsv[:, :, 2]
    return float(((sat < WHITE_S_MAX) & (val > WHITE_V_MIN)).mean())
