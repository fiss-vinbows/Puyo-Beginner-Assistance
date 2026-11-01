"""盤面の4隅(四角形)と 6×13 グリッドの座標変換

ぷよぷよテトリス2の盤面枠は画面上で少し回転して表示される(自分側は右上がり、
相手側は左上がり)ため、Tetris版のような「矩形＋等分割」ではマス位置が
列ごとにズレていく。そこで盤面の可視領域(6列×12段)の4隅を指定してもらい、
透視変換(ホモグラフィ)で「正規化グリッド座標 ⇔ 画面座標」を相互変換する。

正規化グリッド座標系:
  u = 列方向。0 が左端、6 が右端。
  v = 段方向。0 が可視領域の上端(12段目の上辺)、12 が最下段の下辺。
  隠し13段目は v ∈ [-1, 0](上へ外挿)。ゲーム画面では盤面の暗い矩形(可視12段。
  最上段の3列目に「×」の窒息マーク)の直上の帯に当たり、予告おじゃまがここに表示される。

PuyoField の (row, col) との対応: row=0 が13段目 → v = row - 1 + 0.5 が中心。
"""

from __future__ import annotations

import cv2
import numpy as np

from ..engine.puyo_field import COLS, ROWS, VISIBLE_ROWS

Point = tuple[float, float]
Quad = tuple[Point, Point, Point, Point]  # 左上, 右上, 右下, 左下 の順


class BoardGeometry:
    """盤面4隅から作る座標変換。"""

    def __init__(self, quad: Quad) -> None:
        self.quad = tuple((float(x), float(y)) for x, y in quad)
        src = np.array(self.quad, dtype=np.float32)
        dst = np.array([(0, 0), (COLS, 0), (COLS, VISIBLE_ROWS), (0, VISIBLE_ROWS)], dtype=np.float32)
        # 画面 → 正規化
        self._to_grid = cv2.getPerspectiveTransform(src, dst)
        # 正規化 → 画面
        self._to_screen = cv2.getPerspectiveTransform(dst, src)

    def grid_to_screen(self, u: float, v: float) -> Point:
        pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), self._to_screen)[0][0]
        return (float(pt[0]), float(pt[1]))

    def screen_to_grid(self, x: float, y: float) -> Point:
        pt = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), self._to_grid)[0][0]
        return (float(pt[0]), float(pt[1]))

    def cell_center(self, row: int, col: int) -> Point:
        """PuyoField の (row, col) のマス中心の画面座標。row=0 は隠し13段目。"""
        return self.grid_to_screen(col + 0.5, row - (ROWS - VISIBLE_ROWS) + 0.5)

    def cell_size(self) -> float:
        """おおよそのマスの1辺(px)。上辺と右辺の長さから求める。オーバーレイのドット半径などに使う。"""
        tl = np.array(self.grid_to_screen(0, 0))
        tr = np.array(self.grid_to_screen(COLS, 0))
        br = np.array(self.grid_to_screen(COLS, VISIBLE_ROWS))
        w = float(np.linalg.norm(tr - tl))
        h = float(np.linalg.norm(br - tr))
        return (w / COLS + h / VISIBLE_ROWS) / 2

    def warp(
        self, image: np.ndarray, cell_px: int = 32, rows_above: int = 1, origin: tuple[int, int] = (0, 0)
    ) -> np.ndarray:
        """画面画像から盤面を切り出し、傾きを補正した正規化画像を返す。

        rows_above は可視12段の上に何段ぶん外挿するか(1=隠し13段目まで、
        2=予告おじゃまの帯まで)。返り値は幅 COLS*cell_px、高さ (VISIBLE_ROWS+rows_above)*cell_px。
        rows_above=1 のとき、出力の行 r は PuyoField の row に対応する。
        origin は image の左上が画面上のどこにあるか(部分キャプチャした画像を渡す場合)。
        """
        v0 = -rows_above
        rows = VISIBLE_ROWS + rows_above
        # 画像座標 → 画面座標(originぶん平行移動) → 正規化座標 (u, v) → 出力画像座標 (u*cell_px, (v - v0)*cell_px)
        shift = np.array([[1, 0, origin[0]], [0, 1, origin[1]], [0, 0, 1]], dtype=np.float64)
        scale = np.array([[cell_px, 0, 0], [0, cell_px, -v0 * cell_px], [0, 0, 1]], dtype=np.float64)
        matrix = scale @ self._to_grid @ shift
        return cv2.warpPerspective(image, matrix, (COLS * cell_px, rows * cell_px), flags=cv2.INTER_LINEAR)

    def bounding_box(self, rows_above: int = 2) -> tuple[int, int, int, int]:
        """4隅(＋上へ rows_above 段)を含む画面上の外接矩形 (x, y, w, h)。キャプチャ範囲の決定用。"""
        v0 = -rows_above
        corners = [
            self.grid_to_screen(0, v0),
            self.grid_to_screen(COLS, v0),
            self.grid_to_screen(COLS, VISIBLE_ROWS),
            self.grid_to_screen(0, VISIBLE_ROWS),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        x0, y0 = int(np.floor(min(xs))), int(np.floor(min(ys)))
        x1, y1 = int(np.ceil(max(xs))), int(np.ceil(max(ys)))
        return (x0, y0, x1 - x0, y1 - y0)


def detect_board_quads(image: np.ndarray, value_max: int = 110, min_height: int = 300) -> list[Quad]:
    """画面画像から盤面の暗い矩形領域を自動検出し、可視12段の4隅を左から順に返す。

    ぷよぷよテトリス2の盤面内部は暗い半透明の矩形(可視12段ぶん)で、周囲の
    明るい背景・枠とは明度で分離できる。対戦開始直後(盤面がほぼ空)に使うことを
    想定しており、ぷよが多く積まれていると領域が分断されて失敗する。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dark = (hsv[:, :, 2] < value_max).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    quads: list[Quad] = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < min_height:
            continue
        # 盤面は 6:12 の縦長(実測 400×723)。極端に外れた形は除外
        if not (0.35 < w / h < 0.75):
            continue
        # 矩形の充填率(ぷよが少なければ高い)
        if area / (w * h) < 0.6:
            continue
        x, y, w, h = float(x), float(y), float(w), float(h)
        quads.append(((x, y), (x + w, y), (x + w, y + h), (x, y + h)))
    quads.sort(key=lambda q: q[0][0])
    return quads
