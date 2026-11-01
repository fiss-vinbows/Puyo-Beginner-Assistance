"""動画フレーム(PNG)に対して認識→思考→オーバーレイ描画を行い、合成画像を保存する。

実機なしでオーバーレイの見た目(ドット位置・情報パネル)を確認するためのツール。
  venv/Scripts/python tools/preview_overlay.py tests/fixtures/frame_05310.png tests/fixtures/calibration_1080p.json out.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
from PyQt6 import QtGui, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.calibrate import load_calibration  # noqa: E402
from src.engine.ama_client import AmaClient, PlayerState  # noqa: E402
from src.engine.puyo_field import PuyoField  # noqa: E402
from src.engine.tracker import settle_cells  # noqa: E402
from src.overlay.renderer import OverlayDrawData, OverlayLayout, render_overlay  # noqa: E402
from src.vision.board_reader import read_board, read_next  # noqa: E402
from src.vision.color_mapper import ColorMapper  # noqa: E402


def main() -> None:
    frame_path, cal_path, out_path = sys.argv[1:4]
    cal = load_calibration(Path(cal_path))
    frame = cv2.imread(frame_path)
    mapper = ColorMapper()

    states = {}
    pairs_by = {}
    for who, pc in (("self", cal.self_player), ("enemy", cal.enemy_player)):
        board = read_board(frame, pc.geometry())
        n1 = read_next(frame, pc.next_rects[0])
        n2 = read_next(frame, pc.next_rects[1])
        settled = settle_cells(board.cells)
        # このツールでは操作中の組ぷよが不明なので、ネクスト1を現在の組ぷよとみなす
        pairs = [mapper.pair(*n1), mapper.pair(*n2)]
        pairs_by[who] = (n1, n2)
        states[who] = PlayerState(PuyoField.from_rows(mapper.cells_to_rows(settled)), pairs, attack=board.pending_garbage)

    with AmaClient() as client:
        move = client.battle(states["self"], states["enemy"])
        # 2手先: 提案通りに置いた後の盤面でネクネクを置く
        expected = PuyoField.from_rows(move.field_after or [])
        next_move = client.battle(PlayerState(expected, states["self"].pairs[1:] + states["self"].pairs[:1]), states["enemy"])
    print(f"提案 x={move.x} {move.rotation} [{move.label}] {move.think_ms}ms / 2手先 x={next_move.x} {next_move.rotation}")

    child, axis = pairs_by["self"][0]
    n_child, n_axis = pairs_by["self"][1]
    data = OverlayDrawData(
        move.axis_cell, move.child_cell, axis, child, move.label, move.self_analysis, move.enemy_analysis,
        next_axis_cell=next_move.axis_cell, next_child_cell=next_move.child_cell, next_axis_color=n_axis, next_child_color=n_child,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)  # noqa: F841
    image = QtGui.QImage(frame_path)
    painter = QtGui.QPainter(image)
    layout = OverlayLayout(cal.self_player.geometry(), cal.enemy_player.geometry(), origin=(0, 0), dpr=1.0)
    render_overlay(painter, layout, data)
    painter.end()
    image.save(out_path)
    print(f"保存: {out_path}")


if __name__ == "__main__":
    main()
