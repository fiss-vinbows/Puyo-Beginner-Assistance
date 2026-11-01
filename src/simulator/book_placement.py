"""定跡の手(BookMove の2マス)を、支援モードで使う置き場所に直す(2026-10-26、利用者「定跡は支援モードでも使う」)

シミュレーターの規則(SimState)で出現位置からの操作を探し、軸の列・向き・2マス・全自動の操作手順・置いた後の盤面を返す。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..engine.puyo_field import ROWS
from .opening_book import BookMove
from .rules import SimState

# シミュレーターの操作 → 全自動の手順(ama の path と同じ文字。1要素=1入力)
_ACTION_TO_PATH = {"move_left": "LEFT", "move_right": "RIGHT", "rotate_ccw": "CCW", "rotate_cw": "CW"}


@dataclass(frozen=True)
class BookPlacement:
    x: int  # 軸ぷよの列(左から0)
    rotation: str  # 子ぷよの位置 UP/RIGHT/DOWN/LEFT
    axis_cell: tuple[int, int]  # (row, col)。row=0 が13段目
    child_cell: tuple[int, int]
    path: list[str]  # 全自動の操作手順(最後は DROP)
    field_after: list[str]  # 置いて連鎖まで進めた後の盤面(13行、上から)


def to_path(actions: list[str]) -> list[str]:
    """シミュレーターの操作列を全自動の手順にする。同じ操作が続く時は間に NONE(1フレーム離す)を挟む(ama の手順と同じ形)。"""
    out: list[str] = []
    for a in actions:
        step = _ACTION_TO_PATH[a]
        if out and out[-1] == step:
            out.append("NONE")
        out.append(step)
    return out + ["DROP"]


def placement(rows: list[str], pair: str, move: BookMove) -> BookPlacement | None:
    """盤面 rows(13行、上から)に組ぷよ pair(軸子)を、定跡の手 move の2マスへ置く方法。届かなければ None。"""
    st = SimState(seed=0, pairs=[pair] * 3)
    st.board = [list(r) for r in rows]
    actions = st.find_actions(set(move.cells))
    if actions is None:
        return None
    ops = {"move_left": lambda: st.move(-1), "move_right": lambda: st.move(1),
           "rotate_ccw": lambda: st.rotate(-1), "rotate_cw": lambda: st.rotate(1)}
    for a in actions:
        ops[a]()
    # 軸と子の着地マス(下にある方から先に落ちる)
    heights = [st._top_free_row(c) for c in range(len(rows[0]))]
    axis, child = (st.row, st.col), st.child_pos()
    landed = {}
    for name, (r, c) in sorted((("axis", axis), ("child", child)), key=lambda t: -t[1][0]):
        landed[name] = (heights[c], c)
        heights[c] -= 1
    x, rotation = st.col, st.rotation
    st.drop()
    if any(r < 0 or r >= ROWS for r, _ in landed.values()):
        return None
    return BookPlacement(x, rotation, landed["axis"], landed["child"], to_path(actions), ["".join(row) for row in st.board])
