"""ゲーム上の色名 → 思考エンジン(ama)の4スロット(R/Y/G/B)への写像

ぷよぷよテトリス2の対戦では5色(赤・黄・緑・青・紫)のうち4色が使われ、
どの4色かは対戦によって変わりうる。ama は色を4種類しか扱えないため、
その対戦で見えた色に出現順で R/Y/G/B のスロットを割り当てる。

見た目の色とスロット名は一致しない(例: 紫が 'B' になる)。オーバーレイなど
利用者に見せる部分では、スロット名ではなく元の色名や画面上の位置を使うこと。
"""

from __future__ import annotations

from ..engine.puyo_field import COLORS, EMPTY, GARBAGE
from .cell_classifier import EMPTY as V_EMPTY
from .cell_classifier import GAME_COLORS
from .cell_classifier import GARBAGE as V_GARBAGE

# 優先的に同名スロットへ割り当てる(赤→R など)。5色目だけ空きスロットに入る
_PREFERRED = {"red": "R", "yellow": "Y", "green": "G", "blue": "B", "purple": "B"}


class ColorMapper:
    def __init__(self) -> None:
        self._map: dict[str, str] = {}

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self._map)

    def fork(self) -> "ColorMapper":
        """割当のコピー。変換を試してから採用する(途中で失敗した仮の登録を残さない)ために使う。"""
        other = ColorMapper()
        other._map = dict(self._map)
        return other

    def slot(self, color: str) -> str:
        """ゲーム色名をスロット文字へ。未登録の色は空きスロットに割り当てる。5色目以降は ValueError。"""
        if color == V_EMPTY:
            return EMPTY
        if color == V_GARBAGE:
            return GARBAGE
        if color not in GAME_COLORS:
            raise ValueError(f"未知の色: {color!r}")
        if color in self._map:
            return self._map[color]
        used = set(self._map.values())
        preferred = _PREFERRED[color]
        if preferred not in used:
            chosen = preferred
        else:
            free = [s for s in COLORS if s not in used]
            if not free:
                raise ValueError(f"5色目が出現しました: {color!r} (ama は4色までしか扱えません)")
            chosen = free[0]
        self._map[color] = chosen
        return chosen

    def color_of(self, slot: str) -> str | None:
        for color, s in self._map.items():
            if s == slot:
                return color
        return None

    def rows_to_cells(self, rows: list[str]) -> list[list[str]]:
        """ama の13行文字列を board_reader の cells(色名) に戻す。未割当のスロットは unknown。"""
        from .cell_classifier import UNKNOWN

        out: list[list[str]] = []
        for row in rows:
            line = []
            for ch in row:
                if ch == EMPTY:
                    line.append(V_EMPTY)
                elif ch == GARBAGE:
                    line.append(V_GARBAGE)
                else:
                    line.append(self.color_of(ch) or UNKNOWN)
            out.append(line)
        return out

    def cells_to_rows(self, cells: list[list[str]]) -> list[str]:
        """board_reader の cells(色名) を PuyoField.from_rows 用の文字列(13行)へ変換する。"""
        return ["".join(self.slot(c) for c in row) for row in cells]

    def pair(self, child: str, axis: str) -> str:
        """ネクストの(子, 軸)を ama の組ぷよ表記「軸子」にする。"""
        return self.slot(axis) + self.slot(child)
