"""ぷよぷよの盤面表現

ぷよぷよテトリス2のぷよぷよモードは 6列×12段(+隠し13段目) のフィールド。
このモジュールでは盤面を「上から数えた行」(row=0が隠し13段目、row=12が最下段)
×「左から数えた列」(col=0〜5) の2次元リストで持ち、各マスは以下の1文字で表す。

  'R' 赤, 'Y' 黄, 'G' 緑, 'B' 青, '#' おじゃま, '.' 空

思考エンジン(ama)は4色固定で、この4文字をそのまま使う。実機の色
(赤・緑・青・黄・紫)はどの4色が使われるかで変わるため、画像認識側で
「実際の色 → R/Y/G/B のいずれか」へ写像してからこの表現に落とす。
"""

from __future__ import annotations

from dataclasses import dataclass, field

COLS = 6
ROWS = 13  # 隠し段(13段目)を含む
VISIBLE_ROWS = 12

COLORS = ("R", "Y", "G", "B")
GARBAGE = "#"
EMPTY = "."
VALID_CELLS = COLORS + (GARBAGE, EMPTY)


@dataclass
class PuyoField:
    """盤面。grid[row][col] で row=0 が最上段(隠し13段目)、row=12 が最下段。"""

    grid: list[list[str]] = field(default_factory=lambda: [[EMPTY] * COLS for _ in range(ROWS)])

    @classmethod
    def from_rows(cls, rows: list[str]) -> "PuyoField":
        """上から順に並んだ文字列の行から作る。12行(可視部分のみ)でも13行でもよい。"""
        if len(rows) == VISIBLE_ROWS:
            rows = [EMPTY * COLS] + list(rows)
        if len(rows) != ROWS:
            raise ValueError(f"行数は{VISIBLE_ROWS}または{ROWS}である必要があります: {len(rows)}")
        grid = []
        for row in rows:
            if len(row) != COLS:
                raise ValueError(f"各行は{COLS}文字である必要があります: {row!r}")
            for ch in row:
                if ch not in VALID_CELLS:
                    raise ValueError(f"不正なマス文字: {ch!r}")
            grid.append(list(row))
        return cls(grid)

    def to_rows(self) -> list[str]:
        """ama ブリッジへ渡す13行の文字列(上から順)。"""
        return ["".join(row) for row in self.grid]

    def height(self, col: int) -> int:
        """その列に積まれているぷよの数(最下段から数えた高さ)。"""
        for row in range(ROWS):
            if self.grid[row][col] != EMPTY:
                return ROWS - row
        return 0

    def is_game_over(self) -> bool:
        """3列目(col=2)の12段目が埋まっていれば窒息。"""
        return self.grid[ROWS - VISIBLE_ROWS][2] != EMPTY

    def __str__(self) -> str:
        return "\n".join(self.to_rows())


def xy_to_row_col(x: int, y: int) -> tuple[int, int]:
    """ama座標(x=左から0, y=最下段が0)を、この表現の(row, col)へ変換する。"""
    return (ROWS - 1 - y, x)
