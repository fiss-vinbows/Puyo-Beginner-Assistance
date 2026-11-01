"""YamaNEKO の6手目までの定跡(Puyo-YamaNEKO の opening_tree.exe --export が書き出す JSON)を引く。

形式(version 1):
  patterns[形] = {"book": 1・2手目の4マス [文字, 列, 段(下から)], "start": 2手目まで置いた盤面(13行、上から),
                  "rows": [[3〜6手目のツモ8文字, 3〜6手目に置く2マス(各6文字: 列・段(13進、下から)・文字), 6手目の点], ...]}
  文字 A〜D は「出てきた色の順」の記号。色の名前の付け替えで同じになる列は代表の1行だけなので、
  引く側は文字→色の割り当てを全通り(24通り)試して一致する行を探す。
  3手目の手は3〜5手目のツモで決まり、4〜6手目の手は6手目のツモまで見えた上での最善。

使い方: 今の盤面とこれまでの組ぷよ列(第1手から、見えている分まで)を渡すと、定跡どおりに進んでいる間は
次に置く2マス (マス, 色) を返す。定跡から外れた(盤面が一致しない・表に無い)ら None。
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path

from ..engine.puyo_field import COLORS, COLS, EMPTY, ROWS
from ..paths import resource_root
from .rules import apply_gravity, resolve_chain

BOOK_PATH = resource_root() / "assets" / "opening_6moves.json"
MAX_TURN = 6  # 定跡は6手目まで


@dataclass(frozen=True)
class BookMove:
    cells: frozenset  # {((row, col), 色), ...} 置く2マス(row=0 が13段目)
    pattern: str  # 形("AA-AB")
    turn: int  # 何手目か(1始まり)
    value: int | None  # 6手目の盤面の点(3手目以降。3手目は6手目のツモ次第で変わるので行の値の1つ)


def _decode(code: str) -> list[tuple[tuple[int, int], str]]:
    """"30A31B" → [((row, col), 文字), ...]"""
    out = []
    for j in range(0, len(code), 3):
        x, y, letter = int(code[j]), int(code[j + 1], 13), code[j + 2]
        out.append(((ROWS - 1 - y, x), letter))
    return out


def _sorted_pair(pair: str) -> str:
    return "".join(sorted(pair))


class OpeningBook:
    def __init__(self, data: dict) -> None:
        if data.get("version") != 1:
            raise ValueError(f"定跡の形式が違います(version={data.get('version')})")
        self.patterns: dict[str, dict] = {}
        for name, pat in data["patterns"].items():
            # ツモ(ペアごとに文字を並べ替えた8文字・6文字)→ 行
            full: dict[str, tuple[str, int]] = {}
            prefix: dict[str, tuple[str, int]] = {}
            for tsumo, moves, value in pat["rows"]:
                key = "".join(_sorted_pair(tsumo[i : i + 2]) for i in range(0, 8, 2))
                full[key] = (moves, value)
                prefix.setdefault(key[:6], (moves, value))
            self.patterns[name] = {
                "book": [(letter, x, y) for letter, x, y in pat["book"]],
                "start": [list(row) for row in pat["start"]],
                "full": full,
                "prefix": prefix,
            }

    @classmethod
    def load(cls, path: Path = BOOK_PATH) -> "OpeningBook | None":
        """ファイルが無ければ None(定跡なしで動く)。"""
        try:
            return cls(json.loads(path.read_text(encoding="utf-8")))
        except FileNotFoundError:
            return None

    def lookup(self, board: list[list[str]], pairs: list[str], turn: int) -> BookMove | None:
        """turn は今操作している組ぷよの番号(0始まり)。pairs は第1手からの組ぷよ列(見えている分まで)。"""
        if turn >= MAX_TURN or len(pairs) < turn + 1:
            return None
        for name, pat in self.patterns.items():
            for perm in itertools.permutations(COLORS):
                color = dict(zip("ABCD", perm))
                move = self._lookup_one(name, pat, color, board, pairs, turn)
                if move is not None:
                    return move
        return None

    def _lookup_one(self, name, pat, color, board, pairs, turn) -> BookMove | None:  # noqa: ANN001
        letter = {c: l for l, c in color.items()}
        letters = [_sorted_pair("".join(letter[c] for c in p)) for p in pairs]
        # 1・2手目の色の形が合うか(組ぷよは軸・子の順を問わない)
        if len(letters) < 2 or letters[0] != _sorted_pair(name[:2]) or letters[1] != _sorted_pair(name[3:]):
            return None
        book = [((ROWS - 1 - y, x), color[l]) for l, x, y in pat["book"]]
        if turn == 0:
            if not _board_equals(board, {}):
                return None
            return BookMove(frozenset(book[:2]), name, 1, None)
        if turn == 1:
            if not _board_equals(board, dict(book[:2])):
                return None
            return BookMove(frozenset(book[2:]), name, 2, None)
        # 3〜6手目: 3〜5手目のツモ(3手目)/3〜6手目のツモ(4手目以降)で行を引く
        if turn == 2:
            if len(letters) < 5:
                return None
            row = pat["prefix"].get("".join(letters[2:5]))
        else:
            if len(letters) < 6:
                return None
            row = pat["full"].get("".join(letters[2:6]))
        if row is None:
            return None
        moves, value = row
        # 盤面が「start + これまでの定跡の手」と一致する時だけ(定跡から外れたら使わない)
        expected = [[color.get(c, EMPTY) if c != EMPTY else EMPTY for c in r] for r in pat["start"]]
        for k in range(turn - 2):
            for (r, c), l in _decode(moves[k * 6 : (k + 1) * 6]):
                expected[r][c] = color[l]
            expected = resolve_chain(apply_gravity(expected)).final
        if expected != board:
            return None
        cells = frozenset(((rc, color[l]) for rc, l in _decode(moves[(turn - 2) * 6 : (turn - 1) * 6])))
        return BookMove(cells, name, turn + 1, value)


def _board_equals(board: list[list[str]], cells: dict) -> bool:
    return all(board[r][c] == cells.get((r, c), EMPTY) for r in range(ROWS) for c in range(COLS))
