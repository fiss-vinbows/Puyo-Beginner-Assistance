"""シミュレーターの定跡(src/simulator/opening_book.py)のテスト。"""

import unittest

from src.engine.puyo_field import COLS, EMPTY, ROWS
from src.simulator.opening_book import OpeningBook
from src.simulator.rules import SimState

# 1行だけの小さな定跡(形 AA-AB: 1段目に A A A B。3〜6手目のツモ CC DD CD BB)
DATA = {
    "version": 1,
    "patterns": {
        "AA-AB": {
            "book": [["A", 0, 0], ["A", 1, 0], ["A", 2, 0], ["B", 3, 0]],
            "start": ["......"] * 12 + ["AAAB.."],
            "rows": [["CCDDCDBB", "40C41C50D51D42C43D31B32B", 1234]],
        }
    },
}
# 文字 → 色の割り当てが A=Y, B=G, C=R, D=B になるツモ(定跡の文字の付け替えでも引けること)
PAIRS = ["YY", "GY", "RR", "BB", "BR", "GG"]


def play(state: SimState, book: OpeningBook) -> list:
    """定跡の手を順に置き、各手番の手を返す。"""
    moves = []
    for _ in range(6):
        move = book.lookup(state.board, state.pairs[: state.turn + 3], state.turn)
        moves.append(move)
        if move is None:
            break
        actions = state.find_actions(set(move.cells))
        assert actions is not None, move
        for a in actions:
            {"move_left": lambda: state.move(-1), "move_right": lambda: state.move(1),
             "rotate_ccw": lambda: state.rotate(-1), "rotate_cw": lambda: state.rotate(1)}[a]()
        state.drop()
    return moves


class OpeningBookTest(unittest.TestCase):
    def test_follows_book_through_six_moves_with_color_relabeling(self):
        book = OpeningBook(DATA)
        s = SimState(seed=1, pairs=list(PAIRS))
        moves = play(s, book)
        self.assertTrue(all(m is not None for m in moves))
        self.assertEqual([m.turn for m in moves], [1, 2, 3, 4, 5, 6])
        self.assertEqual(moves[0].cells, frozenset({((ROWS - 1, 0), "Y"), ((ROWS - 1, 1), "Y")}))
        self.assertEqual(moves[2].cells, frozenset({((ROWS - 1, 4), "R"), ((ROWS - 2, 4), "R")}))
        self.assertEqual(moves[2].value, 1234)
        bottom = "".join(s.board[ROWS - 1])
        self.assertEqual(bottom, "YYYGRB")

    def test_off_book_board_returns_none(self):
        book = OpeningBook(DATA)
        s = SimState(seed=1, pairs=list(PAIRS))
        s.board = [[EMPTY] * COLS for _ in range(ROWS)]
        s.board[ROWS - 1][5] = "Y"  # 定跡と違う所に置いた
        s.board[ROWS - 2][5] = "Y"
        s.turn = 1
        self.assertIsNone(book.lookup(s.board, s.pairs[:4], 1))

    def test_unknown_tsumo_and_after_six_moves(self):
        book = OpeningBook(DATA)
        s = SimState(seed=1, pairs=["YY", "GY", "RR", "RR", "BR", "GG"])  # 4手目が表に無い
        moves = play(s, book)
        self.assertEqual([m is not None for m in moves], [True, True, False])
        self.assertIsNone(book.lookup(s.board, PAIRS + ["RR"] * 3, 6))


if __name__ == "__main__":
    unittest.main()
