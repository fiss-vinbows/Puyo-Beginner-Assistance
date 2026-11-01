"""シミュレーションモードの規則(src/simulator/rules.py)のテスト。"""

import unittest

from src.engine.puyo_field import COLS, EMPTY, ROWS
from src.simulator.rules import SimState, parse_pairs, resolve_chain


def board_from(rows: list[str]) -> list[list[str]]:
    """下詰めで盤面を作る(rows は上から)。"""
    rows = ["." * COLS] * (ROWS - len(rows)) + rows
    return [list(r) for r in rows]


class SimStateTest(unittest.TestCase):
    def test_same_seed_gives_same_sequence_and_first_two_pairs_use_three_colors(self):
        a = SimState(seed=123)
        b = SimState(seed=123)
        a.ensure_pairs(50)
        b.ensure_pairs(10)
        b.ensure_pairs(50)
        self.assertEqual(a.pairs[:50], b.pairs[:50])
        self.assertLessEqual(len(set("".join(a.pairs[:2]))), 3)

    def test_drop_places_vertical_pair_in_column_3(self):
        s = SimState(seed=1, pairs=["RG"])
        s.drop()
        self.assertEqual(s.board[ROWS - 1][2], "R")  # 軸が下
        self.assertEqual(s.board[ROWS - 2][2], "G")
        self.assertEqual(s.turn, 1)

    def test_move_is_blocked_by_wall(self):
        s = SimState(seed=1)
        for _ in range(5):
            s.move(-1)
        self.assertEqual(s.col, 0)
        s.rotate(-1)  # 子が左の壁にぶつかる → 軸が右へずれる
        self.assertEqual((s.col, s.rotation), (1, "LEFT"))

    def test_quick_turn_when_both_sides_blocked(self):
        s = SimState(seed=1)
        s.board = board_from(["...#.#"] * 12)  # 5列目だけ空いた溝(高さ12)
        s.col = 4
        s.row = 1
        self.assertFalse(s.rotate(1))
        self.assertTrue(s.rotate(1))
        self.assertEqual(s.rotation, "DOWN")

    def test_mawashi_over_height_12_column(self):
        # 3列目の高さ11で床蹴りして浮かせ、高さ12の4列目を越えて5列目に置く
        s = SimState(seed=1, pairs=["RG"])
        s.board = board_from(["...#.."] + ["..##.."] * 11)
        self.assertFalse(s.move(1))
        actions = s.find_actions({((ROWS - 1, 4), "G"), ((ROWS - 2, 4), "R")})  # 子が下
        self.assertIsNotNone(actions)
        for a in actions:
            {"move_left": lambda: s.move(-1), "move_right": lambda: s.move(1),
             "rotate_ccw": lambda: s.rotate(-1), "rotate_cw": lambda: s.rotate(1)}[a]()
        s.drop()
        self.assertEqual((s.board[ROWS - 1][4], s.board[ROWS - 2][4]), ("G", "R"))

    def test_quick_turn_at_spawn_crosses_both_sides(self):
        # 2列目・4列目が高さ12: 出現位置でクイックターンして浮き、1列目へ越える
        s = SimState(seed=1, pairs=["RG"])
        s.board = board_from([".#.#.."] * 12)
        self.assertIsNotNone(s.find_actions({((ROWS - 1, 0), "R"), ((ROWS - 2, 0), "G")}))

    def test_height_13_column_cannot_be_crossed(self):
        s = SimState(seed=1, pairs=["RG"])
        s.board = board_from(["...#.."] * 13)
        self.assertIsNone(s.find_actions({((ROWS - 1, 4), "R"), ((ROWS - 2, 4), "G")}))

    def test_chain_score_and_all_clear(self):
        # 赤4個を縦に並べる → 1連鎖 40点、全消し
        s = SimState(seed=1, pairs=["RR"])
        s.board = board_from(["..R...", "..R..."])
        outcome = s.drop()
        self.assertEqual(outcome.count, 1)
        self.assertEqual(outcome.score, 40)
        self.assertTrue(s.all_clear)

    def test_two_chain(self):
        board = board_from(["Y.....", "RYYY..", "RRR..."])
        result = resolve_chain(board)
        self.assertEqual(result.count, 2)
        self.assertEqual(result.score, 40 + 40 * 8)
        self.assertTrue(all(c == EMPTY for row in result.final for c in row))

    def test_undo_and_reset_same_keeps_sequence(self):
        s = SimState(seed=5)
        first = list(s.upcoming(3))
        s.drop()
        self.assertTrue(s.undo())
        self.assertEqual(s.turn, 0)
        self.assertTrue(all(c == EMPTY for row in s.board for c in row))
        s.set_upcoming(["BB", "GG"])
        again = s.reset(same_sequence=True)
        self.assertEqual(again.upcoming(2), ["BB", "GG"])
        self.assertNotEqual(first[:2], ["BB", "GG"]) if first[:2] != ["BB", "GG"] else None

    def test_game_over_when_spawn_cell_filled(self):
        s = SimState(seed=1)
        s.board = board_from(["..R..."] * 12)
        self.assertTrue(s.game_over)
        self.assertIsNone(s.drop())

    def test_parse_pairs(self):
        self.assertEqual(parse_pairs("rg, BY  yy"), ["RG", "BY", "YY"])
        with self.assertRaises(ValueError):
            parse_pairs("RX")

    def test_garbage(self):
        s = SimState(seed=1)
        s.drop_garbage(8)
        self.assertEqual(sum(c == "#" for row in s.board for c in row), 8)


if __name__ == "__main__":
    unittest.main()
