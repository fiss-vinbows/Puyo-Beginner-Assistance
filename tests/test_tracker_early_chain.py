"""spawn(ネクスト入替)無しの消去検出(分析文書 2026-09-19 第2弾)。

実機 2026-09-19: 相手の連鎖は相手の操作中に起き、相手のネクストは連鎖が終わるまで入れ替わらないため、
chain_start が連鎖終了後に「44->8 の1段」として出ていた。操作中でも確定盤面より減った観測が
連続したら消去の疑いとして取り込み待ちに入り、通常の連鎖検出に渡す。
"""

from __future__ import annotations

import unittest

from src.engine.tracker import PlayerTracker, Snapshot
from tests.test_tracker import grid


class EarlyChainTest(unittest.TestCase):
    def make(self, board=None) -> PlayerTracker:
        board = board if board is not None else grid(["RRRYYY", "GGGPPP"])
        tr = PlayerTracker(
            settled=board, current_pair=("red", "yellow"), next1=("green", "yellow"), next2=("purple", "red"),
            settling=False, last_take_in_reason="simple",
        )
        tr._settled_count = sum(c != "empty" for row in board for c in row)
        return tr

    def feed(self, tr, t, board, n1=None, n2=None, eyes=frozenset()):
        return tr.update(Snapshot(t, board, n1 or tr.next1, n2 or tr.next2, 0, eyes))

    def start(self):
        tr = self.make()
        board = grid(["RRRYYY", "GG...."])
        events = []
        for t in (1.0, 1.04, 1.08):
            events += self.feed(tr, t, board)
        return tr, board, events

    def test_start_without_next_change(self) -> None:
        tr, _, ev = self.start()
        self.assertEqual([e.kind for e in ev].count("chain_start"), 1)
        self.assertIn("spawn無しで検出", [e.detail for e in ev if e.kind == "chain_start"][0])
        self.assertFalse(any(e.kind == "spawn" for e in ev))  # spawn を捏造しない
        self.assertTrue(tr.chain_active)
        self.assertEqual(tr.turn, 0)

    def test_single_frame_loss_not_chain(self) -> None:
        tr = self.make()
        ev = self.feed(tr, 1.0, grid(["RRRYYY", "GG...."]))
        ev += self.feed(tr, 1.04, tr.settled)
        self.assertFalse(any(e.kind == "chain_start" for e in ev))

    def test_net_two_loss_can_start(self) -> None:
        # 引き金の2個が未取込なら4個消しでも純減は2個
        tr = self.make(grid(["RRRYYY"]))
        ev = []
        for t in (1.0, 1.04, 1.08):
            ev += self.feed(tr, t, grid(["RYYY.."]))
        self.assertEqual([e.kind for e in ev].count("chain_start"), 1)

    def test_gap_breaks_confirmation(self) -> None:
        # 観測の空白(0.2秒超)をまたいだ2フレームでは確定しない
        tr = self.make()
        reduced = grid(["RRRYYY", "GG...."])
        ev = self.feed(tr, 1.0, reduced) + self.feed(tr, 1.5, reduced)
        self.assertFalse(any(e.kind == "chain_start" for e in ev))

    def test_late_spawn_preserves_chain_and_bonus(self) -> None:
        # 連鎖の途中で spawn が来ても、連鎖の状態(ボーナス・発火前盤面)を消さない
        tr = self.make()
        tr.all_clear = True
        reduced = grid(["RRRYYY", "GG...."])
        for t in (1.0, 1.04, 1.08):
            self.feed(tr, t, reduced)
        before = tr.chain_before
        ev = []
        for t in (1.12, 1.16):
            ev += self.feed(tr, t, reduced, ("purple", "red"), ("red", "red"))
        self.assertTrue(tr.chain_active)
        self.assertTrue(tr.chain_bonus)
        self.assertEqual(tr.chain_before, before)
        self.assertEqual([e.kind for e in ev].count("spawn"), 1)
        self.assertFalse(any(e.kind == "chain_start" for e in ev))

    def test_no_turn_ready_with_consumed_old_pair(self) -> None:
        # 連鎖が終わっても spawn が来るまでは、消費済みの current_pair で turn_ready を出さない
        tr, board, _ = self.start()
        ev = []
        for t in (1.12, 1.20, 1.50):
            ev += self.feed(tr, t, board)
        self.assertTrue(any(e.kind == "chain_end" for e in ev))
        self.assertFalse(any(e.kind == "turn_ready" for e in ev))

    def test_turn_ready_after_late_spawn(self) -> None:
        # 連鎖終了後に spawn が来たら通常どおり手番が進む
        tr, board, _ = self.start()
        for t in (1.12, 1.20, 1.50):
            self.feed(tr, t, board)
        ev = []
        for t in (1.6, 1.64, 1.68, 2.0, 2.04, 2.4):
            ev += self.feed(tr, t, board, ("purple", "red"), ("red", "red"))
        kinds = [e.kind for e in ev]
        self.assertIn("spawn", kinds)
        self.assertIn("turn_ready", kinds)
        self.assertEqual(tr.current_pair, ("green", "yellow"))

    def test_garbage_removal_alone_not_chain(self) -> None:
        tr = self.make(grid(["RRRYYY", "######"]))
        ev = []
        for t in (1.0, 1.04, 1.08):
            ev += self.feed(tr, t, grid(["RRRYYY"]))
        self.assertFalse(any(e.kind == "chain_start" for e in ev))

    def test_falling_gap_not_chain(self) -> None:
        tr = self.make()
        board = grid(["RRRYYY", "......", "GGGPPP"])
        ev = []
        for t in (1.0, 1.04, 1.08):
            ev += self.feed(tr, t, board)
        self.assertFalse(any(e.kind == "chain_start" for e in ev))

    def test_trigger_candidate_becomes_chain_before(self) -> None:
        # 設置直後の「組ぷよ2個が加わり4連結がある」観測を引き金込みの盤面として保持する
        tr = self.make(grid(["RRR...", "GGG..."]))
        placed = grid(["RRRR..", "GGGY.."])  # 操作中の (red, yellow) を縦に置いて赤4連結
        self.feed(tr, 1.0, placed)
        for t in (1.04, 1.08, 1.12):
            self.feed(tr, t, grid(["GGGY.."]))
        self.assertTrue(tr.chain_active)
        self.assertEqual(tr.chain_before_source, "observed_trigger_candidate")
        self.assertEqual(tr.chain_before, placed)

    def test_chain_observation_is_recorded(self) -> None:
        # 段ごとに消えた個数と直近の観測盤面を残す(残りの連鎖の見積もりに使う)
        tr = self.make(grid(["RRRYYY", "GGGPPP"]))
        step1 = grid(["RRRYYY", "GG...."])
        for t in (1.0, 1.04, 1.08):
            self.feed(tr, t, step1)
        self.assertEqual(tr.chain_step_drops, [4])
        self.assertEqual(tr.chain_observed, step1)
        step2 = grid(["RR....", "GG...."])
        for t in (1.5, 1.54):
            self.feed(tr, t, step2)
        self.assertEqual(tr.chain_step_drops, [4, 4])
        self.assertEqual(tr.chain_observed, step2)

    def test_each_separated_step_emits_event(self) -> None:
        # 段が分かれた消去ごとに chain_step を通知する(1019cac では合流した場合しか通知されず、
        # 実機 18:35 のログで chain_step が 0 件だった。分析文書第3弾 P0-1)
        tr = self.make(grid(["RRRYYY", "GGGPPP"]))
        events = []
        for t, b in [(1.0, grid(["RRRYYY", "GG...."])), (1.04, grid(["RRRYYY", "GG...."])), (1.08, grid(["RRRYYY", "GG...."])),
                     (1.5, grid(["RR....", "GG...."])), (1.54, grid(["RR....", "GG...."]))]:
            events += self.feed(tr, t, b)
        self.assertEqual(tr.chain_step_drops, [4, 4])
        self.assertEqual(len([e for e in events if e.kind == "chain_step"]), 2)

    def test_flash_frame_restores_board_before_pop(self) -> None:
        # 消えるぷよは約0.3秒間「見える/消える」を交互に繰り返す(録画 2026-09-13 55.1s)。
        # 消えたフレーム B が直前のフレーム A の部分集合で、差分が A の4連結(+その上に乗って浮いたぷよ)なら、
        # A が引き金込みの発火直前盤面。最初の消えたフレームで chain_start を出し、A を chain_before に残す
        tr = self.make(grid(["RRRYYY", "GGGPPP"]))  # 確定盤面(引き金なし)
        with_trigger = grid(["RRRYYY", "GGGPPP", "..GG..", "..B..."])  # 相手が緑2個を置いて緑5連結、その上に青
        self.feed(tr, 1.00, with_trigger)
        self.feed(tr, 1.03, with_trigger)
        popped = grid(["RRRYYY", "...PPP"])  # 緑が消えた瞬間(青は浮くので settle_cells で落ちる)
        ev = self.feed(tr, 1.07, popped)
        self.assertEqual([e.kind for e in ev], ["chain_start"])
        self.assertIn("点滅から発火直前盤面を復元", ev[0].detail)
        self.assertTrue(tr.chain_active)
        self.assertEqual(tr.chain_before_source, "flash_frame")
        self.assertEqual(tr.chain_before, with_trigger)
        # 点滅で A が再び見えても二重に開始しない。消え終わって落下・安定したら段として数える
        self.feed(tr, 1.10, with_trigger)
        ev = self.feed(tr, 1.13, popped) + self.feed(tr, 1.17, popped)
        self.assertNotIn("chain_start", [e.kind for e in ev])
        self.assertIn("chain_step", [e.kind for e in ev])

    def test_flash_needs_subset_and_poppable_group(self) -> None:
        # 減ったフレームが直前の盤面の部分集合でない(別の場所が変わった)、または消えた組が無い時は復元しない
        tr = self.make(grid(["RRRYYY", "GGGPPP"]))
        self.feed(tr, 1.00, grid(["RRRYYY", "GGGPPP", "...GG."]))  # 緑は3連結+2で繋がらない
        ev = self.feed(tr, 1.03, grid(["RRRYYY", "GG.PPP"]))  # 4連結が無い盤面からの減少
        self.assertNotIn("chain_start", [e.kind for e in ev])
        ev = self.feed(tr, 1.07, grid(["RRRYYB", "GGGPPP"]))  # 色が変わった(部分集合でない)
        self.assertNotIn("chain_start", [e.kind for e in ev])

    def test_unknown_recovery_duration(self) -> None:
        tr = self.make()
        self.feed(tr, 1.0, tr.settled, ("unknown", "unknown"))
        ev = self.feed(tr, 2.0, tr.settled)
        recovered = [e for e in ev if e.kind == "next_recovered"]
        self.assertEqual(len(recovered), 1)
        self.assertIn("1.000", recovered[0].detail)


if __name__ == "__main__":
    unittest.main()


class ChainReplayTest(unittest.TestCase):
    """発火直前盤面から再現した連鎖を終了判定と結果に使う(実機 2026-09-20 14:56: 14連鎖が 3+3+3+2+1 に割れた)。"""

    def make(self, board) -> PlayerTracker:
        tr = PlayerTracker(
            settled=board, current_pair=("red", "yellow"), next1=("green", "yellow"), next2=("purple", "red"),
            settling=False, last_take_in_reason="simple",
        )
        tr._settled_count = sum(c != "empty" for row in board for c in row)
        return tr

    def feed(self, tr, t, board):
        return tr.update(Snapshot(t, board, tr.next1, tr.next2, 0, frozenset()))

    def test_mid_chain_stable_board_does_not_end_chain(self) -> None:
        from src.engine.chain_sim import simulate_chain

        # 2連鎖: 1列目の赤4個が消えると、その上の緑が床の緑3個の隣に落ちて4連結
        before = grid(["RGGG..", "R.....", "R.....", "R.....", "G....."])
        exp = simulate_chain(before)
        self.assertEqual((exp.steps, exp.score), (2, 40 + 4 * 10 * 8))
        tr = self.make(grid(["RGGG..", "R.....", "R....."]))  # 引き金(縦の赤緑)未取込の確定盤面
        self.feed(tr, 1.00, before)
        self.feed(tr, 1.03, before)
        # 赤が消えた瞬間(点滅の消えたフレーム)。上の緑は浮いているので settle_cells で除かれる
        ev = self.feed(tr, 1.07, grid([".GGG.."]))
        self.assertEqual([e.kind for e in ev], ["chain_start"])
        self.assertIn("再現 2連鎖", ev[0].detail)
        self.assertEqual(tr.chain_expected.steps, 2)
        # 段の間の落下中: 落ちてくる緑が読めず、色ぷよが再現後(0個)より多い盤面が0.5秒安定しても終了しない
        falling = grid([".GGG.."])
        ev = []
        for t in (1.2, 1.3, 1.4, 1.5, 1.6, 1.7):
            ev += self.feed(tr, t, falling)
        self.assertNotIn("chain_end", [e.kind for e in ev])
        self.assertTrue(tr.chain_active)
        # 連鎖後の盤面(空)が安定したら終了。連鎖数・点数は再現した値
        empty = grid([])
        ev = []
        for t in (2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6):
            ev += self.feed(tr, t, empty)
        ends = [e for e in ev if e.kind == "chain_end"]
        self.assertEqual(len(ends), 1)
        self.assertIn("chain=2", ends[0].detail)
        self.assertIn("score=360", ends[0].detail)
        self.assertEqual((tr.chain_steps, tr.chain_score), (2, 360))
        self.assertFalse(tr.chain_active)

    def test_next_pair_landing_after_chain_ends_chain(self) -> None:
        # 連鎖後の盤面の上に次の組ぷよが着地して色ぷよが増えても、待ち切り(4秒)を待たずに終了する
        # (実機 2026-09-27 12:25: 1連鎖の終了が毎回4秒遅れ、その間の組ぷよが自由落下した)
        before = grid(["RRRGB.", "R....."])  # 赤5個が消えて GB が残る1連鎖
        tr = self.make(grid(["RRRGB."]))
        self.feed(tr, 1.00, before)
        self.feed(tr, 1.03, before)
        ev = self.feed(tr, 1.07, grid(["...GB."]))
        self.assertEqual([e.kind for e in ev], ["chain_start"])
        landed = grid(["...GBY", ".....P"])  # 次の組ぷよ(黄紫)が6列目に着地
        ev = []
        for t in (1.3, 1.4, 1.5, 1.6, 1.7, 1.8):
            ev += self.feed(tr, t, landed)
        self.assertIn("chain_end", [e.kind for e in ev])
        self.assertFalse(tr.chain_active)

    def test_real_all_clear_is_confirmed_by_replay(self) -> None:
        # 本物の全消し: 発火直前盤面から再現した連鎖の結果が空で、実画面も空なら全消しボーナスを持つ
        before = grid(["RRRR.."])  # 赤4個だけの盤面を消す1連鎖
        tr = self.make(grid(["RR...."]))  # 引き金(赤赤)未取込の確定盤面
        self.feed(tr, 1.00, before)
        self.feed(tr, 1.03, before)
        ev = self.feed(tr, 1.07, grid([]))
        self.assertIn("再現 1連鎖", ev[0].detail)
        ev = []
        t = 1.3
        while t < 3.0:
            ev += self.feed(tr, t, grid([]))
            t += 0.1
        self.assertIn("all_clear", [e.kind for e in ev])
        self.assertTrue(tr.all_clear)

    def test_board_vanishing_without_replay_is_not_all_clear(self) -> None:
        # 負け: 点滅なしで盤面が一気に消える(引き金も再現も無い)。空盤面だけで全消しにしない
        # (実機 2026-09-27: 負けを全消しと誤認し、次の試合をボーナス持ちとして戦った)
        board = grid(["RGBRGB", "GBRGBR", "BRGBRG", "RGBRGB", "GBRGBR"])
        tr = self.make(board)
        ev = []
        t = 1.0
        while t < 3.0:
            ev += self.feed(tr, t, grid([]))
            t += 0.05
        kinds = [e.kind for e in ev]
        self.assertIn("chain_end", kinds)
        self.assertNotIn("all_clear", kinds)
        self.assertIn("all_clear_unconfirmed", kinds)
        self.assertFalse(tr.all_clear)

    def test_eyeless_effect_is_not_flash(self) -> None:
        # 演出(目の無い紫)が本物の紫とつながって4連結に見え、次のフレームで消えても点滅(連鎖)としない
        # (実機 2026-09-27 14:31 25.60s: 相手の攻撃の紫の円形の演出。連鎖していないのに1連鎖とされた)
        from src.engine.tracker import ROWS

        base = grid(["GGBBYY", "PP...."])  # 左下に本物の紫2個
        tr = self.make(base)

        def row(r: int) -> int:  # grid は下の段から並べるので、上から数えた行番号に直す
            return ROWS - 1 - r

        effect = grid(["GGBBYY", "PPPP..", "..PP.."])  # 3・4列目の紫4個は演出(目が無い)
        eyeless = frozenset({(row(1), 2), (row(1), 3), (row(2), 2), (row(2), 3)})
        for t in (1.00, 1.03):
            tr.update(Snapshot(t, effect, tr.next1, tr.next2, 0, eyeless))
        ev = tr.update(Snapshot(1.07, base, tr.next1, tr.next2, 0, frozenset()))
        self.assertNotIn("chain_start", [e.kind for e in ev])
        self.assertFalse(tr.chain_active)

    def test_wrong_replay_times_out(self) -> None:
        # 再現が間違っていた(実際はもっと早く終わった)時は、減少が無いまま一定時間経てば取り込む
        from src.engine.tracker import CHAIN_EXPECTED_MAX_WAIT_SECONDS

        before = grid(["RGGG..", "R.....", "R.....", "R.....", "G....."])
        tr = self.make(grid(["RGGG..", "R.....", "R....."]))
        self.feed(tr, 1.00, before)
        self.feed(tr, 1.03, before)
        self.feed(tr, 1.07, grid([".GGG.."]))
        stuck = grid([".GGG.."])
        ev = []
        t = 1.2
        while t < 1.2 + CHAIN_EXPECTED_MAX_WAIT_SECONDS + 0.5:
            ev += self.feed(tr, t, stuck)
            t += 0.1
        self.assertIn("chain_end", [e.kind for e in ev])
        self.assertNotIn("再現", [e.detail for e in ev if e.kind == "chain_end"][0])
