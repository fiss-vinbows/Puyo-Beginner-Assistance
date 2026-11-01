"""状態追跡(tracker)のテスト。合成した認識結果の時系列で検証する。"""

from __future__ import annotations

import unittest

from src.engine.puyo_field import COLS, ROWS
from src.engine.tracker import (
    NEXT_STABLE_FRAMES,
    OBSERVED_STABLE_FRAMES,
    PLACEMENT_FALLBACK_SECONDS,
    POPPABLE_MAX_WAIT_SECONDS,
    SETTLE_SECONDS,
    PlayerTracker,
    count_puyos,
    drop_top_eyeless,
    Snapshot,
    settle_cells,
)

E = "empty"
R, Y, G, P, B = "red", "yellow", "green", "purple", "blue"
GB = "garbage"
FPS = 30.0


def grid(rows_text: list[str]) -> list[list[str]]:
    """'R.YG#.' 形式の下からの行(足りない上側は空)を cells に変換する。"""
    m = {"R": R, "Y": Y, "G": G, "P": P, "B": B, "#": GB, ".": E}
    cells = [[E] * COLS for _ in range(ROWS)]
    for i, text in enumerate(rows_text):
        row = ROWS - 1 - i
        cells[row] = [m[c] for c in text]
    return cells


class Feeder:
    """同じ認識結果を指定フレーム数(または秒数)流し込むヘルパー。"""

    def __init__(self) -> None:
        self.tracker = PlayerTracker()
        self.t = 0.0
        self.events = []

    def feed(self, cells, next1, next2, pending=0, frames=1, seconds=None):
        n = frames if seconds is None else int(seconds * FPS) + 1
        for _ in range(n):
            self.t += 1 / FPS
            evs = self.tracker.update(Snapshot(self.t, cells, next1, next2, pending))
            self.events.extend(evs)
        return [e.kind for e in self.events]

    def kinds(self):
        return [e.kind for e in self.events]

    def begin(self, base, current, next1, next2):
        """盤面 base を取り込んだ直後(操作中)の状態にする(開始時の推定を経ずに途中の局面から始める)。"""
        tr = self.tracker
        tr.settled = [row[:] for row in base]
        tr._settled_count = count_puyos(base)
        tr.settling = False
        tr.current_pair, tr.next1, tr.next2 = current, next1, next2
        tr.turn = 1
        tr.last_take_in_reason = "stable"

    def falling(self, cells, pair, next1, next2, frames=1):
        """開始前の枠の入れ替わりでは spawn を出さない(2026-09-20)ので、第1手が落ちてくるフレームを流して
        開始(spawn)させる。pair=(子, 軸) を3列目の 11〜10段目、次のフレームで 10〜9段目(落下)に置く。"""
        for top in (2, 3):
            c = [row[:] for row in cells]
            c[top][2], c[top + 1][2] = pair[0], pair[1]
            kinds = self.feed(c, next1, next2, frames=frames)
        return kinds


class HiddenRowUnannouncedChainTest(unittest.TestCase):
    def test_hidden_row_is_not_counted_as_loss(self) -> None:
        # 2026-10-24(実機 07:08・07:11): 確定盤面の13段目(画面に見えない)に2個あると、観測では毎フレーム「2個減った」に見え、
        # 予告なしの消去の疑いで取り込み待ちに入り、思考の手を送らずに窒息した。13段目は比べない
        rows = ["....PP", "....GY", "PGGBGY", "PPBPYP", "YGGGYP", "YPBPBB", "YBBYBP", "GPYGGP", "PPYGYY", "GBYPPY", "GYGGYG",
                "GYBBGY", "BBYYGP"]
        base = grid(list(reversed(rows)))
        f = Feeder()
        f.begin(base, (G, G), (P, B), (B, Y))
        seen = [row[:] for row in base]
        seen[0] = [E] * COLS
        kinds = f.feed(seen, (P, B), (B, Y), frames=10)
        self.assertFalse(f.tracker.settling)
        self.assertNotIn("chain_start", kinds)


class SettleCellsTest(unittest.TestCase):
    def test_floating_removed(self) -> None:
        cells = grid(["RR....", "......", "..G...", "..G..."])  # 3列目の2個は浮いている
        settled = settle_cells(cells)
        self.assertEqual(settled[ROWS - 1][0], R)
        self.assertEqual(settled[ROWS - 3][2], E)

    def test_drop_top_eyeless_keeps_cells_below_eyed_puyo(self) -> None:
        cells = grid(["RG#...", "Y.Y...", "P.B...", "..G..."])
        # 1列目の最上部の紫・2列目のおじゃまの上には何も無い・3列目は目のある緑の下の青が目なし
        eyeless = frozenset({(ROWS - 3, 0), (ROWS - 2, 0), (ROWS - 3, 2)})
        out = drop_top_eyeless(cells, eyeless)
        self.assertEqual(out[ROWS - 3][0], E)  # 最上部の目なしは消す
        self.assertEqual(out[ROWS - 2][0], E)  # その下も目なしが続く限り消す
        self.assertEqual(out[ROWS - 1][0], R)
        self.assertEqual(out[ROWS - 3][2], B)  # 目のあるぷよの下(連鎖の文字の下など)は残す
        self.assertIs(drop_top_eyeless(cells, frozenset()), cells)


class ObservedCountTest(unittest.TestCase):
    """observed_count: 全自動モードが着地を素早く知るための、毎フレーム更新の観測個数。"""

    def test_updates_every_frame_including_before_take_in(self) -> None:
        f = Feeder()
        f.begin(grid(["RRGG.."]), (R, Y), (G, G), (P, R))
        f.feed(grid(["RRGG.."]), (G, G), (P, R), frames=1)
        self.assertEqual(f.tracker.observed_count, 4)
        # 着地した瞬間(まだ確定盤面には取り込まれていない)に +2 される
        f.feed(grid(["RRGGRY"]), (G, G), (P, R), frames=1)
        self.assertEqual(f.tracker.observed_count, 6)
        # 浮いている(落下中の)組ぷよは数えない
        falling = grid(["RRGG.."])
        falling[2][2], falling[3][2] = R, Y
        f.feed(falling, (G, G), (P, R), frames=1)
        self.assertEqual(f.tracker.observed_count, 4)


class TrackerFlowTest(unittest.TestCase):
    def test_spawn_then_turn_ready_after_settle(self) -> None:
        f = Feeder()
        # 開始: 盤面空、ネクスト安定
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        # 操作中の組ぷよが見える(浮いていて、落ちてくる) → 初回の current 推定
        cells = grid([])
        cells[1][2], cells[2][2] = P, P
        f.feed(cells, (R, Y), (G, G), frames=1)
        self.assertIsNone(f.tracker.current_pair)  # 1フレームでは確定しない(開始前の飾りの誤認防止)
        cells = grid([])
        cells[2][2], cells[3][2] = P, P
        f.feed(cells, (R, Y), (G, G), frames=1)
        self.assertEqual(f.tracker.current_pair, (P, P))
        self.assertIn("turn_ready", f.kinds())  # 第1手の手番がすぐ始まる
        self.assertIn("spawn", f.kinds())
        # 設置 → 枠が入れ替わる(spawn) → 盤面が安定し取り込まれる
        placed = grid(["..P...", "..P..."])
        f.feed(placed, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertFalse(f.tracker.settling)
        self.assertEqual(f.tracker.settled, placed)
        self.assertEqual(f.tracker.current_pair, (R, Y))

    def test_next_change_is_spawn_and_current_is_previous_next(self) -> None:
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        # 開始前の枠の入れ替わりだけでは spawn にしない(出現しない組ぷよの手番ができて提案がずれた 2026-09-20)
        self.assertNotIn("spawn", f.kinds())
        self.assertIn("next_prestart", f.kinds())
        self.assertIsNone(f.tracker.current_pair)
        # 第1手(旧ネクスト)が落ちてきたら開始。向きは枠どおり(子, 軸)
        f.falling(grid([]), (R, Y), (G, G), (P, R))
        spawns = [e for e in f.events if e.kind == "spawn"]
        self.assertEqual(len(spawns), 1)
        self.assertIn("開始前の枠の第1手と一致", spawns[0].detail)
        self.assertEqual(f.tracker.current_pair, (R, Y))
        self.assertEqual(f.tracker.next1, (G, G))

    def test_confirmed_start_does_not_wait_for_movement(self) -> None:
        # 対戦の開始を確認済み(相手の初手を検出)なら、第1手と同じ色の2個が見えた最初のフレームで確定する
        # (実機 2026-09-27: 動くのを待って初手が 0.7〜1.3秒遅れ、相手に数手の差をつけられた)
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)  # 開始: 第1手 (R, Y) は枠から消える
        f.tracker.start_confirmed = True
        cells = grid([])
        cells[1][2], cells[2][2] = R, Y  # 出現位置に座ったまま(まだ動いていない)
        f.feed(cells, (G, G), (P, R), frames=1)
        self.assertEqual(f.tracker.current_pair, (R, Y))
        self.assertIn("turn_ready", f.kinds())
        # 色が第1手と違えば(前の試合の残り・飾り)、確認済みでも従来どおり動くのを待つ
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.tracker.start_confirmed = True
        cells = grid([])
        cells[1][2], cells[2][2] = B, B
        f.feed(cells, (G, G), (P, R), frames=1)
        self.assertIsNone(f.tracker.current_pair)

    def test_start_detected_by_axis_puyo(self) -> None:
        # 盤面が空の試合開始で、3列目12段目に初手の軸ぷよ(枠の第1手の軸の色)が1個現れたら、その時点で開始とする
        # (2個が見えるまで待つと約0.3秒遅れた。実機 2026-10-11)。ただし入れ替わりを見ていない時は、枠の第1手が出現したのか
        # (同じフレームで出現と入れ替わり)、もう2手目なのか(リセットの後・検出の遅れ)を軸だけでは区別できないので、
        # 2個が見えるまで待つ(実機 2026-10-24)。枠がまだ入れ替わっていなければネクネクは未読
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        cells = grid([])
        cells[1][2] = Y  # (子, 軸) = (赤, 黄) の軸が12段目に出た(子は13段目で見えない)
        f.feed(cells, (R, Y), (G, G), frames=2)
        self.assertNotIn("spawn", f.kinds())
        cells = grid([])
        cells[1][2], cells[2][2] = R, Y  # 子が12段目に下りて2個とも見えた
        f.feed(cells, (R, Y), (G, G), frames=2)
        spawns = [e for e in f.events if e.kind == "spawn"]
        self.assertEqual(len(spawns), 1)
        self.assertIn("軸ぷよ", spawns[0].detail)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((R, Y), (G, G), None))
        self.assertIn("turn_ready", f.kinds())
        # 枠が入れ替わったらネクネクを補う(spawn は増えない)
        f.feed(cells, (G, G), (P, P), frames=NEXT_STABLE_FRAMES)
        self.assertEqual(tr.next2, (P, P))
        self.assertEqual(f.kinds().count("spawn"), 1)

    def test_start_by_axis_after_frame_swap(self) -> None:
        # 直前の入れ替わりが開始だった(第1手は枠から消えた組、枠は第2・3手)
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)  # 開始の入れ替わり
        cells = grid([])
        cells[1][2] = Y
        f.feed(cells, (G, G), (P, R), frames=2)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((R, Y), (G, G), (P, R)))

    def test_start_trusts_frame_first_over_misread_spawn(self) -> None:
        # 実機 2026-10-25 12:33: 初手 (紫, 黄) の出現の瞬間を黄黄と読み、相手の初手も黄黄だった。直前の入れ替わりで枠から消えた組と
        # 1色でも同じなら枠の読みを信じる(先読みと合わず初手が約0.95秒遅れ、黄黄で考えた手を置いた)
        f = Feeder()
        f.feed(grid([]), (P, Y), (R, R), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (R, R), (Y, Y), frames=NEXT_STABLE_FRAMES)  # 開始の入れ替わり(第1手は紫黄)
        f.tracker.start_hint = ((Y, Y), (R, R), (Y, Y))  # 相手の初手の読み違い
        cells = grid([])
        cells[1][2], cells[2][2] = Y, Y  # 子も黄と読んだ
        f.feed(cells, (R, R), (Y, Y), frames=2)
        spawns = [e for e in f.events if e.kind == "spawn"]
        self.assertEqual(len(spawns), 1)
        self.assertIn("枠から消えた組", spawns[0].detail)
        self.assertEqual((f.tracker.current_pair, f.tracker.next1, f.tracker.next2), ((P, Y), (R, R), (Y, Y)))
        # 2個が見えて決める側(相手)も、枠から消えた組と1色同じなら枠の読みにする
        f = Feeder()
        f.feed(grid([]), (B, Y), (R, R), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (R, R), (Y, Y), frames=NEXT_STABLE_FRAMES)
        tr = f.tracker
        tr.start_confirmed = True
        tr._detect_start_by_axis = lambda *a: False  # type: ignore[method-assign]  # 2個で決める側だけを試す
        f.feed(cells, (R, R), (Y, Y), frames=1)
        cand = [e for e in f.events if e.kind == "spawn"]
        self.assertEqual(tr.current_pair, (B, Y))
        self.assertIn("出現の読みは", cand[0].detail)
        # 入れ替わりから時間が経っていれば(枠の第1手を信じる根拠が弱い)、従来どおり見えた2個
        f = Feeder()
        f.feed(grid([]), (B, Y), (R, R), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (R, R), (Y, Y), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (R, R), (Y, Y), seconds=0.6)
        f.tracker._detect_start_by_axis = lambda *a: False  # type: ignore[method-assign]
        f.tracker.start_confirmed = True
        f.feed(cells, (R, R), (Y, Y), frames=3)
        cells2 = grid([])
        cells2[2][2], cells2[3][2] = Y, Y  # 1段下がった
        f.feed(cells2, (R, R), (Y, Y), frames=1)
        self.assertEqual(f.tracker.current_pair, (Y, Y))

    def test_start_after_reset_uses_visible_pair(self) -> None:
        # 実機 2026-10-24 07:08: 開始の入れ替わりの直後に新しい試合のリセットが入り、作り直した追跡は入れ替わり後の枠
        # (第2・3手)しか見ていない。軸の色が枠の第1手(実は2手目)の軸と同じでも、見えた2個が違えば見えた2個を初手にする
        f = Feeder()
        f.feed(grid([]), (P, B), (Y, P), frames=NEXT_STABLE_FRAMES)  # 第1手 (青, 紫) は既に枠から消えている
        cells = grid([])
        cells[1][2] = P
        f.feed(cells, (P, B), (Y, P), frames=2)
        self.assertNotIn("spawn", f.kinds())  # 軸だけでは決めない(以前は枠の第1手 (紫, 青) で開始した)
        cells = grid([])
        cells[1][2], cells[2][2] = B, P
        f.feed(cells, (P, B), (Y, P), frames=2)
        spawns = [e for e in f.events if e.kind == "spawn"]
        self.assertEqual(len(spawns), 1)
        self.assertIn("見えた2個", spawns[0].detail)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((B, P), (P, B), (Y, P)))

    def test_start_detected_late_after_swap_uses_visible_pair(self) -> None:
        # 実機 2026-10-24 13:47 の5試合目: 入れ替わりから0.51秒後に検出し(窓は0.5秒)、見えている一番上(子の紫)を軸として
        # 枠の第1手(実は2手目)の紫紫で開始した。初手 (紫, 赤) を置いた後、2手目の紫紫が出ても枠が変わらず自由落下した
        f = Feeder()
        f.feed(grid([]), (P, R), (P, P), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (P, P), (B, R), frames=NEXT_STABLE_FRAMES)  # 開始の入れ替わり(第1手は紫赤)
        f.feed(grid([]), (P, P), (B, R), seconds=0.6)
        cells = grid([])
        cells[1][2], cells[2][2] = P, R
        f.feed(cells, (P, P), (B, R), frames=2)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((P, R), (P, P), (B, R)))
        # 相手の初手が分かっていれば、子が見えていても2個で照合して相手の初手で開始する
        f = Feeder()
        f.feed(grid([]), (P, P), (B, R), frames=NEXT_STABLE_FRAMES)
        f.tracker.start_hint = ((P, R), (P, P), (B, R))
        f.feed(cells, (P, P), (B, R), frames=2)
        spawns = [e for e in f.events if e.kind == "spawn"]
        self.assertIn("相手の初手と同じ", spawns[0].detail)
        self.assertEqual(f.tracker.current_pair, (P, R))

    def test_start_not_detected_from_eyeless_mark(self) -> None:
        # 窒息の位置の×印・演出(色だが目が無い)は軸ぷよとみなさない
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        cells = grid([])
        cells[1][2] = Y
        for _ in range(4):
            f.t += 1 / FPS
            f.events.extend(f.tracker.update(Snapshot(f.t, cells, (R, Y), (G, G), 0, eyeless_cells=frozenset({(1, 2)}))))
        self.assertNotIn("spawn", f.kinds())
        self.assertIsNone(f.tracker.current_pair)

    def test_start_uses_enemy_first_pair_hint(self) -> None:
        # 相手の初手(両者のツモは同じ)を候補にする: 枠の読みと違っても、軸の色が合えば相手の初手で開始する
        f = Feeder()
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.tracker.start_hint = ((R, Y), (G, G), (P, R))
        cells = grid([])
        cells[1][2] = Y
        f.feed(cells, (G, G), (P, R), frames=2)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((R, Y), (G, G), (P, R)))

    def test_aa_aa_start_is_turn_zero(self) -> None:
        # 開始で第1手と第2手が同じ組ぷよ(AA-AA)だと、開始の入れ替わりでネクスト枠は変わらずネクネクだけが変わる。
        # 手番を進めず開始前の入れ替わりとし、軸ぷよの出現で初手(turn 0)を始める(実機 2026-10-11: 2手目と判定して約0.6秒遅れた)
        f = Feeder()
        f.feed(grid([]), (G, G), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, Y), frames=NEXT_STABLE_FRAMES)  # 開始: 第1手 (緑緑) が出て、枠は (緑緑, 紫黄)
        self.assertNotIn("spawn", f.kinds())
        cells = grid([])
        cells[1][2] = G
        f.feed(cells, (G, G), (P, Y), frames=2)
        spawns = [e for e in f.events if e.kind == "spawn"]
        self.assertEqual(len(spawns), 1)
        self.assertIn("turn=0", spawns[0].detail)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((G, G), (G, G), (P, Y)))

    def test_start_ignores_color_seen_before_swap(self) -> None:
        # 入れ替わりより前から3列目12段目に見えている色(窒息の位置の×印を色ぷよと読んだもの)は軸ぷよとみなさない
        # (実機 2026-10-11: ×印を赤と読み、入れ替わりの直後に枠の第1手(実は2手目)の赤赤で開始と判定した)
        f = Feeder()
        mark = grid([])
        mark[1][2] = R
        f.feed(mark, (Y, Y), (R, R), frames=NEXT_STABLE_FRAMES + 2)  # 開始前から赤い印が見えている
        f.feed(mark, (R, R), (G, Y), frames=NEXT_STABLE_FRAMES + 2)  # 開始の入れ替わり(第1手は黄黄)
        self.assertNotIn("spawn", f.kinds())
        self.assertIsNone(f.tracker.current_pair)
        # 黄黄の軸が現れたら黄黄で開始する
        cells = grid([])
        cells[1][2] = Y
        f.feed(cells, (R, R), (G, Y), frames=2)
        tr = f.tracker
        self.assertEqual((tr.current_pair, tr.next1, tr.next2), ((Y, Y), (R, R), (G, Y)))

    def test_same_pair_twice_in_a_row_is_still_a_spawn(self) -> None:
        f = Feeder()
        # 試合中(手番あり): ネクスト=ネクネク=(黄,黄)。設置後もネクスト枠は(黄,黄)のままでネクネクだけ変わる
        f.begin(grid(["R....."]), (B, B), (Y, Y), (Y, Y))
        f.tracker.turn = 3
        f.feed(grid(["R....."]), (Y, Y), (Y, Y), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (Y, Y), (R, G), frames=NEXT_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("spawn"), 1)
        self.assertEqual(f.tracker.current_pair, (Y, Y))
        self.assertEqual(f.tracker.next1, (Y, Y))
        self.assertEqual(f.tracker.next2, (R, G))
        # 次の設置ではネクストが(赤,緑)に変わる通常の spawn
        f.feed(grid([]), (R, G), (P, P), frames=NEXT_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (Y, Y))

    def test_next_flicker_is_ignored(self) -> None:
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        # 1〜2フレームだけ別の色に見える(枠の色の混入)
        f.feed(grid([]), (R, R), (G, G), frames=NEXT_STABLE_FRAMES - 1)
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        self.assertNotIn("spawn", f.kinds())

    def test_moving_pair_during_play_does_not_change_settled(self) -> None:
        f = Feeder()
        base = grid(["RRGG..", "YY...."])
        f.feed(base, (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)  # spawn → 取り込み
        self.assertEqual(f.tracker.settled, base)
        self.assertFalse(f.tracker.settling)
        # 操作中: 組ぷよが柱の上に乗って「下に空きが無い」ように見える
        moving = grid(["RRGG..", "YYY...", "..Y..."])
        f.feed(moving, (G, G), (P, R), seconds=1.0)
        self.assertEqual(f.tracker.settled, base)
        self.assertNotIn("chain_start", f.kinds())

    def test_chain_detected_and_turn_ready_after_chain(self) -> None:
        f = Feeder()
        base = grid(["RRRGGG", "GGG..."])
        f.begin(base, (R, Y), (G, G), (P, R))
        # 次の設置 → spawn → 消去アニメで盤面が減る → 落下 → 安定
        f.feed(grid(["RRRGGG", "GGGR..", "...R.."]), (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("spawn"), 1)
        f.feed(grid(["...GGG", "GGG..."]), (P, R), (Y, Y), seconds=0.25)  # 1段目の消去(連鎖の1段は0.2s超)
        self.assertTrue(f.tracker.chain_active)
        f.feed(grid(["GGGGGG"]), (P, R), (Y, Y), seconds=0.25)  # 2段目
        f.feed(grid([]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        kinds = f.kinds()
        self.assertIn("chain_end", kinds)
        self.assertEqual(f.tracker.chain_steps, 2)
        self.assertEqual(kinds.count("turn_ready"), 1)  # begin() は取り込みを経ないので連鎖後の1回だけ
        self.assertFalse(f.tracker.chain_active)

    def test_next_turn_after_chain_is_taken_in_without_waiting(self) -> None:
        # 実機 2026-10-10 14:08: 自分の連鎖が終わった後、次の組ぷよが出ても盤面は連鎖の終わりと同じなのに、
        # 安定待ち(0.34秒)をしてから取り込んでいた。連鎖の終わりの盤面を次の手番の予想盤面にして即取り込む
        f = Feeder()
        base = grid(["RRRGGG", "GGG..."])
        f.begin(base, (R, Y), (G, G), (P, R))
        f.feed(grid(["RRRGGG", "GGGR..", "...R.."]), (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES)  # 設置 → spawn
        f.feed(grid(["...GGG", "GGG..."]), (P, R), (Y, Y), seconds=0.25)
        f.feed(grid(["GGGGGG"]), (P, R), (Y, Y), seconds=0.25)
        f.feed(grid(["B....."]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)  # 連鎖の終わり(1個残る)
        self.assertIn("chain_end", f.kinds())
        n_ready = f.kinds().count("turn_ready")
        # 次の組ぷよ(P,R)が出る(枠が入れ替わる)。盤面は連鎖の終わりのまま
        f.feed(grid(["B....."]), (Y, Y), (G, R), frames=NEXT_STABLE_FRAMES + 2)
        self.assertEqual(f.kinds().count("turn_ready"), n_ready + 1)
        self.assertIn("(expected)", [e.detail for e in f.events if e.kind == "turn_ready"][-1])

    def test_simple_placement_is_taken_in_without_waiting(self) -> None:
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)  # 開始時の盤面を取り込む
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        # 設置(+2個、連鎖なし、予告なし) → 観測が2フレーム一致した時点で turn_ready
        f.feed(grid(["RRGGRY"]), (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES + OBSERVED_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertIn("(simple)", f.events[-1].detail)

    def test_placement_with_landed_garbage_is_taken_in_without_waiting(self) -> None:
        # 実機 2026-10-11 11:07: おじゃまの後の手番は出現から取り込みまで中央0.37秒(安定待ち)で、組ぷよが自由落下した。
        # ゲームはおじゃまが降り終わってから次の組ぷよを出すので、浮いているのが操作中の組ぷよだけなら即取り込む
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        landed = grid(["RRGGRY", "######"])  # 設置の2個 + 降り終わったおじゃま1段
        moving = [row[:] for row in landed]
        moving[1][2], moving[2][2] = G, G  # 出現した組ぷよ(浮いている)
        f.feed(moving, (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES + OBSERVED_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertIn("(garbage)", f.events[-1].detail)
        self.assertEqual(f.tracker.settled, landed)

    def test_garbage_landing_rechecked_every_frame(self) -> None:
        # 2026-10-26(実機 20:32 の 2P 465s): 盤面の読みが変わった瞬間に浮いているもの(演出・落下中のおじゃま)が見えていると、
        # その後に消えても読みが変わらないので安定待ちの0.3秒を待ち、高い積みで組ぷよが自由落下した。毎フレーム見直す
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        landed = grid(["RRGGRY", "######"])
        with_effect = [row[:] for row in landed]
        with_effect[3][0] = GB  # 宙に浮いたおじゃま(演出・落下中)
        f.feed(with_effect, (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES + OBSERVED_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("turn_ready"), 1)  # 浮いているおじゃまがある間は取り込まない
        t0 = f.t
        f.feed(landed, (P, R), (Y, Y), frames=3)  # 浮いていたものが消えた(確定盤面の読みは同じ)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertIn("(garbage)", [e for e in f.events if e.kind == "turn_ready"][-1].detail)
        self.assertLess(f.t - t0, SETTLE_SECONDS)

    def test_falling_garbage_is_not_taken_in(self) -> None:
        # おじゃまがまだ落ちている(浮いている)間は即取り込みしない
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (G, G), (P, R), pending=1, seconds=SETTLE_SECONDS + 0.1)  # 予告が見えている
        falling = grid(["RRGGRY", "......", "######"])
        f.feed(falling, (P, R), (Y, Y), pending=1, frames=NEXT_STABLE_FRAMES + OBSERVED_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("turn_ready"), 1)

    def test_spawned_pair_on_tall_third_column_is_not_taken_in(self) -> None:
        # 実機 2026-09-15: 3列目が11段目まで積まれていると、スポーン直後の組ぷよ(12・13段目の
        # 3列目)が積みに接触して浮いていないように見え、+2個の簡易取り込みで12段目3列目に
        # 混入した。ルール上そのマスが埋まると負けなので、確定盤面では常に空として扱う
        f = Feeder()
        col3 = ["..R...", "..G...", "..R...", "..G...", "..R...", "..G...", "..R...", "..G...", "..R...", "..G...", "..R..."]
        base = grid(col3)  # 3列目が11段目まで
        f.feed(base, (P, P), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (Y, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        # 前の組ぷよ(P,P)を4列目に縦置き(+2個)。同時に新しい組ぷよ(Y,Y)が3列目の12・13段目に出現し、
        # 12段目の1個が積みの真上に接触して見える
        observed = grid(["..RP..", "..GP.."] + col3[2:] + ["..Y..."])
        f.feed(observed, (G, G), (R, R), frames=NEXT_STABLE_FRAMES + OBSERVED_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertIn("(simple)", f.events[-1].detail)
        self.assertEqual(f.tracker.settled[1][2], E)  # 12段目3列目は空のまま
        self.assertEqual(f.tracker.settled[0][2], E)
        self.assertEqual(f.tracker.settled[ROWS - 1][3], P)  # 設置した2個は取り込まれている
        self.assertEqual(f.tracker.settled[ROWS - 2][3], P)

    def test_placement_that_pops_waits_for_chain(self) -> None:
        f = Feeder()
        base = grid(["RRR..."])
        f.feed(base, (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        # 赤4個がそろう設置 → 消える前の盤面では取り込まない
        f.feed(grid(["RRRR..", "...Y.."]), (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES + OBSERVED_STABLE_FRAMES)
        self.assertEqual(f.kinds().count("turn_ready"), 1)
        f.feed(grid(["...Y.."]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertIn("chain_end", f.kinds())

    def test_expected_board_is_taken_in_immediately(self) -> None:
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        expected = grid(["RRGG..", "RY...."])
        f.tracker.set_expected(expected)
        # 設置後に連鎖が起きる盤面でも、期待通りなら1フレームで取り込む
        f.feed(expected, (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES + 1)
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertIn("(expected)", f.events[-1].detail)
        self.assertEqual(f.tracker.settled, expected)

    def test_garbage_rise_event(self) -> None:
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(grid([]), (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["RRGGRY", "######"]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1 + NEXT_STABLE_FRAMES / FPS)
        self.assertIn("garbage_rise", f.kinds())

    def test_spawned_pair_touching_stack_is_stripped(self) -> None:
        from src.engine.tracker import strip_moving_pair

        prev = grid(["RRGG..", "RR....", "YY...."])
        # 設置した黄2個(4列目)に加え、3列目に高く積まれた上でスポーンした組ぷよ(緑・紫)が接触して見える
        observed = grid(["RRGG..", "RR.Y..", "YYGY..", "..P..."])
        stripped = strip_moving_pair(observed, prev, (P, G))
        self.assertEqual(stripped, grid(["RRGG..", "RR.Y..", "YY.Y.."]))
        # 増分が大きすぎる(起動直後など)場合は触らない
        many = grid(["RRGG..", "RRYY..", "YYGGPP"])
        self.assertEqual(strip_moving_pair(many, grid([]), (P, G)), many)

    def test_waits_while_poppable_group_remains(self) -> None:
        # 連鎖の段間: 消えて落下した後の盤面に次に消える4連結が残っていれば取り込まない
        f = Feeder()
        base = grid(["RRRGGG", ".GG..."])
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["RRRGGG", "RGG...", "Y....."]), (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES)  # 赤4個
        mid = grid(["YGGGGG"])  # 赤が消えて落下した直後(緑5連結が次に消える)
        f.feed(mid, (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.2)
        self.assertEqual(f.kinds().count("turn_ready"), 1)
        f.feed(grid(["Y....."]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        self.assertEqual(f.kinds().count("turn_ready"), 2)

    def test_persistent_fake_poppable_group_does_not_block_forever(self) -> None:
        # 誤認識で「消えるはずの4連結」が見え続ける(全消し後の「ぜんけし」表示等)と、
        # spawn のたびに待ち時間がリセットされて永遠に取り込めなかった(実機 2026-09-16、17秒停止)。
        # 4連結が見え続けている時間は spawn をまたいで数え、上限を超えたら取り込む
        f = Feeder()
        base = grid(["RRGG.."])
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        # 偽の4連結(YYYY)が見える盤面で、手番が1秒ごとに進む
        fake = grid(["YYYY..", "RRGG.."])
        f.feed(fake, (G, G), (P, R), seconds=1.0)  # spawn 1
        f.feed(fake, (P, R), (Y, Y), seconds=1.0)  # spawn 2
        f.feed(fake, (Y, Y), (R, R), seconds=1.0)  # spawn 3
        # 4連結が見え始めてから POPPABLE_MAX_WAIT_SECONDS 経過した時点で取り込まれる
        ready = [e.t for e in f.events if e.kind == "turn_ready"]
        self.assertEqual(len(ready), 1)
        self.assertLess(ready[0], 0.5 + POPPABLE_MAX_WAIT_SECONDS + 0.2)

    def test_all_clear_flag(self) -> None:
        # 連鎖で盤面が空になったら全消しボーナス持ち。次の連鎖で消費される
        f = Feeder()
        f.feed(grid([]), (Y, Y), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY.."]), (Y, Y), (G, P), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY..", "..YY.."]), (G, P), (R, R), frames=3)
        f.feed(grid([]), (G, P), (R, R), seconds=SETTLE_SECONDS + 0.1)
        self.assertIn("all_clear", f.kinds())
        self.assertTrue(f.tracker.all_clear)
        f.feed(grid(["GRRR.."]), (R, R), (G, G), seconds=SETTLE_SECONDS + 0.1)
        self.assertTrue(f.tracker.all_clear)  # 設置しただけでは消えない
        f.feed(grid(["GRRRRG"]), (G, G), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["G....G"]), (Y, Y), (R, R), seconds=SETTLE_SECONDS + 0.1)  # 設置(spawn)→消去
        self.assertFalse(f.tracker.all_clear)  # 連鎖で消費

    def test_chain_bonus_kept_while_chain_runs(self) -> None:
        # 全消しボーナス持ちの連鎖: chain_start で all_clear は落ちるが、その連鎖に+30個が乗ることは
        # chain_bonus で保持し、chain_end で消える(分析文書 2026-09-19 P1-1)。発火直前の盤面も残す
        f = Feeder()
        f.feed(grid([]), (Y, Y), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY.."]), (Y, Y), (G, P), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY..", "..YY.."]), (G, P), (R, R), frames=3)
        f.feed(grid([]), (G, P), (R, R), seconds=SETTLE_SECONDS + 0.1)
        self.assertTrue(f.tracker.all_clear)
        self.assertFalse(f.tracker.chain_bonus)
        before = grid(["GRRR.."])
        f.feed(before, (R, R), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["GRRRRG"]), (G, G), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["G....G"]), (Y, Y), (R, R), seconds=0.25)  # 消去中
        self.assertTrue(f.tracker.chain_active)
        self.assertFalse(f.tracker.all_clear)
        self.assertTrue(f.tracker.chain_bonus)
        self.assertIsNotNone(f.tracker.chain_before)
        self.assertGreaterEqual(count_puyos(f.tracker.chain_before), count_puyos(before))  # 消える前の確定盤面
        self.assertIn("(全消しボーナス付き)", [e.detail for e in f.events if e.kind == "chain_start"][-1])
        f.feed(grid(["G....G"]), (Y, Y), (R, R), seconds=SETTLE_SECONDS + 0.1)
        self.assertIn("chain_end", f.kinds())
        self.assertFalse(f.tracker.chain_bonus)
        self.assertEqual(f.tracker.last_take_in_reason, "stable")

    def test_observed_cells_eyed_drops_eyeless_color(self) -> None:
        # 着地予測の点・攻撃が当たる演出(白い目の無い色)を空にした観測盤面も残す(実機 2026-10-12 07:00)
        f = Feeder()
        cells = grid(["RY...."])
        cells[ROWS - 2][0] = P  # 赤の上に、目の無い紫(演出)が積まれているように見えた
        f.t += 1 / FPS
        f.tracker.update(Snapshot(f.t, cells, (R, Y), (G, G), 0, eyeless_cells=frozenset({(ROWS - 2, 0)})))
        # 列の最上部に乗った目なしは観測盤面からも除く(drop_top_eyeless、2026-10-20)
        self.assertEqual(count_puyos(f.tracker.observed_cells), 2)
        self.assertEqual(f.tracker.observed_cells_eyed, grid(["RY...."]))
        f.t += 1 / FPS
        f.tracker.update(Snapshot(f.t, grid(["RY...."]), (R, Y), (G, G), 0))
        self.assertIs(f.tracker.observed_cells_eyed, f.tracker.observed_cells)  # 目の無い色が無ければ同じ盤面

    def test_effect_on_top_of_stack_is_not_taken_in(self) -> None:
        # 攻撃の光(白い目の無い色)が積みの最上部に重なると浮いていないと判定され、偽の色ぷよとして確定盤面に入った
        # (実機 2026-10-12: 相手側で5色目 'red'。評価データでは本物のぷよに目なしは0)。最上部の目なしは盤面に入れない
        f = Feeder()
        stack = ["Y.....", "G.....", "Y.....", "G.....", "Y.....", "G.....", "Y.....", "G.....", "Y.....", "G....."]
        f.begin(grid(stack), (B, P), (G, G), (Y, Y))
        placed = grid([s[:4] + "BP" if i == 0 else s for i, s in enumerate(stack)])
        observed = [row[:] for row in placed]
        observed[ROWS - 11][0] = R  # 11段目1列目に光(目なし)
        eyeless = frozenset({(ROWS - 11, 0)})
        for _ in range(NEXT_STABLE_FRAMES + int((SETTLE_SECONDS + 0.1) * FPS)):
            f.t += 1 / FPS
            f.events.extend(f.tracker.update(Snapshot(f.t, observed, (Y, Y), (P, P), 0, eyeless_cells=eyeless)))
        self.assertIn("turn_ready", f.kinds())
        self.assertEqual(f.tracker.settled, placed)  # 光の赤は入らない
        self.assertEqual(f.tracker.last_take_in_reason, "simple")  # +2個の単純な設置として即取り込める

    def test_eyeless_top_kept_during_chain(self) -> None:
        # 連鎖中は消える途中のぷよ(泡で目が無い)を先に消さない(段が割れて再現の点数が使われなかった 録画 09-13 の 88.4s)
        f = Feeder()
        f.begin(grid(["YG....", "YG....", "GY...."]), (B, P), (G, G), (Y, Y))
        f.tracker.chain_active = True
        cells = grid(["YG....", "YG....", "GY...."])
        f.t += 1 / FPS
        f.tracker.update(Snapshot(f.t, cells, (G, G), (Y, Y), 0, eyeless_cells=frozenset({(ROWS - 3, 0)})))
        self.assertEqual(f.tracker.observed_count, 6)

    def test_covered_frame_is_ignored(self) -> None:
        # 全消し直後のキャラクターのカットインが盤面を覆い、顔や服が色ぷよとして読まれた(実機 2026-09-17)。
        # 確定盤面に入りうる位置に「目の無い色セル」が多いフレームは盤面更新に使わない
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))
        f.feed(grid(["RY...."]), (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["RY...."]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)  # GG は 13段目に?(+0)…ここでは spawn だけ
        fake = grid(["RYRYRY", "YRYRYR", "GGGGGG", "RRRRRR"])  # カットインが色ぷよとして読まれた盤面
        eyeless = frozenset((r, c) for r in range(ROWS - 4, ROWS) for c in range(COLS))
        for _ in range(int(2.5 * FPS)):
            f.t += 1 / FPS
            f.events.extend(f.tracker.update(Snapshot(f.t, fake, (P, R), (Y, Y), 0, eyeless_cells=eyeless)))
        self.assertTrue(f.tracker.last_unknown)
        self.assertEqual(count_puyos(f.tracker.settled), 2)  # 偽の盤面は取り込まれない
        self.assertNotIn("chain_start", f.kinds())
        # カットイン中に読めたネクスト(偽の色)も採用しない
        f.t += 1 / FPS
        f.tracker.update(Snapshot(f.t, fake, ("green", "green"), ("green", "green"), 0, eyeless_cells=eyeless))
        self.assertEqual(f.tracker.next1, (P, R))

    def test_all_clear_inferred_from_cutin(self) -> None:
        # 全消しの瞬間はカットインで盤面が覆われ「0個」を見逃す(実機 2026-09-17: 相手の全消しを検出できず)。
        # 小さい盤面からカットインが0.5秒以上続いたら全消しとみなす
        f = Feeder()
        f.feed(grid([]), (Y, Y), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY.."]), (Y, Y), (G, P), seconds=SETTLE_SECONDS + 0.1)
        fake = grid(["RYRYRY", "YRYRYR", "GGGGGG", "RRRRRR"])
        eyeless = frozenset((r, c) for r in range(ROWS - 4, ROWS) for c in range(COLS))
        for _ in range(int(0.7 * FPS)):
            f.t += 1 / FPS
            f.events.extend(f.tracker.update(Snapshot(f.t, fake, (G, P), (R, R), 0, eyeless_cells=eyeless)))
        self.assertFalse(f.tracker.all_clear)  # カットイン中はまだ確定しない
        f.feed(grid([]), (G, P), (R, R), frames=1)  # カットインが終わって盤面が空
        self.assertIn("all_clear", f.kinds())
        self.assertTrue(f.tracker.all_clear)

    def test_cutin_at_game_start_is_not_all_clear(self) -> None:
        # 試合開始の演出(空の盤面からのカットイン)を全消しと誤判定しない(実機 2026-09-17)
        f = Feeder()
        fake = grid(["RYRYRY", "YRYRYR", "GGGGGG", "RRRRRR"])
        eyeless = frozenset((r, c) for r in range(ROWS - 4, ROWS) for c in range(COLS))
        for _ in range(int(1.0 * FPS)):
            f.t += 1 / FPS
            f.events.extend(f.tracker.update(Snapshot(f.t, fake, (G, P), (R, R), 0, eyeless_cells=eyeless)))
        f.feed(grid([]), (G, P), (R, R), frames=3)
        self.assertFalse(f.tracker.all_clear)
        self.assertNotIn("all_clear", f.kinds())

    def test_cutin_on_big_board_is_not_all_clear(self) -> None:
        f = Feeder()
        base = grid(["RRGGPP", "GGPPRR", "PPRRGG"])
        f.feed(base, (Y, Y), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (Y, Y), (G, P), seconds=SETTLE_SECONDS + 0.1)
        fake = grid(["RYRYRY", "YRYRYR", "GGGGGG", "RRRRRR"])
        eyeless = frozenset((r, c) for r in range(ROWS - 4, ROWS) for c in range(COLS))
        for _ in range(int(0.7 * FPS)):
            f.t += 1 / FPS
            f.events.extend(f.tracker.update(Snapshot(f.t, fake, (G, P), (R, R), 0, eyeless_cells=eyeless)))
        f.feed(grid([]), (G, P), (R, R), frames=3)
        self.assertFalse(f.tracker.all_clear)  # 18個の盤面からのカットインは全消しではない(大連鎖の演出等)

    def test_all_clear_survives_spawn_before_chain_end(self) -> None:
        # 連鎖で盤面が空になった直後に次の spawn が来ても全消しボーナスを記録する(実機 2026-09-17)
        f = Feeder()
        f.feed(grid([]), (Y, Y), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY.."]), (Y, Y), (G, P), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY..", "..YY.."]), (G, P), (R, R), frames=3)
        f.feed(grid([]), (G, P), (R, R), frames=3)  # 消えた直後(まだ chain_end 前)
        f.feed(grid([]), (R, R), (G, G), seconds=SETTLE_SECONDS + 0.1)  # 次の spawn
        self.assertTrue(f.tracker.all_clear)

    def test_all_clear_banner_does_not_block_board_update(self) -> None:
        # 全消し後に盤面上部へ重なる「全消し」バナーは黄・おじゃま・不明として読まれる(実機 2026-09-16 の
        # 生の認識結果そのまま)。浮いているので確定盤面には影響させず、下の設置は取り込む
        f = Feeder()
        f.feed(grid([]), (Y, Y), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid(["..YY.."]), (Y, Y), (G, P), seconds=SETTLE_SECONDS + 0.1)  # turn 1
        banner = [[E] * COLS for _ in range(ROWS)]
        for row, text in ((1, "....G."), (2, ".YYYG."), (3, ".??#Y."), (4, "..??..")):
            banner[row] = [{"G": G, "Y": Y, "#": "garbage", "?": "unknown", ".": E}[c] for c in text]
        banner[ROWS - 1] = [E, E, Y, Y, E, E]
        banner[ROWS - 2] = [E, E, G, P, E, E]
        f.feed(banner, (G, P), (R, R), seconds=SETTLE_SECONDS + 0.1)  # turn 2: GP を横に乗せた
        self.assertEqual(f.kinds().count("turn_ready"), 2)
        self.assertEqual(count_puyos(f.tracker.settled), 4)  # バナーは取り込まれない

    def test_placement_detected_from_board_while_next_is_unknown(self) -> None:
        # 演出でネクストが読めない間に設置しても手番が進むこと(実機 2026-09-17: 3手置いても5〜7秒進まなかった)
        U = "unknown"
        B = "blue"
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))  # spawn 1: 操作中 RY
        f.feed(grid(["RY...."]), (P, R), (B, B), seconds=SETTLE_SECONDS + 0.1)  # RY 設置 → spawn 2: 操作中 GG
        self.assertEqual(f.tracker.current_pair, (G, G))
        self.assertEqual(f.kinds().count("spawn"), 2)
        # ネクストが読めなくなり、その間に GG を置いた
        f.feed(grid(["RY...."]), (U, U), (U, U), seconds=0.2)
        f.feed(grid(["RYGG.."]), (U, U), (U, U), seconds=PLACEMENT_FALLBACK_SECONDS + SETTLE_SECONDS + 0.1)
        self.assertEqual(f.kinds().count("spawn"), 3)
        self.assertEqual(f.tracker.current_pair, (P, R))  # 旧ネクストが操作中に
        self.assertEqual(f.tracker.next1, (B, B))
        self.assertIsNone(f.tracker.next2)
        self.assertEqual(count_puyos(f.tracker.settled), 4)  # 設置は取り込まれる
        # ネクストが読めるようになったらネクネクを補う(spawn は増えない)
        f.feed(grid(["RYGG.."]), (B, B), (Y, Y), frames=3)
        self.assertEqual(f.tracker.next2, (Y, Y))
        self.assertEqual(f.kinds().count("spawn"), 3)

    def test_no_fallback_while_next_is_readable(self) -> None:
        # ネクストが読めている時は盤面だけで手番を進めない(操作中の組ぷよが積みに触れても誤検出しない)
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))  # spawn 1: 操作中 RY
        f.feed(grid(["RY...."]), (G, G), (P, R), seconds=1.0)  # ネクストは変わらないまま盤面だけ +2
        self.assertEqual(f.kinds().count("spawn"), 1)

    def test_spawn_wait_diagnostic_when_next_does_not_change(self) -> None:
        # 診断(2026-10-10): 着地後に枠が変わらず spawn を検出できない状態が続いたら spawn_wait を記録する。
        # 手番の判定は変えない(spawn は増えない)
        from src.engine.tracker import SPAWN_WAIT_DIAG_MAX, SPAWN_WAIT_DIAG_SECONDS

        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))  # spawn 1: 操作中 RY
        f.feed(grid(["RY...."]), (G, G), (P, R), seconds=SPAWN_WAIT_DIAG_SECONDS + 1.0)
        waits = [e for e in f.events if e.kind == "spawn_wait"]
        self.assertTrue(waits)
        self.assertLessEqual(len(waits), SPAWN_WAIT_DIAG_MAX)
        self.assertIn("枠の読み=", waits[0].detail)
        self.assertIn("増えた=", waits[0].detail)
        self.assertEqual(f.kinds().count("spawn"), 1)

    def test_no_spawn_wait_when_spawn_is_detected(self) -> None:
        # ふつうに枠が入れ替わって spawn を検出した時は記録しない
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))  # spawn 1: 操作中 RY
        f.feed(grid(["RY...."]), (G, G), (P, R), seconds=0.2)  # 着地
        f.feed(grid(["RY...."]), (P, R), (B, B), seconds=1.0)  # 枠が入れ替わる = spawn 2
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertNotIn("spawn_wait", f.kinds())

    def test_three_identical_pairs_advance_while_falling_flickers(self) -> None:
        # 実機 2026-10-10 14:08: 緑赤が3組続き、自由落下中の組ぷよの見え方がマスの境目で揺れて(緑赤 → 緑 → 緑緑赤)
        # 「浮いているのが2色ちょうど」の条件では0.3秒続かず、3.5秒以上手番を進められなかった。
        # 浮いているぷよがどれも次の組ぷよの色なら、見え方が揺れても盤面検出で手番が進む
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, R), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, R), (G, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, R), (G, R))  # spawn 1: 操作中 RY、枠は GR,GR
        f.feed(grid(["RY...."]), (G, R), (G, R), seconds=SETTLE_SECONDS + 0.1)  # RY 設置 → 枠は変わらず
        self.assertEqual(f.kinds().count("spawn"), 1)
        looks = []
        for top in (3, 4, 5, 6, 7, 8):
            a = grid(["RY...."])
            a[top][2], a[top + 1][2] = G, R  # 2マスに見える
            b = grid(["RY...."])
            b[top][2] = G  # 境目にかかって1マスだけ見える
            c = grid(["RY...."])
            c[top][2], c[top + 1][2], c[top - 1][2] = G, R, G  # 3マスに見える
            looks += [a, b, c]
        for cells in looks:
            f.feed(cells, (G, R), (G, R), frames=2)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (G, R))
        self.assertIn("盤面から設置を検出", [e.detail for e in f.events if e.kind == "spawn"][-1])

    def test_three_identical_pairs_advance_despite_noise_frames(self) -> None:
        # 実機 2026-10-10 15:18: 紫紫が3組続き、相手の攻撃の演出が紫に見える・関係ない赤が浮いて見える・落下中の組ぷよが
        # 積みに触れて「増えた」が4個に見えるフレームのたびに計測がやり直しになり、自由落下した。雑音では計測をやり直さない
        f = Feeder()
        f.feed(grid([]), (R, Y), (P, P), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (P, P), (P, P), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (P, P), (P, P))  # spawn 1: 操作中 RY、枠は PP,PP
        f.feed(grid(["RY...."]), (P, P), (P, P), frames=2)  # RY 設置(枠は変わらない)
        frames = []
        for top in (2, 3, 4, 5, 6, 7):
            a = grid(["RY...."])
            a[top][2], a[top + 1][2] = P, P  # 次の組ぷよ PP が落ちてくる
            noise_red = grid(["RY...."])
            noise_red[top][2], noise_red[0][5] = P, R  # 関係ない赤が浮いて見える
            touching = grid(["RY....", "PP...."])  # 落下中の組ぷよが積みに触れて見える(増えた4個)
            frames += [a, noise_red, touching]
        for cells in frames:
            f.feed(cells, (P, P), (P, P), frames=1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (P, P))
        self.assertIn("盤面から設置を検出", [e.detail for e in f.events if e.kind == "spawn"][-1])

    def test_three_identical_pairs_advance_by_board(self) -> None:
        # 同一ツモが3組続く(A,A,A): 2組目を置いても両枠は A,A のまま変わらない。盤面に A の2個が固定され、
        # 新しい組ぷよ(A)が浮いて見えたら設置とみなして手番を進める(実機 2026-09-19 20:06: BG,BG,BG で
        # 2組目の手番が3組目の出現まで検出されず提案が1手遅れた)。後から枠が変わっても二重に数えない
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (G, G), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (G, G))  # spawn 1: 操作中 RY、枠は GG,GG
        f.feed(grid(["RY...."]), (G, G), (G, G), seconds=SETTLE_SECONDS + 0.1)  # RY 設置 → 枠は変わらず(GG,GG)
        # 通常の検出は動かない(枠が同じ)。まだ次の組ぷよが見えていないので進めない
        self.assertEqual(f.kinds().count("spawn"), 1)
        # 次の組ぷよ GG が出現(浮いている)。盤面は RY のまま → 盤面検出で手番が進む
        with_pair = grid(["RY...."])
        with_pair[0][2], with_pair[1][2] = G, G
        f.feed(with_pair, (G, G), (G, G), seconds=PLACEMENT_FALLBACK_SECONDS + 0.1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (G, G))
        self.assertEqual((f.tracker.next1, f.tracker.next2), ((G, G), (G, G)))
        self.assertIn("盤面から設置を検出", [e.detail for e in f.events if e.kind == "spawn"][-1])
        # 確定盤面には RY だけが取り込まれ、浮いている GG は入らない
        self.assertEqual(f.tracker.settled, grid(["RY...."]))
        # 枠が後から (GG, PR) に変わる = 3組目が出た時の表示。既に進めた手番なので spawn を二重に数えない
        f.feed(with_pair, (G, G), (P, R), frames=NEXT_STABLE_FRAMES + 1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertIn("next_sync", f.kinds())
        self.assertEqual(f.tracker.next2, (P, R))
        # 2組目の GG を置き、3組目 GG が出現 → 枠は (PR, BB) に変わるので通常の検出
        f.feed(grid(["RY....", "GG...."]), (P, R), (B, B), frames=NEXT_STABLE_FRAMES + 1)
        self.assertEqual(f.kinds().count("spawn"), 3)
        self.assertEqual(f.tracker.current_pair, (G, G))

    def test_same_pair_spawn_after_chain_floating(self) -> None:
        # 2026-10-24(実機 08:05 turn=34〜38): 同じ組ぷよが続くとネクスト枠の表示が変わらない。連鎖の後の出現待ちでは盤面からの
        # 検出が止まっていて、出現を一度も検出できずに窒息した。新しい組ぷよが浮いて見えたら出現とみなす
        f = Feeder()
        f.begin(grid(["BB...R"]), (B, B), (B, B), (B, B))
        f.feed(grid(["BBBB.R"]), (B, B), (B, B), frames=4)  # 3・4列目に横置き → 4個つながる
        f.feed(grid([".....R"]), (B, B), (B, B), seconds=SETTLE_SECONDS + 0.2)  # 消えて連鎖の終わり
        self.assertIn("chain_end", f.kinds())
        n_spawn, n_ready = f.kinds().count("spawn"), f.kinds().count("turn_ready")
        appear = grid([".....R"])
        appear[0][2], appear[1][2] = B, B  # 次の組ぷよ(同じ BB)が出現位置に浮いて見える。枠は BB,BB のまま
        f.feed(appear, (B, B), (B, B), frames=5)
        self.assertEqual(f.kinds().count("spawn"), n_spawn + 1)
        self.assertIn("連鎖の後、同じ組ぷよが続き", [e.detail for e in f.events if e.kind == "spawn"][-1])
        self.assertEqual(f.kinds().count("turn_ready"), n_ready + 1)

    def test_same_pair_spawn_after_chain_resting_on_stack(self) -> None:
        # 3列目が11段なら、新しい組ぷよは出現位置(12段目)で積みに乗り、浮いて見えない。出現位置に確定盤面には無いネクストの色が
        # 見えたことでも出現とみなす
        col = [R, Y] * 6
        rows = ["".join("." if c != 2 else ("R" if col[i] == R else "Y") for c in range(COLS)) for i in range(11)]
        base = grid(rows)
        f = Feeder()
        f.begin(base, (G, G), (G, G), (G, G))
        tr = f.tracker
        tr._await_spawn_after_chain = True  # 連鎖の終わりを取り込んだ直後の出現待ち
        seen = [row[:] for row in base]
        seen[1][2] = G  # 新しい組ぷよの軸が12段目に乗って見える(子は13段目で見えない)
        f.feed(seen, (G, G), (G, G), frames=5)
        self.assertEqual(f.kinds().count("spawn"), 1)
        self.assertEqual(tr.turn, 2)

    def test_spawn_during_chain_ends_chain_without_expected_wait(self) -> None:
        # 2026-10-24(実機 08:13 turn=40): 点滅から再現した連鎖は3連鎖、実際は2連鎖で終わった。見込みの3段目を最大4秒待つ間に
        # 組ぷよが2つ出て、1つは自由落下した。連鎖中に出現を検出したら、盤面が SPAWNED_CHAIN_QUIET_SECONDS 変わらなければ終える
        from src.engine.chain_sim import simulate_chain
        from src.engine.tracker import CHAIN_EXPECTED_MAX_WAIT_SECONDS, SPAWNED_CHAIN_QUIET_SECONDS

        f = Feeder()
        rest = grid(["RYGB.."])
        f.begin(rest, (R, Y), (G, G), (P, R))
        tr = f.tracker
        tr.chain_active, tr.settling = True, True
        tr.chain_expected = simulate_chain(grid(["RRRR.."]))  # 見込みでは全部消える(実際は4個残る)
        tr._settled_count = count_puyos(rest)
        tr._chain_last_change_t = f.t
        f.feed(rest, (G, G), (P, R), seconds=1.2)
        self.assertNotIn("chain_end", f.kinds())  # 出現が無ければ見込みを待つ
        f.feed(rest, (P, R), (B, B), frames=NEXT_STABLE_FRAMES + 2)  # 次の組ぷよが出現(枠が入れ替わる)
        self.assertIn("spawn", f.kinds())
        self.assertIn("chain_end", f.kinds())
        self.assertIn("turn_ready", f.kinds())
        self.assertLess(f.t, CHAIN_EXPECTED_MAX_WAIT_SECONDS)
        self.assertFalse(tr.chain_active)

        # 盤面が変わった直後の出現(枠の読み違いで連鎖の途中に出現を検出した場合)は、すぐには終えない
        g = Feeder()
        g.begin(rest, (R, Y), (G, G), (P, R))
        tr = g.tracker
        tr.chain_active, tr.settling = True, True
        tr.chain_expected = simulate_chain(grid(["RRRR.."]))
        tr._settled_count = count_puyos(rest) + 4
        g.feed(grid(["RYGB..", "RRRR.."]), (G, G), (P, R), seconds=0.5)
        g.feed(rest, (P, R), (B, B), frames=NEXT_STABLE_FRAMES + 2)  # 盤面が変わったのと同時に出現
        self.assertNotIn("chain_end", g.kinds())
        g.feed(rest, (P, R), (B, B), seconds=SPAWNED_CHAIN_QUIET_SECONDS + 0.1)
        self.assertIn("chain_end", g.kinds())

    def test_spawn_after_reproduced_chain_takes_in_result_without_stable_board(self) -> None:
        # 2026-10-24(実機 16:47 turn=36): 高い積みの1連鎖の後、消えた位置に重なる「1れんさ!」の文字と×印をぷよと読み、
        # 盤面の読みが毎フレーム揺れて連鎖の終わりを取り込めず、次の組ぷよを連鎖中として操作しないまま窒息した。
        # 再現した連鎖の段数まで消えたのを見た後に出現を検出したら、盤面が落ち着くのを待たずに再現した連鎖後の盤面で取り込む
        from src.engine.chain_sim import simulate_chain

        f = Feeder()
        rest = grid(["RYGB.."])
        f.begin(rest, (R, Y), (G, G), (P, R))
        tr = f.tracker
        tr.chain_active, tr.settling = True, True
        tr.chain_expected = simulate_chain(grid(["RYGB..", "PPPP.."]))  # 紫4個が消えて1連鎖
        tr.chain_steps = 1  # 1段目が消えたのは見た
        tr._last_chain_step_t = f.t
        tr._settled_count = count_puyos(rest)
        text1 = grid(["RYGB..", "Y....."])  # 文字をぷよと読んだ(毎フレーム変わる)
        text2 = grid(["RYGB..", "YY...."])
        for i in range(8):  # 段が消えてから0.2秒以上たって出現する
            f.feed(text1 if i % 2 else text2, (G, G), (P, R), frames=1)
        self.assertNotIn("chain_end", f.kinds())  # 出現が無ければ従来どおり盤面を待つ
        for i in range(NEXT_STABLE_FRAMES + 2):
            f.feed(text1 if i % 2 else text2, (P, R), (B, B), frames=1)  # 次の組ぷよが出現(盤面は揺れたまま)
        self.assertIn("spawn", f.kinds())
        self.assertIn("chain_end", f.kinds())
        self.assertIn("turn_ready", f.kinds())
        self.assertFalse(tr.chain_active)
        self.assertEqual(tr.settled[1:], rest[1:])  # 文字を読んだ黄ではなく、再現した連鎖後の盤面

        # 再現した段数まで見ていない時(連鎖の途中で出現を読み違えた可能性)は、従来どおり盤面が落ち着くのを待つ
        g = Feeder()
        g.begin(rest, (R, Y), (G, G), (P, R))
        tr = g.tracker
        tr.chain_active, tr.settling = True, True
        tr.chain_expected = simulate_chain(grid(["RYGB..", "PPPP.."]))
        tr.chain_steps = 0
        tr._last_chain_step_t = g.t - 1.0
        tr._settled_count = count_puyos(rest)
        for i in range(NEXT_STABLE_FRAMES + 2):
            g.feed(text1 if i % 2 else text2, (P, R), (B, B), frames=1)
        self.assertIn("spawn", g.kinds())
        self.assertNotIn("chain_end", g.kinds())

    def test_reproduced_chain_waits_for_remaining_steps(self) -> None:
        # 2026-10-26(実機 20:32 の 2P 233s): 再現は12連鎖、6段目の後に盤面の減少が5秒見えず、4秒で打ち切って「6連鎖」と確定した。
        # 再現した段数が残っている間は、残りの段数 × CHAIN_EXPECTED_SECONDS_PER_STEP まで待つ
        import types

        from src.engine.tracker import CHAIN_EXPECTED_MAX_WAIT_SECONDS

        f = Feeder()
        rest = grid(["RYGB..", "RYGB.."])
        f.begin(rest, (R, Y), (G, G), (P, R))
        tr = f.tracker
        tr.chain_active, tr.settling = True, True
        tr.chain_expected = types.SimpleNamespace(steps=12, final=grid(["R....."]), score=73230)
        tr.chain_steps = 6
        tr._settled_count = count_puyos(rest)
        tr._chain_last_change_t = f.t
        f.feed(rest, (G, G), (P, R), seconds=CHAIN_EXPECTED_MAX_WAIT_SECONDS + 1.0)
        self.assertNotIn("chain_end", f.kinds())  # 残り6段なので4秒では打ち切らない
        self.assertTrue(tr.chain_active)

    def test_spawn_during_chain_forces_end_when_board_never_settles(self) -> None:
        # 2026-10-25(実機 19:46 の 2P 981.65s): 1連鎖の段を観測で数えられず(chain_steps=0)、相手の攻撃の演出で盤面も
        # 揺れ続け、出現から2.6秒「連鎖中」のまま組ぷよが自由落下した。出現から SPAWNED_CHAIN_FORCE_END_SECONDS たてば
        # 盤面を待たずに再現した連鎖後の盤面で取り込む
        from src.engine.chain_sim import simulate_chain
        from src.engine.tracker import SPAWNED_CHAIN_FORCE_END_SECONDS

        f = Feeder()
        rest = grid(["RYGB.."])
        f.begin(rest, (R, Y), (G, G), (P, R))
        tr = f.tracker
        tr.chain_active, tr.settling = True, True
        tr.chain_expected = simulate_chain(grid(["RYGB..", "PPPP.."]))
        tr.chain_steps = 0  # 段を数えられていない
        tr._last_chain_step_t = f.t - 1.0
        tr._settled_count = count_puyos(rest)
        tr._chain_last_change_t = f.t
        flicker = [grid(["RYGB..", "Y....."]), grid(["RYGB..", "YY...."])]  # 演出を色と読んで毎フレーム揺れる
        for i in range(NEXT_STABLE_FRAMES + 2):
            f.feed(flicker[i % 2], (P, R), (B, B), frames=1)  # 次の組ぷよが出現
        self.assertIn("spawn", f.kinds())
        self.assertNotIn("chain_end", f.kinds())  # 出現の直後はまだ終えない(枠の読み違いへの備え)
        spawn_t = next(e.t for e in f.events if e.kind == "spawn")
        while f.t - spawn_t < SPAWNED_CHAIN_FORCE_END_SECONDS + 0.05:
            f.feed(flicker[int(f.t * 30) % 2], (P, R), (B, B), frames=1)
        self.assertIn("chain_end", f.kinds())
        self.assertIn("turn_ready", f.kinds())
        self.assertFalse(tr.chain_active)
        self.assertEqual(tr.settled[1:], rest[1:])  # 揺れた読みではなく再現した連鎖後の盤面
        end_t = next(e.t for e in f.events if e.kind == "chain_end")
        self.assertLess(end_t - spawn_t, SPAWNED_CHAIN_FORCE_END_SECONDS + 0.1)

    def test_no_board_spawn_when_next_frames_differ(self) -> None:
        # ネクスト枠の2つが違う組ぷよなら、出現すれば表示が必ず変わる。表示が変わらない間は、上の方に次の組ぷよの色が
        # 浮いて見えても(相手の攻撃の光の演出)盤面からは出現と判定しない(実機 2026-10-12 07:24 試合2 turn=22)
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))  # spawn 1: 操作中 RY、枠は GG,PR
        f.feed(grid(["RY...."]), (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)  # RY 設置、次の組ぷよはまだ出ない
        effect = grid(["RY...."])
        effect[0][2], effect[1][2] = G, G  # 演出が上の方で緑に見えた
        f.feed(effect, (G, G), (P, R), seconds=PLACEMENT_FALLBACK_SECONDS + 0.3)
        self.assertEqual(f.kinds().count("spawn"), 1)
        # 本当に出現すれば枠が変わり、通常の検出で手番が進む
        f.feed(effect, (P, R), (B, B), frames=NEXT_STABLE_FRAMES + 1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (G, G))

    def test_four_identical_pairs_then_frame_change_is_real_spawn(self) -> None:
        # A,A,A,A,X: 盤面検出で2組目に進んだ後、2組目を置いて3組目が出た時に枠が (A,A)→(A,X) と変わる。
        # これは「枠が追いついた」のではなく本物の spawn(操作中の組ぷよは既に置かれている)
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (G, G), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (G, G), frames=1)  # spawn 1: 操作中 RY、枠 GG,GG
        f.feed(grid(["RY...."]), (G, G), (G, G), seconds=SETTLE_SECONDS + 0.1)
        with_pair = grid(["RY...."])
        with_pair[0][2], with_pair[1][2] = G, G
        f.feed(with_pair, (G, G), (G, G), seconds=PLACEMENT_FALLBACK_SECONDS + 0.1)  # 盤面検出: turn 2 (GG)
        self.assertEqual(f.kinds().count("spawn"), 2)
        f.feed(grid(["RY....", "GG...."]), (G, G), (G, G), frames=2)  # 2組目 GG を設置
        # 3組目 GG が出現、枠は (GG, PR) に変わる → 本物の spawn(turn 3)
        f.feed(grid(["RY....", "GG...."]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES + 1)
        self.assertEqual(f.kinds().count("spawn"), 3)
        self.assertNotIn("next_sync", f.kinds())
        self.assertEqual(f.tracker.current_pair, (G, G))
        self.assertEqual((f.tracker.next1, f.tracker.next2), ((G, G), (P, R)))

    def test_initial_pair_placed_advances_turn_even_if_next_unchanged(self) -> None:
        # 試合開始: 落ちてくる赤赤から操作中を推定(initial)。ネクスト枠は (黄紫, 黄紫)。
        # 赤赤が置かれても枠は変わらない(2組目と3組目が同じ)ので、取り込んだ時点で手番を進める
        # (実機 2026-09-19 22:46: 置いた赤赤を操作中として「隣に置けば全消し」を提示した)
        f = Feeder()
        f.feed(grid([]), (Y, P), (Y, P), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, R), (Y, P), (Y, P))  # 3列目を落下中(浮いていて段が下がる)
        self.assertEqual(f.tracker.current_pair, (R, R))
        # 赤赤が着地し、次の黄紫が落ちてくる(枠は変わらないので、次の組ぷよの出現を設置の証拠にする)
        f.falling(grid([".RR..."]), (Y, P), (Y, P), (Y, P), frames=int(PLACEMENT_FALLBACK_SECONDS * FPS) + 2)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (Y, P))
        self.assertEqual((f.tracker.next1, f.tracker.next2), ((Y, P), (Y, P)))
        self.assertIn("turn_ready", f.kinds())
        # 枠が (黄紫, 青青) に変わる = 3組目が見えた。操作中の黄紫は未設置なので枠の追いつき(二重計上しない)
        f.feed(grid([".RR..."]), (Y, P), (B, B), frames=NEXT_STABLE_FRAMES + 1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.next2, (B, B))

    def test_initial_pair_with_changing_next_is_not_double_counted(self) -> None:
        # 開始時の推定の後、枠が普通に変わる場合は通常の spawn だけ(取り込み側で二重に進めない)
        f = Feeder()
        f.feed(grid([]), (Y, P), (G, G), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, R), (Y, P), (G, G))
        f.feed(grid([".RR..."]), (G, G), (B, B), frames=NEXT_STABLE_FRAMES + 1)  # 着地と同時に枠が変わる
        f.feed(grid([".RR..."]), (G, G), (B, B), seconds=SETTLE_SECONDS + 0.1)
        self.assertEqual(f.kinds().count("spawn"), 2)
        self.assertEqual(f.tracker.current_pair, (Y, P))

    def test_transient_overcount_is_not_a_chain(self) -> None:
        # 設置の瞬間に落下中のおじゃまと操作中の組ぷよが重なって多く見え、次のフレームで本当の個数に戻る。
        # 消える組が無い盤面からの「減少」は消去ではない(実機 2026-09-19 23:14: 偽の連鎖で提案が遅れた)
        f = Feeder()
        base = grid(["RRGG..", "YYPP.."])
        f.feed(base, (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)  # spawn → 取り込み(8個)
        self.assertFalse(f.tracker.settling)
        f.feed(base, (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES)  # 設置 → spawn(取り込み待ち)
        placed = grid(["RRGGRY", "YYPP.."])  # 本当の設置後(10個、消える組なし)
        overcount = grid(["RRGGRY", "YYPP..", "..GG..", "..##.."])  # 一瞬: 操作中の組ぷよ+落下中のおじゃまが重なる(14個)
        f.feed(overcount, (P, R), (Y, Y), frames=2)
        f.feed(placed, (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        self.assertNotIn("chain_start", f.kinds())
        self.assertIn("turn_ready", f.kinds()[-3:])
        self.assertEqual(f.tracker.settled, placed)

    def test_real_chain_after_placement_is_still_detected(self) -> None:
        # 設置で4連結ができた(消える組がある)盤面からの減少は従来どおり連鎖
        f = Feeder()
        base = grid(["RRR...", "YYPP.."])
        f.feed(base, (R, Y), (G, G), frames=NEXT_STABLE_FRAMES)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES)  # 操作中 GG → 設置 → spawn
        f.feed(grid(["RRRR..", "YYPPG."]), (P, R), (Y, Y), frames=2)  # 赤4連結(消える前)
        f.feed(grid(["....G.", "YYPPG."]), (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        self.assertIn("chain_start", f.kinds())

    def test_touching_pair_without_new_pair_is_not_placement(self) -> None:
        # 枠が読めていて変わらず、操作中の組ぷよが積みに触れて「固定されたように」見えても、
        # 次の組ぷよが浮いて見えなければ設置とみなさない
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(grid([]), (G, G), (P, R), frames=NEXT_STABLE_FRAMES)
        f.falling(grid([]), (R, Y), (G, G), (P, R))
        f.feed(grid(["RY...."]), (G, G), (P, R), seconds=1.0)
        self.assertEqual(f.kinds().count("spawn"), 1)

    def test_hidden_row_inferred_from_single_visible_puyo(self) -> None:
        # 2列目が11段まで積まれた状態で(赤,黄)を縦置き → 黄は12段目に見え、赤は13段目で見えない
        f = Feeder()
        base = grid(["." * COLS] * 0)
        rows = ["......"] * 11
        rows = [".Y...." if i % 2 == 0 else ".G...." for i in range(11)]
        base = grid(rows)
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)  # spawn: current=(R,Y)
        f.feed(base, (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES)  # spawn: (R,Y) を設置
        placed = grid(rows + [".Y...."])  # 12段目に黄だけ見える
        f.feed(placed, (P, R), (Y, Y), seconds=SETTLE_SECONDS + 0.1)
        self.assertEqual(f.tracker.settled[0][1], R)  # 13段目に赤があると推定
        self.assertEqual(f.tracker.settled[1][1], Y)
        # 12段目が空いたら記憶は消える
        f.feed(grid(rows), (Y, Y), (G, R), frames=NEXT_STABLE_FRAMES)
        f.feed(grid(rows), (Y, Y), (G, R), seconds=SETTLE_SECONDS + 0.1)
        self.assertEqual(f.tracker.settled[0][1], E)

    def test_expected_with_hidden_row_matches_visible_part(self) -> None:
        f = Feeder()
        rows = [".Y...." if i % 2 == 0 else ".G...." for i in range(11)]
        base = grid(rows)
        f.feed(base, (R, Y), (G, G), seconds=SETTLE_SECONDS + 0.1)
        f.feed(base, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.1)
        expected = grid(rows + [".Y....", ".R...."])  # 13段目に赤(amaの置いた後の盤面)
        f.tracker.set_expected(expected)
        f.feed(grid(rows + [".Y...."]), (P, R), (Y, Y), frames=NEXT_STABLE_FRAMES + 1)
        self.assertIn("(expected)", f.events[-1].detail)
        self.assertEqual(f.tracker.settled, expected)

    def test_pending_garbage_requires_stability(self) -> None:
        f = Feeder()
        f.feed(grid([]), (R, Y), (G, G), pending=0, frames=NEXT_STABLE_FRAMES)
        f.feed(grid([]), (R, Y), (G, G), pending=4, frames=1)
        self.assertEqual(f.tracker.pending_garbage, 0)
        f.feed(grid([]), (R, Y), (G, G), pending=4, seconds=0.3)
        self.assertEqual(f.tracker.pending_garbage, 4)
        self.assertIn("pending", f.kinds())




class HiddenRowChainTest(unittest.TestCase):
    """連鎖の再現に13段目(画面に見えない)の記憶を入れる(2026-10-24)"""

    def test_flash_board_includes_hidden_row(self) -> None:
        # 実機 08:13 turn=39: 13段目(2列目)に黄がある盤面で、引き金の(黄, 青)の黄を3列目12段目(出現位置で読めない)に置いた。
        # 点滅から作り直した盤面に両方の黄が入らず再現は3連鎖、実際は2連鎖で終わり、見込みの3段目を待つ間に組ぷよが自由落下した
        from src.engine.chain_sim import _groups, simulate_chain

        base = grid(list(reversed(["RR....", "YBY.GG", "RBB.YB", "GRRBGB", "RYYYGB", "RRBRBR", "GYRGRY", "GBBRBB", "GRGRYB", "RBYGYG", "RBBYBY", "GGGBBY"])))
        base[0][1] = Y  # 13段目の黄
        f = Feeder()
        f.begin(base, (B, Y), (Y, G), (R, B))
        tr = f.tracker
        tr._hidden = list(base[0])
        placed = [row[:] for row in base]
        placed[1][2], placed[3][3] = Y, B  # (黄, 青)を3・4列目に横置き
        seen = [row[:] for row in placed]
        seen[0] = [E] * COLS  # 画面には13段目が見えない
        seen[1][2] = E  # 出現位置も読まない
        popped = [row[:] for row in seen]
        for g in _groups(seen):
            for r, c in g:
                popped[r][c] = E
        f.feed(seen, (Y, G), (R, B), frames=3)
        f.feed(settle_cells(popped), (Y, G), (R, B), frames=1)
        self.assertIn("chain_start", f.kinds())
        self.assertEqual(tr.chain_before, placed)
        self.assertEqual(tr.chain_expected.steps, 2)  # 読めないマスを補わないと3連鎖
        self.assertEqual(simulate_chain(seen).steps, 3)

    def test_hidden_row_after_chain_follows_replay(self) -> None:
        # 1列目の一番下の1個だけが消えると、13段目の赤は12段目に落ちて見える。記憶の13段目に赤を残すと2個に増える
        from src.engine.chain_sim import simulate_chain

        col = ["R"] + ["G" if i % 2 else "Y" for i in range(11)]
        rows = [col[i] + ("RRR..." if i == 0 else ".....") for i in range(12)]
        before = grid(rows)
        before[0][0] = R  # 13段目の赤
        exp = simulate_chain(before)
        self.assertEqual(exp.steps, 1)
        after = [row[:] for row in exp.final]
        self.assertEqual((after[0][0], after[1][0]), (E, R))
        f = Feeder()
        f.begin(before, (R, Y), (G, G), (P, R))
        tr = f.tracker
        tr._hidden = list(before[0])
        tr.chain_active, tr.settling = True, True
        tr.chain_expected, tr.chain_steps = exp, 1
        tr._settled_count = count_puyos(after)
        tr._chain_last_change_t = f.t
        f.feed(after, (G, G), (P, R), seconds=SETTLE_SECONDS + 0.2)
        self.assertIn("chain_end", f.kinds())
        self.assertEqual(tr.settled, after)
        self.assertEqual(tr._hidden, [E] * COLS)


class TestCarry(unittest.TestCase):
    """攻撃の端数点(2026-10-19、YamaNEKO の --count-carry-self 用)"""

    def test_drop_rows(self) -> None:
        from src.engine.tracker import placement_drop_rows

        empty = grid([])
        # 空の盤面の1列目に縦置き: 12段落ちる
        self.assertEqual(placement_drop_rows(empty, grid(["R.....", "R....."])), 12)
        # 2段の列(1列目)と1段の列(2列目)にまたがる横置き: 低い方(2列目、段1)に着地して 11段
        before = grid(["RY....", "R....."])
        self.assertEqual(placement_drop_rows(before, grid(["RY....", "RGG..."])), 11)
        # おじゃまだけ増えた・色ぷよが2個でない時は数えない
        self.assertIsNone(placement_drop_rows(before, grid(["RY....", "R#...."])))
        self.assertIsNone(placement_drop_rows(empty, grid(["RRR..."])))

    def test_carry_after_chain(self) -> None:
        from src.engine.tracker import carry_after_chain

        # 端数 50点 + 引き金の落下 10段 + 連鎖 1440点 = 1500点 → 21個、余り 30点
        self.assertEqual(carry_after_chain(50, 10, 1440), 30)
        # 連鎖の点が分からない時は 0
        self.assertEqual(carry_after_chain(50, 10, None), 0)
        # マージンタイムで1個 25点の時は 25点の余り(1500点 → 60個、余り 0点)
        self.assertEqual(carry_after_chain(50, 10, 1440, 25), 0)

    def test_margin_target_point(self) -> None:
        from src.engine.margin_time import margin_target_point

        self.assertEqual([margin_target_point(s) for s in (0, 95.9, 96, 111, 112, 128, 255, 256, 400)], [70, 70, 52, 52, 34, 25, 2, 1, 1])


if __name__ == "__main__":
    unittest.main()
