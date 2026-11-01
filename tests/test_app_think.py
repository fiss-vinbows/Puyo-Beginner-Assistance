"""AssistWorker の思考まわり: 相手側の認識エラーが自分の提案を止めないこと。"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

from src.autoplay_support import AVAILABLE as AUTOPLAY_AVAILABLE  # noqa: E402
from src.app import AssistWorker  # noqa: E402
from src.engine.ama_client import BattleMove  # noqa: E402
from src.engine.puyo_field import COLS, ROWS  # noqa: E402
from src.engine.tracker import PlayerTracker  # noqa: E402
from src.overlay.renderer import OverlayDrawData  # noqa: E402
from src.vision.cell_classifier import EMPTY  # noqa: E402


def _empty_cells() -> list[list[str]]:
    return [[EMPTY] * COLS for _ in range(ROWS)]


def _board_with(rows: list[str]) -> list[list[str]]:
    """下から並べた行(R/Y/G/B/P と .)の盤面(色名の13行)。"""
    names = {"R": "red", "Y": "yellow", "G": "green", "B": "blue", "P": "purple"}
    cells = _empty_cells()
    for i, row in enumerate(rows):
        for c, ch in enumerate(row):
            if ch != ".":
                cells[ROWS - 1 - i][c] = names[ch]
    return cells


def _tracker(current: tuple[str, str], n1: tuple[str, str], n2: tuple[str, str]) -> PlayerTracker:
    tr = PlayerTracker()
    tr.settled = _empty_cells()
    tr.current_pair, tr.next1, tr.next2 = current, n1, n2
    return tr


class BuildStatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # calibration/client は _build_states では使わない
        self.worker = AssistWorker(calibration=None, client=None, debug_log_path=None)  # type: ignore[arg-type]

    def test_enemy_fifth_color_does_not_block_self(self) -> None:
        # 実機 2026-09-15: 相手ネクストの黄が赤と誤認識され(この試合の4色は緑紫黄青)、
        # 5色目エラーで自分の思考が5手番スキップされた。相手側の失敗は相手なし(None)にする
        me = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        en = _tracker(("red", "purple"), ("green", "yellow"), ("purple", "green"))
        my_state, en_state, my_pairs, en_pairs = self.worker._build_states(me, en, me.settled, [me.current_pair, me.next1, me.next2])
        self.assertIsNone(en_state)
        self.assertEqual(len(my_pairs), 3)
        self.assertEqual(en_pairs, [])
        self.assertNotIn("red", self.worker.mapper.mapping)  # 誤認識の色を割り当てに残さない

    def test_enemy_ok_gives_battle_state(self) -> None:
        me = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        en = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        _, en_state, _, en_pairs = self.worker._build_states(me, en, me.settled, [me.current_pair, me.next1, me.next2])
        self.assertIsNotNone(en_state)
        self.assertEqual(len(en_pairs), 3)

    def test_self_fifth_color_still_raises(self) -> None:
        me = _tracker(("green", "purple"), ("yellow", "blue"), ("red", "yellow"))
        en = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        with self.assertRaises(ValueError):
            self.worker._build_states(me, en, me.settled, [me.current_pair, me.next1, me.next2])


class FormTextTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.worker = AssistWorker(calibration=None, client=None, debug_log_path=None)  # type: ignore[arg-type]

    def _move(self, form: str | None) -> BattleMove:
        return BattleMove(0, "UP", 0, (12, 0), (11, 0), "build", None, None, 0, form=form)

    def test_aim_then_transition(self) -> None:
        self.assertEqual(self.worker._form_text(self._move("GTR")), "狙い: GTR")
        self.worker._last_form = "GTR"
        self.assertEqual(self.worker._form_text(self._move("GTR")), "狙い: GTR")
        self.assertEqual(self.worker._form_text(self._move("SGTR")), "移行: GTR → フキゲンGTR")
        self.assertEqual(self.worker._form_text(self._move(None)), "定型を離脱: GTR")


class PrecomputeKeyTest(unittest.TestCase):
    def test_precomputed_requires_matching_pairs(self) -> None:
        # 先読みは「期待盤面」だけでなく「その時の組ぷよ」も一致した時だけ使う(分析文書 2026-09-17 P0-C)
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        worker = AssistWorker(calibration=None, client=None, debug_log_path=None)  # type: ignore[arg-type]
        expected = _empty_cells()
        move = BattleMove(0, "UP", 0, (12, 0), (11, 0), "build", None, None, 0)
        worker._precomputed = (expected, (("red", "green"), ("blue", "yellow")), move)
        me = _tracker(("red", "green"), ("blue", "yellow"), ("green", "green"))
        pre = worker._precomputed
        self.assertTrue(pre[0] == me.settled and pre[1] == (me.current_pair, me.next1))
        me = _tracker(("green", "red"), ("blue", "yellow"), ("green", "green"))  # 軸子が逆
        self.assertFalse(pre[1] == (me.current_pair, me.next1))


class FakeClient:
    """battle の応答を差し替えられる偽クライアント(呼び出し内容も記録する)。"""

    def __init__(self, moves: list[BattleMove]) -> None:
        self.moves = list(moves)
        self.calls: list[dict] = []

    def battle(self, my_state, en_state, **kwargs) -> BattleMove:
        self.calls.append({"my": my_state, "en": en_state, **kwargs})
        return self.moves.pop(0)

    def solo(self, field, pairs, **kwargs) -> BattleMove:
        self.calls.append({"solo": field, "pairs": list(pairs), **kwargs})
        return self.moves.pop(0)


def _move(label: str, axis=(12, 0), child=(11, 0), fire_reason=None) -> BattleMove:
    return BattleMove(0, "UP", 0, axis, child, label, None, None, 0, fire_reason=fire_reason)


class RethinkPolicyTest(unittest.TestCase):
    """手番途中の再計算: 表示中のラベルと攻撃の大きさで許可を決める(分析文書 2026-09-19 P0-1)。"""

    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _worker(self, moves: list[BattleMove]) -> tuple[AssistWorker, FakeClient, list]:
        client = FakeClient(moves)
        worker = AssistWorker(calibration=None, client=client, debug_log_path=None)  # type: ignore[arg-type]
        worker.trackers["self"] = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        worker.trackers["enemy"] = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        emitted: list = []
        worker.draw_data_ready.connect(lambda d: emitted.append(d))
        return worker, client, emitted

    def test_rethink_same_placement_updates_label_and_note(self) -> None:
        # 再計算の結果が同じ置き場所でも、判断の意味(ラベル・理由)は最新に更新する
        worker, client, emitted = self._worker([_move("main", (12, 2), (12, 3), fire_reason="counter")])
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="harass")
        self.assertTrue(worker._think(1.0, _frame(), note="相手の消去開始を検出: 提案を更新"))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(emitted[-1].label, "main")
        self.assertEqual(emitted[-1].axis_cell, (12, 2))
        self.assertIn("相手の消去開始を検出", emitted[-1].note)

    def test_enemy_firing_shown_until_fresh_think(self) -> None:
        # 相手の連鎖中は撃っている連鎖(点滅から再現)をパネルに出し、連鎖が終わった後に依頼した思考が届くまで残す
        # (実機 2026-10-25 12:23: 6連鎖を撃っている間、撃つ前の盤面の解析の「9連鎖」のままだった)
        from src.engine.chain_sim import ChainResult

        worker, _, emitted = self._worker([])
        me, en = worker.trackers["self"], worker.trackers["enemy"]
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="build")
        en.chain_expected = ChainResult(steps=6, score=18530, cleared=44, final=_empty_cells())
        en.chain_active = True
        worker._on_enemy_chain(en, "chain_start")
        self.assertEqual(emitted[-1].enemy_firing, (6, 18530, True))
        worker._emit_move(me, _move("build"))  # 連鎖中に届いた思考でも撃っている連鎖のまま
        self.assertEqual(emitted[-1].enemy_firing, (6, 18530, True))
        en.chain_active, en.chain_steps, en.chain_score = False, 6, 18530
        worker._think_submitted = 1.0
        worker._on_enemy_chain(en, "chain_end")
        self.assertEqual(emitted[-1].enemy_firing, (6, 18530, False))
        worker._emit_move(me, _move("build"))  # 連鎖中に依頼した思考: 撃った連鎖を残す
        self.assertEqual(emitted[-1].enemy_firing, (6, 18530, False))
        worker._think_submitted = worker._enemy_fire_end + 0.1  # 連鎖の後に依頼した思考: 解析に戻す
        worker._emit_move(me, _move("build"))
        self.assertIsNone(emitted[-1].enemy_firing)

    def test_rethink_keeps_build_display_when_still_build(self) -> None:
        # 構築中に再計算して構築のままなら、乱数で変わっただけの手を出さず表示を維持する
        worker, client, emitted = self._worker([_move("build", (12, 4), (12, 5))])
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="build")
        worker._think(1.0, _frame(), note="相手の連鎖が終了: 提案を更新")
        self.assertEqual(emitted[-1].axis_cell, (12, 2))
        self.assertEqual(emitted[-1].label, "build")

    def test_rethink_kept_display_requests_precompute_again(self) -> None:
        # 相手の消去開始などで先読みを捨てた後、再計算の結果が構築のまま(表示を維持)でも、表示中の手で次の手を
        # 先読みし直す(しないと次の手番が思考待ちで硬直した。実機 2026-10-10 15:28)
        worker, client, emitted = self._worker([_move("build", (12, 4), (12, 5)), _move("build", (12, 0), (11, 0))])
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="build")
        shown = _move("build", (12, 2), (12, 3))
        shown.field_after = ["......"] * 13
        worker._last_move = shown
        worker._precomputed = None
        worker._think(1.0, _frame(), note="相手の消去開始を検出: 提案を更新")
        self.assertEqual(len(client.calls), 2)  # 再計算 + 先読み
        self.assertIsNotNone(worker._precomputed)
        self.assertIsNotNone(worker._precomputed_attack)

    def test_anticipated_precompute_is_requested_when_enemy_can_fire(self) -> None:
        # 攻撃されていない時、相手が今すぐ撃てる攻撃が大きければ、それを受けた状態での次の手も先に計算しておく
        from src.engine.ama_client import AttackInfo, FieldAnalysis

        alt_move = _move("accept", (12, 1), (11, 1))
        alt_move.path = ["LEFT", "DROP"]
        worker, client, emitted = self._worker([_move("build", (12, 4), (12, 5)), _move("build", (12, 0), (11, 0)), alt_move])
        worker.autoplay.enabled = True
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="build")
        shown = _move("build", (12, 2), (12, 3))
        shown.field_after = ["......"] * 13
        big = AttackInfo(count=4, score=2000, garbage=28, x=0, rotation="UP", frame=0, all_clear=False)
        shown.enemy_analysis = FieldAnalysis(immediate=big, max=big, harass=big, potential=0, garbage_max=28)
        worker._last_move = shown
        worker._precomputed = None
        worker._think(1.0, _frame(), note="相手の連鎖が終了: 提案を更新")
        self.assertEqual(len(client.calls), 3)  # 再計算 + 先読み + 相手の発火を想定した先読み
        self.assertGreaterEqual(client.calls[2]["my"].attack, 28)  # 想定した攻撃を受けた状態で計算
        self.assertIsNotNone(worker._precomputed_alt)
        self.assertEqual(worker._precomputed_alt[3], 28)

    def test_rethink_does_not_use_stale_precompute(self) -> None:
        # 再計算(note あり)では、相手の状態が古い先読みを一度表示する経路を使わない
        worker, client, emitted = self._worker([_move("main", (12, 4), (12, 5), fire_reason="counter")])
        me = worker.trackers["self"]
        worker._precomputed = (me.settled, (me.current_pair, me.next1), _move("build", (12, 0), (12, 1)))
        worker._think(1.0, _frame(), note="相手の消去開始を検出: 提案を更新")
        self.assertEqual([d.axis_cell for d in emitted], [(12, 4)])

    def test_new_turn_does_not_keep_previous_turn_display(self) -> None:
        # 手番の開始(new_turn)と再計算の理由が重なっても、前の手番の表示を「構築のまま維持」しない
        # (実機 2026-09-19 113.55s)
        worker, client, emitted = self._worker([_move("build", (12, 4), (12, 5))])
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="build")
        worker._think(1.0, _frame(), note="相手の消去開始を検出: 提案を更新", new_turn=True)
        self.assertEqual(emitted[-1].axis_cell, (12, 4))

    def test_incoming_estimate_is_carried_to_next_think(self) -> None:
        # 相手の連鎖中にブリッジが返した着弾見込みを保持し、次の思考では予告(アイコン数)より大きい方を渡す
        big = _move("accept")
        big.incoming, big.incoming_source = 50, "live"
        worker, client, _ = self._worker([big, _move("build")])
        worker.trackers["self"].pending_garbage = 2
        worker._think(1.0, _frame())
        self.assertEqual(worker._incoming_carry, 50)
        worker._think(2.0, _frame())
        self.assertEqual(client.calls[1]["my"].attack, 50)

    def test_carry_kept_while_enemy_chain_continues(self) -> None:
        # 相手の連鎖中に予告の表示が0になっても(自分の相殺)、着弾見込みは解除しない
        # (実機 2026-09-27 14:23: 相手の4連鎖中に48個を解除し、ama が着弾2個と見て判断した)
        worker, _, _ = self._worker([])
        me, en = worker.trackers["self"], worker.trackers["enemy"]
        worker._incoming_carry = 48
        me.pending_garbage = 0
        en.chain_active = True
        worker._release_carry_if_pending_cleared(1.0, me)
        self.assertEqual(worker._incoming_carry, 48)
        # 相手の連鎖が終わって予告も無ければ解除する
        en.chain_active = False
        worker._release_carry_if_pending_cleared(2.0, me)
        self.assertEqual(worker._incoming_carry, 0)

    def test_incoming_estimate_is_revised_downward(self) -> None:
        # 見積もりは max で保持せず最新値で置き換える(148→60 の下方訂正が届く。分析文書第3弾 P0-3)
        first, second = _move("accept"), _move("accept")
        first.incoming, first.incoming_source = 148, "live"
        second.incoming, second.incoming_source = 60, "live"
        worker, client, _ = self._worker([first, second, _move("build")])
        worker._think(1.0, _frame())
        self.assertEqual(worker._incoming_carry, 148)
        worker._think(2.0, _frame())
        self.assertEqual(worker._incoming_carry, 60)
        worker._think(3.0, _frame())
        self.assertEqual(client.calls[2]["my"].attack, 60)

    def test_no_valid_placement_clears_display_and_continues(self) -> None:
        # ブリッジの「置ける手が無い」は例外で止めず、提案を消して観測を続ける(crash_log 2026-09-19 の4件)
        from src.engine.ama_client import NoValidPlacementError

        class Raising(FakeClient):
            def battle(self, my_state, en_state, **kwargs):
                raise NoValidPlacementError("ama ブリッジがエラーを返しました: no valid placement")

        client = Raising([])
        worker = AssistWorker(calibration=None, client=client, debug_log_path=None)  # type: ignore[arg-type]
        worker.trackers["self"] = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        worker.trackers["enemy"] = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        emitted: list = []
        worker.draw_data_ready.connect(lambda d: emitted.append(d))
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="build")
        self.assertTrue(worker._think(1.0, _frame()))
        self.assertIsNone(worker._last_data)
        self.assertEqual(emitted[-1], None)
        self.assertFalse(worker._think_pending)

    def test_spawn_promotes_next_dots_or_hides_stale_dots(self) -> None:
        # 手番の切り替わり: 前の手番のドット(置いたツモの色)を出したままにしない。
        # 2手先の提案があり新しい組ぷよの色と一致すればそれを先に出し、無ければドットを消す
        worker, _, emitted = self._worker([])
        me = worker.trackers["self"]
        me.current_pair = ("yellow", "blue")  # (子, 軸)
        worker._last_data = OverlayDrawData(
            axis_cell=(12, 2), child_cell=(12, 3), axis_color="purple", child_color="green", label="build",
            next_axis_cell=(12, 0), next_child_cell=(11, 0), next_axis_color="blue", next_child_color="yellow",
        )
        worker._on_self_spawn(me)
        d = emitted[-1]
        self.assertFalse(d.waiting)
        self.assertEqual((d.axis_cell, d.child_cell), ((12, 0), (11, 0)))
        self.assertEqual((d.child_color, d.axis_color), ("yellow", "blue"))
        self.assertIsNone(d.next_axis_cell)
        # 2手先が無い(または色が合わない)時はドットを隠す
        me.current_pair = ("red", "red")
        worker._on_self_spawn(me)
        self.assertTrue(emitted[-1].waiting)

    def test_self_chain_start_hides_stale_dots_but_keeps_next(self) -> None:
        # 自分の発火(chain_start)を検出したら、置き終わった組ぷよのドットを連鎖の間出したままにしない
        # (実機 2026-09-20 10:20)。2手先の提案は残し、次の spawn で色が合えば差し替えられる
        worker, _, emitted = self._worker([])
        me = worker.trackers["self"]
        worker._last_data = OverlayDrawData(
            axis_cell=(12, 2), child_cell=(12, 3), axis_color="purple", child_color="green", label="build",
            next_axis_cell=(12, 0), next_child_cell=(11, 0), next_axis_color="blue", next_child_color="yellow",
        )
        worker._on_self_chain_start()
        d = emitted[-1]
        self.assertTrue(d.waiting)
        self.assertEqual(d.next_axis_cell, (12, 0))
        # 連鎖後の spawn: 2手先の色が新しい組ぷよと一致すれば提案として出る
        me.current_pair = ("yellow", "blue")
        worker._on_self_spawn(me)
        d = emitted[-1]
        self.assertFalse(d.waiting)
        self.assertEqual((d.axis_cell, d.child_cell), ((12, 0), (11, 0)))
        # 提案が無い時は何もしない
        worker._last_data = None
        n = len(emitted)
        worker._on_self_chain_start()
        self.assertEqual(len(emitted), n)

    def test_self_chain_hides_building_chain_info(self) -> None:
        # 自分の連鎖の消化中は、組んでいる連鎖の情報をパネルに出さない(利用者 2026-10-25)。連鎖が終われば戻す
        from src.engine.chain_sim import simulate_chain

        worker, _, emitted = self._worker([])
        me = worker.trackers["self"]
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="purple", child_color="green", label="build")
        me.chain_expected = simulate_chain(_board_with(["RRRR.."]))
        worker._on_self_chain_start()
        self.assertTrue(emitted[-1].chaining)
        # 撃っている連鎖(点滅から再現した値)を自分の欄に出す(利用者 2026-10-25)
        self.assertEqual(emitted[-1].self_firing, (1, 40, True))
        me.chain_steps, me.chain_score = 1, 40
        worker._on_self_chain_end(me)
        self.assertFalse(emitted[-1].chaining)
        self.assertEqual(emitted[-1].self_firing, (1, 40, False))
        self.assertEqual(emitted[-1].note, "")  # 再現どおりなら訂正のメモは出さない
        # 連鎖数が分からない終わり方でも戻す
        me.chain_expected = None
        worker._on_self_chain_start()
        me.chain_steps = 0
        worker._on_self_chain_end(me)
        self.assertFalse(emitted[-1].chaining)

    def test_self_firing_corrected_when_replay_was_wrong(self) -> None:
        # 再現した連鎖より多い段が消えたら、その場で観測の段数に直し、終わりに訂正をメモに出す(利用者「誤った点数は訂正」)
        from src.engine.chain_sim import simulate_chain

        worker, _, emitted = self._worker([])
        me = worker.trackers["self"]
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="purple", child_color="green", label="build")
        me.chain_expected = simulate_chain(_board_with(["RRRR.."]))  # 再現では1連鎖40点
        me.chain_active = True
        worker._on_self_chain_start()
        me.chain_steps = 1
        worker._on_self_chain_step(me)
        self.assertEqual(emitted[-1].self_firing, (1, 40, True))  # 再現どおりなら変えない
        me.chain_steps = 2
        worker._on_self_chain_step(me)
        self.assertEqual(emitted[-1].self_firing, (2, 0, True))  # 再現より多い段: 観測の段数(点数は不明)
        me.chain_active = False
        me.chain_steps, me.chain_score = 2, 360
        worker._on_self_chain_end(me)
        self.assertEqual(emitted[-1].self_firing, (2, 360, False))
        self.assertEqual(emitted[-1].note, "点数を訂正: 1連鎖 40点 → 2連鎖 360点")

    def test_new_game_reset_waits_longer_while_holding_all_clear(self) -> None:
        # 全消し直後の空盤面を「新しい試合」と誤認して全消しボーナスと色割当を失わない(実機 2026-09-20 13:56)
        from src.engine.tracker import Snapshot

        def snaps(n: int) -> dict:
            cells = [[EMPTY] * COLS for _ in range(ROWS)]
            for i in range(n):
                cells[ROWS - 1 - i][0] = "red"
            s = Snapshot(t=0.0, cells=cells, next1=("red", "red"), next2=("red", "red"), pending_garbage=0, eyeless_cells=frozenset())
            return {"self": s, "enemy": s}

        worker, _, _ = self._worker([])
        worker._check_new_game(0.0, snaps(10))  # 積みがあった
        me = worker.trackers["self"]
        me.all_clear = True
        worker._check_new_game(1.0, snaps(0))
        worker._check_new_game(2.5, snaps(0))  # 1秒経過でもボーナス持ちならリセットしない
        self.assertIs(worker.trackers["self"], me)
        self.assertTrue(me.all_clear)
        worker._check_new_game(12.0, snaps(0))  # 長く空のままなら本当に新しい試合
        self.assertIsNot(worker.trackers["self"], me)
        # ボーナス無しなら従来どおり1秒
        worker._check_new_game(20.0, snaps(10))
        me = worker.trackers["self"]
        worker._check_new_game(21.0, snaps(0))
        worker._check_new_game(22.1, snaps(0))
        self.assertIsNot(worker.trackers["self"], me)

    def test_game_over_clears_all_clear_bonus_so_new_game_resets(self) -> None:
        # 全消しの直後に試合が終わると、ボーナス持ちの10秒待ちの間に次の試合が始まりリセットされなかった
        # (実機 2026-10-20 22:28 の 115.0s)。負け(12個以上が一気に消えて全消しにならない)でボーナスを消す
        from src.engine.tracker import Snapshot

        def snaps(n: int) -> dict:
            cells = [[EMPTY] * COLS for _ in range(ROWS)]
            for i in range(n):
                cells[ROWS - 1 - i][0] = "red"
            s = Snapshot(t=0.0, cells=cells, next1=("red", "red"), next2=("red", "red"), pending_garbage=0, eyeless_cells=frozenset())
            return {"self": s, "enemy": s}

        worker, _, _ = self._worker([])
        worker._check_new_game(0.0, snaps(10))
        me, en = worker.trackers["self"], worker.trackers["enemy"]
        me.all_clear = True
        en.chain_dropped = 5  # 小さい盤面の消去(点滅を見逃した全消しの可能性)ではボーナスを消さない
        worker._on_game_over(0.5, "enemy", en)
        self.assertTrue(me.all_clear)
        en.chain_dropped = 43  # 相手の負け
        worker._on_game_over(0.5, "enemy", en)
        self.assertFalse(me.all_clear)
        worker._check_new_game(1.0, snaps(0))
        worker._check_new_game(2.1, snaps(0))  # 通常の1秒でリセットされる
        self.assertIsNot(worker.trackers["self"], me)

    def test_opening_preview_before_start_and_adopted_at_first_turn(self) -> None:
        # 開始前(操作中の組ぷよ無し・盤面空・枠に第1手と第2手)に空盤面+2組で初手を計算して表示し、
        # 開始の出現で第1手・第2手が一致すれば先読みとして使う(実機 2026-09-20 15:50)。計算は思考スレッドで行う(2026-10-11)
        first = _move("build", (12, 2), (11, 2))
        first.field_after = None
        worker, client, emitted = self._worker([first, _move("build", (12, 2), (11, 2))])
        me = worker.trackers["self"]
        me.current_pair, me.next1, me.next2 = None, ("green", "purple"), ("yellow", "blue")
        worker._opening_preview(0.5, me)
        self.assertEqual(len(client.calls), 1)
        self.assertIn("solo", client.calls[0])
        self.assertEqual(len(client.calls[0]["pairs"]), 2)
        self.assertIn("開始前", emitted[-1].note)
        self.assertEqual((emitted[-1].child_color, emitted[-1].axis_color), ("green", "purple"))
        # 同じ枠なら再計算しない
        worker._opening_preview(0.6, me)
        self.assertEqual(len(client.calls), 1)
        # 開始: 第1手が出現(枠はまだ入れ替わっていない時はネクネク未読)→ 出現で計算済みの初手を先読みとして置く
        me.current_pair, me.next1, me.next2 = ("green", "purple"), ("yellow", "blue"), None
        worker._on_self_spawn(me)
        self.assertIsNotNone(worker._precomputed)
        self.assertIs(worker._precomputed[2], first)
        # 開始後は開始前の計算をしない
        worker._opening_preview(2.1, me)
        self.assertEqual(len(client.calls), 1)

    def test_opening_preview_uses_first_pair_after_frame_swap(self) -> None:
        # 開始で枠が入れ替わった後(第1手は _prestart_first、枠は第2・3手)の解釈でも計算する
        # (実機 2026-09-27 12:25: 第2手を第1手とみなし、ネクストと違う色で提案した)。枠がまだ入れ替わっていない解釈と両方計算する
        two = _move("build", (12, 0), (11, 0))
        three = _move("build", (12, 2), (11, 2))
        worker, client, emitted = self._worker([two, three])
        me = worker.trackers["self"]
        me.current_pair, me.next1, me.next2 = None, ("purple", "red"), ("purple", "purple")
        me._prestart_first = ("blue", "blue")
        worker._opening_preview(0.5, me)
        self.assertEqual([len(c["pairs"]) for c in client.calls], [2, 3])
        self.assertEqual((emitted[-1].child_color, emitted[-1].axis_color), ("blue", "blue"))  # 表示は入れ替わりの後の解釈
        # 入れ替わりが開始だった: 第1手 (青青) が出現 → 3組で計算した手を使う
        me.current_pair, me.next1, me.next2 = ("blue", "blue"), ("purple", "red"), ("purple", "purple")
        worker._on_self_spawn(me)
        self.assertEqual(worker._precomputed[1], (("blue", "blue"), ("purple", "red")))
        self.assertIs(worker._precomputed[2], three)

    def test_opening_result_after_start_is_sent(self) -> None:
        # 開始の後に初手の計算結果が届いても、出現した組ぷよに合えばその場で送る
        worker, client, emitted = self._worker([])
        me = worker.trackers["self"]
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        me.current_pair, me.next1, me.next2 = ("green", "purple"), ("yellow", "blue"), None
        me.observed_cells = _empty_cells()
        move = _move("build", (12, 0), (11, 0))
        move.path = ["LEFT", "LEFT", "DROP"]
        worker._apply_opening({"t": 0.5, "seq": (("green", "purple"), ("yellow", "blue"))}, move, None)
        self.assertEqual(sent, [["LEFT", "LEFT", "DROP"]])

    def test_autoplay_sends_precomputed_path_at_spawn(self) -> None:
        # 先読みの盤面どおりなら、組ぷよが出た時点で操作を送る(思考を待つと硬直する。実機 2026-09-27)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        me.current_pair, me.next1 = ("yellow", "blue"), ("green", "yellow")
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append((turn, path)) or True
        move = _move("build")
        move.path = ["LEFT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        expected[ROWS - 1][1] = "red"
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        me.observed_count = 2  # 予想した盤面と同じ
        me.observed_cells = [row[:] for row in expected]
        worker._on_self_spawn(me)
        self.assertEqual(sent, [(me.turn, ["LEFT", "DROP"])])
        # 相手の攻撃が絡む局面では送らない(思考が受け・潰しに変えることがあり、キーは取り消せない)
        sent.clear()
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        me.pending_garbage = 3
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        me.pending_garbage = 0
        # 盤面が予想と違う(連鎖・おじゃま・置き間違い)なら送らず、通常の思考に任せる
        sent.clear()
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        me.observed_count = 8
        me.observed_cells = _empty_cells()
        for c in range(4):
            me.observed_cells[ROWS - 1][c] = "red"
            me.observed_cells[ROWS - 2][c] = "blue"
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        # 個数は同じでも盤面が違う(前の手が置き場所からずれた)なら送らない(実機 2026-10-10 15:28: 別の盤面用の
        # 手順を送って次の手もずれた)
        sent.clear()
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        me.observed_count = 2
        me.observed_cells = _empty_cells()
        me.observed_cells[ROWS - 1][3] = "red"
        me.observed_cells[ROWS - 1][4] = "red"
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        me.observed_cells = [row[:] for row in expected]
        # 組ぷよが先読みと違う時も送らない
        worker._precomputed = (expected, (("red", "red"), ("blue", "blue")), move)
        me.observed_count = 2
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])

    def test_autoplay_waits_for_think_when_precomputed_move_clears(self) -> None:
        # 消す手(単発・連鎖)の先読みは出現時点で送らず、即表示もせず、3組目まで見た思考の手を送る
        # (実機 2026-10-11 07:23: 2組しか見えない先読みが緑4個の単発を選んで送り、思考は積む手だった)
        fresh = _move("build", (12, 2), (11, 2))
        fresh.path = ["DROP"]
        worker, client, emitted = self._worker([fresh])
        me = worker.trackers["self"]
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("harass", (12, 1), (12, 2))
        pre_move.path = ["RIGHT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        me.observed_count = 1
        me.observed_cells = [row[:] for row in expected]
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])  # 出現時点では送らない
        # 手番開始の思考でも先読みの手を即送信せず、思考の手を送る
        me.settling = False
        worker._precomputed = (me.settled, (me.current_pair, me.next1), pre_move)
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertEqual(sent, [["DROP"]])
        self.assertEqual(emitted[-1].axis_cell, (12, 2))

    def test_autoplay_sends_precomputed_under_same_attack(self) -> None:
        # 相手の攻撃中でも、先読みを計算した時と攻撃の状況(着弾見込み・相手の連鎖中)が同じなら出現時点で送る
        # (実機 2026-10-10 15:28: 相手の連鎖中に出た手番は送信まで中央0.25秒・最大1.25秒硬直した)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        en = worker.trackers["enemy"]
        me.current_pair, me.next1 = ("yellow", "blue"), ("green", "yellow")
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append((turn, path)) or True
        move = _move("build")
        move.path = ["LEFT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        expected[ROWS - 1][1] = "red"
        me.observed_count = 2
        me.observed_cells = [row[:] for row in expected]
        en.chain_active = True
        worker._incoming_carry = 30
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        worker._precomputed_attack = worker._attack_snapshot(me)
        worker._on_self_spawn(me)
        self.assertEqual(sent, [(me.turn, ["LEFT", "DROP"])])
        # 相手の連鎖が進んで予告の数が増えても、見積もった着弾量が同じなら送る(実機 2026-10-10 15:50)
        sent.clear()
        worker.autoplay._played_turn = None
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        me.pending_garbage = 12
        worker._on_self_spawn(me)
        self.assertEqual(sent, [(me.turn, ["LEFT", "DROP"])])
        # 同じ連鎖の間は、PBA 側の着弾見込みが後から更新されても送る(先読みもブリッジが同じ連鎖を再現して見積もっている。
        # キーを送った手の再計算の前に先読みを計算するため 2026-10-11)
        sent.clear()
        worker.autoplay._played_turn = None
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        worker._incoming_carry = 60
        worker._on_self_spawn(me)
        self.assertEqual(sent, [(me.turn, ["LEFT", "DROP"])])
        # 別の連鎖が始まっていたら送らない(思考が受け・返しに変えうる)
        sent.clear()
        worker.autoplay._played_turn = None
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        worker._attack_count += 1
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        # 相手の連鎖が終わった後は、着弾見込みが変わっていたら送らない
        en.chain_active = False
        worker._incoming_carry = 30
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        worker._precomputed_attack = worker._attack_snapshot(me)
        worker._incoming_carry = 60
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        en.chain_active = True
        # 攻撃の状況が記録されていない先読み(従来)は、攻撃中には送らない
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        worker._precomputed_attack = None
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])

    def test_autoplay_sends_anticipated_move_when_enemy_fires(self) -> None:
        # 相手の発火を想定した先読み: 相手の連鎖が始まって通常の先読みを捨てた後でも、想定した着弾量と見積もりが
        # 近ければ出現時点で送る(利用者「相手がこの場面で発火してきたらどうなるかをあらかじめ想定できないか」)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        en = worker.trackers["enemy"]
        me.current_pair, me.next1 = ("yellow", "blue"), ("green", "yellow")
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append((turn, path)) or True
        alt = _move("accept")
        alt.path = ["RIGHT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        me.observed_count = 1
        me.observed_cells = [row[:] for row in expected]
        worker._precomputed = None  # 相手の消去開始で捨てた
        worker._precomputed_alt = (expected, (me.current_pair, me.next1), alt, 30)
        en.chain_active = True
        worker._incoming_carry = 0
        worker._incoming_at_think = 34  # 見積もり 34個(想定 30個、差は許容内)
        worker._on_self_spawn(me)
        self.assertEqual(sent, [(me.turn, ["RIGHT", "DROP"])])
        # 見積もりが想定と大きく違えば送らない
        sent.clear()
        worker.autoplay._played_turn = None
        worker._precomputed_alt = (expected, (me.current_pair, me.next1), alt, 30)
        worker._incoming_at_think = 90
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        # 相手が連鎖中でなければ使わない
        worker._incoming_at_think = 30
        en.chain_active = False
        me.pending_garbage = 5
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])

    def test_drop_release_reason_is_logged(self) -> None:
        # 下キーを離した理由(観測個数・手番・連鎖)をログに残し、直後に離した時は観測盤面も残す
        # (実機 2026-09-27 14:52: 落下11msで離して組ぷよが自然落下した。理由が分からなかった)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        captured = {}
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: captured.setdefault("placed", placed) is None or True
        me.observed_count = 10
        me.last_observed_text = "......"
        worker._autoplay_send(me, ["LEFT", "DROP"])
        placed = captured["placed"]
        self.assertFalse(placed())
        self.assertEqual(logs, [])
        me.observed_count = 12
        self.assertTrue(placed())
        self.assertIn("観測個数 10→12", logs[-1])
        self.assertIn("送信時の観測盤面", logs[-1])

    def test_drop_release_waits_for_target_cells(self) -> None:
        # 提案の置き場所が分かっている時は、その2マスが組ぷよの色で埋まるまで下キーを離さない
        # (実機 2026-09-27 14:57: 高い列の真上を通る組ぷよや演出で個数が2個増え、42〜63msで離して自然落下した)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        worker._log = lambda text: None
        captured = {}
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: captured.setdefault("placed", placed) is None or True
        me.current_pair = ("green", "purple")  # (子, 軸)
        move = _move("build", (12, 5), (11, 5))  # 6列目の最下段に軸、その上に子
        me.observed_count = 10
        me.observed_cells = _empty_cells()
        worker._autoplay_send(me, ["RIGHT", "DROP"], move)
        placed = captured["placed"]
        # 5列目の上に組ぷよが接して個数が増えても、提案のマスが空なら離さない
        me.observed_count = 12
        self.assertFalse(placed())
        # 提案の2マスが組ぷよの色で埋まったら離す
        cells = _empty_cells()
        cells[12][5] = "purple"
        cells[11][5] = "green"
        me.observed_cells = cells
        self.assertTrue(placed())

    def test_drop_release_waits_while_pair_is_floating(self) -> None:
        # 提案の2マスが埋まって見えても(着地予測の点を組ぷよと読んだ)、落下中の組ぷよが浮いて見えている間は下キーを離さない
        # (実機 2026-10-12 07:18 turn=22: 5段落とす位置で落下開始から24msで離し、自然落下して次の出現まで1.93秒)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        captured = {}
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: captured.setdefault("placed", placed) is None or True
        me.current_pair = ("green", "purple")  # (子, 軸)
        move = _move("build", (12, 5), (11, 5))
        me.observed_count = 10
        me.observed_cells = _empty_cells()
        worker._autoplay_send(me, ["RIGHT", "DROP"], move)
        placed = captured["placed"]
        cells = _empty_cells()
        cells[12][5] = "purple"  # 着地予測の点を組ぷよと読んだ
        cells[11][5] = "green"
        me.observed_cells = cells
        me.floating_now = [(3, 5, "purple"), (2, 5, "green")]  # 本物の組ぷよはまだ上で落下中
        self.assertFalse(placed())
        self.assertTrue(any("落下中の組ぷよが浮いている" in text for text in logs))
        me.floating_now = [(1, 0, "red")]  # 組ぷよと違う色の浮いた演出は見ない
        self.assertTrue(placed())

    def _send_traced(self):
        """置きミスの診断の試験用: 6列目に縦に置く手を送り、保存した画像の名前を集める。"""
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        saved: list[str] = []
        worker._trace_saver = type("Saver", (), {"save": lambda self, path, frame, params=None: saved.append(path.name) or True})()
        captured = {}
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: captured.setdefault("placed", placed) is None or True
        me.current_pair = ("green", "purple")  # (子, 軸)
        move = _move("build", (12, 5), (11, 5))  # 6列目の最下段に軸、その上に子
        me.observed_count = 10
        me.observed_cells = _empty_cells()
        worker._autoplay_send(me, ["RIGHT", "RIGHT", "RIGHT", "DROP"], move)
        return worker, me, logs, saved, captured["placed"]

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_move_miss_is_traced(self) -> None:
        # 提案の2マス以外に着地した手は、キーの時刻・組ぷよの動き・画面を残す
        # (実機 2026-10-24 14:01 turn=17: 最後の右移動が効かず5列目に着地して4連鎖が暴発した。原因がログから分からなかった)
        worker, me, logs, saved, _ = self._send_traced()
        t_send = worker._move_trace["t_send"]
        worker.autoplay.sender.timed.append((worker._t0_perf + t_send + 0.010, "RIGHT", True))
        me.floating_now = [(1, 2, "purple"), (0, 2, "green")]
        worker._record_move_trace(t_send + 0.016, _frame())
        me.floating_now = [(3, 4, "purple"), (2, 4, "green")]  # 5列目で止まった
        worker._record_move_trace(t_send + 0.100, _frame())
        cells = _empty_cells()
        cells[12][4] = "purple"
        cells[11][4] = "green"
        me.observed_cells = cells
        me.floating_now = []
        me.turn += 1  # 下キーを離す前に手番が進んだ
        worker._record_move_trace(t_send + 0.300, _frame())
        self.assertIsNone(worker._move_trace)
        text = "\n".join(logs)
        self.assertIn("置きミスの診断(turn=", text)
        self.assertIn("6列1段purple", text)
        self.assertIn("+10RIGHT↓", text)
        self.assertIn("[3列13段・3列12段]", text)
        self.assertIn("[5列11段・5列10段]", text)
        self.assertEqual(len(saved), 3)
        self.assertTrue(saved[0].startswith("movemiss_self_"))

    def test_move_on_target_is_not_traced(self) -> None:
        # 提案の2マスに着地した手は何も残さない
        worker, me, logs, saved, placed = self._send_traced()
        cells = _empty_cells()
        cells[12][5] = "purple"
        cells[11][5] = "green"
        me.observed_cells = cells
        self.assertTrue(placed())
        worker._record_move_trace(1.0, _frame())
        self.assertIsNone(worker._move_trace)
        self.assertFalse(any("置きミスの診断" in text for text in logs))
        self.assertEqual(saved, [])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_display_follows_keys_already_sent(self) -> None:
        # 先読みの手順を送った後に思考が別の手を選んでも、キーは取り消せないので表示は送った手のままにする
        # (実機 2026-09-27 12:01: 送った手順と表示が食い違う設置が 49/428件)
        pre_move = _move("build", (12, 1), (11, 1))
        pre_move.path = ["LEFT", "DROP"]
        worker, client, emitted = self._worker([_move("harass", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        worker.autoplay.enabled = True
        worker.autoplay._played_turn = me.turn  # この手番は送信済み
        worker._precomputed = (me.settled, (me.current_pair, me.next1), pre_move)
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertEqual(emitted[-1].axis_cell, (12, 1))  # 送った手のまま
        self.assertIn("実行済み", emitted[-1].note)
        # 送っていない手番なら従来どおり対応の手に差し替える(全自動では先読みどおりなら取り込み時点で送るので、支援モードで確かめる)
        worker.autoplay.enabled = False
        worker.autoplay._played_turn = None
        worker._precomputed = (me.settled, (me.current_pair, me.next1), pre_move)
        client.moves.append(_move("harass", (12, 5), (11, 5)))
        self.assertTrue(worker._think(2.0, _frame(), new_turn=True))
        self.assertEqual(emitted[-1].axis_cell, (12, 5))

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_precompute_uses_sent_move_under_attack(self) -> None:
        # 相手の攻撃中に出現時点で先読みの手を送った後、思考が別の手を選んでも、表示と次の手の先読みは送った手で行う
        # (実機 2026-10-11 07:07: 思考の手の盤面で先読みし、次の手番で盤面が合わず約0.6秒自由落下した)
        sent_move = _move("accept", (12, 4), (12, 5))
        sent_move.path = ["RIGHT", "CCW", "RIGHT", "DROP"]
        sent_move.field_after = ["......"] * 13
        fresh = _move("accept", (12, 5), (11, 5))
        fresh.path = ["RIGHT", "RIGHT", "DROP"]
        fresh.field_after = None  # 思考の手で先読みしようとすると先読みが作られない(区別のため)
        fresh.incoming = 30
        worker, client, emitted = self._worker([fresh, _move("build", (12, 0), (11, 0))])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent: list = []
        worker.autoplay.enabled = True
        def play(turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw):
            # 本物と同じく、同じ手番では二度送らない
            if worker.autoplay._played_turn == turn:
                return False
            worker.autoplay._played_turn = turn
            sent.append(path)
            return True

        worker.autoplay.play = play
        worker.trackers["enemy"].chain_active = True
        self.assertTrue(worker._autoplay_send(me, sent_move.path, sent_move))  # 出現時点で送った
        me.settling = False
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertEqual(sent, [sent_move.path])  # 二度は送らない
        self.assertEqual(emitted[-1].axis_cell, (12, 4))  # 表示は送った手
        self.assertTrue(any("送った手を実行中" in text for text in logs))
        self.assertEqual(worker._incoming_carry, 30)  # 着弾量の見積もりは思考のものを使う
        # 次の手の先読みは送った手の盤面で計算する
        self.assertIs(worker._last_move, sent_move)
        self.assertIsNotNone(worker._precomputed)
        # 送っていない手番なら思考の手を出す
        worker._sent_move = None
        worker.autoplay._played_turn = None
        client.moves = [fresh, _move("build", (12, 0), (11, 0))]
        self.assertTrue(worker._think(2.0, _frame(), new_turn=True))
        self.assertEqual(emitted[-1].axis_cell, (12, 5))

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_committed_rethink_requests_precompute_first(self) -> None:
        # キーを送り終えた手の再計算(相手の連鎖の開始など)では、次の手の先読みを再計算より先に依頼する
        # (実機 2026-10-11 11:07: 再計算の後に先読みしていたので次の組ぷよの出現に間に合わず、自由落下した)
        class FakeWorker:
            def __init__(self) -> None:
                self.submits: list = []

            def submit(self, kind, ctx, fn, urgent=False):
                self.submits.append((kind, ctx, urgent))

            def cancel(self, kind):
                return False

            def poll(self):
                return []

        worker, client, emitted = self._worker([])
        me = worker.trackers["self"]
        sent_move = _move("build", (12, 2), (11, 2))
        sent_move.path = ["DROP"]
        sent_move.field_after = ["......"] * 13
        fw = FakeWorker()
        worker._think_worker = fw
        worker.autoplay.enabled = True
        worker.autoplay._played_turn = me.turn
        worker._sent_move = (me.turn, sent_move)
        worker.trackers["enemy"].chain_active = True
        self.assertTrue(worker._think(1.0, _frame(), note="相手の消去開始を検出: 提案を更新"))
        kinds = [(k, u) for k, _, u in fw.submits]
        self.assertEqual(kinds, [("main", True), ("pre", True)])  # 後から urgent で入れた先読みが先に実行される
        self.assertIs(fw.submits[1][1]["move"], sent_move)
        self.assertIsNotNone(worker._pre_inflight)
        # 再計算の結果が届いても、同じ前提の先読みはもう一度依頼しない
        worker._dispatch_precompute(me, worker.trackers["enemy"], sent_move)
        self.assertEqual(len(fw.submits), 2)
        # 送っていない手番の再計算では先に依頼しない(従来どおり)
        fw.submits.clear()
        worker.autoplay._played_turn = None
        worker._think(2.0, _frame(), note="相手の消去開始を検出: 提案を更新")
        self.assertEqual([k for k, _, _ in fw.submits], ["main"])

    def test_late_precompute_is_sent_at_next_turn(self) -> None:
        # 出現の後に届いた先読みでも、ちょうど次の手番で組ぷよ・盤面が先読みどおりなら、捨てずにすぐ送る
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append((turn, path)) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][3] = "red"
        me.turn = 5
        me.observed_cells = [row[:] for row in expected]
        me.observed_count = 1
        ctx = {"key": (4, None, None, None, ""), "move": pre_move, "expected": expected,
               "pairs": (me.current_pair, me.next1), "attack": worker._attack_snapshot(me)}
        worker._apply_precompute(ctx, pre_move, None)
        self.assertEqual(sent, [(5, ["LEFT", "LEFT", "DROP"])])
        self.assertTrue(any("出現の後に届いた" in text for text in logs))
        # 2手番以上前の先読みは使わない
        sent.clear()
        worker.autoplay._played_turn = None
        ctx["key"] = (3, None, None, None, "")
        worker._apply_precompute(ctx, pre_move, None)
        self.assertEqual(sent, [])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_enemy_chain_event_refreshes_precompute_for_sent_move(self) -> None:
        # 相手の連鎖の開始・終了では、この手番のキーを送り終えていれば、再計算を待たずに次の手の先読みをすぐやり直す
        # (実機 2026-10-11 11:29: 自分の連鎖中は再計算が保留され、先読みが無いまま出現した手番が4手)
        class FakeWorker:
            def __init__(self) -> None:
                self.submits: list = []

            def submit(self, kind, ctx, fn, urgent=False):
                self.submits.append((kind, ctx, urgent))

            def cancel(self, kind):
                return False

            def poll(self):
                return []

        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        sent_move = _move("main", (12, 2), (11, 2))
        sent_move.path = ["DROP"]
        sent_move.field_after = ["......"] * 13
        fw = FakeWorker()
        worker._think_worker = fw
        worker.autoplay.enabled = True
        worker.autoplay._played_turn = me.turn
        worker._sent_move = (me.turn, sent_move)
        worker.trackers["enemy"].chain_active = True
        worker._refresh_precompute_for_sent()
        self.assertEqual([(k, u) for k, _, u in fw.submits], [("pre", True)])
        # 送っていない手番では何もしない
        fw.submits.clear()
        worker.autoplay._played_turn = None
        worker._refresh_precompute_for_sent()
        self.assertEqual(fw.submits, [])

    def test_precompute_kept_when_board_taken_in_during_same_turn(self) -> None:
        # 同じ手番のうちに盤面の取り込みが進んだ(自分の連鎖中など)だけなら、届いた先読みを捨てずに残す
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["DROP"]
        expected = _empty_cells()
        ctx = {"key": (me.turn, me.current_pair, me.next1, me.next2, "古い盤面"), "move": pre_move, "expected": expected,
               "pairs": (me.next1, me.next2), "attack": worker._attack_snapshot(me)}
        worker._apply_precompute(ctx, pre_move, None)
        self.assertIsNotNone(worker._precomputed)
        self.assertEqual(worker._precomputed[1], (me.next1, me.next2))

    def test_precompute_is_retried_at_turn_ready(self) -> None:
        # 出現の瞬間は置いたぷよが読めず盤面が違っても、取り込みの時点で先読みどおりなら、思考を待たずに送る
        # (実機 2026-10-11 11:29: 出現時点の「盤面が違う」が0.04秒後の取り込みでは先読みどおりだった)
        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        expected = _empty_cells()
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        me.observed_cells = None  # 出現の瞬間: 盤面がまだ読めていない
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        me.observed_cells = [row[:] for row in expected]  # 取り込みの時点では先読みどおり
        me.settled = [row[:] for row in expected]
        me.settling = False
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertEqual(sent[0], ["LEFT", "LEFT", "DROP"])
        self.assertTrue(any("取り込み時点で送信" in text for text in logs))

    def test_mismatched_precompute_is_retried_every_frame(self) -> None:
        # 出現の瞬間だけ盤面の読みが違っても(相手の攻撃の演出など)、取り込みを待たずに次のフレームで先読みどおりに
        # 読めた時点で送る。違ったマスはログに残す(実機 2026-10-11 21:30: 相手の連鎖中に40手、うち27手は取り込みでは一致)
        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        observed = [row[:] for row in expected]
        observed[ROWS - 1][0] = "green"  # 出現の瞬間: 1段目1列目を読み違えた
        me.observed_cells = observed
        worker._now = 10.0
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])
        self.assertTrue(any("違うマス: 1段目1列目 期待red 読みgreen" in text for text in logs))
        worker._now = 10.03
        worker._retry_mismatched_precompute(me)  # 次のフレームもまだ違う: 送らず比べ直しを続ける
        self.assertEqual(sent, [])
        self.assertIsNotNone(worker._mismatch_retry)
        me.observed_cells = [row[:] for row in expected]  # 次のフレームでは先読みどおり
        worker._now = 10.06
        worker._retry_mismatched_precompute(me)
        self.assertEqual(sent, [["LEFT", "LEFT", "DROP"]])
        self.assertIsNone(worker._mismatch_retry)
        self.assertTrue(any("出現の60ms後で送信" in text for text in logs))

    def test_precompute_sent_when_mismatch_is_only_eyeless_color(self) -> None:
        # 出現の瞬間に空のはずのマスを色と読んでも、それが白い目の無い色(着地予測の点・攻撃の演出)だけなら先読みどおりとみなし、
        # 出現時点で送る(実機 2026-10-12 07:00: 「盤面が違う」で送れなかった8手の多く)
        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        worker._log = lambda text: None
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        observed = [row[:] for row in expected]
        observed[ROWS - 2][0] = "purple"  # 演出を紫と読んだ
        me.observed_cells = observed
        me.observed_cells_eyed = [row[:] for row in expected]  # 目の無い色を空にすると先読みどおり
        worker._on_self_spawn(me)
        self.assertEqual(sent, [["LEFT", "LEFT", "DROP"]])

    def test_spawn_ghost_is_ignored_at_spawn(self) -> None:
        # 出現の直後に3列目の積みのすぐ上の2マスに出る着地予測の点(組ぷよと同じ色)は空として比べる。
        # 違う色なら着地予測の点ではないので送らない(実機 2026-10-12 07:00: turn25 紫/赤など3手)
        from src.app import strip_spawn_ghost

        expected = _empty_cells()
        expected[ROWS - 1][2] = "green"  # 3列目の高さ1
        observed = [row[:] for row in expected]
        observed[ROWS - 2][2] = "red"     # 軸の着地予測(2段目)
        observed[ROWS - 3][2] = "purple"  # 子の着地予測(3段目)
        pair = ("purple", "red")
        self.assertEqual(strip_spawn_ghost(observed, expected, pair), expected)
        self.assertNotEqual(strip_spawn_ghost(observed, expected, ("yellow", "blue")), expected)
        moved = [row[:] for row in observed]
        moved[ROWS - 2][2], moved[ROWS - 2][3] = "empty", "red"  # 3列目以外のマスは除かない
        self.assertNotEqual(strip_spawn_ghost(moved, expected, pair), expected)

        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        worker._log = lambda text: None
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        me.current_pair = pair
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        me.observed_cells = observed
        me.observed_cells_eyed = observed  # 点の光沢を目と読んだ(目の有無では除けない)
        worker._on_self_spawn(me)
        self.assertEqual(sent, [["LEFT", "LEFT", "DROP"]])

    def test_precompute_not_sent_when_eyed_board_also_differs(self) -> None:
        # 目のある色(本物のぷよ)が違うなら、目の無い色を空にしても違うので送らない
        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        worker._log = lambda text: None
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        expected = _empty_cells()
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        observed = _empty_cells()
        observed[ROWS - 1][3] = "blue"
        me.observed_cells = observed
        me.observed_cells_eyed = observed
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_place_cancel_passes_fall_rows(self) -> None:
        # 設置キャンセル(2026-10-21 仮): 有効にすると、狙える置き方では確定した盤面の高さから落ちる段数を求めて渡す。
        # 画面の観測盤面には出現した組ぷよが3列目の上に見えているが、それでは判定しない
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        calls: list = []
        worker.autoplay.enabled = True
        worker.autoplay.keymap.place_cancel = "move"
        worker.autoplay.play = lambda turn, path, **kw: calls.append(kw) or True
        me.settled = _empty_cells()
        observed = _empty_cells()
        observed[1][2] = "red"  # 出現した組ぷよ(3列目の12段目)
        me.observed_cells = observed
        move = _move("build", (12, 3), (12, 4))  # 1列右へ動かして4・5列目に横置き
        worker._autoplay_send(me, ["RIGHT", "DROP"], move)
        self.assertEqual(calls[-1].get("cancel_fall_rows"), 11)
        # 段差があれば狙わない
        me.settled[12][4] = "blue"
        worker.autoplay._played_turn = None
        worker._autoplay_send(me, ["RIGHT", "DROP"], move)
        self.assertNotIn("cancel_fall_rows", calls[-1])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_step_cancel_passes_plan(self) -> None:
        # 段差キャンセル(2026-10-25): 有効にすると、縦置きを1段低い隣の列へ置く手では確定した盤面の高さから手順を求めて渡す
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        calls: list = []
        worker.autoplay.enabled = True
        worker.autoplay.keymap.place_cancel = "off"
        worker.autoplay.keymap.step_cancel = True
        worker.autoplay.play = lambda turn, path, **kw: calls.append(kw) or True
        me.settled = _empty_cells()
        me.settled[12][2] = "blue"  # 3列目だけ1段高い
        me.observed_cells = _empty_cells()
        move = _move("build", (12, 3), (11, 3))  # 4列目に縦置き(子が上)
        worker._autoplay_send(me, ["RIGHT", "DROP"], move)
        self.assertEqual(calls[-1].get("step_plan"), (10, [], ["CW", "CCW", "RIGHT"]))
        # 切っていれば渡さない
        worker.autoplay.keymap.step_cancel = False
        worker.autoplay._played_turn = None
        worker._autoplay_send(me, ["RIGHT", "DROP"], move)
        self.assertNotIn("step_plan", calls[-1])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_step_cancel_uses_precomputed_board(self) -> None:
        # 出現時点で送る先読みの手は、段差キャンセルの高さを先読みの盤面で数える(実機 2026-10-25 08:46 turn=49:
        # 確定した盤面の更新が出現の後に届き、1手前の盤面で落ちる段数を見積もって回転と横移動が着地の後に入った)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        calls: list = []
        worker.autoplay.enabled = True
        worker.autoplay.keymap.place_cancel = "off"
        worker.autoplay.keymap.step_cancel = True
        worker.autoplay.play = lambda turn, path, **kw: calls.append(kw) or True
        me.settled = _empty_cells()  # まだ1手前の盤面(3列目が空)
        expected = _empty_cells()
        expected[12][2] = "blue"  # 先読みの盤面では3列目が1段
        me.observed_cells = [row[:] for row in expected]
        move = _move("build", (12, 3), (11, 3))  # 4列目に縦置き(子が上)
        move.path = ["RIGHT", "DROP"]
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        self.assertIsNone(worker._send_precomputed(me))
        self.assertEqual(calls[-1].get("step_plan"), (10, [], ["CW", "CCW", "RIGHT"]))

    def test_harass_wait_has_deadline(self) -> None:
        # 先読みの手が小さい消去なら3組目まで見た思考を待つが、待ちには上限がある(2026-10-21): 出現から HARASS_WAIT_MAX_SEC たっても
        # 手を送っていなければ先読みの手をそのまま送る(実機 2026-10-12 12:13 turn=34: 思考の手が送られないまま2秒たって自由落下した)
        from src.app import HARASS_WAIT_MAX_SEC

        worker, client, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("harass", (12, 1), (12, 2))
        pre_move.path = ["RIGHT", "DROP"]
        expected = _empty_cells()
        expected[ROWS - 1][0] = "red"
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        me.observed_cells = [row[:] for row in expected]
        worker._now = 10.0
        worker._on_self_spawn(me)
        self.assertEqual(sent, [])  # 出現時点では思考を待つ
        self.assertIsNotNone(worker._mismatch_retry)
        worker._now = 10.0 + HARASS_WAIT_MAX_SEC / 2
        worker._retry_mismatched_precompute(me)
        self.assertEqual(sent, [])  # 上限の前はまだ待つ
        worker._now = 10.0 + HARASS_WAIT_MAX_SEC + 0.01
        worker._retry_mismatched_precompute(me)
        self.assertEqual(sent, [["RIGHT", "DROP"]])  # 上限を過ぎたら先読みの手を送る
        self.assertIsNone(worker._mismatch_retry)
        self.assertTrue(any("先読みの小さい消去をそのまま送りました" in text for text in logs))

    def test_blocked_move_is_sent_after_deadline(self) -> None:
        # 保険(2026-10-24): 思考の手が届いた時に盤面の確定待ちで送れなかったら、出現から BLOCKED_SEND_SEC たっても確定待ちのままなら送る
        # (実機 07:11 turn=35: 確定待ちのまま組ぷよが出現位置で固定されて窒息した)
        from src.app import BLOCKED_SEND_SEC

        worker, client, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        move = _move("harass", (2, 2), (2, 3))
        move.path = ["RIGHT", "DROP"]
        worker._self_spawn_t = 10.0
        me.settling = True
        worker._autoplay_move(me, move)
        self.assertEqual(sent, [])
        worker._now = 10.0 + BLOCKED_SEND_SEC / 2
        worker._send_blocked_move(me)
        self.assertEqual(sent, [])  # 上限の前はまだ待つ
        worker._now = 10.0 + BLOCKED_SEND_SEC + 0.01
        worker._send_blocked_move(me)
        self.assertEqual(sent, [["RIGHT", "DROP"]])
        self.assertIsNone(worker._blocked_move)
        self.assertTrue(any("思考の手を送りました" in text for text in logs))
        # 自分の連鎖中・手番が進んだ時は送らない
        sent.clear()
        worker._blocked_move = (me.turn, move)
        me.chain_active = True
        worker._send_blocked_move(me)
        self.assertEqual(sent, [])
        self.assertIsNone(worker._blocked_move)

    def test_mismatch_retry_stops_at_turn_ready(self) -> None:
        # 取り込み時点の確定した盤面でも違えば本当に盤面が違うので、毎フレームの比べ直しをやめて思考の手に任せる
        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        worker._log = lambda text: None
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        expected = _empty_cells()
        worker._precomputed = (expected, (me.current_pair, me.next1), pre_move)
        actual = _empty_cells()
        actual[ROWS - 1][3] = "blue"  # 本当に違う盤面(おじゃま・置き間違いなど)
        me.observed_cells = actual
        worker._on_self_spawn(me)
        self.assertIsNotNone(worker._mismatch_retry)
        me.settled = [row[:] for row in actual]
        me.settling = False
        worker._think(1.0, _frame(), new_turn=True)
        self.assertIsNone(worker._mismatch_retry)
        self.assertNotIn(["LEFT", "LEFT", "DROP"], sent)

    def test_field_edge_offset_follows_horizontal_shake(self) -> None:
        # 横揺れの確認用: 盤面(暗い)と右の枠(明るい)の境目の横位置を、校正した位置からのずれで返す
        import numpy as np

        from src.app import field_edge_offset
        from src.capture.geometry import BoardGeometry

        geo = BoardGeometry(((100, 100), (580, 100), (580, 1060), (100, 1060)))  # 1マス80px
        frame = np.full((1200, 800, 3), 200, np.uint8)
        frame[100:1060, 100:580] = 30
        self.assertEqual(field_edge_offset(frame, geo), 0)
        self.assertEqual(field_edge_offset(np.roll(frame, 6, axis=1), geo), 6)
        self.assertEqual(field_edge_offset(np.roll(frame, -5, axis=1), geo), -5)
        # 壁際のぷよ(盤面の内側)の輪郭は境目と取り違えない
        frame[300:900, 540:570] = 255  # 壁から10px内側まで(実機のぷよの輪郭は壁から離れている)
        self.assertEqual(field_edge_offset(frame, geo), 0)

    def test_boards_match_ignores_garbage_color_misread(self) -> None:
        # 相手の攻撃の演出中は、おじゃまが色ぷよに読まれることがある。期待盤面がおじゃまのマスは、埋まっていれば何色でもよい
        from src.app import boards_match

        expected = _empty_cells()
        expected[ROWS - 1][0] = "garbage"
        expected[ROWS - 1][1] = "red"
        observed = [row[:] for row in expected]
        observed[ROWS - 1][0] = "green"  # おじゃまを緑と読んだ
        self.assertTrue(boards_match(observed, expected))
        observed[ROWS - 1][1] = "blue"  # 色ぷよの色が違うのは別の盤面
        self.assertFalse(boards_match(observed, expected))
        observed = [row[:] for row in expected]
        observed[ROWS - 1][0] = EMPTY  # おじゃまのマスが空なのも別の盤面
        self.assertFalse(boards_match(observed, expected))

    def test_turn_ready_retry_compares_settled_board(self) -> None:
        # 取り込み時点でもう一度送る時は、確定した盤面と比べる(観測盤面は演出でまだ揺れていることがある。実機 2026-10-11: 17手)
        worker, client, _ = self._worker([_move("build", (12, 5), (11, 5))])
        me = worker.trackers["self"]
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        pre_move = _move("build", (12, 0), (11, 0))
        pre_move.path = ["LEFT", "LEFT", "DROP"]
        worker._precomputed = (me.settled, (me.current_pair, me.next1), pre_move)
        worker.trackers["enemy"].chain_active = True  # 相手の攻撃中(即表示の経路は使わない場面)
        worker._precomputed_attack = worker._attack_snapshot(me)
        noisy = _empty_cells()
        noisy[ROWS - 1][3] = "red"  # 観測盤面はまだ揺れている
        me.observed_cells = noisy
        me.settling = False
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertEqual(sent[0], ["LEFT", "LEFT", "DROP"])

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_replan_after_landing_when_board_differs(self) -> None:
        # 置いた組ぷよが着地した後、観測盤面が先読みと違えば(置き場所のずれ・おじゃまの落下)、その盤面で次の手をすぐ計算し直す
        # (実機 2026-10-11: 盤面が本当に違った手番が41手あり、出現の後に考え直す間に組ぷよが自由落下した)
        class FakeWorker:
            def __init__(self) -> None:
                self.submits: list = []

            def submit(self, kind, ctx, fn, urgent=False):
                self.submits.append((kind, ctx, urgent))

            def cancel(self, kind):
                return False

            def poll(self):
                return []

        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent_move = _move("build", (12, 2), (11, 2))
        sent_move.path = ["DROP"]
        sent_move.field_after = ["......"] * 13
        fw = FakeWorker()
        worker._think_worker = fw
        worker.autoplay.enabled = True
        worker.autoplay._played_turn = me.turn
        worker._sent_move = (me.turn, sent_move)
        worker._landed_turn = me.turn
        worker._precomputed = (_empty_cells(), (me.next1, me.next2), _move("build"))  # 先読みは空の盤面を想定
        landed = _empty_cells()
        landed[ROWS - 1][4] = "green"  # 実際は別の場所に置かれた
        landed[ROWS - 1][5] = "purple"
        me.observed_cells = landed
        me.last_observed_text = "置いた後"
        worker._replan_after_landing()  # 1回目は盤面が落ち着いたかを見るだけ
        self.assertEqual(fw.submits, [])
        worker._replan_after_landing()
        self.assertEqual([(k, u) for k, _, u in fw.submits], [("pre", True)])
        self.assertEqual(fw.submits[0][1]["expected"], landed)
        self.assertTrue(any("計算し直します" in text for text in logs))
        # 原因の手がかり: 送った手の2マス(3列目)に組ぷよが無い = 置き場所のずれ、違うマスは2つ(2026-10-21)
        line = next(text for text in logs if "計算し直します" in text)
        self.assertIn("置いた2マス: ずれ", line)
        self.assertIn("違うマス2(うちおじゃま0)", line)
        self.assertIn("1段目5列目 期待empty 読みgreen", line)
        # 同じ盤面では二度やらない
        worker._replan_after_landing()
        self.assertEqual(len(fw.submits), 1)

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_replan_after_landing_ignores_eyeless_effect(self) -> None:
        # 着地の瞬間の攻撃が当たる演出(白い目の無い赤)を、計算し直しの期待盤面に入れない(実機 2026-10-12 07:33 の turn=24:
        # 4列目12段目の演出の赤を入れた盤面で計算し直し、次の出現で「盤面が違う」となって組ぷよが自然落下した)
        class FakeWorker:
            def __init__(self) -> None:
                self.submits: list = []

            def submit(self, kind, ctx, fn, urgent=False):
                self.submits.append((kind, ctx, urgent))

            def cancel(self, kind):
                return False

            def poll(self):
                return []

        def setup(pre_cells):
            worker, _, _ = self._worker([])
            me = worker.trackers["self"]
            worker._log = lambda text: None
            sent_move = _move("build", (12, 2), (11, 2))
            sent_move.path = ["DROP"]
            sent_move.field_after = ["......"] * 13
            fw = FakeWorker()
            worker._think_worker = fw
            worker.autoplay.enabled = True
            worker.autoplay._played_turn = me.turn
            worker._sent_move = (me.turn, sent_move)
            worker._landed_turn = me.turn
            worker._precomputed = (pre_cells, (me.next1, me.next2), _move("build"))
            landed = _empty_cells()
            landed[ROWS - 1][2] = "green"
            landed[ROWS - 2][2] = "yellow"
            effect = [row[:] for row in landed]
            effect[ROWS - 1][3] = "red"  # 演出の赤(目が無い)
            me.observed_cells = effect
            me.observed_cells_eyed = landed
            me.last_observed_text = "置いた後"
            worker._replan_after_landing()
            worker._replan_after_landing()
            return fw, landed

        # 先読みどおりに置けていて、違いが演出だけなら計算し直さない
        expected = _empty_cells()
        expected[ROWS - 1][2] = "green"
        expected[ROWS - 2][2] = "yellow"
        fw, _ = setup(expected)
        self.assertEqual(fw.submits, [])
        # 本当に違う時も、演出を空にした盤面で計算し直す
        fw, landed = setup(_empty_cells())
        self.assertEqual(len(fw.submits), 1)
        self.assertEqual(fw.submits[0][1]["expected"], landed)

    def test_all_clear_from_banner(self) -> None:
        # 「全消し」の文字がはっきり見えたら全消しを持っているとする(連鎖の終わりの盤面を読み違えても、双方が全消しの構えが働く)
        worker, _, _ = self._worker([])
        worker._log = lambda text: None
        en = worker.trackers["enemy"]
        self.assertFalse(en.all_clear)
        worker._all_clear_from_banner(1.0, "enemy")
        self.assertTrue(en.all_clear)

    def test_main_fire_precompute_is_sent_at_spawn(self) -> None:
        # 本線の発火(main)の先読みは出現時点で送る(小さい消去 harass だけ思考を待つ。2026-10-11 11:07: 本線の発火まで
        # 待って0.2〜0.3秒遅れた)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        move = _move("main", (12, 1), (11, 1))
        move.path = ["LEFT", "DROP"]
        expected = _empty_cells()
        worker._precomputed = (expected, (me.current_pair, me.next1), move)
        me.observed_cells = [row[:] for row in expected]
        me.observed_count = 0
        worker._on_self_spawn(me)
        self.assertEqual(sent, [["LEFT", "DROP"]])

    def test_spawn_logs_why_keys_were_not_sent(self) -> None:
        # 出現時点で送れなかった時は理由を残す(次の実機ログで残った原因を特定するため)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        worker.autoplay.enabled = True
        worker._precomputed = None
        worker._on_self_spawn(me)
        self.assertTrue(any("出現時点では送りません" in text and "先読みが無い" in text for text in logs))
        logs.clear()
        worker._pre_inflight = {"key": None}
        worker._on_self_spawn(me)
        self.assertTrue(any("先読みを計算中" in text for text in logs))

    def test_drop_is_not_released_by_chain_detection(self) -> None:
        # 落下中に「自分の連鎖中」と見えても下キーを離さない(相手の攻撃の演出を消去と誤検出し、0msで離して自由落下した。
        # 実機 2026-10-11 11:19 turn=25)。13段目にかかる置き場所は見える方の1マスで着地を判定する(turn=24)
        worker, _, _ = self._worker([])
        me = worker.trackers["self"]
        worker._log = lambda text: None
        captured = {}
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: captured.setdefault("placed", placed) is None or True
        me.current_pair = ("green", "purple")  # (子, 軸)
        move = _move("build", (1, 0), (0, 0))  # 軸が12段目、子が13段目(見えない)
        me.observed_count = 10
        me.observed_cells = _empty_cells()
        worker._autoplay_send(me, ["LEFT", "DROP"], move)
        placed = captured["placed"]
        me.chain_active = True
        self.assertFalse(placed())
        cells = _empty_cells()
        cells[1][0] = "purple"  # 見える方(軸)が着地
        me.observed_cells = cells
        self.assertTrue(placed())

    def test_instant_precomputed_not_sent_under_attack(self) -> None:
        # 相手の攻撃中は「先読み一致」の即表示(=全自動では即送信)を使わず、思考した ama の手を送る
        # (実機 2026-09-27 12:40: 相手の5連鎖中に先読みの手を送って単発になり、ama の受けの手は捨てられた)
        pre_move = _move("build", (12, 1), (11, 1))
        pre_move.path = ["LEFT", "DROP"]
        fresh = _move("accept", (12, 5), (11, 5))
        fresh.path = ["RIGHT", "DROP"]
        worker, client, emitted = self._worker([fresh])
        me = worker.trackers["self"]
        sent: list = []
        worker.autoplay.enabled = True
        worker.autoplay.play = lambda turn, path, placed=None, early_drop_sec=0.0, piece_cols=None, cancel_fall_rows=None, **_kw: sent.append(path) or True
        me.settling = False  # 手番開始(turn_ready)の状態
        worker.trackers["enemy"].chain_active = True
        worker._precomputed = (me.settled, (me.current_pair, me.next1), pre_move)
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertEqual(sent, [["RIGHT", "DROP"]])
        self.assertEqual(emitted[-1].axis_cell, (12, 5))

    def test_instant_build_kept_when_thought_is_build_like(self) -> None:
        # 支援モード: 相手の連鎖中、先読みの構築の手を表示した後の思考が「伸ばす」(中身は構築)なら置き場所を変えず、
        # 判断の内容だけ更新する。返しの準備のように置き場所に意味がある時は差し替える
        # (2026-10-12、利用者「相手の発火に対して何をすべきかわからなくなる。提示が振動する」)
        from src.engine.ama_client import action_text

        for reason, steps, expected_cell, expected_action in (
            ("EXTEND", 0, (12, 1), "伸ばす"),
            ("COUNTER_PREP", 3, (12, 5), "返しの準備 @3"),
        ):
            pre_move = _move("build", (12, 1), (11, 1))
            fresh = _move("accept", (12, 5), (11, 5))
            fresh.reason, fresh.plan_steps, fresh.incoming = reason, steps, 40
            worker, client, emitted = self._worker([fresh])
            me = worker.trackers["self"]
            me.settling = False
            worker.trackers["enemy"].chain_active = True
            worker._precomputed = (me.settled, (me.current_pair, me.next1), pre_move)
            self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
            self.assertEqual(emitted[-1].axis_cell, expected_cell, reason)
            self.assertEqual(emitted[-1].label, "accept", reason)
            self.assertIn(expected_action, emitted[-1].action, reason)
        # 受けて組み続ける時は受ける量を出す
        accept = _move("accept")
        accept.reason = "BUILD_CHAIN"
        self.assertEqual(action_text(accept, 5), "受ける(5個は受けて組み続ける)")

    def test_think_returns_false_when_input_missing(self) -> None:
        # 入力不足(ネクスト未認識)なら False を返し、_loop が再計算の理由を次の tick へ持ち越せる
        worker, client, _ = self._worker([])
        worker.trackers["self"].next1 = None
        self.assertFalse(worker._think(1.0, _frame()))
        self.assertEqual(client.calls, [])

    def test_rethink_switches_display_when_response_changes(self) -> None:
        # 潰しを表示中に相手の大きい攻撃(着弾見込みが増えた)で再計算し、本線発火に変わったら表示を切り替える
        worker, client, emitted = self._worker([_move("main", (12, 4), (12, 5), fire_reason="counter")])
        worker._last_data = OverlayDrawData(axis_cell=(12, 2), child_cell=(12, 3), axis_color="green", child_color="purple", label="harass")
        client.moves[0].incoming = 40  # 前回の思考時(0個)から見込みが増えた
        worker._think(1.0, _frame(), note="相手が本線を発火: 提案を更新")
        self.assertEqual(emitted[-1].label, "main")
        self.assertEqual(emitted[-1].axis_cell, (12, 4))
        self.assertIn("相手が本線を発火", emitted[-1].note)

    def test_flash_frame_board_is_passed_as_chain_field(self) -> None:
        # 点滅から復元した発火直前盤面があれば chain_field として渡す。無ければ渡さない(見積もりで代用しない)
        from src.engine.think_input import build_think_input

        worker, client, _ = self._worker([_move("build")])
        en = worker.trackers["enemy"]
        en.chain_active, en.chain_step_drops, en.chain_observed = True, [4], _empty_cells()
        before = _empty_cells()
        before[12][0] = before[12][1] = before[12][2] = before[11][0] = "green"
        en.chain_before, en.chain_before_source = before, "flash_frame"
        inp = build_think_input(worker.mapper, worker.trackers["self"], en, worker.trackers["self"].settled,
                                [worker.trackers["self"].current_pair, worker.trackers["self"].next1, worker.trackers["self"].next2])
        self.assertIsNotNone(inp.en_state.chain_field)
        self.assertEqual(inp.en_state.chain_field.to_rows()[-1][:3], "GGG")
        en.chain_before_source = "settled_without_trigger"
        inp = build_think_input(worker.mapper, worker.trackers["self"], en, worker.trackers["self"].settled,
                                [worker.trackers["self"].current_pair, worker.trackers["self"].next1, worker.trackers["self"].next2])
        self.assertIsNone(inp.en_state.chain_field)

    def test_haipuyo_history_is_passed(self) -> None:
        # 配ぷよ読み(2026-10-23): 試合の最初から置いた組ぷよ(組ぷよと同じ文字)と置いた数、相手の置いた数を要求に入れる。
        # 切っている時・履歴が欠けている時は入れない
        from src.engine.think_input import build_think_input

        worker, client, _ = self._worker([_move("build")])
        me, en = worker.trackers["self"], worker.trackers["enemy"]
        me.turn, en.turn = 2, 3
        c1, c2 = me.current_pair
        me.pair_by_turn = {0: (c1, c2), 1: (c2, c1), 2: me.current_pair}
        src = [me.current_pair, me.next1, me.next2]
        inp = build_think_input(worker.mapper, me, en, me.settled, src, haipuyo=True)
        req = inp.my_state.to_json()
        self.assertEqual(req["placed"], 2)
        self.assertEqual(len(req["history"]), 2)
        # 同じ色は組ぷよと同じ文字になる
        self.assertEqual(req["history"][0], req["queue"][0])
        self.assertEqual(req["history"][1], req["queue"][0][::-1])
        self.assertEqual(inp.en_state.to_json()["placed"], 3)
        off = build_think_input(worker.mapper, me, en, me.settled, src)
        self.assertNotIn("history", off.my_state.to_json())
        self.assertNotIn("placed", off.en_state.to_json())
        # 先読みの要求(次の手番の手を次・次々の組で先に計算する): 今の組ぷよも置いた扱いにする(2026-10-24 実機の不具合)
        pre = build_think_input(inp.mapper, me, en, me.settled, [me.next1, me.next2], haipuyo=True, ahead=True).my_state.to_json()
        self.assertEqual(pre["placed"], 3)
        self.assertEqual(pre["history"], req["history"] + [req["queue"][0]])
        self.assertEqual(pre["queue"][0], req["queue"][1])
        del me.pair_by_turn[1]
        gap = build_think_input(worker.mapper, me, en, me.settled, src, haipuyo=True)
        self.assertNotIn("history", gap.my_state.to_json())

    def test_peek_sends_placed(self) -> None:
        # 相手のツモを覗く(2026-11-01): 配ぷよ読みなしでも自分と相手の置いた数を入れる(履歴は入れない)。先読みの要求は1つ先。
        # 相手の操作中の組ぷよが読めない時は、ネクストの位置がずれるので相手の置いた数を入れない
        from src.engine.think_input import build_think_input

        worker, client, _ = self._worker([_move("build")])
        me, en = worker.trackers["self"], worker.trackers["enemy"]
        me.turn, en.turn = 4, 5
        src = [me.current_pair, me.next1, me.next2]
        inp = build_think_input(worker.mapper, me, en, me.settled, src, peek=True)
        req = inp.my_state.to_json()
        self.assertEqual(req["placed"], 4)
        self.assertNotIn("history", req)
        self.assertEqual(inp.en_state.to_json()["placed"], 5)
        self.assertEqual(len(inp.en_state.to_json()["queue"]), 3)
        pre = build_think_input(inp.mapper, me, en, me.settled, [me.next1, me.next2], ahead=True, peek=True)
        self.assertEqual(pre.my_state.to_json()["placed"], 5)
        off = build_think_input(worker.mapper, me, en, me.settled, src)
        self.assertNotIn("placed", off.my_state.to_json())
        self.assertNotIn("placed", off.en_state.to_json())
        en.current_pair = None
        no_cur = build_think_input(worker.mapper, me, en, me.settled, src, peek=True)
        self.assertEqual(no_cur.my_state.to_json()["placed"], 4)
        self.assertNotIn("placed", no_cur.en_state.to_json())

    def test_tracker_records_pair_history(self) -> None:
        # 追跡は手番ごとの操作中の組ぷよを覚える(試合ごとに追跡を作り直すので試合の最初からの履歴)
        tr = PlayerTracker()
        tr.turn, tr.pair_by_turn = 2, {0: ("red", "red"), 1: ("blue", "green")}
        self.assertEqual(tr.pair_history(), [("red", "red"), ("blue", "green")])
        tr.pair_by_turn = {1: ("blue", "green")}
        self.assertIsNone(tr.pair_history())

    def test_enemy_chain_bonus_is_passed_to_engine(self) -> None:
        # 相手が全消しボーナス付きの連鎖中: chain_start で all_clear は落ちるが chain_bonus で渡す
        worker, client, _ = self._worker([_move("build")])
        en = worker.trackers["enemy"]
        en.chain_active, en.chain_dropped, en.chain_bonus, en.all_clear = True, 12, True, False
        worker._think(1.0, _frame())
        en_state = client.calls[0]["en"]
        self.assertTrue(en_state.chain_bonus)
        self.assertTrue(en_state.to_json()["chain_bonus"])


def _frame():
    import numpy as np

    return np.zeros((8, 8, 3), dtype=np.uint8)


class AsyncThinkTest(unittest.TestCase):
    """思考を思考スレッドで行い、観測ループを止めない(2026-09-27: 思考のたびに観測が約0.4秒止まっていた)。"""

    def _worker(self, delay: float):
        import time as _time

        from src.engine.background import ThinkWorker

        class SlowClient(FakeClient):
            def battle(self, my_state, en_state, **kwargs):
                _time.sleep(delay)
                return super().battle(my_state, en_state, **kwargs)

        client = SlowClient([_move("build", (12, 2), (11, 2))])
        worker = AssistWorker(calibration=None, client=client, debug_log_path=None)  # type: ignore[arg-type]
        worker.enable_precompute = False
        worker.trackers["self"] = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        worker.trackers["enemy"] = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        worker._think_worker = ThinkWorker()
        worker._client_lock = worker._think_worker.lock
        emitted: list = []
        worker.draw_data_ready.connect(lambda d: emitted.append(d))
        self.addCleanup(worker._think_worker.stop)
        return worker, emitted

    def _wait_results(self, worker, timeout: float = 2.0) -> None:
        import time as _time

        end = _time.perf_counter() + timeout
        while _time.perf_counter() < end:
            worker._poll_think_results()
            if worker._think_worker._results.empty() and not worker._think_worker._order and not worker._think_worker.lock.locked():
                return
            _time.sleep(0.01)

    def test_think_returns_without_waiting_and_result_is_applied(self) -> None:
        import time as _time

        worker, emitted = self._worker(delay=0.3)
        start = _time.perf_counter()
        self.assertTrue(worker._think(1.0, _frame(), new_turn=True))
        self.assertLess(_time.perf_counter() - start, 0.1)  # 思考(0.3秒)を待たずに観測へ戻る
        self.assertEqual(emitted, [])
        _time.sleep(0.4)
        self._wait_results(worker)
        self.assertEqual(emitted[-1].axis_cell, (12, 2))

    def test_stale_result_is_discarded(self) -> None:
        # 思考中に手番が進んだら、古い組ぷよの結果は表示も操作もしない
        import time as _time

        worker, emitted = self._worker(delay=0.2)
        worker._think(1.0, _frame(), new_turn=True)
        worker.trackers["self"].turn += 1
        _time.sleep(0.3)
        self._wait_results(worker)
        self.assertEqual(emitted, [])


class AttackFrameEstimateTest(unittest.TestCase):
    def test_remaining_moves_shrink_with_elapsed_time(self) -> None:
        # 相手の連鎖が終わるまでに置ける手数は、経過時間とともに減る(固定の「3手」にしない)
        from src.engine.chain_sim import ChainResult
        from src.engine.think_input import CHAIN_STEP_SEC, MY_MOVE_SEC, estimate_attack_frames
        from src.engine.tracker import PlayerTracker

        en = PlayerTracker(chain_active=True, chain_started_t=10.0)
        en.chain_expected = ChainResult(steps=6, score=9100, cleared=28, final=_empty_cells())
        total = 6 * CHAIN_STEP_SEC
        self.assertEqual(estimate_attack_frames(en, 10.0), int(total // MY_MOVE_SEC))
        self.assertEqual(estimate_attack_frames(en, 10.0 + total - 1.5), 1)
        self.assertEqual(estimate_attack_frames(en, 10.0 + total + 1.0), 0)
        # 再現できていない連鎖・時刻が無い時は見積もらない(ブリッジの固定値)
        self.assertIsNone(estimate_attack_frames(en, None))
        en.chain_expected = None
        self.assertIsNone(estimate_attack_frames(en, 12.0))


class MainWindowTest(unittest.TestCase):
    def test_window_constructs(self) -> None:
        # 設定画面が組み立てられること(選択肢の定義変更で起動できなくなった 2026-09-16 の再発防止)
        from src.app import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = MainWindow()
        # 本線発火の目安は 2026-10-23 に無くした(利用者)。設置キャンセルの切り替えがある
        self.assertFalse(hasattr(window, "fire_combo"))
        self.assertTrue(hasattr(window.side_settings["1p"], "place_cancel"))
        self.assertTrue(hasattr(window.side_settings["2p"], "haipuyo"))
        window.close()

    def test_place_cancel_switch_overrides_keymap(self) -> None:
        # 画面の設置キャンセルの切り替えは keymap の place_cancel を上書きする(なし → off、ありで off → move)
        off = AssistWorker(calibration=None, client=None, debug_log_path=None, place_cancel=False)  # type: ignore[arg-type]
        self.assertEqual(off.autoplay.keymap.place_cancel, "off")
        on = AssistWorker(calibration=None, client=None, debug_log_path=None, place_cancel=True)  # type: ignore[arg-type]
        self.assertNotEqual(on.autoplay.keymap.place_cancel, "off")


if __name__ == "__main__":
    unittest.main()


class PreholdActionTest(unittest.TestCase):
    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_only_when_spawn_timing_is_known(self) -> None:
        # 先行入力は、次の組ぷよの先読みの最初が左右移動で、置いた手で消えず・おじゃまも降らない時だけ(2026-10-25 試験)
        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        worker = AssistWorker(calibration=None, client=None, debug_log_path=None)  # type: ignore[arg-type]
        me = _tracker(("red", "green"), ("blue", "yellow"), ("green", "green"))
        me.turn = 5
        worker.trackers["self"] = me
        worker.trackers["enemy"] = _tracker(("red", "green"), ("blue", "yellow"), ("green", "green"))
        move = BattleMove(0, "UP", 0, (12, 0), (11, 0), "build", None, None, 0)
        move.path = ["LEFT", "NONE", "LEFT", "DROP"]
        worker._precomputed = (_empty_cells(), (me.next1, me.next2), move)
        self.assertEqual(worker._prehold_action(5), "LEFT")  # 回転なしで1列目(壁)まで
        self.assertIsNone(worker._prehold_action(4))  # 別の手番
        move.x = 1
        self.assertIsNone(worker._prehold_action(5))  # 壁でない(押しっぱなしで行き過ぎる)
        move.x = 0
        me.pending_garbage = 3  # おじゃまが降る: 出現の時刻が読めない
        self.assertIsNone(worker._prehold_action(5))
        me.pending_garbage = 0
        chain = _empty_cells()
        for r in range(ROWS - 4, ROWS):
            chain[r][0] = "red"  # 置いた手で消える
        worker._precomputed = (chain, (me.next1, me.next2), move)
        self.assertIsNone(worker._prehold_action(5))
        move.path = ["CW", "LEFT", "DROP"]
        worker._precomputed = (_empty_cells(), (me.next1, me.next2), move)
        self.assertIsNone(worker._prehold_action(5))  # 最初が回転
        move.path = ["LEFT", "DROP"]
        worker._precomputed = (_empty_cells(), (me.current_pair, me.next1), move)
        self.assertIsNone(worker._prehold_action(5))  # 次に出る組ぷよ用の先読みでない

class FirstInstanceTest(unittest.TestCase):
    def test_second_instance_is_not_first(self) -> None:
        # 2回目に起動した PBA は 2P から始める(1P と 2P を start.bat だけで同時に動かすため)
        from src import app

        name = f"Local\\PBA_test_instance_check_{os.getpid()}"  # 同時に走る別のテストと重ならないよう
        saved = app._instance_mutex
        try:
            self.assertTrue(app._claim_first_instance(name))
            self.assertFalse(app._claim_first_instance(name))
        finally:
            app._instance_mutex = saved


class SideChoiceTest(unittest.TestCase):
    def test_both_runs_1p_and_2p_with_swapped_calibration(self) -> None:
        # 1P・2P の両方に思考エンジンを選ぶと1つの PBA で動かす。2P は対戦用キャリブレーションの自分と相手を入れ替える
        from PyQt6 import QtWidgets

        from src import app
        from src.capture.calibrate import load_calibration

        _ = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        window = app.MainWindow()
        window.side_settings["2p"].engine_combo.setCurrentIndex(list(app.ENGINE_CHOICES).index("ama"))
        window.calibration = load_calibration(Path(__file__).parent / "fixtures" / "calibration_1080p.json")
        self.assertEqual(window._sides(), ["1p", "2p"])
        self.assertFalse(window._calibration_for("1p").self_on_right())
        self.assertTrue(window._calibration_for("2p").self_on_right())
        window.close()


class FifthColorDiagnosisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_fifth_color_mid_game_logs_where_and_saves_frame(self) -> None:
        # 試合の途中で5色目が出て思考できなかった時、どこに未割当の色があるかを記録し画面を保存する(手番ごとに1回)
        # (実機 2026-10-20 22:28 の 336s は画像が無く原因を確かめられなかった)
        worker = AssistWorker(calibration=None, client=None, debug_log_path=None)  # type: ignore[arg-type]
        logs: list[str] = []
        saved: list = []
        worker._log = logs.append  # type: ignore[method-assign]
        worker._saver.save = lambda path, frame, *a: saved.append(path)  # type: ignore[method-assign]
        me = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        for r in range(ROWS - 6, ROWS):
            me.settled[r] = ["green", "purple", "yellow", "blue", "green", "purple"]
        me.settled[ROWS - 7][0] = "red"  # 盤面に5色目
        worker.trackers["self"] = me
        worker.trackers["enemy"] = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        for c in ("green", "purple", "yellow", "blue"):
            worker.mapper.slot(c)
        self.assertFalse(worker._think(1.0, _frame()))
        self.assertTrue(any("5色目の診断" in m and "未割当 ['red']" in m for m in logs), logs)
        self.assertEqual(len(saved), 1)
        self.assertIn("fifth_color_self_", str(saved[0]))
        worker._think(1.1, _frame())  # 同じ手番では二度は残さない
        self.assertEqual(len(saved), 1)


class TickStatsTest(unittest.TestCase):
    def test_reports_every_period(self) -> None:
        # 観測の間隔・画面取得の待ち・処理時間を一定時間ごとにまとめる(2026-10-21)
        from src.app import TICK_STATS_SECONDS, TickStats

        st = TickStats()
        reports = []
        t = 0.0
        while t < TICK_STATS_SECONDS * 2 + 1:
            r = st.add(t, t, 0.002, 0.015)
            if r:
                reports.append(r)
            t += 0.03
        self.assertEqual(len(reports), 2)
        self.assertIn("間隔 中央 30ms", reports[0])
        self.assertIn("画面取得の待ち 中央 2ms", reports[0])
        self.assertIn("1回の処理(取得込み) 中央 15ms", reports[0])


class ReplanWhilePrecomputeInFlightTest(unittest.TestCase):
    """着地の時点で先読みがまだ計算中なら、その先読みが想定している盤面と比べる(2026-10-21)。
    実機 2026-10-20 22:28: 着地した盤面が先読みと違うとした手番 183/323 の多くは、先読みが届く前に着地しただけだった。"""

    def setUp(self) -> None:
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def _setup(self, inflight_expected):
        class FakeWorker:
            def __init__(self) -> None:
                self.submits: list = []

            def submit(self, kind, ctx, fn, urgent=False):
                self.submits.append((kind, ctx, urgent))

            def cancel(self, kind):
                return False

            def poll(self):
                return []

        worker = AssistWorker(calibration=None, client=FakeClient([]), debug_log_path=None)  # type: ignore[arg-type]
        worker.trackers["self"] = _tracker(("green", "purple"), ("yellow", "blue"), ("green", "yellow"))
        worker.trackers["enemy"] = _tracker(("blue", "purple"), ("green", "yellow"), ("purple", "green"))
        me = worker.trackers["self"]
        logs: list[str] = []
        worker._log = logs.append
        sent_move = _move("build", (12, 2), (11, 2))
        sent_move.path = ["DROP"]
        sent_move.field_after = ["......"] * 13
        fw = FakeWorker()
        worker._think_worker = fw
        worker.autoplay.enabled = True
        worker.autoplay._played_turn = me.turn
        worker._sent_move = (me.turn, sent_move)
        worker._landed_turn = me.turn
        worker._precomputed = None
        worker._pre_inflight = {"key": worker._think_key(me), "move": sent_move, "expected": inflight_expected,
                                "pairs": (me.next1, me.next2), "attack": None}
        landed = _empty_cells()
        landed[ROWS - 1][2] = "purple"
        landed[ROWS - 2][2] = "green"
        me.observed_cells = landed
        me.observed_cells_eyed = landed
        me.last_observed_text = "置いた後"
        worker._replan_after_landing()  # 1回目は盤面が落ち着いたかを見るだけ
        worker._replan_after_landing()
        return fw, logs, landed

    def test_landed_as_inflight_expected_waits(self) -> None:
        expected = _empty_cells()
        expected[ROWS - 1][2] = "purple"
        expected[ROWS - 2][2] = "green"
        fw, logs, _ = self._setup(expected)
        self.assertEqual(fw.submits, [])  # 計算中の先読みどおりなので依頼し直さない
        self.assertFalse(any("計算し直します" in text for text in logs))

    @unittest.skipUnless(AUTOPLAY_AVAILABLE, "全自動のプラグイン(PT-plugins)がありません")
    def test_landed_elsewhere_replans_against_inflight(self) -> None:
        expected = _empty_cells()
        expected[ROWS - 1][0] = "purple"
        expected[ROWS - 2][0] = "green"
        fw, logs, landed = self._setup(expected)
        self.assertEqual([(k, u) for k, _, u in fw.submits], [("pre", True)])
        self.assertEqual(fw.submits[0][1]["expected"], landed)
        line = next(text for text in logs if "計算し直します" in text)
        self.assertIn("違うマス4", line)  # 計算中の先読みの盤面と比べた違い
