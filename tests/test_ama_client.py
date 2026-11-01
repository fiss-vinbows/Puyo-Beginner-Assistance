"""ama ブリッジクライアントのテスト

座標変換・リクエスト組み立ての単体テストと、ブリッジ実行ファイルが
存在する環境でのみ実行する統合テストからなる。
"""

from __future__ import annotations

import json
import unittest

from src.engine.ama_client import AMA_BRIDGE_EXE, AmaClient, PlayerState, build_request, parse_move
from src.engine.puyo_field import COLS, ROWS, PuyoField, xy_to_row_col

EMPTY_ROWS = ["......"] * 12


class XyToRowColTest(unittest.TestCase):
    def test_bottom_left_is_last_row(self) -> None:
        self.assertEqual(xy_to_row_col(0, 0), (ROWS - 1, 0))

    def test_top_hidden_row(self) -> None:
        self.assertEqual(xy_to_row_col(5, 12), (0, 5))


class PuyoFieldTest(unittest.TestCase):
    def test_from_12_rows_adds_hidden_row(self) -> None:
        f = PuyoField.from_rows(EMPTY_ROWS[:-1] + ["RRGG.."])
        self.assertEqual(len(f.to_rows()), 13)
        self.assertEqual(f.to_rows()[-1], "RRGG..")
        self.assertEqual(f.height(0), 1)
        self.assertEqual(f.height(4), 0)

    def test_rejects_invalid_char(self) -> None:
        with self.assertRaises(ValueError):
            PuyoField.from_rows(EMPTY_ROWS[:-1] + ["RRGGXX"])

    def test_game_over(self) -> None:
        rows = list(EMPTY_ROWS)
        rows[0] = "..R..."  # 12段目の3列目
        self.assertTrue(PuyoField.from_rows(rows).is_game_over())
        self.assertFalse(PuyoField.from_rows(EMPTY_ROWS).is_game_over())


class BuildRequestTest(unittest.TestCase):
    def test_request_shape(self) -> None:
        req = build_request(PuyoField.from_rows(EMPTY_ROWS), ["RG", "BY"], width=100)
        self.assertEqual(len(req["field"]), ROWS)
        self.assertTrue(all(len(r) == COLS for r in req["field"]))
        self.assertEqual(req["queue"], ["RG", "BY"])
        self.assertEqual(req["width"], 100)
        self.assertNotIn("depth", req)

    def test_rejects_bad_pair(self) -> None:
        with self.assertRaises(ValueError):
            build_request(PuyoField.from_rows(EMPTY_ROWS), ["R#"])


class ParseMoveTest(unittest.TestCase):
    def test_parse(self) -> None:
        move = parse_move(
            {"x": 3, "r": "LEFT", "score": 100, "axis": [3, 1], "child": [2, 1], "ms": 5, "candidates": [{"x": 3, "r": "LEFT", "score": 100}]}
        )
        self.assertEqual(move.x, 3)
        self.assertEqual(move.rotation, "LEFT")
        self.assertEqual(move.axis_cell, (ROWS - 2, 3))
        self.assertEqual(move.child_cell, (ROWS - 2, 2))
        self.assertEqual(move.candidates, [(3, "LEFT", 100)])

    def test_error(self) -> None:
        with self.assertRaises(RuntimeError):
            parse_move({"error": "boom"})



@unittest.skipUnless(AMA_BRIDGE_EXE.exists(), "ama ブリッジが未ビルド")
class AmaClientIntegrationTest(unittest.TestCase):
    def test_suggest_on_partial_field(self) -> None:
        # 最下段 RRGG.. と 2段目 R..... に対して GB を置く。
        rows = list(EMPTY_ROWS)
        rows[-2] = "R....."
        rows[-1] = "RRGG.."
        field = PuyoField.from_rows(rows)
        with AmaClient(AMA_BRIDGE_EXE) as client:
            move = client.suggest(field, ["GB", "RR"])
        self.assertIn(move.rotation, ("UP", "RIGHT", "DOWN", "LEFT"))
        self.assertEqual(len(move.candidates), 22)
        # 着地マスは空きマスで、その下は埋まっているか最下段
        for row, col in (move.axis_cell, move.child_cell):
            self.assertEqual(field.grid[row][col], ".")
        self.assertLess(move.think_ms, 5000)

    def test_bridge_ignores_pba_options(self) -> None:
        # 2026-10-21: ama は手を加えず素の状態で使う(PBA に搭載する思考エンジンは YamaNEKO だけ)。初心者モード・開幕の形の
        # 選択・本線発火の目安は ama のブリッジでは読まない(要求にあっても無視し、エラーにもしない)
        rows = list(EMPTY_ROWS)
        for r in range(1, 12):
            rows[r] = ".R...."
        field = PuyoField.from_rows(rows)
        with AmaClient(AMA_BRIDGE_EXE) as client:
            client._send({"field": field.to_rows(), "queue": ["GB", "YY"], "beginner": True, "forms": ["NOPE"], "form_weight": 150})
            move = parse_move(client._recv())  # 読まない項目があってもエラーにしない
        self.assertIn((1, "DOWN"), [(x, r) for x, r, _ in move.candidates])  # 13段目を使う置き方も候補に残る
        self.assertIn(move.form, (None, "GTR", "FRON", "SGTR"))  # 上流の3形だけ

    def test_bridge_returns_input_path(self) -> None:
        # 全自動モード用: 置き場所までの操作手順(1要素=1フレームの入力)を返す
        rows = list(EMPTY_ROWS)
        rows[-1] = "RRRR.."
        rows[-2] = "G....."
        with AmaClient(AMA_BRIDGE_EXE) as client:
            move = client.suggest(PuyoField.from_rows(rows), ["RG", "BY", "GG"])
        self.assertTrue(move.path)
        self.assertEqual(move.path[-1], "DROP")  # 最後は設置
        known = {"LEFT", "RIGHT", "CW", "CCW", "CW_LEFT", "CW_RIGHT", "CCW_LEFT", "CCW_RIGHT", "DROP", "NONE", "WAIT", "TOUCH"}
        self.assertTrue(set(move.path) <= known, move.path)
        # 移動が要る手なら左右の入力が手順に入っている
        if move.x != 2 or move.rotation != "UP":
            self.assertTrue({"LEFT", "RIGHT", "CW", "CCW", "CW_LEFT", "CW_RIGHT", "CCW_LEFT", "CCW_RIGHT"} & set(move.path))

    def test_solo_mode(self) -> None:
        rows = list(EMPTY_ROWS)
        rows[-1] = "RRGG.."
        with AmaClient(AMA_BRIDGE_EXE) as client:
            move = client.solo(PuyoField.from_rows(rows), ["RG", "BY"])
        self.assertEqual(move.label, "build")
        self.assertIsNone(move.enemy_analysis)
        self.assertIsNotNone(move.self_analysis)
        self.assertIsNotNone(move.field_after)
        self.assertEqual(len(move.field_after), 13)

    def test_multiple_requests_reuse_process(self) -> None:
        field = PuyoField.from_rows(EMPTY_ROWS)
        with AmaClient(AMA_BRIDGE_EXE) as client:
            first = client.suggest(field, ["RG", "BY"])
            second = client.suggest(field, ["RG", "BY"])
        self.assertEqual(first.x, second.x)

    def test_ama_restarts_after_many_requests(self) -> None:
        # 2P を素の ama で動かす(2026-10-25): ama のブリッジは問い合わせのたびにメモリが増えるので、一定回数ごとに起動し直す。
        # 起動し直しても同じ盤面なら同じ手を返し、前のプロセスの終わり(None)を読んで失敗しない
        from unittest import mock

        from src.engine import ama_client

        field = PuyoField.from_rows(EMPTY_ROWS)
        with mock.patch.object(ama_client, "AMA_RESTART_REQUESTS", 2), AmaClient(bridge="ama") as client:
            self.assertEqual(client.engine, "ama")
            moves = [client.suggest(field, ["RG", "BY"]) for _ in range(5)]
            self.assertEqual(client._requests, 1)  # 5回目の前に起動し直した(2回ごと)
        self.assertEqual(len({(m.x, m.rotation) for m in moves}), 1)



class ParseBattleTest(unittest.TestCase):
    _ATTACK = {"count": 2, "score": 420, "garbage": 6, "x": 0, "r": "RIGHT", "frame": 3, "all_clear": True}
    _ANALYSIS = {"immediate": _ATTACK, "max": _ATTACK, "harass": _ATTACK, "potential": 2280, "garbage_max": 6}

    def test_parse_battle(self) -> None:
        from src.engine.ama_client import parse_battle

        move = parse_battle(
            {"x": 0, "r": "RIGHT", "eval": 420, "axis": [0, 3], "child": [1, 3], "label": "build",
             "self": self._ANALYSIS, "enemy": self._ANALYSIS, "ms": 10}
        )
        self.assertEqual(move.label, "build")
        self.assertEqual(move.axis_cell, (ROWS - 4, 0))
        self.assertEqual(move.self_analysis.max.count, 2)
        self.assertEqual(move.enemy_analysis.potential, 2280)
        self.assertTrue(move.self_analysis.harass.all_clear)
        self.assertEqual(move.reason, "")  # 理由の無い応答(ama)は空

    def test_parse_battle_reason(self) -> None:
        # 自作AI のブリッジが返す判断の理由(返しの上回り・とどめの確認用)をログ用に読む
        from src.engine.ama_client import parse_battle

        move = parse_battle(
            {"x": 0, "r": "RIGHT", "eval": 420, "axis": [0, 3], "child": [1, 3], "label": "main",
             "reason": "COUNTER_MARGIN", "self": self._ANALYSIS, "enemy": self._ANALYSIS, "ms": 10}
        )
        self.assertEqual(move.reason, "COUNTER_MARGIN")

    def test_parse_battle_plan(self) -> None:
        # 最終的な構想(2026-10-24、YamaNEKO が配ぷよ読みで先のツモを読めた時だけ): 盤面・連鎖・発火の組ぷよのマス
        from src.engine.ama_client import parse_battle

        base = {"x": 0, "r": "RIGHT", "eval": 0, "axis": [0, 3], "child": [1, 3], "label": "build",
                "self": self._ANALYSIS, "enemy": self._ANALYSIS, "ms": 10}
        self.assertIsNone(parse_battle(base).plan_field)
        plan = {"field": ["......"] * 13, "chain": 9, "score": 31140, "steps": 12, "trigger": [[2, 8], [3, 4]]}
        move = parse_battle({**base, "plan": plan})
        self.assertEqual(len(move.plan_field), 13)
        self.assertEqual(move.plan_chain, (9, 31140, 12))
        self.assertEqual(move.plan_trigger, [(ROWS - 9, 2), (ROWS - 5, 3)])
        self.assertEqual(move.choukou, 0)
        self.assertEqual(parse_battle({**base, "choukou": 1}).choukou, 1)  # 跳考の回数(2026-10-24)


@unittest.skipUnless(AMA_BRIDGE_EXE.exists(), "ama ブリッジが未ビルド")
class AmaClientBattleIntegrationTest(unittest.TestCase):
    def _field(self) -> PuyoField:
        rows = list(EMPTY_ROWS)
        rows[-3] = "RB...."
        rows[-2] = "RGB..."
        rows[-1] = "RGGB.."
        return PuyoField.from_rows(rows)

    def test_analyze_scores_follow_tsu_rule(self) -> None:
        with AmaClient(AMA_BRIDGE_EXE) as client:
            a = client.analyze(self._field(), ["RB", "GG"])
        # 赤を1列目に置けば赤4個が消える: 4個×10点×1 = 40点(通ルール)
        self.assertEqual(a.immediate.count, 1)
        self.assertEqual(a.immediate.score, 40)
        self.assertEqual(a.immediate.x, 0)
        self.assertGreaterEqual(a.max.score, a.immediate.score)
        self.assertEqual(a.garbage_max, a.max.score // 70)

    def test_attack_semantics_swapped_for_ama(self) -> None:
        # JSON の attack は「そのプレイヤーに降る予告」。ama 内部では「送った量」なのでブリッジで入れ替える。
        # 入れ替えないと ama は常に「自分が攻撃中」と解釈し、相手の攻撃を受けている分岐に入らなかった
        # (実機 2026-09-17: 相手の7連鎖に無反応で負け)。受けている時のラベルが accept になることで確認
        me_rows = EMPTY_ROWS[:6] + ["R.....", "RG....", "GRB...", "GGRB..", "YYRBB.", "RYYGGB"]
        en_rows = EMPTY_ROWS[:10] + ["RR....", "RRGG.."]
        with AmaClient(AMA_BRIDGE_EXE) as client:
            en = PlayerState(PuyoField.from_rows(en_rows), ["YY", "YY", "YY"])
            attacked = client.battle(PlayerState(PuyoField.from_rows(me_rows), ["YY", "RG", "BB"], attack=20), en)
            calm = client.battle(PlayerState(PuyoField.from_rows(me_rows), ["YY", "RG", "BB"]), en)
        self.assertIn(attacked.label, ("accept", "main", "harass"))
        self.assertNotEqual(attacked.label, "build")
        self.assertEqual(calm.label, "build")

    def test_enemy_chain_replayed_from_flash_frame(self) -> None:
        # 点滅から復元した発火直前の盤面(chain_field)があれば、それを消して連鎖を再現し着弾量を返す。
        # 復元できない(追跡盤面に消える組が無い)時は見積もりで代用せず、何も返さない(2026-09-20 の方針)
        me = ["......", "......", "R.....", "RB....", "GY....", "GY....", "RB....", "GB....", "YYB...", "GYB..G", "GGRYYG", "RBRBGY", "RRBBYY"]
        en = ["......", "......", "......", "......", "......", "......", "..BB..", "..YR.G", "..YY.G", "..RRBB", "GGGBYB", "YYRBGG", "YBRRYG"]
        before = EMPTY_ROWS[:7] + ["B.....", "B.....", "G.....", "G.....", "GB....", "GB...."]  # 引き金込み: 緑4→青4 の2連鎖
        with AmaClient(AMA_BRIDGE_EXE) as client:
            me_s = PlayerState(PuyoField.from_rows(me), ["YB", "GB", "BG"], attack=2)
            flash = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1, chain_field=PuyoField.from_rows(before)))
            none = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1))
            bonus = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1, chain_bonus=True, chain_field=PuyoField.from_rows(before)))
        self.assertEqual(flash.incoming_source, "flash_replay")
        self.assertEqual(flash.incoming_chain, 2)
        self.assertEqual(flash.incoming, (40 + 4 * 10 * 8) // 70)  # 緑4個(1倍)+青4個(8倍)=360点 → 5個
        self.assertIsNone(none.incoming)
        self.assertEqual(bonus.incoming, (360 + 30 * 70) // 70)  # 進行中の連鎖に乗る全消しボーナス+30個

    def test_attack_frame_estimate_is_used(self) -> None:
        # PBA が経過時間から見積もった「相手の連鎖が終わるまでに置ける手数」があれば、固定値の代わりに使う
        # (実機 2026-09-27 15:24: 固定の「3手」のまま相手の6連鎖が終わり、間に合う攻撃を選べなかった)
        me = ["......", "......", "R.....", "RB....", "GY....", "GY....", "RB....", "GB....", "YYB...", "GYB..G", "GGRYYG", "RBRBGY", "RRBBYY"]
        en = ["......", "......", "......", "......", "......", "......", "..BB..", "..YR.G", "..YY.G", "..RRBB", "GGGBYB", "YYRBGG", "YBRRYG"]
        before = EMPTY_ROWS[:7] + ["B.....", "B.....", "G.....", "G.....", "GB....", "GB...."]
        with AmaClient(AMA_BRIDGE_EXE) as client:
            me_s = PlayerState(PuyoField.from_rows(me), ["YB", "GB", "BG"], attack=2)
            est = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1,
                                                  chain_field=PuyoField.from_rows(before), attack_frame_est=0))
            fixed = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1,
                                                    chain_field=PuyoField.from_rows(before)))
        self.assertEqual(est.diag["attack_frame"], 0)
        self.assertEqual(fixed.diag["attack_frame"], 1)  # 2連鎖 → max(1, min(4, 2/2))

    def test_fire_score_is_ignored_by_ama(self) -> None:
        # 本線発火の目安は YamaNEKO だけが読む(2026-10-21 に ama のブリッジから外した)。送った要求はそのまま残す
        rows = list(EMPTY_ROWS)
        rows[-3] = "R....."
        rows[-2] = "RGG..."
        rows[-1] = "RGG..."
        me = PlayerState(PuyoField.from_rows(rows), ["RB", "BY", "GG"])
        en = PlayerState(PuyoField.from_rows(EMPTY_ROWS), ["RG", "BY", "GG"])
        with AmaClient(AMA_BRIDGE_EXE) as client:
            move = client.battle(me, en, fire_score=300)
        self.assertIsNone(move.fire_reason)
        req = json.loads(move.request)
        self.assertEqual((req["mode"], req["fire_score"], req["self"]["queue"]), ("battle", 300, ["RB", "BY", "GG"]))

    def test_battle_returns_label_and_both_analyses(self) -> None:
        from src.engine.ama_client import LABEL_TEXT_JA, PlayerState

        enemy_rows = list(EMPTY_ROWS)
        enemy_rows[-1] = "YYY..."
        with AmaClient(AMA_BRIDGE_EXE) as client:
            move = client.battle(
                PlayerState(self._field(), ["RB", "GG"]),
                PlayerState(PuyoField.from_rows(enemy_rows), ["YG", "RR"]),
            )
        self.assertIn(move.label, LABEL_TEXT_JA)
        self.assertEqual(move.enemy_analysis.immediate.count, 1)  # 黄を置けば黄4個
        for row, col in (move.axis_cell, move.child_cell):
            self.assertEqual(self._field().grid[row][col], ".")

class DiagTextTest(unittest.TestCase):
    def test_diag_text_shows_why_not_returned(self) -> None:
        # 攻撃を受けている時の判断材料(実機 2026-09-27 12:44: 206個に8連鎖で返さなかった理由の切り分け)
        from src.engine.ama_client import diag_text

        text = diag_text({
            "enemy_attack": 206, "attack_frame": 3, "accept_limit": 12,
            "best_in_time": None,
            "best_any": {"score": 40400, "count": 8, "frame": 5, "send": 577},
        })
        self.assertIn("着弾206個", text)
        self.assertIn("期限3手", text)
        self.assertIn("期限内の最大攻撃=無し", text)
        self.assertIn("8連鎖40400点(送577個,5手後)", text)
        self.assertEqual(diag_text(None), "")




class ResolveBridgeExeTest(unittest.TestCase):
    """思考エンジンのブリッジの場所(2026-10-21 から PBA は YamaNEKO 固定。ama はシミュレーターの比較用に名前で選ぶ)"""

    def _resolve(self, conf: dict | None, bridge: str = "yamaneko"):
        import tempfile
        from pathlib import Path

        from src.engine.ama_client import resolve_bridge_exe

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "engine.json"
            if conf is not None:
                p.write_text(json.dumps(conf), encoding="utf-8")
            return resolve_bridge_exe(p, bridge)

    def test_yamaneko_is_fixed(self) -> None:
        from src.engine.ama_client import YAMANEKO_BRIDGE_EXE

        self.assertEqual(self._resolve(None), YAMANEKO_BRIDGE_EXE)
        self.assertEqual(self._resolve({"bridge": "ama"}), YAMANEKO_BRIDGE_EXE)  # 以前の選択の項目は読まない

    def test_yamaneko_exe_override(self) -> None:
        from src.paths import app_root

        self.assertEqual(self._resolve({"yamaneko_exe": "x/y.exe"}), (app_root() / "x" / "y.exe").resolve())

    def test_ama_by_name(self) -> None:
        self.assertEqual(self._resolve({"yamaneko_exe": "x/y.exe"}, "ama"), AMA_BRIDGE_EXE)

    def test_unknown_is_error(self) -> None:
        with self.assertRaises(ValueError):
            self._resolve(None, "unknown")


def _yamaneko_exe():
    from src.engine.ama_client import YAMANEKO_BRIDGE_EXE

    return YAMANEKO_BRIDGE_EXE


@unittest.skipUnless(_yamaneko_exe().exists(), "自作AI(Puyo-YamaNEKO)のブリッジが未ビルド")
class YamanekoBridgeIntegrationTest(unittest.TestCase):
    """自作AI(yamaneko-bridge.exe)が ama ブリッジと同じ形で答えること(PBA の全自動モードでそのまま使えること)"""

    def _client(self) -> AmaClient:
        return AmaClient(_yamaneko_exe())

    def test_target_point_is_sent(self) -> None:
        # マージンタイム(2026-10-21): おじゃま1個に必要な点を要求に入れ、ブリッジはその点で攻撃を見積もる
        # (赤3個の上に緑3個: RG を縦に置くと 2連鎖 360点。70点なら5個、25点なら14個)
        rows = list(EMPTY_ROWS)
        rows[-2] = "GGG..."
        rows[-1] = "RRR..B"
        me = PlayerState(PuyoField.from_rows(rows), ["RG", "BY", "BB"])
        en = PlayerState(PuyoField.from_rows(EMPTY_ROWS), ["RG", "BY", "GG"])
        with self._client() as client:
            normal = client.battle(me, en)
            margin = client.battle(me, en, target_point=25)
        self.assertNotIn("target_point", json.loads(normal.request))
        self.assertEqual(json.loads(margin.request)["target_point"], 25)
        self.assertEqual((normal.self_analysis.garbage_max, margin.self_analysis.garbage_max), (360 // 70, 360 // 25))

    def test_suggest_returns_input_path(self) -> None:
        rows = list(EMPTY_ROWS)
        rows[-1] = "RRRR.."[:3] + "..."
        rows[-2] = "G....."
        with self._client() as client:
            move = client.suggest(PuyoField.from_rows(rows), ["RG", "BY", "GG"])
        self.assertTrue(move.path)
        self.assertEqual(move.path[-1], "DROP")
        known = {"LEFT", "RIGHT", "CW", "CCW", "CW_LEFT", "CW_RIGHT", "CCW_LEFT", "CCW_RIGHT", "DROP", "NONE", "WAIT", "TOUCH"}
        self.assertTrue(set(move.path) <= known, move.path)
        self.assertEqual(len(move.field_after), 13)

    def test_solo_mode(self) -> None:
        rows = list(EMPTY_ROWS)
        rows[-1] = "RRGG.."
        with self._client() as client:
            move = client.solo(PuyoField.from_rows(rows), ["RG", "BY"])
        self.assertEqual(move.label, "build")
        self.assertIsNotNone(move.self_analysis)

    def test_analyze_scores_follow_tsu_rule(self) -> None:
        rows = list(EMPTY_ROWS)
        rows[-3] = "RB...."
        rows[-2] = "RGB..."
        rows[-1] = "RGGB.."
        with self._client() as client:
            a = client.analyze(PuyoField.from_rows(rows), ["RB", "GG"])
        self.assertEqual(a.immediate.count, 1)
        self.assertEqual(a.immediate.score, 40)
        self.assertEqual(a.immediate.x, 0)
        self.assertGreaterEqual(a.max.score, a.immediate.score)
        self.assertEqual(a.garbage_max, a.max.score // 70)

    def test_attacked_label_is_not_build(self) -> None:
        me_rows = EMPTY_ROWS[:6] + ["R.....", "RG....", "GRB...", "GGRB..", "YYRBB.", "RYYGGB"]
        en_rows = EMPTY_ROWS[:10] + ["RR....", "RRGG.."]
        with self._client() as client:
            en = PlayerState(PuyoField.from_rows(en_rows), ["YY", "YY", "YY"])
            attacked = client.battle(PlayerState(PuyoField.from_rows(me_rows), ["YY", "RG", "BB"], attack=20), en)
        self.assertIn(attacked.label, ("accept", "main", "harass"))

    def test_enemy_chain_replayed_from_flash_frame(self) -> None:
        me = ["......", "......", "R.....", "RB....", "GY....", "GY....", "RB....", "GB....", "YYB...", "GYB..G", "GGRYYG", "RBRBGY", "RRBBYY"]
        en = ["......", "......", "......", "......", "......", "......", "..BB..", "..YR.G", "..YY.G", "..RRBB", "GGGBYB", "YYRBGG", "YBRRYG"]
        before = EMPTY_ROWS[:7] + ["B.....", "B.....", "G.....", "G.....", "GB....", "GB...."]
        with self._client() as client:
            me_s = PlayerState(PuyoField.from_rows(me), ["YB", "GB", "BG"], attack=2)
            flash = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1, chain_field=PuyoField.from_rows(before)))
            none = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1))
            bonus = client.battle(me_s, PlayerState(PuyoField.from_rows(en), ["RR", "BY", "YB"], attack_frame=1, chain_bonus=True, chain_field=PuyoField.from_rows(before)))
        self.assertEqual(flash.incoming_source, "flash_replay")
        self.assertEqual(flash.incoming_chain, 2)
        self.assertEqual(flash.incoming, (40 + 4 * 10 * 8) // 70)
        self.assertIsNone(none.incoming)
        self.assertEqual(bonus.incoming, (360 + 30 * 70) // 70)

    def test_fire_score_target(self) -> None:
        # 赤3個の上に緑3個: RG を縦に置くと赤4→緑4の2連鎖(360点)。自作AIは序盤なので撃たずに構築し、
        # 目安 300点を指定すると発火に差し替わる(ama 用のテストの盤面は緑4個がつながった安定しない盤面なので使わない)。
        # 右端の青は全消しにしないため(全消しになる手は目安が無くても自作AIが選ぶ。2026-10-10)
        rows = list(EMPTY_ROWS)
        rows[-2] = "GGG..."
        rows[-1] = "RRR..B"
        me = PlayerState(PuyoField.from_rows(rows), ["RG", "BY", "BB"])
        en = PlayerState(PuyoField.from_rows(EMPTY_ROWS), ["RG", "BY", "GG"])
        with self._client() as client:
            calm = client.battle(me, en)
            move = client.battle(me, en, fire_score=300)
        self.assertEqual(calm.label, "build")
        self.assertIsNone(calm.fire_reason)
        self.assertEqual(move.fire_reason, "target")
        self.assertIn(move.label, ("main", "harass", "all_clear"))
        self.assertEqual(move.chain_after, (2, 360))

    def test_battle_returns_label_and_both_analyses(self) -> None:
        from src.engine.ama_client import LABEL_TEXT_JA

        rows = list(EMPTY_ROWS)
        rows[-3] = "RB...."
        rows[-2] = "RGB..."
        rows[-1] = "RGGB.."
        field = PuyoField.from_rows(rows)
        enemy_rows = list(EMPTY_ROWS)
        enemy_rows[-1] = "YYY..."
        with self._client() as client:
            move = client.battle(PlayerState(field, ["RB", "GG"]), PlayerState(PuyoField.from_rows(enemy_rows), ["YG", "RR"]))
        self.assertIn(move.label, LABEL_TEXT_JA)
        self.assertEqual(move.enemy_analysis.immediate.count, 1)
        for row, col in (move.axis_cell, move.child_cell):
            self.assertEqual(field.grid[row][col], ".")
        self.assertTrue(move.path)


class StartupTest(unittest.TestCase):
    """思考エンジンの起動(2026-10-13): YamaNEKO の起動時の計測に合わせた待ち時間と、両方モードのスレッド数"""

    def test_search_threads_passed_only_to_yamaneko(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            # 受け取った引数を ready に載せて返すだけの偽のブリッジ
            fake = Path(d) / "yamaneko-fake.bat"
            fake.write_text('@echo {"ready":true,"args":"%*"}\n', encoding="ascii")
            client = AmaClient(fake, search_threads=6)
            self.assertEqual(client.engine, "yamaneko")
            self.assertEqual(client.args[1:], ["--search-threads", "6"])
            client._proc.wait(timeout=5)
            other = Path(d) / "ama-fake.bat"
            other.write_text('@echo {"ready":true}\n', encoding="ascii")
            client2 = AmaClient(other, search_threads=6)
            self.assertEqual(client2.args[1:], [])  # ama にはスレッド数の指定を渡さない
            client2._proc.wait(timeout=5)

    def test_ready_timeout_kills_process(self) -> None:
        # ready が来ないまま待ちの上限を超えたら、エラーにしてプロセスを残さない
        import sys
        from pathlib import Path
        from unittest import mock

        from src.engine import ama_client

        with mock.patch.object(ama_client, "READY_TIMEOUT_SEC", 0.5):
            created = []
            real_popen = ama_client.subprocess.Popen

            def popen(*a, **k):
                p = real_popen(*a, **k)
                created.append(p)
                return p

            with mock.patch.object(ama_client.subprocess, "Popen", popen):
                with self.assertRaises(TimeoutError):
                    AmaClient(Path(sys.executable))  # 標準入力を待ったまま何も出力しない
        self.assertIsNotNone(created[0].wait(timeout=5))

    def test_ready_timeout_is_long_enough_for_calibration(self) -> None:
        from src.engine import ama_client

        self.assertGreaterEqual(ama_client.READY_TIMEOUT_SEC, 60.0)
        self.assertEqual(ama_client.RECV_TIMEOUT_SEC, 10.0)  # 1手ごとの応答の待ちは変えない


class TestPlayerStateCarry(unittest.TestCase):
    """攻撃の端数点は値がある時だけ送る(ama のブリッジは読まない。2026-10-19)"""

    def test_carry_in_json(self) -> None:
        from src.engine.ama_client import PlayerState
        from src.engine.puyo_field import PuyoField

        f = PuyoField.from_rows(["......"] * 13)
        self.assertNotIn("carry", PlayerState(f, ["RB", "GY"]).to_json())
        self.assertEqual(PlayerState(f, ["RB", "GY"], carry=42).to_json()["carry"], 42)


if __name__ == "__main__":
    unittest.main()
