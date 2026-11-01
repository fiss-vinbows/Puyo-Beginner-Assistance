"""思考エンジン(ama ブリッジ)に渡す自分・相手の状態を、追跡結果から作る純粋な部分。

本体(src/app.py)と録画検証(tools/replay_video.py)で同じ入力を作るために切り出した
(分析文書 2026-09-19 P1-3: 録画検証が本体と別の入力で判断していた)。
"""

from __future__ import annotations

from dataclasses import dataclass

from .ama_client import PlayerState
from .puyo_field import PuyoField
from .tracker import PlayerTracker
from ..vision.color_mapper import ColorMapper


# 相手の連鎖1段にかかる秒数(実機 2026-09-27 の実測: 6連鎖 7.9〜8.2秒、5連鎖 6.7秒)
CHAIN_STEP_SEC = 1.3
# 自分の1手にかかる秒数(全自動の実測の中央値。手動なら人間の速さで変わる)
MY_MOVE_SEC = 1.0


def estimate_attack_frames(en: PlayerTracker, now: float | None) -> int | None:
    """相手の連鎖が終わるまでに自分が置ける手数(ama の enemy.attack_frame の見積もり)。

    点滅から段数を再現できた連鎖だけ、(段数 × 1段の秒数 − 経過秒数) ÷ 自分の1手の秒数 で見積もる。
    固定の min(4, 段数/2) では経過時間が反映されず、相手の6連鎖の間ずっと「3手後」の攻撃を狙い続けて
    間に合わなかった(実機 2026-09-27 15:24: 130個に対し「3手後に5連鎖」が6手番続き、連鎖終了時は2連鎖しか無かった)。
    見積もれない時は None(ブリッジは従来の固定値を使う)。
    """
    if now is None or not en.chain_active or en.chain_expected is None or en.chain_expected.steps <= 0:
        return None
    remaining = en.chain_expected.steps * CHAIN_STEP_SEC - (now - en.chain_started_t)
    return max(0, int(remaining // MY_MOVE_SEC))


@dataclass
class ThinkInput:
    my_state: PlayerState
    en_state: PlayerState | None  # 相手の変換に失敗した(または一人用)なら None
    my_pairs: list[str]
    en_pairs: list[str]
    mapper: ColorMapper  # 変換に成功した色の割当(呼び出し側で採用する)
    enemy_error: str = ""  # 相手側の変換エラーの内容(空なら成功)


def build_think_input(
    mapper: ColorMapper,
    me: PlayerTracker,
    en: PlayerTracker,
    my_settled: list[list[str]],
    my_pairs_src: list[tuple[str, str]],
    solo: bool = False,
    incoming_hint: int = 0,
    now: float | None = None,
    haipuyo: bool = False,
    ahead: bool = False,
    peek: bool = False,
) -> ThinkInput:
    """追跡結果から思考入力を作る。

    色の割当は fork した仮のものに対して変換し、成功した分だけ返す mapper に反映する
    (途中で失敗しても仮の登録を残さない)。自分側の変換失敗(5色目など)は ValueError のまま伝える。
    相手側の変換失敗は en_state=None にして、相手を見ない思考に切り替えられるようにする
    (実機 2026-09-15: 相手ネクストの誤認識で自分の思考が5手番止まった)。
    """
    trial = mapper.fork()
    my_pairs = [trial.pair(*p) for p in my_pairs_src]
    my_field = PuyoField.from_rows(trial.cells_to_rows(my_settled))
    # 自分への予告はアイコン数(0〜6)なので、相手の連鎖から見積もった着弾量(incoming_hint)があれば大きい方を使う
    my_state = PlayerState(my_field, my_pairs, attack=max(me.pending_garbage, incoming_hint), all_clear=me.all_clear, carry=me.carry)
    if haipuyo:
        # 配ぷよ読み(2026-10-23): 試合の最初から置いた組ぷよを、組ぷよと同じ色の割当で文字にして送る。履歴が欠けている・
        # 変換できない色がある時は送らない(照合できないだけで思考は止めない。仮の割当は捨てる)
        # ahead: 先読み(次の手番の手を、今の組ぷよを置いた後の盤面と次・次々の組で先に計算する)の要求。今の組ぷよも置いた扱いにする
        # (2026-10-24 実機: 先読みの要求で履歴が今の手番のままだったので、見えている組が履歴の1組先から始まり照合できなかった)
        hist = me.pair_history()
        if hist is not None and ahead:
            hist = hist + [me.current_pair] if me.current_pair is not None else None
        if hist is not None:
            try:
                h_trial = trial.fork()
                my_state.history = [h_trial.pair(*p) for p in hist]
                my_state.placed = me.turn + (1 if ahead else 0)
            except ValueError:
                my_state.history = None
    if peek and my_state.placed is None:
        # 相手のツモを覗く(2026-11-01): 置いた数で、自分と相手の見えている組が配ぷよのどの位置かを合わせる
        my_state.placed = me.turn + (1 if ahead else 0)
    if solo:
        return ThinkInput(my_state, None, my_pairs, [], trial)
    mapper = trial
    trial = mapper.fork()
    try:
        if en.current_pair is not None and en.next1 is not None and en.next2 is not None:
            en_pairs = [trial.pair(*en.current_pair), trial.pair(*en.next1), trial.pair(*en.next2)]
        elif en.next1 is not None and en.next2 is not None:
            en_pairs = [trial.pair(*en.next1), trial.pair(*en.next2)]
        else:
            en_pairs = my_pairs[1:3] if len(my_pairs) >= 3 else my_pairs
        en_field = PuyoField.from_rows(trial.cells_to_rows(en.settled))
        # 連鎖中: 点滅から復元した発火直前の盤面(引き金込み)があればそれを渡す(全連鎖を正確に再現できる)。
        # 無ければ渡さない(途中盤面からの見積もりは 2026-09-20 に撤廃: 数倍ずれて ama の判断を狂わせた)
        chain_field = None
        if en.chain_active and en.chain_before_source == "flash_frame" and en.chain_before is not None:
            chain_field = PuyoField.from_rows(trial.cells_to_rows(en.chain_before))
    except ValueError as exc:
        return ThinkInput(my_state, None, my_pairs, [], mapper, enemy_error=str(exc))
    en_state = PlayerState(
        en_field,
        en_pairs,
        attack=en.pending_garbage,
        attack_frame=1 if en.chain_active else 0,
        all_clear=en.all_clear,
        chain_bonus=en.chain_bonus if en.chain_active else False,
        chain_field=chain_field,
        attack_frame_est=estimate_attack_frames(en, now),
        placed=en.turn if haipuyo else None,
    )
    if peek and en_state.placed is None and en.current_pair is not None and en.next1 is not None and en.next2 is not None:
        # 相手のツモを覗く: 操作中+ネクスト2組が揃って読めている時だけ置いた数を送る(操作中が読めない時に送ると、ネクストの位置が
        # 1つずれる。ネクストが読めない時の相手の組は自分の組の写しなので、送ると食い違いになる)
        en_state.placed = en.turn
    return ThinkInput(my_state, en_state, my_pairs, en_pairs, trial)
