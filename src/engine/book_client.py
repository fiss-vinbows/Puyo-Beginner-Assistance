"""定跡を AI の思考より優先する(2026-10-26、利用者「定跡をありにすると AI 思考よりも定跡を優先。定跡に無い手順になったら AI 思考に戻る」)

思考エンジン(AmaClient)への問い合わせを包み、盤面とこれまでの組ぷよが定跡どおりの間は、エンジンの答えの置き場所を
定跡の手に差し替える(分析・判断材料などはエンジンの答えのまま)。通常の思考・2手先の先読み・開始前の初手のどれも
この包みを通るので、すべてに定跡が効く。定跡を引くには第1手からの組ぷよ(history)が要る。無い時は定跡を使わない。
"""

from __future__ import annotations

from dataclasses import replace

from ..simulator.book_placement import placement
from ..simulator.opening_book import OpeningBook
from .ama_client import BattleMove, PlayerState
from .puyo_field import PuyoField


class BookClient:
    """AmaClient と同じ使い方。battle・solo だけ定跡を先に引き、それ以外はそのままエンジンへ渡す。"""

    def __init__(self, inner, book: OpeningBook, log=None) -> None:  # noqa: ANN001
        self.inner = inner
        self.book = book
        self._log = log or (lambda msg: None)

    def __getattr__(self, name: str):  # noqa: ANN204
        return getattr(self.inner, name)

    def book_move(self, field: PuyoField, pairs: list[str], history: list[str] | None, move: BattleMove) -> BattleMove:
        """定跡に手があれば move の置き場所を差し替えた写しを返す(無ければ move のまま)。"""
        if history is None or not pairs:
            return move
        rows = field.to_rows()
        board = [list(r) for r in rows]
        turn = len(history)
        found = self.book.lookup(board, list(history) + list(pairs), turn)
        if found is None:
            return move
        place = placement(rows, pairs[0], found)
        if place is None:
            return move
        return replace(
            move, x=place.x, rotation=place.rotation, axis_cell=place.axis_cell, child_cell=place.child_cell,
            label="build", path=place.path, field_after=place.field_after, chain_after=(0, 0), fire_reason=None,
            reason=f"BOOK {found.pattern} {found.turn}", plan_steps=0, plan_field=None, plan_chain=(0, 0, 0), plan_trigger=[],
        )

    def battle(self, me: PlayerState, enemy: PlayerState, **kwargs) -> BattleMove:  # noqa: ANN003
        history = me.history
        if history is not None and not self._haipuyo_enabled():
            # 定跡のために付けた履歴は、配ぷよ読みでないエンジンには送らない
            me = replace(me, history=None, placed=None)
            enemy = replace(enemy, history=None, placed=None)
        move = self.inner.battle(me, enemy, **kwargs)
        return self.book_move(me.field, me.pairs, history, move)

    def solo(self, field: PuyoField, pairs: list[str], history: list[str] | None = None) -> BattleMove:
        move = self.inner.solo(field, pairs)
        return self.book_move(field, pairs, history, move)

    def _haipuyo_enabled(self) -> bool:
        """配ぷよ読みで起動した YamaNEKO なら history をそのまま送る(配ぷよの照合に使う)。"""
        return any(a.startswith("--haipuyo") for a in getattr(self.inner, "args", []))


def is_book_move(move: BattleMove) -> bool:
    return move.reason.startswith("BOOK ")


def book_text(move: BattleMove) -> str:
    """パネルのメモ「定跡 AA-AB の3手目」。"""
    _, pattern, turn = move.reason.split(" ", 2)
    return f"定跡 {pattern} の{turn}手目"
