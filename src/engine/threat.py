"""相手の攻撃の観測記録と、自分の手の速さの計測(新方針の第1段階: 提案は変えず記録だけ)。

分析文書 2026-09-19 P0-2: 「いま発火した攻撃」の量と着弾までの時間が確定しておらず、閾値調整では
解決しない。まず攻撃ごとに、発火直前の盤面・消えた個数・段数・所要時間・自分の予告の変化・
思考側の見積もりを1つの記録(AttackEpisode)にまとめてログに残し、実機で数値を確かめる。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttackEpisode:
    """相手の1回の連鎖(攻撃)の観測記録。chain_start で作り chain_end で締める。"""

    attack_id: int
    started_at: float
    before: list[list[str]]  # 発火直前の相手の確定盤面(引き金の組ぷよは含まれないことが多い)
    bonus: bool  # この連鎖に全消しボーナスが乗っている
    my_pending_at_start: int  # 開始時点の自分への予告
    ended_at: float | None = None
    steps: int = 0  # 観測した段数(点滅で段が合流するため下限)
    dropped: int = 0  # 消えた個数
    my_pending_at_end: int | None = None
    estimates: list[tuple[float, str, int]] = field(default_factory=list)  # (時刻, 根拠, おじゃま個数)

    @property
    def duration(self) -> float | None:
        return None if self.ended_at is None else self.ended_at - self.started_at

    def add_estimate(self, t: float, source: str, garbage: int) -> None:
        """思考側の見積もり(相手の本線など)を記録する。同じ根拠・同じ値の連続は省く。"""
        if self.estimates and self.estimates[-1][1:] == (source, garbage):
            return
        self.estimates.append((t, source, garbage))

    def summary(self) -> str:
        """chain_end 時にログへ出す1行。"""
        dur = "" if self.duration is None else f" 所要{self.duration:.2f}s"
        est = " ".join(f"{src}={g}個@{t:.2f}s" for t, src, g in self.estimates) or "なし"
        pend = "" if self.my_pending_at_end is None else f" 自分の予告 {self.my_pending_at_start}->{self.my_pending_at_end}"
        return (
            f"攻撃#{self.attack_id}: 段数≥{self.steps} 消去{self.dropped}個{dur}"
            f"{' 全消しボーナス付き' if self.bonus else ''}{pend} 見積もり: {est}"
        )


class TurnClock:
    """自分の spawn 間隔から平均手時間を求める(着弾までに置ける手数 M の目安)。

    開始直後や長い停止(演出・思考の空白)は平均を歪めるので、上限を超える間隔は数えない。
    """

    MAX_INTERVAL_SEC = 6.0

    def __init__(self) -> None:
        self._last_spawn_t: float | None = None
        self._intervals: list[float] = []

    def spawn(self, t: float) -> None:
        if self._last_spawn_t is not None:
            gap = t - self._last_spawn_t
            if 0.0 < gap <= self.MAX_INTERVAL_SEC:
                self._intervals.append(gap)
        self._last_spawn_t = t

    @property
    def average(self) -> float | None:
        if not self._intervals:
            return None
        return sum(self._intervals) / len(self._intervals)

    @property
    def samples(self) -> int:
        return len(self._intervals)

    def moves_within(self, seconds: float) -> int | None:
        """seconds の間に置ける手数の目安(平均手時間が未計測なら None)。"""
        avg = self.average
        if avg is None or avg <= 0:
            return None
        return int(seconds // avg)
