"""マージンタイム(2026-10-21、利用者の表): 試合の経過時間でおじゃま1個に必要な点(レート)が下がる。

経過時間は自分の最初の組ぷよが落ち始めてからの秒数。連鎖を起こした時点の値でその連鎖全体を計算する。
YamaNEKO のシミュレーター(battle::margin_target_point)と同じ表。
"""

from __future__ import annotations

# (この秒数未満なら, 1個に必要な点)。256秒からは1点
MARGIN_TABLE: tuple[tuple[int, int], ...] = (
    (96, 70), (112, 52), (128, 34), (144, 25), (160, 16), (176, 12),
    (192, 8), (208, 6), (224, 4), (240, 3), (256, 2),
)


def margin_target_point(elapsed_seconds: float) -> int:
    """経過秒数(整数に切り捨て)での、おじゃま1個に必要な点。"""
    s = int(max(0.0, elapsed_seconds))
    for limit, point in MARGIN_TABLE:
        if s < limit:
            return point
    return 1
