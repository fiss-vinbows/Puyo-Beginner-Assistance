"""実機の配ぷよ(65536通り)をシミュレーターで配る(2026-10-26、利用者「シミュレーターに配ぷよデータを反映させるモード」)

assets/haipuyo.txt.gz: 1行が1通りの配ぷよ(128組 = 256文字、r/y/g/b/p のうち4色)。2文字で1組で、1文字目を軸とみなす
(PBA・思考エンジンの組ぷよ表記「軸子」と同じ順)。色は出てきた順に R/Y/G/B へ割り当てる(シミュレーターは4色固定)。
128組を使い切ったら最初に戻る(ゲームと同じ)。
"""

from __future__ import annotations

import gzip
from functools import lru_cache

from ..engine.puyo_field import COLORS
from ..paths import resource_root

HAIPUYO_PATH = resource_root() / "assets" / "haipuyo.txt.gz"
SEQUENCE_COUNT = 65536
PAIRS_PER_SEQUENCE = 128


@lru_cache(maxsize=1)
def _table() -> tuple[str, ...]:
    with gzip.open(HAIPUYO_PATH, "rt", encoding="ascii") as f:
        return tuple(line.strip() for line in f if line.strip())


def available() -> bool:
    return HAIPUYO_PATH.exists()


def sequence(index: int) -> list[str]:
    """index 番(0〜65535)の配ぷよ128組を、シミュレーターの組ぷよ表記(軸子、R/Y/G/B)で返す。"""
    line = _table()[index % SEQUENCE_COUNT]
    slots: dict[str, str] = {}
    out = []
    for i in range(0, len(line), 2):
        pair = ""
        for ch in line[i:i + 2]:
            if ch not in slots:
                slots[ch] = COLORS[len(slots)]
            pair += slots[ch]
        out.append(pair)
    return out
