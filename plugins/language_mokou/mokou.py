"""情報パネルの「mokou」表記(2026-10-25、利用者「遊びとして」)

基本は日本語と同じで、発火の行(「発火 / 発火中 N連鎖 X点」と「点数を訂正: …」)の連鎖数だけ、11〜19連鎖を
「数の呼び名」にする(「発火 13の雷 81,060点」。2026-10-25 利用者「連鎖数の数はやっぱり入れてほしい」)。表に無い連鎖数は「N連鎖」のまま。描く直前に1行ずつ置き換える(ログは日本語のまま)。
"""

from __future__ import annotations

import re

CHAIN_NAMES: dict[int, str] = {
    11: "刃",
    12: "破戒",
    13: "雷",
    14: "母",
    15: "夜",
    16: "黄昏",
    17: "帳",
    18: "暁",
    19: "夜明け",
}

_CHAIN = re.compile(r"(\d+)連鎖")


def to_mokou(line: str) -> str:
    """パネルの1行を mokou 表記にする。発火の行以外はそのまま返す。"""
    if not line.startswith(("発火", "点数を訂正")):
        return line
    def name(m: re.Match[str]) -> str:
        n = int(m.group(1))
        return f"{n}の{CHAIN_NAMES[n]}" if n in CHAIN_NAMES else m.group(0)

    return _CHAIN.sub(name, line)
