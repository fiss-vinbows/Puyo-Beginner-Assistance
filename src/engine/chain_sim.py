"""盤面から連鎖を再現する(ぷよぷよテトリス2 のルール・得点、ama の core/chain.h と同じ表)

用途: 点滅から復元した発火直前の盤面(引き金込み)から、連鎖数・点数・終了後の盤面を先に求める。
実機 2026-09-20 14:56: 14連鎖(63個→1個)が、段の間の落下で盤面が0.3秒安定するたびに「連鎖終了」と
判定されて 3+3+3+2+1 の5つに割れた。再現した終了後の盤面が観測されるまで連鎖を終えないようにする。

ルール: 13段目(row=0)のぷよは連結に数えず消えない(落ちるだけ)。おじゃまは消える組に隣接すると消える。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..vision.cell_classifier import EMPTY, GAME_COLORS, GARBAGE
from .puyo_field import COLS, ROWS

COLOR_BONUS = (0, 0, 3, 6, 12, 24)
GROUP_BONUS = (0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 10)
POWER = (0, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512)


@dataclass(frozen=True)
class ChainResult:
    steps: int  # 連鎖数
    score: int  # 点数(全消しボーナスは含まない)
    cleared: int  # 消えた個数(おじゃま込み)
    final: list[list[str]]  # 連鎖後の盤面
    step_cleared: tuple[int, ...] = ()  # 段ごとに消えた個数(おじゃま込み)


def _groups(cells: list[list[str]]) -> list[list[tuple[int, int]]]:
    """同色4個以上の連結(13段目は含めない)。"""
    seen = [[False] * COLS for _ in range(ROWS)]
    groups: list[list[tuple[int, int]]] = []
    for r in range(1, ROWS):
        for c in range(COLS):
            color = cells[r][c]
            if color not in GAME_COLORS or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            group: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                group.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 1 <= ny < ROWS and 0 <= nx < COLS and not seen[ny][nx] and cells[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(group) >= 4:
                groups.append(group)
    return groups


def _drop(cells: list[list[str]]) -> list[list[str]]:
    out = [[EMPTY] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        w = ROWS - 1
        for r in range(ROWS - 1, -1, -1):
            if cells[r][c] != EMPTY:
                out[w][c] = cells[r][c]
                w -= 1
    return out


def simulate_chain(cells: list[list[str]]) -> ChainResult:
    """盤面を落下させてから連鎖を最後まで進める。"""
    board = _drop([row[:] for row in cells])
    steps = 0
    score = 0
    cleared = 0
    per_step: list[int] = []
    while True:
        groups = _groups(board)
        if not groups:
            break
        pop: set[tuple[int, int]] = set()
        colors: set[str] = set()
        group_bonus = 0
        for g in groups:
            pop.update(g)
            colors.add(board[g[0][0]][g[0][1]])
            group_bonus += GROUP_BONUS[min(11, len(g))]
        pop_count = len(pop)
        # 隣接するおじゃまも消える(得点の個数には含めない: ama と同じ)
        garbage: set[tuple[int, int]] = set()
        for y, x in pop:
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 1 <= ny < ROWS and 0 <= nx < COLS and board[ny][nx] == GARBAGE:
                    garbage.add((ny, nx))
        power = POWER[min(steps, len(POWER) - 1)]
        score += pop_count * 10 * max(1, min(999, power + COLOR_BONUS[min(len(colors), 5)] + group_bonus))
        for y, x in pop | garbage:
            board[y][x] = EMPTY
        steps += 1
        cleared += pop_count + len(garbage)
        per_step.append(pop_count + len(garbage))
        board = _drop(board)
    return ChainResult(steps=steps, score=score, cleared=cleared, final=board, step_cleared=tuple(per_step))
