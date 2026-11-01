"""シミュレーションモードのゲーム規則と状態(画面に依存しない純粋なロジック)。

支援モード(画像認識)とは別に、自前の盤面でぷよを置いて ama の提案を見ながら練習する。
TBA(../Tetris)の練習画面と同じく自動落下は無く、落下キーで一気に置く。

盤面の表現は PuyoField と同じ1文字(R/Y/G/B/#/.)、row=0 が13段目(隠し段)、row=12 が最下段。
組ぷよは "RG" のような2文字(軸, 子)。子の位置は ama と同じ UP/RIGHT/DOWN/LEFT。

規則(ぷよぷよテトリス2のぷよぷよに合わせた簡略版):
  - 組ぷよは3列目の12段目(軸)・13段目(子)に出る。3列目の12段目が埋まっていたら負け
  - 移動・回転は積みとぶつからない範囲。回転は壁・積みに当たったら反対へ1列ずれる(床では1段上がる)。
    左右とも塞がっていたら同じ回転を2回押すと上下が入れ替わる(クイックターン)
  - 14段目以上に出たぷよは消える。13段目のぷよは連結に数えない(chain_sim と同じ)
  - 得点表は ama(core/chain.h)と同じ。全消しは次の連鎖に +2100点(30個)
  - 初手2組は3色だけから選ぶ(ぷよぷよ通と同じ)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from ..engine.chain_sim import COLOR_BONUS, GROUP_BONUS, POWER
from ..engine.puyo_field import COLORS, COLS, EMPTY, GARBAGE, ROWS, PuyoField

SPAWN_COL = 2
SPAWN_ROW = 1  # 軸ぷよの出現位置(12段目)
ROTATIONS = ("UP", "RIGHT", "DOWN", "LEFT")
# 子ぷよの軸からのずれ (行, 列)
CHILD_OFFSET = {"UP": (-1, 0), "RIGHT": (0, 1), "DOWN": (1, 0), "LEFT": (0, -1)}
ALL_CLEAR_BONUS = 2100
FIRST_PAIRS_3_COLORS = 2


def generate_pairs(rng: random.Random, count: int) -> list[str]:
    """第1手から count 組作る。先頭2組は3色だけから選ぶ。"""
    first_colors = rng.sample(COLORS, 3)
    out = []
    for i in range(count):
        colors = first_colors if i < FIRST_PAIRS_3_COLORS else COLORS
        out.append(rng.choice(colors) + rng.choice(colors))
    return out


def parse_pairs(text: str) -> list[str]:
    """「RG YB BB」のような文字列を組ぷよのリストにする(空白・カンマ区切り。軸→子の順)。"""
    pairs = [p for p in text.replace(",", " ").upper().split() if p]
    for p in pairs:
        if len(p) != 2 or any(ch not in COLORS for ch in p):
            raise ValueError(f"組ぷよは R/Y/G/B の2文字で指定してください: {p!r}")
    return pairs


def _groups(board: list[list[str]]) -> list[list[tuple[int, int]]]:
    """同色4個以上の連結(13段目は含めない)。"""
    seen = [[False] * COLS for _ in range(ROWS)]
    groups = []
    for r in range(1, ROWS):
        for c in range(COLS):
            color = board[r][c]
            if color not in COLORS or seen[r][c]:
                continue
            seen[r][c] = True
            stack = [(r, c)]
            group = []
            while stack:
                y, x = stack.pop()
                group.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 1 <= ny < ROWS and 0 <= nx < COLS and not seen[ny][nx] and board[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(group) >= 4:
                groups.append(group)
    return groups


def apply_gravity(board: list[list[str]]) -> list[list[str]]:
    out = [[EMPTY] * COLS for _ in range(ROWS)]
    for c in range(COLS):
        w = ROWS - 1
        for r in range(ROWS - 1, -1, -1):
            if board[r][c] != EMPTY:
                out[w][c] = board[r][c]
                w -= 1
    return out


@dataclass
class ChainStep:
    """連鎖の1段。board は消える直前の盤面、popped は消えるマス(おじゃま込み)。"""

    board: list[list[str]]
    popped: set[tuple[int, int]]
    score: int  # この段の点数


@dataclass
class ChainOutcome:
    steps: list[ChainStep]
    final: list[list[str]]
    score: int  # 全消しボーナスを含む

    @property
    def count(self) -> int:
        return len(self.steps)


def resolve_chain(board: list[list[str]], all_clear_bonus: bool = False) -> ChainOutcome:
    """落下済みの盤面から連鎖を最後まで進める。all_clear_bonus なら1段目に全消しボーナスを足す。"""
    board = [row[:] for row in board]
    steps: list[ChainStep] = []
    total = 0
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
        power = POWER[min(len(steps), len(POWER) - 1)]
        score = len(pop) * 10 * max(1, min(999, power + COLOR_BONUS[min(len(colors), 5)] + group_bonus))
        if not steps and all_clear_bonus:
            score += ALL_CLEAR_BONUS
        garbage = {
            (ny, nx)
            for y, x in pop
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            if 1 <= ny < ROWS and 0 <= nx < COLS and board[ny][nx] == GARBAGE
        }
        steps.append(ChainStep([row[:] for row in board], pop | garbage, score))
        total += score
        for y, x in pop | garbage:
            board[y][x] = EMPTY
        board = apply_gravity(board)
    return ChainOutcome(steps, board, total)


@dataclass
class Snapshot:
    board: list[list[str]]
    turn: int
    score: int
    all_clear: bool
    max_chain: int
    pairs: list[str]


@dataclass
class SimState:
    """シミュレーションの状態。pairs は第1手からの組ぷよ列(足りなくなったら乱数で足す)。"""

    seed: int
    # 実機の配ぷよの番号(0〜65535。None なら seed の乱数で作る。2026-10-26)
    haipuyo: int | None = None
    board: list[list[str]] = field(default_factory=lambda: [[EMPTY] * COLS for _ in range(ROWS)])
    pairs: list[str] = field(default_factory=list)
    turn: int = 0  # 今操作している組ぷよの番号(0始まり)
    score: int = 0
    all_clear: bool = False  # 全消しボーナスを持っている(次の連鎖で加算)
    max_chain: int = 0
    last_chain: ChainOutcome | None = None
    # 操作中の組ぷよ
    row: int = SPAWN_ROW
    col: int = SPAWN_COL
    rotation: str = "UP"
    _quick_turn: int = 0  # 回せなかった方向(+1=右回転、-1=左回転)。同じ方向にもう一度でクイックターン
    history: list[Snapshot] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ensure_pairs(self.turn + 3)

    @classmethod
    def new(cls, seed: int | None = None, haipuyo: bool = False) -> "SimState":
        """新しい配列。haipuyo なら実機の配ぷよ(65536通り)から1つを選ぶ。"""
        if haipuyo:
            from .haipuyo import SEQUENCE_COUNT

            return cls(seed=0, haipuyo=random.randrange(SEQUENCE_COUNT))
        return cls(seed=random.randrange(1, 10**9) if seed is None else seed)

    # ---- 組ぷよ列 ----
    def ensure_pairs(self, count: int) -> None:
        """足りない分は seed から決まる配列の続きで足す(同じ seed なら何度作っても同じ配列)。
        配ぷよの番号があれば、その配ぷよの続き(128組で最初に戻る)で足す。"""
        if len(self.pairs) < count:
            if self.haipuyo is not None:
                from .haipuyo import sequence

                seq = sequence(self.haipuyo)
                self.pairs.extend(seq[i % len(seq)] for i in range(len(self.pairs), count))
                return
            self.pairs.extend(generate_pairs(random.Random(self.seed), count)[len(self.pairs) :])

    @property
    def current(self) -> str:
        self.ensure_pairs(self.turn + 1)
        return self.pairs[self.turn]

    def upcoming(self, count: int) -> list[str]:
        """操作中を含めて count 組。"""
        self.ensure_pairs(self.turn + count)
        return self.pairs[self.turn : self.turn + count]

    def set_upcoming(self, pairs: list[str]) -> None:
        """操作中の組ぷよから先を pairs に置き換える(その先は乱数で足す)。"""
        self.push_history()
        self.pairs = self.pairs[: self.turn] + list(pairs)
        self.ensure_pairs(self.turn + 3)
        self._respawn()

    # ---- 状態 ----
    @property
    def game_over(self) -> bool:
        return self.board[SPAWN_ROW][SPAWN_COL] != EMPTY

    def field(self) -> PuyoField:
        return PuyoField([row[:] for row in self.board])

    def child_pos(self, row: int | None = None, col: int | None = None, rotation: str | None = None) -> tuple[int, int]:
        dr, dc = CHILD_OFFSET[rotation or self.rotation]
        return ((self.row if row is None else row) + dr, (self.col if col is None else col) + dc)

    def current_cells(self) -> list[tuple[tuple[int, int], str]]:
        """操作中の組ぷよの (マス, 色)。軸→子の順。row が負(14段目以上)のマスも返す。"""
        pair = self.current
        return [((self.row, self.col), pair[0]), (self.child_pos(), pair[1])]

    def _free(self, r: int, c: int) -> bool:
        if not 0 <= c < COLS or r >= ROWS:
            return False
        return r < 0 or self.board[r][c] == EMPTY

    def _fits(self, row: int, col: int, rotation: str) -> bool:
        """軸は13段目(row=0)まで、子は14段目(row=-1)まで上がれる(ama の core/move.cpp と同じ範囲)。"""
        cr, cc = self.child_pos(row, col, rotation)
        return row >= 0 and self._free(row, col) and self._free(cr, cc)

    def _respawn(self) -> None:
        self.row, self.col, self.rotation, self._quick_turn = SPAWN_ROW, SPAWN_COL, "UP", 0

    # ---- 操作 ----
    def move(self, dx: int) -> bool:
        if self.game_over or not self._fits(self.row, self.col + dx, self.rotation):
            return False
        self.col += dx
        self._quick_turn = 0
        # 床蹴りで13段目に浮いた組ぷよは浮いたまま横へ動ける(実機でもすぐには落ちない。ama の
        # core/move.cpp も「どこかで高さ11の列で蹴れれば高さ12の列を越えられる」としている)
        return True

    def rotate(self, direction: int) -> bool:
        """direction=+1 で右回転、-1 で左回転。"""
        if self.game_over:
            return False
        idx = ROTATIONS.index(self.rotation)
        target = ROTATIONS[(idx + direction) % 4]
        dr, dc = CHILD_OFFSET[target]
        # そのまま → 反対へ1列/1段ずらす(壁・床蹴り)
        for row, col in ((self.row, self.col), (self.row - dr, self.col - dc)):
            if self._fits(row, col, target):
                self.row, self.col, self.rotation, self._quick_turn = row, col, target, 0
                return True
        # 左右とも塞がっている: 同じ方向にもう一度押すと2個の位置が入れ替わる(クイックターン)。
        # 軸が子の位置へ移るので、子が上なら軸が13段目に上がる(=浮いて高さ12の列を越えられる。ama と同じ)
        if self._quick_turn == direction:
            flipped = ROTATIONS[(idx + 2) % 4]
            row, col = self.child_pos()
            if self._fits(row, col, flipped):
                self.row, self.col, self.rotation, self._quick_turn = row, col, flipped, 0
                return True
        self._quick_turn = direction
        return False

    def landing_cells(self) -> list[tuple[tuple[int, int], str]]:
        """今の位置から落とした時に2個が乗るマス (マス, 色)。13段目より上に乗るものは row<0。"""
        heights = [self._top_free_row(c) for c in range(COLS)]
        out = []
        cells = self.current_cells()
        # 下にあるほうから先に落とす
        for (r, c), color in sorted(cells, key=lambda t: -t[0][0]):
            out.append(((heights[c], c), color))
            heights[c] -= 1
        return out

    def find_actions(self, target: set[tuple[tuple[int, int], str]], max_depth: int = 24) -> list[str] | None:
        """今の位置から、落とすと target の (マス, 色) の2個になるまでの最短の操作列(幅優先探索)。届かなければ None。
        色まで比べる(マスだけだと上下逆に置く手順でも一致してしまう)。

        操作は "move_left"/"move_right"/"rotate_ccw"/"rotate_cw"(最後の「置く」は含まない)。
        このシミュレーターの規則(床蹴り・回し・クイックターン)で実際に届く手順を返す。
        """
        saved = (self.row, self.col, self.rotation, self._quick_turn)
        ops = {
            "move_left": lambda: self.move(-1),
            "move_right": lambda: self.move(1),
            "rotate_ccw": lambda: self.rotate(-1),
            "rotate_cw": lambda: self.rotate(1),
        }
        try:
            if self.game_over:
                return None
            start = saved
            seen = {start}
            frontier: list[tuple[tuple, list[str]]] = [(start, [])]
            while frontier:
                nxt = []
                for pos, actions in frontier:
                    self.row, self.col, self.rotation, self._quick_turn = pos
                    if set(self.landing_cells()) == target:
                        return actions
                    if len(actions) >= max_depth:
                        continue
                    for name, op in ops.items():
                        self.row, self.col, self.rotation, self._quick_turn = pos
                        op()
                        new = (self.row, self.col, self.rotation, self._quick_turn)
                        if new not in seen:
                            seen.add(new)
                            nxt.append((new, actions + [name]))
                frontier = nxt
            return None
        finally:
            self.row, self.col, self.rotation, self._quick_turn = saved

    def _top_free_row(self, col: int) -> int:
        r = ROWS - 1
        while r >= 0 and self.board[r][col] != EMPTY:
            r -= 1
        return r

    def push_history(self) -> None:
        self.history.append(
            Snapshot([row[:] for row in self.board], self.turn, self.score, self.all_clear, self.max_chain, list(self.pairs))
        )

    def drop(self) -> ChainOutcome | None:
        """操作中の組ぷよを落として置き、連鎖を最後まで進める。負けている時は None。"""
        if self.game_over:
            return None
        self.push_history()
        for (r, c), color in self.landing_cells():
            if r >= 0:  # 14段目以上に出たぷよは消える
                self.board[r][c] = color
        outcome = resolve_chain(self.board, self.all_clear)
        if outcome.count:
            self.all_clear = all(cell == EMPTY for row in outcome.final for cell in row)
            self.max_chain = max(self.max_chain, outcome.count)
        self.board = outcome.final
        self.score += outcome.score
        self.last_chain = outcome
        self.turn += 1
        self.ensure_pairs(self.turn + 3)
        self._respawn()
        return outcome

    def undo(self) -> bool:
        if not self.history:
            return False
        s = self.history.pop()
        self.board, self.turn, self.score, self.all_clear, self.max_chain, self.pairs = (
            s.board, s.turn, s.score, s.all_clear, s.max_chain, s.pairs,
        )
        self.last_chain = None
        self._respawn()
        return True

    def reset(self, same_sequence: bool) -> "SimState":
        """同一配列なら今の組ぷよ列(ツモ順設定を含む)のまま、そうでなければ新しい乱数で最初から。"""
        if same_sequence:
            return SimState(seed=self.seed, haipuyo=self.haipuyo, pairs=list(self.pairs))
        return SimState.new(haipuyo=self.haipuyo is not None)

    def set_board(self, board: list[list[str]]) -> None:
        """盤面を置き換える(盤面編集)。浮いているぷよは落とす。連鎖は次に置いた時に起きる。"""
        self.push_history()
        self.board = apply_gravity([row[:] for row in board])
        self.all_clear = False
        self._respawn()

    def drop_garbage(self, count: int, rng: random.Random | None = None) -> None:
        """おじゃまを count 個降らせる(6個ごとに1段、端数はばらばらの列)。"""
        if count <= 0:
            return
        self.push_history()
        rng = rng or random.Random()
        per_col = [count // COLS] * COLS
        for c in rng.sample(range(COLS), count % COLS):
            per_col[c] += 1
        for c in range(COLS):
            for _ in range(per_col[c]):
                r = self._top_free_row(c)
                if r < 0:
                    break
                self.board[r][c] = GARBAGE
        self._respawn()
