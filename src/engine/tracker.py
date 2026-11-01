"""画面認識の時系列から、片方のプレイヤーのゲーム状態を追跡する

board_reader が返すのは「そのフレームに見えているもの」で、落下中の組ぷよ・
回転の残像・落下中のおじゃま・連鎖の消去アニメなどが混ざる。ここでは連続した
フレームの認識結果を受け取り、以下を導く(画面キャプチャに依存しない純粋な
ロジックで、動画のリプレイや単体テストで検証できる):

  - settled: 確定盤面
  - 現在操作中の組ぷよ / ネクスト / ネクネク
  - 手番の切り替わり(spawn)と、思考してよいタイミング(turn_ready)
  - 連鎖中(chain_active)と連鎖数の推定
  - 予告おじゃま個数(時間的に安定した値)

【盤面の取り込み方針】
ぷよぷよでは、組ぷよを設置した直後にネクスト枠が入れ替わり(=spawn)、その後に
「ちぎりの落下 → 連鎖 → おじゃまの落下」がすべて起きてから操作が始まる。
つまり盤面が本当に変わるのは spawn 直後の一定期間に限られる。一方、操作中は
組ぷよが柱の上を通過したり回転の残像が出たりして、「下に空きが無い」だけでは
落下中のぷよを除外できない。そこで:

  1. spawn を検出したら「取り込み待ち」状態に入る
  2. 観測盤面(各列の下から連続して埋まっている部分)が一定時間変化しなくなったら、
     それを確定盤面として取り込み、turn_ready を出して「操作中」状態へ移る
  3. 操作中は観測盤面を無視する(確定盤面は次の spawn まで変えない)

取り込み待ちの間に観測のぷよ数が減れば連鎖中とみなし、減少の回数で連鎖数を
推定する。

【提示を速くするための早期取り込み】(実機で「盤面が高いと提案が間に合わない」ため)
  - 先読み: 呼び出し側は提案を出した後、「提案通りに置いた後の盤面」を
    set_expected() で渡しておく。spawn 後の観測がそれと一致したら安定待ちなしで
    即取り込む(提案に従って置いている限り、遅延は spawn 検出分だけになる)
  - 連鎖もおじゃま落下も起きない設置(観測 = 確定盤面+2個、同色4連結なし、
    予告おじゃまなし)は安定待ちなしで取り込む

連鎖演出のテキストに隠れたセルが空と誤判定されると列の上側がまとめて欠けるため、
観測は同じ結果が連続フレームで一致した時だけ採用する。

【13段目のぷよ】
13段目に置かれたぷよは画面に描かれないため、認識では常に空に見える。だが思考には
影響する(実機で、13段目に置いた赤が抜けた盤面で計算し「9連鎖」と表示したが実際は
14連鎖だった)。そこで13段目の内容を別に記憶する(_hidden):
  - 先読みが一致した時は、期待盤面(amaの置いた後の盤面。13段目を含む)の13段目を引き継ぐ
  - それ以外でも、設置で色ぷよが1個しか増えず、その列が12段目まで埋まっていれば、
    残りの1個(設置した組ぷよのもう一方の色)が13段目にあると推定する
  - その列の12段目が空になったら(落ちてきて見えるようになる)記憶を消す

色は board_reader のゲーム色名のまま扱う(ama のスロットへの写像は呼び出し側)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .chain_sim import ChainResult, simulate_chain
from .puyo_field import COLS, ROWS
from ..vision.cell_classifier import EMPTY, GARBAGE, GAME_COLORS, UNKNOWN

Pair = tuple[str, str]  # (子, 軸)

# spawn 後、観測盤面がこの秒数変化しなければ取り込んで turn_ready を出す。
# ちぎりの落下はこれより短い
SETTLE_SECONDS = 0.3
# 連鎖の各段の間: 消えるぷよは点滅(フレームごとに現れたり消えたり)するので観測は
# 安定しないが、消えて落下した後〜次の点滅までに約0.3秒の安定した隙間がある。
# その隙間では次に消える同色4連結が盤面に残っているので、4連結がある間は取り込まない
# (実機で11連鎖の途中を「連鎖終了」と誤判定し、連鎖前の盤面で思考してしまった)。
# ただし認識ミスで4連結に見え続ける場合に備え、この秒数を超えたら取り込む
POPPABLE_MAX_WAIT_SECONDS = 2.0
# 発火直前の盤面(引き金込み)を復元できた連鎖は、再現した連鎖後の盤面より色ぷよが多い観測を
# 「連鎖終了」としない(段の間の落下で盤面が0.3秒安定し、14連鎖が 3+3+3+2+1 に割れた 2026-09-20 14:56)。
# ただし復元が間違っていた時のため、減少が無いままこの秒数を超えたら取り込む
CHAIN_EXPECTED_MAX_WAIT_SECONDS = 4.0
# 再現した連鎖の段数が残っている間は、残りの段数 × この秒数まで待つ(上の4秒より長ければ)。相手の大連鎖と同時の時などに
# 盤面の減少が5秒見えず、4秒で打ち切って12連鎖を「6連鎖」と確定し、続きを別の連鎖として数えた(実機 2026-10-26 20:32 の 2P 233s)。
# 連鎖が本当に終わっていれば次の組ぷよが出て SPAWNED_CHAIN_FORCE_END_SECONDS で終えるので、長く待っても自由落下しない
CHAIN_EXPECTED_SECONDS_PER_STEP = 2.0
# 連鎖中に次の組ぷよの出現を検出した後は、盤面がこの秒数変わらなければ再現の見込み・消せる形を待たずに連鎖の終わりとする(2026-10-24)
SPAWNED_CHAIN_QUIET_SECONDS = 1.0
# 再現した連鎖の段数まで消えたのを見た後、最後の段からこの秒数以上たって出現を検出したら、再現した連鎖後の盤面ですぐ取り込む(2026-10-24)
# 連鎖中に出現を検出してからこの秒数たっても連鎖が終わらなければ、盤面の安定を待たずに取り込む(2026-10-25 実機 19:46 の 2P 981.65s:
# 1連鎖の段を観測で数えられず、相手の攻撃の演出で盤面も安定せず、出現から2.6秒「連鎖中」のまま組ぷよが自由落下した。
# ゲームは自分の連鎖とおじゃまの落下が終わってから次の組ぷよを出すので、出現は連鎖が終わった証拠。盤面が安定していれば従来の
# 取り込み(SETTLE_SECONDS の安定待ち)が先に済むよう、それより長く取る。ネクスト枠の読み違いへの備えも兼ねる)
SPAWNED_CHAIN_FORCE_END_SECONDS = 0.5
SPAWN_AFTER_STEP_MIN_SECONDS = 0.2
# ネクストが変わった時、新しいネクストが以前のネクネクと一致しなくても、
# このフレーム数続けば(認識の食い違いが定常化したとみなして)spawn として受け入れる
NEXT_RESYNC_FRAMES = 8
# 予告おじゃまはこの秒数同じ値が続いたら採用する(落下中のおじゃまが帯を横切る誤検出を除く)
PENDING_STABLE_SECONDS = 0.15
# ネクストは同じ値がこのフレーム数続いたら採用する(枠のぷよが移動中は別の色に見える)
NEXT_STABLE_FRAMES = 2
# 観測盤面は同じ結果がこのフレーム数続いたら採用する
OBSERVED_STABLE_FRAMES = 2
# 連鎖では最低4個消える。これ未満の減少は認識の揺れとみなす
CHAIN_MIN_DECREASE = 3
# 連鎖の各段の消去は、減少がこの秒数以内に連続していれば同じ段とみなす
CHAIN_STEP_MERGE_SECONDS = 0.2
# ネクストが読めない(演出がネクスト枠を覆う等)間の設置検出: 操作中に確定盤面より
# 操作中の組ぷよの色2個だけ増えた観測がこの秒数続いたら設置(spawn)とみなす。
# 実機 2026-09-17: 演出中にネクストが読めず、3手置いても手番が進まなかった(5〜7秒の遅れ)
PLACEMENT_FALLBACK_SECONDS = 0.3
# 診断(2026-10-10): 操作中の組ぷよが着地した(観測が確定盤面より2個以上増えた)のに、この秒数たっても次の spawn を
# 検出できない時、数フレームおきにネクスト枠の読み取りと代替検出の条件を spawn_wait として記録する
# (実機 2026-10-10 13:49: 青青を置いた後、同一ツモ(緑紫・緑紫)の次の組ぷよを4.5秒検出できず自由落下した)
SPAWN_WAIT_DIAG_SECONDS = 0.6
# 同一ツモが3組以上続いてネクスト枠が変わらない時の設置検出: 置いた組ぷよが確定盤面に入ってからこの秒数が過ぎ、
# 次の組ぷよの色が浮いて見えたフレームがこの回数あれば設置とみなす(新しい組ぷよが見えたことが証拠なので短くてよい)
SAME_PAIR_FALLBACK_SECONDS = 0.15
SAME_PAIR_EVIDENCE_FRAMES = 3
SAME_PAIR_SPAWN_ROWS = 4  # 新しい組ぷよが出たとみなす範囲(上から何段か。row=0 が13段目)
SPAWN_WAIT_DIAG_EVERY = 3  # 何フレームおきに記録するか
SPAWN_WAIT_DIAG_MAX = 30  # 1回の待ちで記録する上限
# 開始時の組ぷよの推定: 浮いている2個が同じ色のまま段が下がる(または列が変わる)のをこの秒数以内に見たら確定
INITIAL_CANDIDATE_MAX_AGE_SECONDS = 1.5
# 開始を軸ぷよ1個で判定する時: 同じ色が続けて見えるフレーム数と、「直前の入れ替わり」とみなす秒数(2026-10-11)
START_AXIS_FRAMES = 2
START_SWAP_RECENT_SECONDS = 0.5
# spawn(ネクスト入替)無しの消去検出(実機 2026-09-19: 相手の連鎖は相手の操作中に起き、相手のネクストは
# 連鎖が終わるまで入れ替わらないので、spawn 頼みだと連鎖終了後にしか chain_start が出なかった)。
# 確定盤面より減った観測が連続 OBSERVED_STABLE_FRAMES フレーム(間隔 UNANNOUNCED_MAX_GAP_SECONDS 以内)
# 続いたら消去の疑いとして取り込み待ち(settling)に入り、通常の連鎖検出に渡す。
# 引き金の組ぷよ2個が未取込だと4個消しでも純減は2個なので UNANNOUNCED_MIN_LOSS=2
UNANNOUNCED_MIN_LOSS = 2
UNANNOUNCED_MAX_GAP_SECONDS = 0.2
# 引き金込みの盤面候補(設置直後に「操作中の組ぷよの色2個だけ増え、4連結がある」観測)の有効期間
TRIGGER_CANDIDATE_MAX_AGE_SECONDS = 0.5
# 点滅フレームからの発火直前盤面の復元(録画 2026-09-13 の 55.1〜55.4s で確認): 消えるぷよは約0.3秒間
# 「見える/消える」を交互に繰り返し、見えるフレームでは目のある通常のぷよとして読める。
# 直近 FLASH_WINDOW_SECONDS の観測に「今の盤面を含み、差分がちょうど4連結以上の組(+隣のおじゃま)」の
# 盤面があれば、それが引き金込みの発火直前盤面。そこから全連鎖を正確に再現できる
FLASH_WINDOW_SECONDS = 0.6
FLASH_HISTORY_FRAMES = 20
# 演出(全消し直後のキャラクターのカットイン等)が盤面を覆うと、顔や服が色ぷよとして読まれる。
# 本物のぷよには白い目があるので、確定盤面に入りうる位置(各列の最下段から連続して埋まっている
# 部分)にある「目の無い色セル」がこの数以上なら、そのフレームは信用しない
# (実機 2026-09-17: カットインで26セル。通常の盤面は0)
COVERED_MIN_EYELESS = 5
# カットインがこの秒数以上続き、直前の確定盤面がこの個数以下なら「全消し」とみなす。
# 全消しの瞬間はカットインで盤面が覆われて「0個になった」フレームを見逃す(実機 2026-09-17: 相手の
# 全消しを検出できず、+30個の攻撃を想定できなかった)ので、カットイン自体を証拠にする
ALL_CLEAR_CUTIN_SECONDS = 0.5
ALL_CLEAR_CUTIN_MAX_PUYOS = 8
# ただし試合開始の演出もカットインとして読まれる(空の盤面から全消しと誤判定し、試合全体が
# 全消し戦モードになった 2026-09-17)ので、直前の盤面にぷよがあり(この個数以上)、
# カットインが終わった後の盤面が空(新しい組ぷよの2個以下)であることを確認する
ALL_CLEAR_CUTIN_MIN_PUYOS = 2
ALL_CLEAR_AFTER_MAX_PUYOS = 2


@dataclass
class Snapshot:
    """1フレームぶんの認識結果。"""

    t: float  # 秒
    cells: list[list[str]]  # [row][col]、row=0 が13段目
    next1: Pair
    next2: Pair
    pending_garbage: int
    eyeless_cells: frozenset[tuple[int, int]] = frozenset()  # 色だが白い目が無いセル(演出の判定用)


@dataclass
class TrackerEvent:
    kind: str  # "spawn" / "turn_ready" / "chain_start" / "chain_step" / "chain_end" / "all_clear" / "garbage_rise" / "pending"
    t: float
    detail: str = ""


def settle_cells(cells: list[list[str]]) -> list[list[str]]:
    """各列で最下段から連続して埋まっている部分だけを残し、浮いているぷよを消す。"""
    out = [[EMPTY] * COLS for _ in range(ROWS)]
    for col in range(COLS):
        for row in range(ROWS - 1, -1, -1):
            c = cells[row][col]
            if c == EMPTY:
                break
            out[row][col] = c
    return out


# 組ぷよは13段目・12段目の3列目(row=0,1 / col=2)に出現する。ルール上、12段目3列目が
# 埋まった時点で負けなので、生きている試合の確定盤面でこの2マスが埋まることはない。
# 3列目が11段目まで積まれていると、スポーン直後の組ぷよが積みに接触して settle_cells で
# 「浮いていない」と判定され確定盤面に混入した(実機 2026-09-15: 提案が3列目を避けた)ので、
# 確定盤面の候補ではこの2マスを常に空として扱う。
SPAWN_CELLS = ((0, 2), (1, 2))


def clear_spawn_cells(cells: list[list[str]]) -> list[list[str]]:
    """組ぷよの出現位置(13・12段目の3列目)を空にした盤面を返す。"""
    if all(cells[r][c] == EMPTY for r, c in SPAWN_CELLS):
        return cells
    out = [row[:] for row in cells]
    for r, c in SPAWN_CELLS:
        out[r][c] = EMPTY
    return out


def drop_top_eyeless(cells: list[list[str]], eyeless: frozenset[tuple[int, int]]) -> list[list[str]]:
    """各列の最上部に乗った「白い目の無い色のマス」(攻撃の光などの演出)を空にした盤面を返す。

    演出は盤面の上部(11〜12段目)に出るので、積みが高いと settle_cells で浮いていないと判定され、
    偽の色ぷよとして確定盤面・観測個数に入った(演出の赤を期待盤面に入れて窒息 2026-10-12 e79f556、
    相手側で5色目 'red' 2026-10-12 12:48 の 125.9s)。評価データ(vision_eval、36枚)では本物のぷよ1789マスに
    目なしは0、演出を色と読んだ91マスのうち82が目なし。列を上から見て、目のある色ぷよ・おじゃまより
    下は消さない(連鎖中の文字や消える泡の下にある本物のぷよで連鎖の個数を狂わせないため)。"""
    if not eyeless:
        return cells
    out = None
    for col in range(COLS):
        for row in range(ROWS):
            c = cells[row][col]
            if c == EMPTY:
                continue
            if (row, col) not in eyeless:
                break
            if out is None:
                out = [line[:] for line in cells]
            out[row][col] = EMPTY
    return cells if out is None else out


def count_puyos(cells: list[list[str]]) -> int:
    return sum(1 for row in cells for c in row if c != EMPTY)


def free_cells(cells: list[list[str]]) -> int:
    """可視12段の空きマス数(13段目は数えない)。"""
    return sum(1 for row in cells[1:] for c in row if c == EMPTY)


def contains_board(observed: list[list[str]], base: list[list[str]]) -> bool:
    """base のぷよが全て同じ位置・同じ色で observed にあるか(増えた分は後から置いた組ぷよ・おじゃま)。"""
    return all(observed[r][c] == base[r][c] for r in range(ROWS) for c in range(COLS) if base[r][c] != EMPTY)


def count_colored(cells: list[list[str]]) -> int:
    return sum(1 for row in cells for c in row if c in GAME_COLORS)


def count_garbage(cells: list[list[str]]) -> int:
    return sum(1 for row in cells for c in row if c == GARBAGE)


# 攻撃の端数点(2026-10-19、YamaNEKO の --count-carry-self 用): おじゃまは (連鎖の点 + 端数点) ÷ 70 個で、余りは次へ持ち越す。
# 実機は下キーを押しっぱなしで落とすので、落とした段数×1点(落下ボーナス)も端数点に入る
TARGET_POINT = 70


def placement_drop_rows(before: list[list[str]], after: list[list[str]]) -> int | None:
    """置く前と置いた後の盤面から、その手の落下ボーナス(落とした段数)を求める。

    増えた色ぷよがちょうど2個の時だけ数え、それ以外(おじゃまだけ増えた・見逃した手が混ざった)は None。
    段数は YamaNEKO のシミュレーターと同じく、出現位置(13段目)から低い方の着地点まで = 12 − 着地の段(下から0)。
    """
    added = [(r, c) for r in range(ROWS) for c in range(COLS) if after[r][c] in GAME_COLORS and before[r][c] == EMPTY]
    if len(added) != 2:
        return None
    land = min(ROWS - 1 - r for r, _ in added)
    return max(0, 12 - land)


def carry_after_chain(carry: int, drop_rows: int | None, score: int | None, target_point: int = TARGET_POINT) -> int:
    """連鎖が終わった後の端数点。連鎖の点が分からない(引き金を捉えられなかった)時は 0 とみなす。
    target_point はおじゃま1個に必要な点(マージンタイムで下がる)。"""
    if score is None:
        return 0
    return (carry + (drop_rows or 0) + score) % max(1, target_point)


def floating_puyos(cells: list[list[str]], settled: list[list[str]]) -> list[tuple[int, int, str]]:
    """確定盤面に含まれないぷよ (row, col, color)。"""
    return [
        (r, c, cells[r][c])
        for r in range(ROWS)
        for c in range(COLS)
        if cells[r][c] != EMPTY and settled[r][c] == EMPTY
    ]


def has_poppable_group(cells: list[list[str]]) -> bool:
    """同色4個以上の連結があるか(=この盤面は消去が起きる)。"""
    seen = [[False] * COLS for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            color = cells[r][c]
            if color not in GAME_COLORS or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < ROWS and 0 <= nx < COLS and not seen[ny][nx] and cells[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if size >= 4:
                return True
    return False


def poppable_cells(cells: list[list[str]]) -> set[tuple[int, int]]:
    """同色4個以上の連結に属するマスの集合(=次に消えるマス)。"""
    seen = [[False] * COLS for _ in range(ROWS)]
    out: set[tuple[int, int]] = set()
    for r in range(ROWS):
        for c in range(COLS):
            color = cells[r][c]
            if color not in GAME_COLORS or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            group = []
            while stack:
                y, x = stack.pop()
                group.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < ROWS and 0 <= nx < COLS and not seen[ny][nx] and cells[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(group) >= 4:
                out.update(group)
    return out


def strip_moving_pair(settled: list[list[str]], prev: list[list[str]], pair: Pair | None) -> list[list[str]]:
    """盤面が高い時、スポーン直後の組ぷよが積みに接触して「浮いていない」ように見える
    (12段目3列目にスポーンし、その列が11段まで積まれている等)。前回の確定盤面より
    色ぷよが「設置の2個」より多く増えていれば、増えた分のうち操作中の組ぷよの色と
    一致する列の最上部のセル(最大2個)を取り除く。"""
    if pair is None:
        return settled
    added = [
        (r, c)
        for r in range(ROWS)
        for c in range(COLS)
        if settled[r][c] in GAME_COLORS and prev[r][c] == EMPTY
    ]
    extra = len(added) - 2
    # 増分が「設置2個+組ぷよ2個」を超える場合は前回の確定盤面が古い(起動直後など)ので触らない
    if extra <= 0 or extra > 2:
        return settled
    colors = set(pair)
    out = [row[:] for row in settled]
    # 列の最上部にあり、色が操作中の組ぷよと一致するセルを上から順に取り除く
    # (縦向きの組ぷよは1個取り除くと次のセルが最上部になる)
    for _ in range(min(extra, 2)):
        candidates = [
            (r, c)
            for r, c in sorted(added)
            if out[r][c] in colors and (r == 0 or out[r - 1][c] == EMPTY)
        ]
        if not candidates:
            break
        r, c = candidates[0]
        out[r][c] = EMPTY
    return out


_SHORT = {"red": "R", "yellow": "Y", "green": "G", "blue": "B", "purple": "P", GARBAGE: "#", EMPTY: ".", UNKNOWN: "?"}


def has_unknown(cells: list[list[str]]) -> bool:
    return any(c == UNKNOWN for row in cells for c in row)


def has_blocking_unknown(cells: list[list[str]]) -> bool:
    """確定盤面に入りうる位置(各列の最下段から連続して埋まっている部分)に unknown があるか。

    空マスの上に浮いている unknown は無視する: 全消し後に盤面上部へ重なる「全消し」バナーが
    黄・おじゃま・不明として読まれ、盤面更新が次の連鎖まで止まった(実機 2026-09-16)。
    浮いているものは settle_cells でも落とされるので、確定盤面には影響しない。
    """
    for col in range(COLS):
        for row in range(ROWS - 1, -1, -1):
            c = cells[row][col]
            if c == EMPTY:
                break
            if c == UNKNOWN:
                return True
    return False


def is_covered(cells: list[list[str]], eyeless: frozenset[tuple[int, int]]) -> bool:
    """盤面が演出に覆われているか(確定盤面に入りうる位置に目の無い色セルが多い)。"""
    if len(eyeless) < COVERED_MIN_EYELESS:
        return False
    n = 0
    for col in range(COLS):
        for row in range(ROWS - 1, -1, -1):
            if cells[row][col] == EMPTY:
                break
            if (row, col) in eyeless:
                n += 1
    return n >= COVERED_MIN_EYELESS


def pair_known(pair: Pair) -> bool:
    return all(c in GAME_COLORS for c in pair)


@dataclass
class PlayerTracker:
    settled: list[list[str]] = field(default_factory=lambda: [[EMPTY] * COLS for _ in range(ROWS)])
    current_pair: Pair | None = None
    next1: Pair | None = None
    next2: Pair | None = None
    turn: int = 0  # spawn のたびに増える
    # 配ぷよ読み用(2026-10-23): 手番ごとの操作中の組ぷよ(画面の色)。試合ごとに追跡を作り直すので試合の最初からの履歴になる
    pair_by_turn: dict = field(default_factory=dict)
    chain_active: bool = False
    all_clear: bool = False  # 全消しボーナスを持っている(連鎖で盤面が空になった〜次の連鎖まで)
    chain_steps: int = 0  # 進行中(または直前)の連鎖の連鎖数の推定(点滅で段が合流するため過小になりがち)
    chain_dropped: int = 0  # 進行中の連鎖で消えたぷよの個数(連鎖の大きさの目安。4個=1連鎖)
    # 進行中の連鎖が全消しボーナスを消費した(=その連鎖に+30個が乗る)。all_clear は chain_start で
    # 落とすため、連鎖中の見積もりでボーナス分を取り落としていた(分析文書 2026-09-19 P1-1)
    chain_bonus: bool = False
    chain_before: list[list[str]] | None = None  # 進行中(または直前)の連鎖の直前の確定盤面(攻撃の再現・記録用)
    # 進行中の連鎖の観測: 段ごとに消えた個数と、直近の観測盤面(消えた後)。残りの連鎖はこの盤面を消して
    # 見積もる(実機 2026-09-19: 「8個以上消えたら手札の最大攻撃」の代用が 3連鎖を 244個 と見積もった)
    chain_step_drops: list[int] = field(default_factory=list)
    chain_observed: list[list[str]] | None = None
    chain_expected: ChainResult | None = None  # 発火直前盤面から再現した連鎖(復元できた時だけ)
    chain_started_t: float = 0.0  # 連鎖の開始を検出した時刻(相手の連鎖の残り時間の見積もり用)
    chain_score: int = 0  # 直前に終わった連鎖の点数(再現できた時。できなければ 0)
    _chain_last_change_t: float = -1.0  # 連鎖の開始・段の減少を最後に見た時刻
    pending_garbage: int = 0
    settling: bool = True  # spawn 後の取り込み待ち(True) / 操作中(False)
    _observed: list[list[str]] | None = None  # 連続フレームで一致した最新の観測盤面
    _observed_since: float = 0.0  # _observed がその内容になった時刻
    _observed_candidate: tuple[list[list[str]], int] | None = None  # (盤面, 連続フレーム数)
    _settled_count: int = 0
    # _settled_count を「設置後(消去前)の観測」で引き上げた時、その観測に消える4連結があったか。
    # 無ければその後の減少は消去ではない(実機 2026-09-19 23:14: 設置の瞬間に落下中のおじゃま・操作中の
    # 組ぷよが重なって 47個 に見え、次のフレームの 42個 を「5個消えた」と誤認 → 連鎖の安定待ちで提案が
    # 0.3〜0.5秒遅れ、次の組ぷよが出た後に提案が出た)。確定盤面が基準の時(None)は従来どおり
    # 減少をそのまま消去とみなす(相手の引き金の組ぷよは盤面に無いことが多い)
    _raised_baseline_poppable: bool | None = None
    _last_chain_step_t: float = -1.0
    _pending_candidate: tuple[int, float] | None = None  # (値, 最初に見た時刻)
    _next_candidate: tuple[Pair, Pair, int] | None = None  # (next1, next2, 連続フレーム数)
    _expected: list[list[str]] | None = None  # 先読み用: 次の設置後に期待する盤面
    _next_mismatch_frames: int = 0  # 新ネクスト≠旧ネクネク が続いているフレーム数
    _hidden: list[str] = field(default_factory=lambda: [EMPTY] * COLS)  # 13段目(画面に見えない)の記憶
    settling_since: float = 0.0  # 取り込み待ちに入った時刻(診断用)
    last_unknown: bool = False  # 直近の観測に unknown セルがあったか(診断用)
    last_observed_text: str = ""  # 直近の観測盤面(診断用)
    # 直近の観測で「積まれている」と見えたぷよの数(浮いているものは除く)。確定盤面と違い毎フレーム更新する。
    # 全自動モードが「組ぷよが着地したか」を素早く知るために使う(手番の進行を待つと着地の0.7秒後になる)
    observed_count: int = 0
    last_garbage_wait: str = ""  # おじゃまの後の取り込みを安定待ちにした理由(ログ用)
    observed_cells: list[list[str]] | None = None  # 直近の観測盤面(浮いているものを除く。全自動の着地判定用)
    # 直近の観測盤面から、白い目の無い色のマス(着地予測の点・攻撃の演出)を空にしてから落とした盤面。目の無い色が無ければ
    # observed_cells と同じ。出現時点の先読みの盤面の比較に使う(実機 2026-10-12 07:00: 「盤面が違う」8手の多くが、
    # 空のはずの上の方のマスを着地予測の点や攻撃が当たる演出で色と読んだもの)
    observed_cells_eyed: list[list[str]] | None = None
    floating_now: list[tuple[int, int, str]] = field(default_factory=list)  # 直近の観測で浮いているぷよ(行, 列, 色)。全自動の押しっぱなし移動用
    last_raw_text: str = ""  # 直近の生の認識結果(unknown を含む。診断用)
    last_poppable: bool = False  # 直近の観測盤面に消えるはずの4連結があったか(診断用)
    last_take_in_t: float = 0.0  # 最後に確定盤面を取り込んだ時刻(診断用)
    # 確定盤面の由来: "stable"(観測が安定) / "simple"(単純な設置) / "garbage"(設置+降り終わったおじゃま) / "expected"(先読みの期待盤面と一致)。
    # expected は予測由来なので、それに一致した先読みは独立した観測の裏付けではない(分析文書 2026-09-19 P1-4)
    last_take_in_reason: str = ""
    # 消えるはずの4連結が見え続けている期間の開始時刻。spawn ではリセットしない:
    # 誤認識(全消し後の「ぜんけし」表示等)で偽の4連結が見え続けると、spawn のたびに
    # 待ち時間がリセットされて永遠に取り込めなくなる(実機 2026-09-16: 全消し後17秒停止)
    _poppable_since: float | None = None
    _last_placed_pair: Pair | None = None  # 直前の spawn で設置された組ぷよ(子, 軸)
    _next_unknown_since: float | None = None  # ネクストが読めなくなった時刻(診断・代替検出用)
    _covered_since: float | None = None  # カットインに覆われ始めた時刻(全消しの推定用)
    _all_clear_pending: bool = False  # カットインを見た。終わった後の盤面が空なら全消しと確定する
    _placed_candidate: tuple[list[list[str]], float] | None = None  # 代替検出の候補(盤面, 最初に見た時刻)
    # 盤面から設置を検出して手番を進めた後、ネクスト枠の表示がまだ追いついていない(同一ツモの連続で
    # 枠が変わらない、演出で読めない等)。枠が後から変わっても同じ手番を二重に数えない
    _placed_by_board: bool = False
    _covered_skips: int = 0  # 直近の spawn 以降、演出で覆われてネクストを読まなかったフレーム数(診断用)
    # 同一ツモの連続で枠が変わらない時の設置検出の状態(置いた組ぷよが入った確定盤面, 最初に見た時刻, 次の組ぷよが見えたフレーム数)
    _same_pair_wait: tuple[list[list[str]], float, int] | None = None
    _same_pair_prev_placed: list[list[str]] | None = None  # 直前のフレームで「置いた組ぷよが入った」と見えた盤面
    _after_chain_evidence: int = 0  # 連鎖の後、同じ組ぷよが続く時に新しい組ぷよが見えたフレーム数(2026-10-24)
    # 連鎖中に次の組ぷよの出現を検出した(2026-10-24)。組ぷよは連鎖が終わってから出るので、連鎖は終わっている
    _spawned_in_chain: bool = False
    _spawned_in_chain_t: float = 0.0  # その出現を検出した時刻
    _spawn_wait_since: float | None = None  # 着地を見てから spawn を待っている時刻(診断用、spawn でリセット)
    _spawn_wait_frames: int = 0
    _spawn_wait_logged: int = 0
    # 試合開始時に「浮いている2個」から推定した操作中の組ぷよが、まだ設置されていない。
    # この2個が確定盤面に取り込まれた時点で手番を進める(実機 2026-09-19 22:46: 開始直後の赤赤が
    # 置かれた後もネクスト枠が同じ組ぷよ(黄紫,黄紫)で変わらず、置いた赤赤を操作中として
    # 「赤赤を隣に置けば全消し」を提示した)
    _initial_pair_pending: bool = False
    # 開始前(手番も設置も無い)にネクスト枠が入れ替わった時の「これから出現する第1手」(旧 next1)。
    # 開始前の枠の入れ替わり(前の試合の残り→カウントダウンの第1・2手、カウントダウン→開始)を spawn に
    # すると、出現しない組ぷよの手番ができて提案がずれた(実機 2026-09-20 15:50: 存在しない黄緑に提案)
    _prestart_first: Pair | None = None
    # 対戦が始まった別の証拠(相手の初手を検出した)がある。あれば自分の初手は「動いた」のを待たずに確定する
    start_confirmed: bool = False
    # 開始を軸ぷよ1個で判定する(2026-10-11): 開始前の枠の入れ替わりの時刻、相手の初手から分かった自分の
    # (現在, ネクスト, ネクネク)(両者のツモは同じ。app が入れる)、3列目の上で見えた軸ぷよの候補 (色, 連続フレーム数)
    _prestart_swap_t: float | None = None
    start_hint: tuple[Pair, Pair | None, Pair | None] | None = None
    _start_axis_candidate: tuple[str, int] | None = None
    _spawn_color_since: tuple[str, float] | None = None  # 3列目の上に見えている色と、見え始めた時刻
    # 開始時の組ぷよの推定候補 (色(整列), 上端の段, 列, 時刻)。動いている(段が下がる/列が変わる)のを見てから
    # 確定する(開始前の画面の飾りや残像を組ぷよと誤認して、出現しない組ぷよの手番を作らないため 2026-09-20)
    _initial_candidate: tuple[tuple[str, str], int, int, float] | None = None
    # spawn 無しの消去検出(分析文書 2026-09-19 第2弾): (減った観測盤面, 連続フレーム数, 時刻)
    _loss_candidate: tuple[list[list[str]], int, float] | None = None
    # spawn 無しで検出した連鎖の後、次の spawn を待っている(消費済みの current_pair で turn_ready を出さない)
    _await_spawn_after_chain: bool = False
    # 引き金込みの盤面候補(観測盤面, 時刻)。落下中の接触や残像でも成立しうるので「候補」
    _trigger_candidate: tuple[list[list[str]], float] | None = None
    carry: int = 0  # 攻撃の端数点(前の連鎖の余りと落下ボーナス。試合ごとに作り直すので 0 から始まる。2026-10-19)
    # おじゃま1個に必要な点(マージンタイム。本体が毎回の観測で経過時間から入れる。2026-10-21)。端数点の余りに使う。
    # 正しくは連鎖を起こした時点の値だが、連鎖の終わりの値で計算する(連鎖の途中で区切りをまたいだ時だけ数点ずれる)
    target_point: int = TARGET_POINT
    chain_before_source: str = "settled_without_trigger"  # chain_before の由来(flash_frame / observed_trigger_candidate / settled_without_trigger)
    _recent_boards: list[tuple[float, list[list[str]], int]] = field(default_factory=list)  # 点滅検出用の直近の観測(時刻, 盤面, 個数)

    def _watch_unannounced_chain(self, snap: Snapshot, observed: list[list[str]]) -> bool:
        """操作中(settling=False)に確定盤面より減った観測が続いたら、消去の疑いとして取り込み待ちに入る。
        True を返したら settling=True になっている(以後は通常の連鎖検出が減少を chain_start にする)。
        おじゃまだけが減った・浮いたぷよの落下で一時的に欠けた・色ぷよが減っていない、は除外する。
        13段目は画面に見えないので比べない(2026-10-24: 確定盤面の13段目に2個あると毎フレーム「2個減った」に見え、消去の疑いで
        取り込み待ちに入って思考の手を送らず、組ぷよが出現位置で固定されて窒息した。実機 07:08・07:11)。"""
        baseline = self.settled[1:]
        colored_before = sum(c in GAME_COLORS for row in baseline for c in row)
        raw = clear_spawn_cells(snap.cells)[1:]
        colored_now = sum(c in GAME_COLORS for row in raw for c in row)  # 浮いているぷよも含む生観測
        loss = count_puyos(baseline) - count_puyos(observed[1:])
        if (
            not self.last_take_in_reason  # 初回の取り込み前は基準が無い
            or colored_before < 2
            or colored_before - colored_now < UNANNOUNCED_MIN_LOSS
            or loss < UNANNOUNCED_MIN_LOSS
            or count_garbage(observed) > count_garbage(baseline)
        ):
            self._loss_candidate = None
            return False
        prev = self._loss_candidate
        continuous = prev is not None and prev[0] == observed and 0.0 < snap.t - prev[2] <= UNANNOUNCED_MAX_GAP_SECONDS
        n = prev[1] + 1 if continuous else 1
        self._loss_candidate = ([row[:] for row in observed], n, snap.t)
        if n < OBSERVED_STABLE_FRAMES:
            return False
        self.settling = True
        self.settling_since = snap.t
        self._await_spawn_after_chain = True
        self._expected = None  # 連鎖の途中を先読み一致で確定しない
        self._observed = None
        self._observed_candidate = None
        self._observed_since = snap.t
        self._loss_candidate = None
        return True

    def _detect_flash(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent]) -> bool:
        """消去の点滅から発火直前の盤面(引き金込み)を復元して連鎖を開始する。

        今の観測 B に対し、直近の観測 A で「B のぷよは全て A にもあり、A にだけあるマスがちょうど
        A の4連結の組(と隣接するおじゃま)」なら、A が発火直前の盤面、B は消えた瞬間。
        検出したら chain_active にして A を chain_before(由来 flash_frame)に保存し、通常の段の数え方に渡す。
        """
        count = count_puyos(settled)
        # 履歴を更新(今の観測を含める前に判定する)
        history = self._recent_boards
        detected = False
        if not self.chain_active and self.current_pair is not None:
            for t0, board, n in reversed(history):
                if snap.t - t0 > FLASH_WINDOW_SECONDS:
                    break
                if n - count < 4:
                    continue
                if any(settled[r][c] != EMPTY and settled[r][c] != board[r][c] for r in range(ROWS) for c in range(COLS)):
                    continue  # B は A の部分集合でない
                missing = [(r, c) for r in range(ROWS) for c in range(COLS) if board[r][c] != EMPTY and settled[r][c] == EMPTY]
                groups = poppable_cells(board)
                if not groups:
                    continue
                # 消えたマスは「4連結の組」「その隣のおじゃま」「消えた組の上に乗っていて浮いた(settle_cells が
                # 落とした)ぷよ」のどれかでなければならない
                def explained(r: int, c: int) -> bool:
                    if (r, c) in groups or board[r][c] == GARBAGE:
                        return True
                    return any((r2, c) in groups for r2 in range(r + 1, ROWS))
                if not all(explained(r, c) for r, c in missing):
                    continue
                if not any((r, c) in groups for r, c in missing):
                    continue
                # A が発火直前の盤面
                self.chain_active = True
                self.chain_started_t = snap.t
                self.chain_steps = 0
                self.chain_dropped = 0
                self.chain_step_drops = []
                self._last_chain_step_t = -1.0
                self.chain_bonus = self.all_clear
                self.all_clear = False
                # 読めないマス(13段目・出現位置)のぷよを補う(2026-10-24、実機 08:13 turn=39: 13段目の黄と、引き金の組ぷよの
                # 3列目12段目の黄が入らず再現が3連鎖、実際は2連鎖で、見込みの3段目を待つ間に組ぷよが自由落下した。
                # 13段目が埋まった連鎖は見込みのずれが30%、空なら7%)
                self.chain_before = self._restore_invisible(board)
                self.chain_before_source = "flash_frame"
                self.chain_observed = None
                self.chain_expected = simulate_chain(self.chain_before)
                self._chain_last_change_t = snap.t
                self._trigger_candidate = None
                self._settled_count = n  # 段の減少はこの盤面を基準に数える
                self._raised_baseline_poppable = None
                if not self.settling:
                    self.settling = True
                    self.settling_since = snap.t
                    self._await_spawn_after_chain = True
                self._expected = None
                self._observed = None
                self._observed_candidate = None
                self._observed_since = snap.t
                self._loss_candidate = None
                history.clear()
                note = "(全消しボーナス付き)" if self.chain_bonus else ""
                exp = self.chain_expected
                events.append(TrackerEvent("chain_start", snap.t, f"{note}(点滅から発火直前盤面を復元: {n}個→{count}個、再現 {exp.steps}連鎖 {exp.score:,}点)"))
                detected = True
                break
        # 発火直前の候補(A)には、白い目の無いマス(演出)を入れない。本物のぷよは点滅の見えるフレームで
        # 目がある。相手の攻撃の紫の円形の演出が盤面上部で紫ぷよに読まれ、本物の紫とつながって4連結に見え、
        # 演出が消えた次のフレームで「4個消えた」と誤検出した(実機 2026-09-27 14:31: 連鎖していないのに
        # 1連鎖とされ、終了判定の4秒待ちの間、次の組ぷよが思考されず自由落下した)
        board = settled
        if any(settled[r][c] != EMPTY for r, c in snap.eyeless_cells):
            board = [row[:] for row in settled]
            for r, c in snap.eyeless_cells:
                board[r][c] = EMPTY
            board = settle_cells(board)
        history.append((snap.t, [row[:] for row in board], count_puyos(board)))
        if len(history) > FLASH_HISTORY_FRAMES:
            del history[0]
        return detected

    def _remember_trigger(self, snap: Snapshot, observed: list[list[str]]) -> None:
        """設置直後の「確定盤面に操作中の組ぷよの色2個だけが加わり、4連結がある」観測を、
        引き金込みの盤面候補として覚える(chain_start 時に chain_before に使う)。"""
        if self.chain_active or self.current_pair is None:
            return
        observed = self._restore_invisible(observed)  # 確定盤面には記憶の13段目が入っているので、観測にも補って比べる
        added = []
        for r in range(ROWS):
            for c in range(COLS):
                before, after = self.settled[r][c], observed[r][c]
                if before == after:
                    continue
                if before != EMPTY or after not in GAME_COLORS:
                    return  # 既存セルが変わっている(消去・落下中)
                added.append(after)
        pair = self._last_placed_pair if self.settling else self.current_pair
        if pair is not None and sorted(added) == sorted(pair) and has_poppable_group(observed):
            self._trigger_candidate = ([row[:] for row in observed], snap.t)

    def set_expected(self, cells: list[list[str]] | None) -> None:
        """次の spawn 後にこの盤面が観測されたら即取り込む(先読み用)。"""
        self._expected = cells

    def update(self, snap: Snapshot) -> list[TrackerEvent]:
        events: list[TrackerEvent] = []
        # 演出(カットイン)が盤面を覆っている間はネクスト枠も覆われている。偽の色が組ぷよとして
        # 登録され、次の試合の序盤で5色目エラーになった(実機 2026-09-17)ので、ネクストも更新しない
        if not is_covered(clear_spawn_cells(snap.cells), snap.eyeless_cells):
            self._update_next(snap, events)
        else:
            self._covered_skips += 1
        self._update_board(snap, events)
        self._update_pending(snap, events)
        if self.current_pair is not None:
            self.pair_by_turn[self.turn] = self.current_pair
        return events

    def pair_history(self) -> list | None:
        """試合の最初からこれまでに置いた組ぷよ(手番 0〜turn-1、画面の色)。途中の手番が抜けていれば None
        (配ぷよ読みは途中が欠けると照合できないので送らない)。"""
        hist = [self.pair_by_turn.get(t) for t in range(self.turn)]
        return None if any(p is None for p in hist) else hist

    # ---- ネクスト → 手番の切り替わり ----

    def _update_next(self, snap: Snapshot, events: list[TrackerEvent]) -> None:
        if not (pair_known(snap.next1) and pair_known(snap.next2)):
            if self._next_unknown_since is None:
                self._next_unknown_since = snap.t
            return
        unknown_for = 0.0 if self._next_unknown_since is None else snap.t - self._next_unknown_since
        self._next_unknown_since = None
        # 「ネクスト不明N秒」は spawn の注記だと安定待ちの間に 0 に戻ってしまうので、解消した時点で出す
        if unknown_for >= 0.5:
            events.append(TrackerEvent("next_recovered", snap.t, f"unknown_for={unknown_for:.3f}s"))
        if self.next2 is None and self.next1 is not None:
            # 代替検出で手番を進めた後: ネクネクが読めるようになったので補う
            if snap.next1 == self.next1:
                self.next2 = snap.next2
                self._next_candidate = (snap.next1, snap.next2, NEXT_STABLE_FRAMES)
            return
        cand = self._next_candidate
        if cand is not None and cand[0] == snap.next1 and cand[1] == snap.next2:
            self._next_candidate = (cand[0], cand[1], cand[2] + 1)
        else:
            self._next_candidate = (snap.next1, snap.next2, 1)
        if self._next_candidate[2] < NEXT_STABLE_FRAMES:
            return
        n1, n2, _ = self._next_candidate
        if self.next1 is None:
            self.next1, self.next2 = n1, n2
        elif n1 != self.next1 and self.turn == 0 and count_puyos(settle_cells(snap.cells)) == 0:
            # 開始前〜第1手の設置前(何も置かれていない)の枠の入れ替わり: spawn にしない(開始前の画面では
            # 枠が数秒ごとに入れ替わることがあり、spawn にすると出現しない組ぷよの手番ができる 2026-09-20 16:01)。
            # 操作中の組ぷよが未知なら旧 next1 が出現するはずの第1手なので覚えておき、落ちてきた2個と照合する
            self._next_mismatch_frames = 0
            self._prestart_swap_t = snap.t
            if self.current_pair is None:
                self._prestart_first = self.next1
            self.next1, self.next2 = n1, n2
            events.append(TrackerEvent("next_prestart", snap.t, f"first={self._prestart_first} next={n1} nextnext={n2} (開始前の枠の入れ替わり)"))
        elif n1 != self.next1:
            # ネクスト枠が入れ替わった = 組ぷよを設置し、新しい組ぷよが出た。
            # 本物の入れ替わりでは「新しいネクスト = 以前のネクネク」になるはず。
            # そうでなければ認識のちらつき(実機で spawn が3連発した)の可能性が高いので、
            # しばらく続いた場合だけ受け入れる
            if n1 != self.next2:
                self._next_mismatch_frames += 1
                if self._next_mismatch_frames < NEXT_RESYNC_FRAMES:
                    return
            self._next_mismatch_frames = 0
            # 操作中の組ぷよは以前のネクスト
            self._last_placed_pair = self.current_pair
            self.current_pair = self.next1
            self.next1, self.next2 = n1, n2
            self.turn += 1
            self._placed_by_board = False
            self._initial_pair_pending = False
            self._begin_settling(snap.t)
            unknown_note = f" (ネクスト不明{unknown_for:.1f}s)" if unknown_for >= 0.5 else ""
            covered_note = f" (演出で未読{self._covered_skips}フレーム)" if self._covered_skips >= 10 else ""
            self._covered_skips = 0
            events.append(TrackerEvent("spawn", snap.t, f"turn={self.turn} current={self.current_pair} next={self.next1} nextnext={self.next2}{unknown_note}{covered_note}"))
        elif n2 != self.next2 and self.next1 == self.next2:
            if self.turn == 0 and self.current_pair is None and count_puyos(settle_cells(snap.cells)) == 0:
                # 開始前〜開始: 第1手と第2手が同じ組ぷよ(AA-AA など)だと、開始の入れ替わりでネクスト枠の表示が変わらず
                # ネクネクだけが変わる。手番を進めず開始前の入れ替わりとして扱う(実機 2026-10-11: 初手を2手目と判定し、
                # 先読みが無いまま思考を待って約0.6秒遅れた)
                self._next_mismatch_frames = 0
                self._prestart_swap_t = snap.t
                self._prestart_first = self.next1
                self.next2 = n2
                events.append(TrackerEvent("next_prestart", snap.t, f"first={self._prestart_first} next={self.next1} nextnext={n2} (開始前の枠の入れ替わり、同じ組ぷよ)"))
                return
            if self._placed_by_board and not self._current_pair_placed(snap):
                # 盤面から設置を検出して既に手番を進めており、操作中の組ぷよはまだ置かれていない:
                # 枠の表示が追いついただけなので二重に数えない(置かれていれば本物の次の spawn)
                self.next2 = n2
                self._placed_by_board = False
                events.append(TrackerEvent("next_sync", snap.t, f"nextnext={n2} (盤面検出済みの手番にネクスト枠が追いついた)"))
                return
            # ネクストとネクネクが同じ組ぷよ(例: 黄黄・黄黄)だと、設置してもネクスト枠の
            # 表示が変わらず spawn を見落とす(実機で「同じ色が2手連続」の時に提案がズレた)。
            # この場合はネクネクの変化で spawn を検出する(new next1 = 旧 next2 = 旧 next1 なので整合)
            self._next_mismatch_frames = 0
            self._last_placed_pair = self.current_pair
            self.current_pair = self.next1
            self.next2 = n2
            self.turn += 1
            self._initial_pair_pending = False
            self._begin_settling(snap.t)
            covered_note = f" (演出で未読{self._covered_skips}フレーム)" if self._covered_skips >= 10 else ""
            self._covered_skips = 0
            events.append(TrackerEvent("spawn", snap.t, f"turn={self.turn} current={self.current_pair} next={self.next1} nextnext={self.next2} (same-pair){covered_note}"))
        else:
            self._next_mismatch_frames = 0
            if n2 != self.next2:
                self.next2 = n2

    def _current_pair_placed(self, snap: Snapshot) -> bool:
        """観測盤面が「確定盤面+操作中の組ぷよの2個」になっている(=操作中の組ぷよが置かれた)か。"""
        if self.current_pair is None:
            return False
        settled = settle_cells(drop_top_eyeless(clear_spawn_cells(snap.cells), snap.eyeless_cells))
        added = [settled[r][c] for r in range(ROWS) for c in range(COLS) if settled[r][c] in GAME_COLORS and self.settled[r][c] == EMPTY]
        removed = any(self.settled[r][c] != EMPTY and settled[r][c] == EMPTY for r in range(1, ROWS) for c in range(COLS))
        return len(added) == 2 and sorted(added) == sorted(self.current_pair) and not removed

    def _diag_spawn_wait(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent], watching: bool) -> None:
        """診断: 操作中の組ぷよが着地したのに次の spawn を検出できない状態が続いたら、ネクスト枠の読み取りと
        代替検出(_detect_placement_without_next)の各条件を spawn_wait として記録する(手番の判定は変えない)。"""
        if self.chain_active or count_puyos(settled) < count_puyos(self.settled) + 2:
            self._spawn_wait_since = None
            return
        if self._spawn_wait_since is None:
            self._spawn_wait_since = snap.t
            self._spawn_wait_frames = 0
            self._spawn_wait_logged = 0
        self._spawn_wait_frames += 1
        waited = snap.t - self._spawn_wait_since
        if waited < SPAWN_WAIT_DIAG_SECONDS or self._spawn_wait_logged >= SPAWN_WAIT_DIAG_MAX:
            return
        if self._spawn_wait_frames % SPAWN_WAIT_DIAG_EVERY != 0:
            return
        self._spawn_wait_logged += 1
        added = sorted(settled[r][c] for r in range(ROWS) for c in range(COLS) if settled[r][c] in GAME_COLORS and self.settled[r][c] == EMPTY)
        removed = any(self.settled[r][c] != EMPTY and settled[r][c] == EMPTY for r in range(1, ROWS) for c in range(COLS))
        floating = sorted(f[2] for f in floating_puyos(snap.cells, settled) if f[2] in GAME_COLORS)
        cand = self._next_candidate
        placed = None if self._placed_candidate is None else round(snap.t - self._placed_candidate[1], 2)
        covered = is_covered(clear_spawn_cells(snap.cells), snap.eyeless_cells)
        events.append(TrackerEvent(
            "spawn_wait", snap.t,
            f"#{self._spawn_wait_logged} 着地から{waited:.2f}s 枠の読み={snap.next1},{snap.next2} 覚えている現在={self.current_pair} "
            f"次={self.next1} 次々={self.next2} 枠の候補={cand} 枠不明から={self._next_unknown_since} 演出で覆い={covered} "
            f"| 代替検出: 増えた={added}(現在の2色={sorted(self.current_pair) if self.current_pair else None}) 消えた={removed} "
            f"浮いている={floating} 候補の経過={placed} 連鎖監視={watching} 連鎖後の待ち={self._await_spawn_after_chain} "
            f"盤面検出済み={self._placed_by_board}"
        ))

    def _detect_placement_without_next(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent]) -> None:
        """ネクスト枠の変化に頼らない設置検出。操作中に、確定盤面より「操作中の組ぷよの色2個」だけ増えた
        観測が PLACEMENT_FALLBACK_SECONDS 続いたら設置とみなし、ネクスト変化と同じように手番を進める。

        - ネクストが読めない間(演出で枠が隠れる)は従来どおり盤面だけで判定する。
        - ネクストが読めていても枠が変わらない場合がある(同一ツモが3組以上続く: 旧 next1=A、旧 next2=A、
          その先も A なら設置後も両枠は A のまま。実機 2026-09-19 20:06 で BG,BG,BG が続き、2組目の手番が
          3組目の出現まで検出されず提案が1手遅れた)。この場合は「新しい組ぷよ(next1 の2色)が浮いて見える」
          ことを出現の証拠として要求し、接地中でまだ固定されていない2個を設置と誤認しないようにする。
        通常の検出(枠の変化、2フレーム)の方が速いので、枠が変われば通常の検出が先に手番を進める。"""
        if self.current_pair is None or self.next1 is None:
            self._placed_candidate = None
            return
        added = [settled[r][c] for r in range(ROWS) for c in range(COLS) if settled[r][c] in GAME_COLORS and self.settled[r][c] == EMPTY]
        removed = any(self.settled[r][c] != EMPTY and settled[r][c] == EMPTY for r in range(1, ROWS) for c in range(COLS))
        placed_now = len(added) == 2 and sorted(added) == sorted(self.current_pair) and not removed
        next_known = self._next_unknown_since is None
        if next_known and self.next1 != self.next2:
            # 枠が読めていて2つが違う組ぷよなら、出現すれば1つ目の枠に2つ目の組ぷよが入るので表示が必ず変わる。表示が変わらないのは
            # まだ出現していないということなので、盤面からは判定しない(実機 2026-10-12 07:24 試合2 turn=22: 相手の攻撃の光の
            # 演出を「上に浮いている次の組ぷよ」と読み、着地の0.3秒後に出現と判定してキーが空振りし、本物の組ぷよが自然落下した。
            # 21:30 のログでは盤面からの判定70回のうち53回が2つの枠が違う時だった)
            self._placed_candidate = None
            self._same_pair_wait = None
            self._same_pair_prev_placed = None
            return
        if next_known:
            # 枠が読めているのに変わらない(同一ツモが3組以上続く): 置いた組ぷよが確定盤面に入った状態を一度見たら、
            # そこから時間を測り、次の組ぷよ(next1 の色)が浮いて見えるフレームが SAME_PAIR_EVIDENCE_FRAMES 回あれば設置とみなす。
            # 雑音のフレーム(相手の攻撃の演出が紫に見える、関係ない色が混ざる、落下中の組ぷよが積みに触れて「増えた」が4個に
            # 見える)では計測をやり直さない(実機 2026-10-10 14:08・15:18: 揺れや雑音のたびに0.3秒の計測がやり直しになり、
            # 次の組ぷよが3.5秒以上自由落下した)
            self._placed_candidate = None
            if removed:
                self._same_pair_wait = None
                return
            # 起点は「置いた組ぷよが入った盤面」が2フレーム続けて見えた時(1フレームだけの雑音では始めない)。
            # 証拠は「上から SAME_PAIR_SPAWN_ROWS 段以内に浮いている next1 の色」だけ数える(新しい組ぷよは上に出る。
            # 同一ツモでは今落ちている組ぷよも同じ色なので、盤面の中ほどを落ちているぷよを証拠にすると手番を二重に進める)
            st = self._same_pair_wait
            prev_placed = self._same_pair_prev_placed
            self._same_pair_prev_placed = settled if placed_now else None
            if placed_now and prev_placed == settled and (st is None or st[0] != settled):
                st = (settled, snap.t, 0)
            if st is None:
                return
            # 出現位置(3列目の12段目)に、置いた組ぷよが入った盤面には無いネクストの色が見えるのも証拠にする(2026-10-24: 3列目が
            # 11段なら新しい組ぷよは出現位置で積みに乗り、浮いて見えない)
            sr, sc = SPAWN_CELLS[1]
            at_spawn = snap.cells[sr][sc] in self.next1 and st[0][sr][sc] == EMPTY
            if at_spawn or any(r < SAME_PAIR_SPAWN_ROWS and c in self.next1 for r, _, c in floating_puyos(snap.cells, settled)):
                st = (st[0], st[1], st[2] + 1)
            self._same_pair_wait = st
            if snap.t - st[1] < SAME_PAIR_FALLBACK_SECONDS or st[2] < SAME_PAIR_EVIDENCE_FRAMES:
                return
            self._same_pair_wait = None
        else:
            if not placed_now:
                self._placed_candidate = None
                return
            cand = self._placed_candidate
            if cand is None or cand[0] != settled:
                self._placed_candidate = (settled, snap.t)
                return
            if snap.t - cand[1] < PLACEMENT_FALLBACK_SECONDS:
                return
        self._advance_by_board(snap, events, next_known, "盤面から設置を検出")

    def _detect_same_pair_spawn_after_chain(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent]) -> None:
        """連鎖の後の出現待ちで、ネクストとネクネクが同じ組ぷよ(出現してもネクスト枠の表示が変わらない)なら、新しい組ぷよが
        見えたことで出現とみなす(2026-10-24、実機 08:05 turn=34〜38: 黄緑が5手続き、1連鎖の後の出現を一度も検出できずに窒息した)。
        証拠は「上から SAME_PAIR_SPAWN_ROWS 段以内に浮いているネクストの色」か「出現位置(3列目の12段目)に、確定盤面には無い
        ネクストの色」(3列目が11段なら組ぷよは出現位置で積みに乗り、浮いて見えない)。SAME_PAIR_EVIDENCE_FRAMES 回で進める。"""
        if (
            self.chain_active
            or self.current_pair is None
            or self.next1 is None
            or self._next_unknown_since is not None
            or self.next1 != self.next2
        ):
            self._after_chain_evidence = 0
            return
        colors = set(self.next1)
        floating = any(r < SAME_PAIR_SPAWN_ROWS and c in colors for r, _, c in floating_puyos(snap.cells, settled))
        r, c = SPAWN_CELLS[1]
        at_spawn = snap.cells[r][c] in colors and self.settled[r][c] == EMPTY
        self._after_chain_evidence = self._after_chain_evidence + 1 if (floating or at_spawn) else 0
        if self._after_chain_evidence < SAME_PAIR_EVIDENCE_FRAMES:
            return
        self._after_chain_evidence = 0
        self._advance_by_board(snap, events, True, "連鎖の後、同じ組ぷよが続き枠が変わらないので新しい組ぷよが見えたことで検出")

    def _advance_by_board(self, snap: Snapshot, events: list[TrackerEvent], next_known: bool, source: str) -> None:
        """ネクスト枠の変化を使わずに手番を進める(盤面から設置・出現を検出した時)"""
        self._placed_candidate = None
        self._initial_pair_pending = False
        self._last_placed_pair = self.current_pair
        self.current_pair = self.next1
        if next_known:
            # 枠はまだ旧表示のまま: ネクネクは「旧 next2」、その先は枠の2つ目が旧 next2 と同じ表示なら
            # (同一ツモの連続)その値、違えば不明
            old_next2 = self.next2
            self.next1 = old_next2
            self.next2 = snap.next2 if (old_next2 is not None and snap.next1 == old_next2 and pair_known(snap.next2)) else None
            self._placed_by_board = True
            why = f"ネクスト枠は {snap.next1},{snap.next2} のまま"
        else:
            self.next1, self.next2 = self.next2, None
            why = f"ネクスト不明{snap.t - self._next_unknown_since:.1f}s"
        self.turn += 1
        self._begin_settling(snap.t)
        covered_note = f" 演出で未読{self._covered_skips}フレーム" if self._covered_skips >= 10 else ""
        self._covered_skips = 0
        events.append(TrackerEvent("spawn", snap.t, f"turn={self.turn} current={self.current_pair} next={self.next1} nextnext={self.next2} ({source}: {why}{covered_note})"))

    def _begin_settling(self, t: float) -> None:
        self.settling = True
        self.settling_since = t
        self._spawn_wait_since = None
        self._same_pair_wait = None
        self._same_pair_prev_placed = None
        self._await_spawn_after_chain = False
        self._loss_candidate = None
        if self.chain_active:
            # spawn 無しで検出した連鎖の途中(または終了直前)に spawn が来た: 連鎖の状態(chain_before・
            # chain_bonus・消えた個数)を消さない。all_clear・chain_end は _take_in で一度だけ確定する
            # 出現は連鎖が終わった証拠なので、盤面が落ち着いていれば連鎖の終わりの取り込みで見込みの待ちをしない(_spawned_in_chain)
            self._spawned_in_chain = True
            self._spawned_in_chain_t = t
            return
        self._trigger_candidate = None
        # 連鎖で盤面が空になった直後に次の spawn が来ると chain_end の前に連鎖状態が消える。
        # その場合も全消しボーナスは記録する(実機 2026-09-17: 4個→0個の全消しが記録されなかった)
        if self.chain_active and self._settled_count == 0:
            self.all_clear = True
        self.chain_active = False
        self.chain_bonus = False
        self.chain_steps = 0
        self.chain_expected = None
        self._last_chain_step_t = -1.0
        # 取り込み待ちは「spawn 後に観測された盤面」から数える
        self._observed = None
        self._observed_candidate = None
        self._observed_since = t

    # ---- 盤面 ----

    def _update_board(self, snap: Snapshot, events: list[TrackerEvent]) -> None:
        # 再現した連鎖の段数まで消えたのを見た後に次の組ぷよが出た: 連鎖は終わっているので、盤面が落ち着くのを待たずに
        # 再現した連鎖後の盤面で取り込む(実機 2026-10-24 16:47 turn=36: 高い積みの1連鎖の後、消えた位置に重なる
        # 「1れんさ!」の文字と×印をぷよと読んで盤面の読みが毎フレーム揺れ、連鎖中のまま組ぷよが自由落下して窒息した)。
        # 段が消えたのと同時の出現(SPAWN_AFTER_STEP_MIN_SECONDS 未満。段の途中・ネクスト枠の読み違いのことがある)と、
        # 再現が全消しの時(偽の全消しを避けるため、空の盤面を見て確定する)は従来どおり盤面を待つ
        exp = self.chain_expected
        if (
            self.chain_active and self._spawned_in_chain and exp is not None and self.chain_steps == exp.steps
            and self._spawned_in_chain_t - self._last_chain_step_t >= SPAWN_AFTER_STEP_MIN_SECONDS and count_puyos(exp.final) > 0
        ):
            self._take_in(snap, [row[:] for row in exp.final], events, "chain_result")
            return
        if self.chain_active and self._spawned_in_chain and snap.t - self._spawned_in_chain_t >= SPAWNED_CHAIN_FORCE_END_SECONDS:
            # 再現した連鎖後の盤面があればそれを(全消しの再現は偽の全消しを避けるため使わない)、無ければ直近の観測を取り込む
            if exp is not None and count_puyos(exp.final) > 0:
                board = exp.final
            else:
                board = self.observed_cells if self.observed_cells is not None else self.settled
            self._take_in(snap, [row[:] for row in board], events, "spawned_in_chain")
            return
        # 出現位置の組ぷよは確定盤面に関係ないので、そこが unknown でも盤面更新を止めない
        cells = clear_spawn_cells(snap.cells)
        # 演出に覆われている(全消し直後のキャラクターのカットイン等: 顔や服が色ぷよとして読まれ、
        # 偽の盤面が取り込まれた 2026-09-17)フレームは unknown と同じく盤面更新に使わない
        covered = is_covered(cells, snap.eyeless_cells)
        self.last_unknown = covered or has_blocking_unknown(cells)
        self.last_raw_text = "\n".join("".join(_SHORT.get(c, "?") for c in row) for row in cells)
        if covered:
            if self._covered_since is None:
                self._covered_since = snap.t
            elif (
                not self.all_clear
                and self.turn >= 1
                and snap.t - self._covered_since >= ALL_CLEAR_CUTIN_SECONDS
                and ALL_CLEAR_CUTIN_MIN_PUYOS <= self._settled_count <= ALL_CLEAR_CUTIN_MAX_PUYOS
            ):
                self._all_clear_pending = True
        else:
            self._covered_since = None
        if self.last_unknown:
            self._loss_candidate = None
            self._observed_candidate = None  # 遮蔽・不明の前後を連続フレームと数えない
            return
        # 積みの最上部に乗った演出(目の無い色)は盤面に入れない(drop_top_eyeless)。連鎖中は除く: 消える途中のぷよは
        # 泡になって目が無くなり、最上部のものを先に消すと段が割れて数えられ、再現の点数が採用されなかった(録画 09-13 の 88.4s)
        if not self.chain_active:
            cells = drop_top_eyeless(cells, snap.eyeless_cells)
        if self._all_clear_pending:
            self._all_clear_pending = False
            if count_puyos(settle_cells(cells)) <= ALL_CLEAR_AFTER_MAX_PUYOS:
                self.all_clear = True
                events.append(TrackerEvent("all_clear", snap.t, "(カットインから推定)"))
        settled = settle_cells(cells)
        self.last_observed_text = "\n".join("".join(c[0].upper() if c != EMPTY else "." for c in row) for row in settled)
        self.observed_count = count_puyos(settled)
        self.observed_cells = settled
        eyeless = [(r, c) for r, c in snap.eyeless_cells if cells[r][c] != EMPTY]
        if eyeless:
            eyed = [row[:] for row in cells]
            for r, c in eyeless:
                eyed[r][c] = EMPTY
            self.observed_cells_eyed = settle_cells(eyed)
        else:
            self.observed_cells_eyed = settled
        self.floating_now = floating_puyos(cells, settled)

        # 操作中の組ぷよが未知(開始直後)なら、まず3列目の上に現れた軸ぷよ1個で開始を判定し、だめなら浮いている2個から読む
        if self.current_pair is None and not self._detect_start_by_axis(snap, settled, events):
            self._infer_current_pair(snap, settled, events)

        self._remember_trigger(snap, settled)
        if self._detect_flash(snap, settled, events):
            return
        if not self.settling:
            watching = self._watch_unannounced_chain(snap, settled)
            self._diag_spawn_wait(snap, settled, events, watching)
            if not watching:
                # 連鎖終了後に旧 current_pair で手番を進めない(次の spawn を待つ)
                if not self._await_spawn_after_chain:
                    self._detect_placement_without_next(snap, settled, events)
                else:
                    self._detect_same_pair_spawn_after_chain(snap, settled, events)
                return

        settled = strip_moving_pair(settled, self.settled, self.current_pair)
        settled = self._apply_hidden(settled)

        # 先読みが当たった: 1フレームでも一致すれば即取り込む(13段目は見えないので比較から除く)
        if not self.chain_active and self._expected is not None and settled[1:] == self._expected[1:] and self._observed_candidate is not None:
            self._take_in(snap, [row[:] for row in self._expected], events, "expected")
            return

        cand = self._observed_candidate
        if cand is not None and cand[0] == settled:
            self._observed_candidate = (cand[0], cand[1] + 1)
        else:
            self._observed_candidate = (settled, 1)
        if self._observed_candidate[1] < OBSERVED_STABLE_FRAMES:
            return

        if settled != self._observed:
            self._observed = settled
            self._observed_since = snap.t
            count = count_puyos(settled)
            min_loss = UNANNOUNCED_MIN_LOSS if self._await_spawn_after_chain and not self.chain_active else CHAIN_MIN_DECREASE
            if (
                count <= self._settled_count - min_loss
                and not self.chain_active
                and self._raised_baseline_poppable is False
                and count >= count_puyos(self.settled)
            ):
                # 引き上げた基準の観測には消える組が無く、確定盤面のぷよも減っていない = 一時的な見かけの
                # 増加(落下中のおじゃま・操作中の組ぷよの重なり)が消えただけ。消去ではないので基準を戻す
                # (本物の消去なら確定盤面のぷよが巻き込まれて確定個数を下回る)
                self._settled_count = count
                self._raised_baseline_poppable = None
                return
            if count <= self._settled_count - min_loss:
                # 確定盤面より減った = 消去が起きた
                if not self.chain_active:
                    self.chain_active = True
                    self.chain_started_t = snap.t
                    self.chain_steps = 0
                    self.chain_dropped = 0
                    self.chain_step_drops = []
                    self._last_chain_step_t = -1.0
                    self.chain_bonus = self.all_clear  # この連鎖にボーナスが乗る
                    self.all_clear = False  # 連鎖で全消しボーナスは消費される
                    self._recent_boards.clear()
                    trigger = self._trigger_candidate
                    if trigger is not None and 0.0 <= snap.t - trigger[1] <= TRIGGER_CANDIDATE_MAX_AGE_SECONDS:
                        self.chain_before = [row[:] for row in trigger[0]]
                        self.chain_before_source = "observed_trigger_candidate"
                        self.chain_expected = simulate_chain(self.chain_before)
                    else:
                        self.chain_before = [row[:] for row in self.settled]
                        self.chain_before_source = "settled_without_trigger"
                        self.chain_expected = None  # 引き金が無い盤面では再現できない
                    self._chain_last_change_t = snap.t
                    self._trigger_candidate = None
                    note = "(全消しボーナス付き)" if self.chain_bonus else ""
                    if self._await_spawn_after_chain:
                        note += "(spawn無しで検出)"
                    if self.chain_expected is not None:
                        note += f"(引き金込み盤面から再現 {self.chain_expected.steps}連鎖 {self.chain_expected.score:,}点)"
                    events.append(TrackerEvent("chain_start", snap.t, note))
                dropped_now = self._settled_count - count
                self.chain_dropped += dropped_now
                # 残りの連鎖の見積もりに使う途中盤面は、浮いているぷよ(落下中)が無い観測だけ
                # (落下中の上部が欠けた盤面を消すと実際には起きない連鎖を数え、見込みが数倍揺れた)
                if not floating_puyos(cells, settled):
                    self.chain_observed = [row[:] for row in settled]
                if snap.t - self._last_chain_step_t > CHAIN_STEP_MERGE_SECONDS:
                    self.chain_steps += 1
                    self.chain_step_drops.append(dropped_now)
                else:
                    self.chain_step_drops[-1] += dropped_now  # 点滅で段が合流した分は同じ段に足す
                # 通知は段が分かれた場合も合流した場合も出す(1019cac で else 側だけになり、実機 18:35 のログで
                # chain_step が 0 件 = 8個消去の再評価・枠更新・攻撃記録が動かなかった。分析文書第3弾 P0-1)
                events.append(TrackerEvent("chain_step", snap.t, f"chain={self.chain_steps} puyos {self._settled_count}->{count}"))
                self._last_chain_step_t = snap.t
                self._chain_last_change_t = snap.t
                self._settled_count = count
                return
            # 連鎖もおじゃま落下も起きない単純な設置なら安定待ちなしで取り込む
            if (
                not self.chain_active
                and count == self._settled_count + 2
                and count_garbage(settled) == count_garbage(self.settled)
                and self.pending_garbage == 0
                and not has_poppable_group(settled)
            ):
                self._take_in(snap, settled, events, "simple")
            elif not self.chain_active and self._garbage_landed_reason(settled) is None:
                # 設置の2個に加えておじゃまが降り終わった盤面: ゲームはおじゃまが降り終わってから次の組ぷよを出すので、
                # 出現した時点で浮いているのが操作中の組ぷよだけなら盤面は確定している。安定待ち(0.3秒)をしない
                # (実機 2026-10-11 11:07: おじゃまの後の手番は出現から取り込みまで中央0.37秒、その間に組ぷよが自由落下した)
                self._take_in(snap, settled, events, "garbage")
            elif count > self._settled_count and not self.chain_active:
                # 設置後(消去前)の盤面。連鎖の減少はここを基準に数える
                self._settled_count = count
                self._raised_baseline_poppable = has_poppable_group(settled)
            return

        since = snap.t - self._observed_since
        if not self.chain_active and self.settling and count_garbage(settled) > count_garbage(self.settled):
            # 盤面の読みが変わらなくても、浮いているもの(落下中のおじゃま・演出)は毎フレーム変わるので、おじゃまが降り終わったかを
            # 毎フレーム見直す(読みが変わった瞬間だけ見ていたので、その時に浮いているものが残っていると安定待ちの0.3秒を待ち、
            # 高い積みで組ぷよが自由落下した。実機 2026-10-26 20:32 の 2P 465s: 出現から取り込みまで0.53秒)
            why = self._garbage_landed_reason(settled)
            if why is None:
                self._take_in(snap, settled, events, "garbage")
                return
            self.last_garbage_wait = why
        if since < SETTLE_SECONDS:
            return
        self.last_poppable = has_poppable_group(settled)
        # 連鎖中に次の組ぷよが出た(連鎖は終わっているはず)なら、盤面が SPAWNED_CHAIN_QUIET_SECONDS 変わらなければ待たない
        # (ネクスト枠の読み違いで連鎖の途中に出現を検出することもあるので、すぐには終えない。段の間の点滅は約0.5秒)
        spawned = self.chain_active and self._spawned_in_chain and since >= SPAWNED_CHAIN_QUIET_SECONDS
        if self.last_poppable and not spawned:
            if self._poppable_since is None:
                self._poppable_since = self._observed_since
            if since < POPPABLE_MAX_WAIT_SECONDS and snap.t - self._poppable_since < POPPABLE_MAX_WAIT_SECONDS:
                return  # まだ消える(連鎖が続く)
        else:
            self._poppable_since = None
        # 再現した連鎖の見込みより早く終わっても、次の組ぷよが出ていれば待たない(2026-10-24、実機 08:13 turn=40: 再現は3連鎖、実際は
        # 2連鎖で終わり、見込みの3段目を4秒待つ間に組ぷよが2つ出て、1つは自由落下した)
        if (
            self.chain_active
            and not spawned
            and self.chain_expected is not None
            and count_colored(settled[1:]) > count_colored(self.chain_expected.final[1:])
            and not self._landed_on_chain_result(settled)
        ):
            remaining = max(0, self.chain_expected.steps - self.chain_steps)
            wait = max(CHAIN_EXPECTED_MAX_WAIT_SECONDS, remaining * CHAIN_EXPECTED_SECONDS_PER_STEP)
            if snap.t - self._chain_last_change_t < wait:
                return  # 再現した連鎖後の盤面より色ぷよが多い = まだ連鎖の途中(段の間の落下で安定して見えただけ)
            self.chain_expected = None  # 減少が無いまま待ち切った: 再現が間違っていたので観測の段数で確定する

        # 観測が安定した: 確定盤面として取り込む
        self._take_in(snap, settled, events, "stable")

    def _garbage_landed_reason(self, settled: list[list[str]]) -> str | None:
        """設置の2個とおじゃまが降り終わった盤面なら None。そうでなければ、まだ取り込めない理由(ログ用)。"""
        if count_garbage(settled) <= count_garbage(self.settled):
            return "おじゃまが増えていない"
        if count_colored(settled) != count_colored(self.settled) + 2:
            return f"色ぷよが{count_colored(settled) - count_colored(self.settled):+d}個(設置の2個と合わない)"
        if has_poppable_group(settled):
            return "消える組がある"
        if any(color == GARBAGE for _, _, color in self.floating_now):
            return "落下中のおじゃまがある"
        floating = sum(1 for _, _, color in self.floating_now if color in GAME_COLORS)
        if floating > 2:
            return f"浮いている色が{floating}個"
        return None

    def _with_hidden(self, board: list[list[str]]) -> list[list[str]]:
        """記憶している13段目のぷよを盤面に重ねた写し(記憶は変えない)。列の12段目が空なら重ねない(落ちたはず)。"""
        cols = [c for c in range(COLS) if self._hidden[c] != EMPTY and board[1][c] != EMPTY and board[0][c] == EMPTY]
        if not cols:
            return board
        out = [row[:] for row in board]
        for c in cols:
            out[0][c] = self._hidden[c]
        return out

    def _restore_invisible(self, board: list[list[str]]) -> list[list[str]]:
        """引き金を置いた盤面(点滅の直前・引き金の候補)に、読めないマス(13段目・3列目12段目の出現位置)のぷよを補う。
        13段目は記憶から重ねる。引き金の組ぷよが1個しか見えなければ、もう1個は読めないマスに着地したと推定する
        (横の列か真上で、着地できる読めないマスが1つに絞れる時だけ)。記憶は変えない。"""
        out = self._with_hidden(board)
        pair = self._last_placed_pair if self.settling else self.current_pair
        if pair is None:
            return out
        added = [(r, c) for r in range(ROWS) for c in range(COLS) if out[r][c] in GAME_COLORS and self.settled[r][c] == EMPTY]
        if len(added) != 1 or out[added[0][0]][added[0][1]] not in pair:
            return out
        r, c = added[0]
        other = pair[1] if out[r][c] == pair[0] else pair[0]
        hidden_cells = [(0, cc) for cc in range(COLS)] + [SPAWN_CELLS[1]]
        cands = []
        for cc in (c - 1, c, c + 1):
            if not 0 <= cc < COLS:
                continue
            land = max((rr for rr in range(ROWS) if out[rr][cc] == EMPTY), default=-1)  # 列の一番下の空きマス
            if land >= 0 and (land, cc) in hidden_cells and (cc != c or land == r - 1):
                cands.append((land, cc))
        if len(cands) != 1:
            return out
        out = [row[:] for row in out]
        out[cands[0][0]][cands[0][1]] = other
        return out

    def _apply_hidden(self, settled: list[list[str]]) -> list[list[str]]:
        """記憶している13段目のぷよを観測盤面に重ねる。列の12段目が空なら(落ちたので)記憶を消す。"""
        out = None
        for col in range(COLS):
            if self._hidden[col] == EMPTY:
                continue
            if settled[1][col] == EMPTY:
                self._hidden[col] = EMPTY
                continue
            if out is None:
                out = [row[:] for row in settled]
            out[0][col] = self._hidden[col]
        return out if out is not None else settled

    def _infer_hidden_placement(self, settled: list[list[str]]) -> list[list[str]]:
        """設置で色ぷよが1個しか増えず、その列が12段目まで埋まっていれば、
        組ぷよのもう一方が13段目にあると推定して盤面に加える。"""
        if self._last_placed_pair is None:
            return settled
        added = [
            (r, c)
            for r in range(1, ROWS)
            for c in range(COLS)
            if settled[r][c] in GAME_COLORS and self.settled[r][c] == EMPTY
        ]
        if len(added) != 1:
            return settled
        r, c = added[0]
        if r != 1 or settled[0][c] != EMPTY:
            return settled
        child, axis = self._last_placed_pair
        visible = settled[r][c]
        # 回転なしの縦置きなら下が軸・上が子。見えている方と違う色を13段目とみなす
        # (同色なら同じ色)
        hidden_color = child if visible == axis else axis
        self._hidden[c] = hidden_color
        out = [row[:] for row in settled]
        out[0][c] = hidden_color
        return out

    def _landed_on_chain_result(self, settled: list[list[str]]) -> bool:
        """再現した連鎖後の盤面の上に、次の組ぷよ(色ぷよ2個まで)やおじゃまが乗っただけの観測か。

        連鎖が終わると次の組ぷよが出て着地し、色ぷよが再現後より増える。個数だけで判定すると
        4秒待ち切るまで連鎖中のままになり、その間の組ぷよが思考されず自由落下した
        (実機 2026-09-27 12:25: 1連鎖の終了が3回とも4.0秒後)。連鎖の途中なら次に消える4個以上が
        余分に残っているので、増えた色ぷよが2個以下なら連鎖は終わっている。"""
        final = self.chain_expected.final
        # 13段目は見えない(再現では13段目に残っても、記憶とずれることがある)ので12段目まで比べる
        same = all(settled[r][c] == final[r][c] for r in range(1, ROWS) for c in range(COLS) if final[r][c] != EMPTY)
        return same and count_colored(settled[1:]) - count_colored(final[1:]) <= 2

    def _take_in(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent], reason: str) -> None:
        self._spawned_in_chain = False
        if reason == "expected":
            # 期待盤面には13段目が含まれているので、その内容を記憶に引き継ぐ
            self._hidden = list(settled[0])
        elif not self.chain_active:
            settled = self._infer_hidden_placement(settled)
        elif self.chain_expected is not None and self.chain_expected.steps >= self.chain_steps:
            # 連鎖の終わり: 下が消えると13段目のぷよも落ちるので、再現した連鎖後の盤面の13段目を記憶にする
            # (12段目が空の列は、見えている盤面と再現がずれているので記憶しない)
            final = self.chain_expected.final
            self._hidden = [final[0][c] if settled[1][c] != EMPTY else EMPTY for c in range(COLS)]
            settled = [list(self._hidden)] + [row[:] for row in settled[1:]]
        garbage_before = count_garbage(self.settled)
        garbage_after = count_garbage(settled)
        placed_initial = False
        if self._initial_pair_pending and self.current_pair is not None:
            added = sorted(settled[r][c] for r in range(ROWS) for c in range(COLS) if settled[r][c] in GAME_COLORS and self.settled[r][c] == EMPTY)
            placed_initial = added == sorted(self.current_pair)
            self._initial_pair_pending = False
        # 攻撃の端数点: 連鎖なしで置いた手は落とした段数を足す(連鎖の手は連鎖の終わりで数える)
        if not self.chain_active:
            rows = placement_drop_rows(self.settled, settled)
            if rows is not None:
                self.carry += rows
        prev_settled = self.settled
        self.settled = settled
        self.last_take_in_t = snap.t
        self.last_take_in_reason = reason
        self._poppable_since = None
        self._settled_count = count_puyos(settled)
        self._raised_baseline_poppable = None
        self._recent_boards.clear()  # 点滅検出の履歴は確定盤面より後の観測だけ
        self.settling = False
        # 自分の連鎖の終わりの取り込みなら、その盤面を次の手番の予想盤面にする(次の組ぷよが出ても盤面は変わらない
        # ので安定待ちせずに取り込む。おじゃまが降れば一致しないので従来どおり。実機 2026-10-10 14:08: 連鎖の後の
        # 取り込みが毎回0.34秒の安定待ちになり、設置が遅れていた)
        self._expected = [row[:] for row in settled] if self.chain_active else None
        if self.chain_active:
            self.chain_active = False
            self.chain_bonus = False
            exp = self.chain_expected
            if self._settled_count == 0:
                # 全消しは「発火直前盤面から再現した連鎖の結果が空」で、実画面も空の時だけ確定する。
                # 負けの時も盤面が(点滅なしで)一気に消えて連鎖として扱われるので、空盤面だけで確定すると
                # 次の試合を全消しボーナス持ちとして戦い、新しい試合の検出も止まった
                # (実機 2026-09-27: 4つのログの負け8回すべてで偽の全消し)
                if exp is not None and exp.steps >= 1 and count_puyos(exp.final) == 0:
                    self.all_clear = True
                    events.append(TrackerEvent("all_clear", snap.t))
                else:
                    reason = "発火直前盤面を再現できない" if exp is None else f"再現後に{count_puyos(exp.final)}個残る"
                    events.append(TrackerEvent("all_clear_unconfirmed", snap.t, f"盤面は空だが全消しにしない({reason}。負け・試合終了の可能性)"))
            # 攻撃の端数点: 引き金の手の落下ボーナス(置く前の確定盤面 → 発火直前の盤面)と連鎖の点を足した余り
            trigger_rows = placement_drop_rows(prev_settled, self.chain_before) if self.chain_before is not None else None
            self.carry = carry_after_chain(self.carry, trigger_rows, exp.score if exp is not None else None, self.target_point)
            if exp is not None and exp.steps >= self.chain_steps:
                # 復元した盤面から再現した連鎖数・点数を確定値にする(観測の段数は点滅の合流・落下の安定で不正確)
                self.chain_steps = exp.steps
                self.chain_score = exp.score
                detail = f"chain={exp.steps} dropped={self.chain_dropped} score={exp.score} (再現)"
            else:
                self.chain_score = 0
                detail = f"chain={self.chain_steps} dropped={self.chain_dropped}"
            self.chain_expected = None
            events.append(TrackerEvent("chain_end", snap.t, detail))
        if garbage_after > garbage_before:
            note = f"(安定待ちで取り込み: {self.last_garbage_wait})" if reason == "stable" and self.last_garbage_wait else ""
            events.append(TrackerEvent("garbage_rise", snap.t, f"garbage {garbage_before}->{garbage_after}{note}"))
        self.last_garbage_wait = ""
        if placed_initial and self.next1 is not None:
            # 開始時に推定した組ぷよが置かれた: ネクスト枠が変わっていなくても手番を進める
            # (枠が変わっていれば _update_next が先に進めているので、ここには来ない)
            self._last_placed_pair = self.current_pair
            self.current_pair = self.next1
            old_next2 = self.next2
            self.next1 = old_next2
            self.next2 = snap.next2 if (old_next2 is not None and snap.next1 == old_next2 and pair_known(snap.next2)) else None
            self.turn += 1
            self._placed_by_board = True
            events.append(TrackerEvent("spawn", snap.t, f"turn={self.turn} current={self.current_pair} next={self.next1} nextnext={self.next2} (開始時の組ぷよが置かれた: 枠は {snap.next1},{snap.next2} のまま)"))
        if self.current_pair is not None and not self._await_spawn_after_chain:
            events.append(TrackerEvent("turn_ready", snap.t, f"turn={self.turn} puyos={self._settled_count} ({reason})"))

    def _detect_start_by_axis(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent]) -> bool:
        """盤面が空の試合開始で、3列目の上(13〜11段目)に初手の軸ぷよの色が現れたら、その時点で開始とする。開始したら True。

        初手の組ぷよは軸が12段目(見える)・子が13段目(見えない)に出る。2個が見えるまで待つと、開始直後のゆっくりした落下で
        子が12段目に下りるまで約0.3秒かかり、さらに相手の初手の検出を待つので、キーを送るのが本当の開始から約0.45秒遅れた
        (実機 2026-10-10〜11 の50試合)。開始前の画面ではこのマスにぷよは出ない。窒息の位置の×印・演出は目が無いので除く。
        初手の候補は、相手の初手(start_hint)・直前の入れ替わりで枠から消えた組(_prestart_first)・枠の第1手(next1)。"""
        if self.turn != 0 or self.next1 is None or count_puyos(settled) != 0:
            return False
        color = None
        for r in range(0, 3):
            c = snap.cells[r][2]
            if c in GAME_COLORS and (r, 2) not in snap.eyeless_cells:
                color = c
                break
        if color is None:
            self._start_axis_candidate = None
            self._spawn_color_since = None
            return False
        # その色がいつから見えているか。開始より前から見えている色(窒息の位置の×印などを色ぷよと読んだもの)は軸ぷよではない
        # (実機 2026-10-11: ×印を赤と読み、初手を取り違えて2手目が約5秒操作されなかった。72試合中2試合)
        since = self._spawn_color_since
        if since is None or since[0] != color:
            self._spawn_color_since = since = (color, snap.t)
        cand = self._start_axis_candidate
        frames = cand[1] + 1 if cand is not None and cand[0] == color else 1
        self._start_axis_candidate = (color, frames)
        if frames < START_AXIS_FRAMES:
            return False
        recent_swap = self._prestart_swap_t is not None and snap.t - self._prestart_swap_t <= START_SWAP_RECENT_SECONDS
        if recent_swap:
            if since[1] < self._prestart_swap_t - 1e-9:
                return False  # 入れ替わりより前から見えていた色
        elif snap.t - since[1] > START_SWAP_RECENT_SECONDS:
            return False  # 入れ替わりが無いのに前から見えている色
        # 子も見えていれば(出現位置で12段目に下りた・検出が遅れた)2個で照合する。見えている一番上は子なので軸の色では照合できない
        seen = self._visible_start_pair(snap)
        options: list[tuple[str, Pair | None, Pair | None, Pair | None]] = []
        # 直前の入れ替わりで枠から消えた組を、相手の初手より先に試す。出現の瞬間の色の読みは外れやすく(軸が黄色だと子も黄色と
        # 読む)、相手の初手もその読み違いのことがある(実機 2026-10-25 12:33: 紫黄・青黄を黄黄と読み、先読みと合わず初手が約0.95秒
        # 遅れた。10月のログで相手の初手の読み違い 11回)。枠の読みは大きく安定している
        if recent_swap and self._prestart_first is not None:
            options.append(("直前の入れ替わりで枠から消えた組", self._prestart_first, self.next1, self.next2))
        if self.start_hint is not None:
            options.append(("相手の初手と同じ", *self.start_hint))
        if not recent_swap and seen is not None:
            # 入れ替わりを見ていない: 同じフレームで出現と入れ替わりが起きたなら枠の第1手が出現した(ネクストは枠の第2手、
            # ネクネクは未読)。ただし入れ替わりの後に追跡を作り直した(新しい試合のリセット)時や、検出が遅れて入れ替わりから
            # 時間が経った時は、枠の第1手はもう2手目で、取り違えると2手目が出ても枠が変わらず出現を検出できない(実機 2026-10-24
            # 00:58・07:08・07:49・13:47、2手目が最長7秒自由落下した)。軸だけでは区別できないので2個が見えるまで待ち、
            # 見えた2個が枠の第1手と違えば見えた2個を初手にする(ネクスト・ネクネクは枠の第1・2手)
            # 向きまで比べる(07:08 は初手 (青, 紫)・2手目 (紫, 青) で色の組は同じだった)
            if seen == self.next1:
                options.append(("枠の第1手", self.next1, self.next2, None))
            else:
                options.append(("見えた2個(枠の第1手は2手目)", seen, self.next1, self.next2))
        for source, cur, n1, n2 in options:
            if cur is None:
                continue
            framed = source == "直前の入れ替わりで枠から消えた組"
            if seen is not None:
                # 枠から消えた組は、見えた2個と1色でも同じなら信じる(見えた2個の片方が読み違いのことがある)
                if cur != seen and not (framed and set(cur) & set(seen)):
                    continue
                if cur != seen:
                    source += f"(見えた2個は {seen})"
            elif cur[1] != color and not (framed and color in cur):  # Pair は (子, 軸)
                continue
            self.current_pair, self.next1, self.next2 = cur, n1, n2
            self._start_axis_candidate = None
            self._initial_candidate = None
            self._prestart_first = None
            self._initial_pair_pending = True
            self.settling = False
            self._settled_count = 0
            events.append(TrackerEvent("spawn", snap.t, f"turn={self.turn} current={cur} next={n1} nextnext={n2} (initial: 3列目に軸ぷよが出現、{source})"))
            events.append(TrackerEvent("turn_ready", snap.t, f"turn={self.turn} puyos=0 (initial)"))
            return True
        return False

    @staticmethod
    def _visible_start_pair(snap: Snapshot) -> Pair | None:
        """試合開始で3列目の上に見えている初手の2個 (子, 軸)。出現時は子が軸の真上。一番上の色ぷよの真下も色ぷよの時だけ返す"""
        for r in range(0, 3):
            c = snap.cells[r][2]
            if c in GAME_COLORS and (r, 2) not in snap.eyeless_cells:
                below = snap.cells[r + 1][2]
                if below in GAME_COLORS and (r + 1, 2) not in snap.eyeless_cells:
                    return (c, below)
                return None
        return None

    def _infer_current_pair(self, snap: Snapshot, settled: list[list[str]], events: list[TrackerEvent]) -> None:
        floating = floating_puyos(snap.cells, settled)
        colored = [f for f in floating if f[2] in GAME_COLORS]
        if len(colored) != 2:
            return
        (r1, c1, k1), (r2, c2, k2) = sorted(colored)
        if (r1 == r2 and abs(c1 - c2) == 1) or (c1 == c2 and r2 - r1 == 1):
            colors = tuple(sorted((k1, k2)))
            first = self._prestart_first
            # 対戦の開始を確認済み(相手の初手を検出)で、2個が開始前の枠の第1手と同じ色なら、動くのを待たずに
            # 確定する。全自動では検出するまで操作を送らないので、動くのを待つと自然落下で1段下がるまで
            # (0.7〜1.3秒)初手が遅れ、相手に数手の差をつけられた(実機 2026-09-27)
            # 直前の入れ替わりで枠から消えた組と1色同じなら、枠の読みを信じる。出現の瞬間の色の読みは外れやすい
            # (実機 2026-10-25 12:33: 紫黄・青黄を黄黄と読み、自分の初手の候補にも使われて先読みと合わず約0.95秒遅れた)
            recent_swap = self._prestart_swap_t is not None and snap.t - self._prestart_swap_t <= START_SWAP_RECENT_SECONDS
            framed = first is not None and (tuple(sorted(first)) == colors or (recent_swap and bool(set(first) & set(colors))))
            confirmed_start = self.start_confirmed and framed
            if not confirmed_start:
                cand = self._initial_candidate
                if cand is None or cand[0] != colors or snap.t - cand[3] > INITIAL_CANDIDATE_MAX_AGE_SECONDS:
                    self._initial_candidate = (colors, r1, c1, snap.t)
                    return
                if r1 <= cand[1] and c1 == cand[2]:
                    return  # まだ動いていない(落ちてくる組ぷよなら段が下がるか列が変わる)
            self._initial_candidate = None
            # 縦なら上が子、横なら位置だけでは軸が分からないので仮に左を軸とする
            self.current_pair = (k1, k2) if c1 == c2 else (k2, k1)
            note = "initial"
            if first is not None and sorted(first) == sorted((k1, k2)):
                self.current_pair = first  # 開始前の枠で見えていた第1手(子・軸の向きも枠どおり)
                note = "initial: 開始前の枠の第1手と一致"
            elif framed:
                note = f"initial: 開始前の枠の第1手(出現の読みは {self.current_pair})"
                self.current_pair = first
            if confirmed_start:
                note += "、対戦開始を確認済みなので動くのを待たない"
            self._prestart_first = None
            self._initial_pair_pending = True
            events.append(TrackerEvent("spawn", snap.t, f"turn={self.turn} current={self.current_pair} ({note})"))
            # 盤面が空(試合開始)なら取り込むものが無いので、この組ぷよの手番をすぐ始める
            # (以前は turn_ready が出ず、第1手には提案が無かった)。途中起動で盤面があれば通常の取り込みに任せる
            if self.next1 is not None and count_puyos(settled) == 0:
                self.settling = False
                self._settled_count = 0
                events.append(TrackerEvent("turn_ready", snap.t, f"turn={self.turn} puyos=0 (initial)"))

    # ---- 予告おじゃま(安定した値だけ採用) ----

    def _update_pending(self, snap: Snapshot, events: list[TrackerEvent]) -> None:
        if self._pending_candidate is None or self._pending_candidate[0] != snap.pending_garbage:
            self._pending_candidate = (snap.pending_garbage, snap.t)
        elif snap.t - self._pending_candidate[1] >= PENDING_STABLE_SECONDS and snap.pending_garbage != self.pending_garbage:
            self.pending_garbage = snap.pending_garbage
            events.append(TrackerEvent("pending", snap.t, f"pending={self.pending_garbage}"))
