"""ama (https://github.com/citrus610/ama) をサブプロセスとして呼び出すクライアント。

ama は C++ 製の強豪ぷよぷよAIで、本プロジェクトでは 1P(連鎖構築)用の
ビームサーチ(beam::search_multi)だけを使う。ama 本体には外部から盤面を渡す
入口が無いため、external/ama-main/bridge/main.cpp に「標準入出力で1行1メッセージの
JSONをやり取りするブリッジ」を追加してあり、このモジュールはそれと通信する。

プロトコル(1行1JSON)。詳細は external/ama-main/bridge/main.cpp 冒頭のコメント参照。
  build   : {"field", "queue"} -> {"x", "r", "score", "axis", "child", "candidates", "ms"}
  analyze : {"mode": "analyze", "field", "queue"} -> 発火したら何連鎖・何点か(凝視・点数表示用)
  battle  : {"mode": "battle", "self": Player, "enemy": Player} -> 相手を凝視した対戦思考の手
  エラー時は {"error": "..."}。起動直後にブリッジから {"ready": true} が1行送られてくる。

ama の座標は x=左から0、y=最下段が0。PuyoField の (row, col) へは
puyo_field.xy_to_row_col で変換する。
"""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

from ..paths import app_root, is_frozen
from .puyo_field import PuyoField, xy_to_row_col

# external/ama-main で `mingw32-make PEXT=true bridge` を実行すると生成される。README参照。
# 実行ファイル形式(PBA.exe)では、同じフォルダに置いた ama-bridge.exe を使う。
AMA_BRIDGE_EXE = (
    app_root() / "ama-bridge.exe"
    if is_frozen()
    else app_root() / "external" / "ama-main" / "bin" / "bridge" / "ama-bridge.exe"
)

# 自作AI(Puyo-YamaNEKO、ama ブリッジと同じ形で話す yamaneko-bridge.exe)。既定はこのフォルダと並べて置いた場所
YAMANEKO_BRIDGE_EXE = app_root().parent / "Puyo-YamaNEKO" / "engine" / "bin" / "yamaneko-bridge.exe"
# PBA が使う思考エンジンは自作AIの YamaNEKO に固定(2026-10-21)。ama はシミュレーターの比較とテストと、
# メイン画面の「2P の思考エンジン」で 2P に素の ama を選んだ時だけ使う(2026-10-25、実機で YamaNEKO 対 ama)。
# YamaNEKO の場所を変える時は config/engine.json に {"yamaneko_exe": "../Puyo-YamaNEKO/engine/bin/yamaneko-bridge.exe"}
# (相対パスは PBA の直下から)
ENGINE_CONFIG = app_root() / "config" / "engine.json"
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000  # Windows のプロセスの優先度「通常以下」
# 配ぷよ読み(2026-10-23)で YamaNEKO のブリッジに渡す起動引数: 撃つ・返す判断に相手の本当の先のツモを使い、構築は12組先まで本当のツモで読む
# (シミュレーターの評価用 seed: 深さ8で ama 戦 65.8 → 76.8%・潰し用AI 戦 69.0 → 77.5%。1人モードで深さ13なら125,000点に12試行すべて届いた)。
# 使ってよいかは相手との取り決め次第(CPU 戦の試験なら問題ない)。支援モードとシミュレーターの配ぷよモードで使う
HAIPUYO_BRIDGE_ARGS = ["--haipuyo-threat", "--haipuyo-depth", "12"]
# 相手のツモを覗く(2026-11-01): 相手も同じ配ぷよなら、相手が先に進んでいる時の相手のネクストが自分の先のツモになる。
# 要求に自分と相手の置いた数(placed)を入れる。配ぷよの表を使わないので、配ぷよ読みのできないテトリス2でも使える可能性がある
# (相手と配ぷよが同じかは未確認。食い違いを見つけたら YamaNEKO がその試合では使わない)。
# シミュレーター: 配ぷよ読みありの YamaNEKO に評価用 seed で 29.0 → 43.5%、ama 戦は差なし
PEEK_BRIDGE_ARGS = ["--peek-enemy"]


def resolve_bridge_exe(config_path: Path | None = None, bridge: str = "yamaneko") -> Path:
    """起動するブリッジの実行ファイル。bridge は "yamaneko"(既定)か "ama"(シミュレーターの比較用)。"""
    if bridge == "ama":
        return AMA_BRIDGE_EXE
    if bridge != "yamaneko":
        raise ValueError(f'bridge は "ama" か "yamaneko" です: {bridge!r}')
    path = ENGINE_CONFIG if config_path is None else config_path
    exe = None
    if path.exists():
        with open(path, encoding="utf-8") as f:
            exe = json.load(f).get("yamaneko_exe")
    if not exe:
        return YAMANEKO_BRIDGE_EXE
    p = Path(exe)
    return p if p.is_absolute() else (app_root() / p).resolve()


# ブリッジからの応答をこの秒数待っても届かない場合は応答不能とみなす。
# Windowsのパイプはselectで待てないため、専用の読み取りスレッド+
# タイムアウト付きキューで実現する(Cold Clear 2クライアントと同じ方式)。
RECV_TIMEOUT_SEC = 10.0
# ama のブリッジを起動し直す問い合わせの回数(メモリの増加を抑える。YamaNEKO の battle.exe の AmaController と同じ 300 回)
AMA_RESTART_REQUESTS = 300
# 起動直後の {"ready": true} だけはこの秒数まで待つ(2026-10-13)。YamaNEKO は起動時に思考時間を測ってビーム幅を
# 決めるので、遅い PC や CPU が混んでいる時は10秒を超える(負荷のある状態で2スレッドだと11.9秒かかった)
READY_TIMEOUT_SEC = 60.0


@dataclass
class AmaMove:
    """ama が返した最善手。"""

    x: int  # 軸ぷよの列(左から0)
    rotation: str  # 子ぷよの位置: "UP" / "RIGHT" / "DOWN" / "LEFT"
    score: int  # ama の評価値(高いほど良い)
    axis_cell: tuple[int, int]  # 軸ぷよの着地マス (row, col)。row=0が隠し13段目
    child_cell: tuple[int, int]  # 子ぷよの着地マス (row, col)
    think_ms: int  # 思考にかかった時間(ミリ秒)
    candidates: list[tuple[int, str, int]]  # 全候補 (x, rotation, score) をスコア降順で
    field_after: list[str] | None = None  # 提案通りに置いて連鎖まで進めた後の盤面(13行、上から)
    form: str | None = None  # 設置後の盤面に最も合う開幕の形(FORM_TEXT_JA のいずれか。無ければ None)
    path: list[str] = dc_field(default_factory=list)  # 置き場所までの操作手順(1要素=1フレームの入力。全自動モード用)


@dataclass
class AttackInfo:
    """「この手順で発火すると何連鎖・何点か」の1件。count=0 なら該当なし。"""

    count: int  # 連鎖数
    score: int  # 点数
    garbage: int  # おじゃまぷよ換算(個)
    x: int  # 発火に使う1手目の軸ぷよの列
    rotation: str
    frame: int  # 何手目で発火するか(0=今の組ぷよで発火)
    all_clear: bool


@dataclass
class FirePoint:
    """1個置くと連鎖が起きる発火点。cell は置いたぷよが乗るマス (row, col)。"""

    cell: tuple[int, int]
    color: str  # ama のスロット文字(R/Y/G/B)
    count: int
    score: int


@dataclass
class FieldAnalysis:
    """盤面の凝視結果(analyze モードの応答)。"""

    immediate: AttackInfo  # 今の組ぷよで発火できる最大
    max: AttackInfo  # 手持ちの組ぷよ(ネクストまで)で発火できる最大
    harass: AttackInfo  # 3連鎖以下の潰し最大
    potential: int  # 将来の本線の見込み点(静止探索)
    garbage_max: int  # max のおじゃま換算(個)
    fire_points: list[FirePoint] = dc_field(default_factory=list)  # 点数の高い順
    # 組んでいる連鎖: 1つの列に同じ色を3個まで足すと起きる最大の連鎖数(見えている組ぷよに発火の色が無くても数える)。
    # YamaNEKO のブリッジだけが返す(ama のブリッジでは 0)
    potential_chain: int = 0


@dataclass
class BattleMove:
    """battle モードの応答。相手を凝視した上での手と、その意味(label)。"""

    x: int
    rotation: str
    eval: int
    axis_cell: tuple[int, int]
    child_cell: tuple[int, int]
    label: str  # "main"(本線発火) / "harass"(潰し) / "all_clear" / "accept"(受け) / "build"(構築) / "prepare"(発火準備)
    self_analysis: FieldAnalysis | None
    enemy_analysis: FieldAnalysis | None  # 一人用モードでは None
    think_ms: int
    field_after: list[str] | None = None  # 提案通りに置いて連鎖まで進めた後の盤面(13行、上から)
    chain_after: tuple[int, int] = (0, 0)  # その時に起きる連鎖 (連鎖数, 点数)
    form: str | None = None  # 設置後の盤面に最も合う開幕の形(無ければ None)
    fire_reason: str | None = None  # PBA の発火判定が構築を発火に差し替えた理由(danger/kill/target)
    reason: str = ""  # ブリッジの判断の理由(自作AI の COUNTER_MARGIN・FINISH など。ama は空。ログ用)
    plan_steps: int = 0  # 手順(準備の手を含む攻撃)を進める判断なら、この手を含めた手数(撃つのは最後の手)。それ以外は0
    incoming: int | None = None  # 相手が連鎖中の時、自分に降ると見積もったおじゃま(個)
    incoming_source: str = ""  # その根拠(replay/live/pending/max_attack)
    incoming_chain: int = 0  # 見積もった相手の連鎖数
    path: list[str] = dc_field(default_factory=list)  # 置き場所までの操作手順(1要素=1フレームの入力。全自動モード用)
    diag: dict | None = None  # 攻撃を受けている時の判断材料(ブリッジの "diag"。ログ用)
    # 最終的な構想(2026-10-24。YamaNEKO が配ぷよ読みで本当の先のツモを読めた時だけ): 撃つ連鎖の発火の組ぷよを置いた直後
    # (消える前)の盤面(13行、上から)、(連鎖数, 点数, 今の手を含めて何手目に撃つか)、発火の組ぷよの2マス (row, col)
    plan_field: list[str] | None = None
    plan_chain: tuple[int, int, int] = (0, 0, 0)
    plan_trigger: list[tuple[int, int]] = dc_field(default_factory=list)
    # 跳考(2026-10-24): 出現位置で接地している時に、子ぷよを下へ回して持ち上げる操作を手順に挟んだ回数(YamaNEKO だけ。ログ用)
    choukou: int = 0
    # ブリッジに送った要求(JSON の1行)。ログに残し、場面をブリッジでそのまま再現するのに使う(2026-10-12: ログの盤面は画面の色、
    # ツモはブリッジの文字で書かれていて対応を推測するしかなく、実機の判断を再現できなかった)
    request: str = ""


def diag_text(diag: dict | None) -> str:
    """ブリッジの診断(本線で返せたか)をログ用の1行にする。"""
    if not diag:
        return ""

    def brief(a: dict | None) -> str:
        if not a:
            return "無し"
        return f"{a['count']}連鎖{a['score']}点(送{a['send']}個,{a['frame']}手後)"

    return (
        f"判断材料: 着弾{diag['enemy_attack']}個 期限{diag['attack_frame']}手 受け上限{diag['accept_limit']}個 "
        f"期限内の最大攻撃={brief(diag.get('best_in_time'))} 期限を問わない最大={brief(diag.get('best_any'))}"
    )


LABEL_TEXT_JA = {
    "main": "本線発火",
    "harass": "潰し",
    "all_clear": "All Green",  # 両言語とも英語表記(利用者の指定 2026-10-25)
    "accept": "受け",
    "build": "構築",
    "prepare": "発火準備(数手先で発火)",
}

# 自作AI の「受け」のうち、中身は構築の手(伸ばす・受けて組み続ける)の理由。支援モードでは先読みの構築の手と同じ扱いにする
# (2026-10-12、利用者「相手の発火に対して何をすべきかわからなくなる。提示が振動する」: 相手の連鎖中に先読みの手を表示した後、
# 中身は同じ構築なのに「受け」なので別の構築の手に差し替えていた。11:36 のログで差し替え69回のうち34回)
BUILD_LIKE_REASONS = ("EXTEND", "EXTEND_MARGIN", "BUILD_CHAIN", "ACCEPT")


def is_build_like(move: "BattleMove") -> bool:
    """置き場所に意味の無い構築の手か(構築、または自作AI の「伸ばす・受けて組み続ける」)。"""
    return move.label == "build" or (move.label == "accept" and move.reason in BUILD_LIKE_REASONS)


def action_text(move: "BattleMove", incoming: int = 0) -> str:
    """支援モードの表示: 自作AI の判断の理由ごとに、何をすべきかを言葉で返す(該当しない・理由の無い ama は空)。
    incoming は受ける量(相手の連鎖の見積もり・確定した予告)。"""
    r, n = move.reason, move.plan_steps
    if r in ("EXTEND", "EXTEND_MARGIN"):
        return "伸ばす(まだ返せない。着弾までに連鎖を伸ばす)"
    if move.label == "accept" and r in ("BUILD_CHAIN", "ACCEPT"):
        return f"受ける({incoming}個は受けて組み続ける)" if incoming > 0 else "受ける(受けて組み続ける)"
    if r in ("COUNTER_PREP", "COUNTER_PREP_MARGIN"):
        return f"返しの準備 @{n}" if n >= 2 else "返す(撃つ)"
    if r == "ATTACKED_KILL":
        return f"撃つ準備 @{n}(倒しきれる)" if n >= 2 else "撃つ(倒しきれる)"
    if r in ("COUNTER", "COUNTER_MARGIN", "COUNTER_SMALL", "COUNTER_KILL", "COUNTER_PARTIAL"):
        return "返す(撃つ)"
    return ""


# PBA の発火判定(ブリッジ)が構築を発火に差し替えた理由の表示
FIRE_REASON_TEXT_JA = {
    "target": "目安の連鎖に到達",
}


def _parse_attack(d: dict) -> AttackInfo:
    return AttackInfo(
        count=int(d["count"]),
        score=int(d["score"]),
        garbage=int(d["garbage"]),
        x=int(d["x"]),
        rotation=str(d["r"]),
        frame=int(d["frame"]),
        all_clear=bool(d.get("all_clear", False)),
    )


def parse_analysis(response: dict) -> FieldAnalysis:
    if "error" in response:
        raise RuntimeError(f"ama ブリッジがエラーを返しました: {response['error']}")
    return FieldAnalysis(
        immediate=_parse_attack(response["immediate"]),
        max=_parse_attack(response["max"]),
        harass=_parse_attack(response["harass"]),
        potential=int(response["potential"]),
        garbage_max=int(response["garbage_max"]),
        potential_chain=int(response.get("potential_chain", 0)),
        fire_points=[
            FirePoint(cell=xy_to_row_col(int(p["x"]), int(p["y"])), color=str(p["color"]), count=int(p["count"]), score=int(p["score"]))
            for p in response.get("fire_points", [])
        ],
    )


def parse_battle(response: dict) -> BattleMove:
    if "error" in response:
        raise RuntimeError(f"ama ブリッジがエラーを返しました: {response['error']}")
    ax, ay = response["axis"]
    cx, cy = response["child"]
    return BattleMove(
        x=int(response["x"]),
        rotation=str(response["r"]),
        eval=int(response["eval"]),
        axis_cell=xy_to_row_col(ax, ay),
        child_cell=xy_to_row_col(cx, cy),
        label=str(response["label"]),
        self_analysis=parse_analysis(response["self"]),
        enemy_analysis=parse_analysis(response["enemy"]),
        think_ms=int(response.get("ms", 0)),
        field_after=list(response["field_after"]) if "field_after" in response else None,
        chain_after=(int(response["chain_after"]["count"]), int(response["chain_after"]["score"])) if "chain_after" in response else (0, 0),
        form=response.get("form") or None,
        fire_reason=response.get("fire_reason") or None,
        reason=str(response.get("reason") or ""),
        plan_steps=int(response.get("plan_steps") or 0),
        plan_field=list(response["plan"]["field"]) if "plan" in response else None,
        plan_chain=(int(response["plan"]["chain"]), int(response["plan"]["score"]), int(response["plan"]["steps"])) if "plan" in response else (0, 0, 0),
        plan_trigger=[xy_to_row_col(x, y) for x, y in response["plan"].get("trigger", [])] if "plan" in response else [],
        choukou=int(response.get("choukou") or 0),
        incoming=response.get("incoming"),
        incoming_source=str(response.get("incoming_source", "")),
        incoming_chain=int(response.get("incoming_chain", 0)),
        path=[str(p) for p in response.get("path", [])],
        diag=response.get("diag"),
    )


@dataclass
class PlayerState:
    """battle モードに渡す片側プレイヤーの状態。"""

    field: PuyoField
    pairs: list[str]  # 2組以上
    attack: int = 0  # このプレイヤーに降る予定の予告おじゃま(個)
    attack_frame: int = 0  # 相手の連鎖が終わるまでの手数目安(0=攻撃されていない)
    dropping: int = 0  # 落下中のおじゃま(個)
    all_clear: bool = False
    bonus: int = 0
    chain_bonus: bool = False  # 進行中の連鎖に全消しボーナス(+30個)が乗っている(相手側)
    # 点滅から復元した発火直前の盤面(相手側、引き金込み、13行)。ブリッジはこれを消して連鎖を再現する
    chain_field: PuyoField | None = None
    # 相手の連鎖が終わるまでに自分が置ける手数の見積もり(相手側。None ならブリッジの固定値)
    attack_frame_est: int | None = None
    # 攻撃の端数点(前の連鎖の余りと落下ボーナス。None なら送らない。YamaNEKO は --count-carry-self で使う。ama は読まない)
    carry: int | None = None
    # 配ぷよ読み用(2026-10-23。None なら送らない。YamaNEKO は --haipuyo-threat / --haipuyo-depth で使う。ama は読まない):
    # 試合の最初から置いた組ぷよ(pairs と同じ文字)と、置いた数
    history: list[str] | None = None
    placed: int | None = None

    def to_json(self) -> dict:
        d = build_request(self.field, self.pairs)
        d.update(
            attack=self.attack,
            attack_frame=self.attack_frame,
            dropping=self.dropping,
            all_clear=self.all_clear,
            bonus=self.bonus,
            chain_bonus=self.chain_bonus,
        )
        if self.chain_field is not None:
            d["chain_field"] = self.chain_field.to_rows()
        if self.attack_frame_est is not None:
            d["attack_frame_est"] = self.attack_frame_est
        if self.carry is not None:
            d["carry"] = self.carry
        if self.history is not None:
            d["history"] = list(self.history)
        if self.placed is not None:
            d["placed"] = self.placed
        return d


class NoValidPlacementError(RuntimeError):
    """ブリッジが「置ける手が無い(no valid placement)」を返した。盤面の誤認・ゲームオーバー・移動制限などで
    起きる。他の RuntimeError(タイムアウト・通信破損)と違い、観測を続けて次の状態で再探索すればよい
    (分析文書第3弾 E: 一括で握りつぶさず、この応答だけ分ける。crash_log 2026-09-19 に4件)。"""


def parse_move(response: dict) -> AmaMove:
    """ブリッジの応答(dict)を AmaMove に変換する。純粋関数にして単体テスト可能にしている。"""
    if "error" in response:
        if response["error"] == "no valid placement":
            raise NoValidPlacementError(f"ama ブリッジがエラーを返しました: {response['error']}")
        raise RuntimeError(f"ama ブリッジがエラーを返しました: {response['error']}")
    ax, ay = response["axis"]
    cx, cy = response["child"]
    return AmaMove(
        x=int(response["x"]),
        rotation=str(response["r"]),
        score=int(response["score"]),
        axis_cell=xy_to_row_col(ax, ay),
        child_cell=xy_to_row_col(cx, cy),
        think_ms=int(response.get("ms", 0)),
        candidates=[(int(c["x"]), str(c["r"]), int(c["score"])) for c in response.get("candidates", [])],
        field_after=list(response["field_after"]) if "field_after" in response else None,
        form=response.get("form") or None,
        path=[str(p) for p in response.get("path", [])],
    )


# ブリッジが返す開幕の形(ama の beam/form.h の names)の表示名
FORM_TEXT_JA = {"GTR": "GTR", "FRON": "Fron", "SGTR": "フキゲンGTR"}


def build_request(
    field: PuyoField,
    pairs: list[str],
    width: int | None = None,
    depth: int | None = None,
) -> dict:
    """ブリッジへ送るリクエストを組み立てる。pairs は "RG" のような2文字(軸, 子)のリスト。"""
    for pair in pairs:
        if len(pair) != 2 or any(ch not in "RYGB" for ch in pair):
            raise ValueError(f"組ぷよは R/Y/G/B の2文字で指定してください: {pair!r}")
    req: dict = {"field": field.to_rows(), "queue": list(pairs)}
    if width is not None:
        req["width"] = width
    if depth is not None:
        req["depth"] = depth
    return req


class AmaClient:
    """ama ブリッジのプロセスを1つ起動して使い回すクライアント。

    支援モード開始時に生成し、終了時に close() を呼ぶこと。
    """

    def __init__(self, exe_path: Path | None = None, bridge: str = "yamaneko", search_threads: int | None = None,
                 extra_args: list[str] | None = None, custom_args: list[str] | None = None,
                 low_priority: bool = False) -> None:
        # 省略時は YamaNEKO。bridge="ama" はシミュレーターの比較用。
        # search_threads は YamaNEKO の探索スレッド数(省略時はコア数。両方モードで2つが CPU を取り合わないよう半分を渡す)
        if exe_path is None:
            exe_path = resolve_bridge_exe(bridge=bridge)
        self.exe_path = exe_path
        # 思考エンジンの種類(実行ファイル名から): "yamaneko"(自作AI)/ "ama"
        self.engine = "yamaneko" if exe_path.name.startswith("yamaneko") else "ama"
        if not exe_path.exists():
            raise FileNotFoundError(
                f"思考エンジンのブリッジの実行ファイルが見つかりません: {exe_path}\n"
                "ama: external/ama-main で `mingw32-make PEXT=true bridge` を実行してください。\n"
                "自作AI(yamaneko): Puyo-YamaNEKO で `sh engine/build.sh` を実行してください。"
            )
        # pythonw(コンソール無し)から起動した場合にパイプ通信が不安定になる
        # 事象がTetris版で確認されているため、コンソールを一切生成させない。
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        # 優先度を「通常以下」にする(2026-10-26、利用者「移動ミスは CPU の負荷によるものではないか」): 使用率は下がらないが、
        # ゲームが CPU を要る時はそちらが先に動く。起動し直した時(ama の300回ごと)も同じ優先度
        self.low_priority = low_priority
        if low_priority and sys.platform == "win32":
            creation_flags |= BELOW_NORMAL_PRIORITY_CLASS
        args = [str(exe_path)]
        if search_threads is not None and self.engine == "yamaneko":
            args += ["--search-threads", str(max(1, search_threads))]
        if extra_args and self.engine == "yamaneko":
            args += list(extra_args)  # 配ぷよ読みなど YamaNEKO だけの起動引数
        if custom_args:
            args += list(custom_args)  # 画面で登録した思考エンジンの起動引数(2026-10-26)
        self.args = args
        self._creation_flags = creation_flags
        self._requests = 0  # 今のプロセスへの問い合わせの回数(ama は AMA_RESTART_REQUESTS 回ごとに起動し直す)
        self._start()

    def _start(self) -> None:
        self._proc = subprocess.Popen(
            self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=self._creation_flags,
            cwd=str(self.exe_path.parent),  # config.json を実行ファイルの隣から読ませる
        )
        # 読み取りのキューとスレッドはプロセスごとに作る(起動し直した時、前のプロセスの終わりの None を混ぜない)
        self._read_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread = threading.Thread(target=self._reader_loop, args=(self._proc, self._read_queue), daemon=True)
        self._reader_thread.start()
        self._requests = 0

        try:
            ready = self._recv(READY_TIMEOUT_SEC)
            if not ready.get("ready"):
                raise RuntimeError(f"ama ブリッジから予期しない初期メッセージ: {ready}")
        except Exception:
            # 起動に失敗したプロセスを残さない(以前は待ちの上限を超えると動いたまま残った)
            self._proc.kill()
            raise

    @staticmethod
    def _reader_loop(proc: subprocess.Popen, read_queue: "queue.Queue[str | None]") -> None:
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                read_queue.put(line)
        except (OSError, ValueError):
            pass
        read_queue.put(None)

    def _send(self, msg: dict) -> None:
        # ama のブリッジは問い合わせのたびにメモリが増える(1回 約6MB。PBA の1試合で100〜150回)ので、一定回数ごとに
        # 起動し直す(起動は約10ms。応答を待っている問い合わせは無い時点なので途中の状態は失わない。ama は問い合わせごとに完結)
        if self.engine == "ama" and self._requests >= AMA_RESTART_REQUESTS:
            self.close()
            self._start()
        self._requests += 1
        assert self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def _recv(self, timeout: float = RECV_TIMEOUT_SEC) -> dict:
        try:
            line = self._read_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"ama ブリッジから{timeout}秒間応答がありませんでした") from None
        if line is None:
            raise RuntimeError("ama ブリッジのプロセスが終了しました(予期しないクラッシュの可能性)")
        return json.loads(line)

    def suggest(
        self,
        field: PuyoField,
        pairs: list[str],
        width: int | None = None,
        depth: int | None = None,
    ) -> AmaMove:
        """盤面と組ぷよ列(先頭が現在の組ぷよ)を渡し、最善手を同期的に得る。

        ama の探索は1回の呼び出しで完結する(Cold Clear 2のような継続思考ではない)。
        実測で1手あたり数十〜200ms程度。
        """
        self._send(build_request(field, pairs, width, depth))
        return parse_move(self._recv())

    def analyze(self, field: PuyoField, pairs: list[str]) -> FieldAnalysis:
        """盤面を凝視し「発火したら何連鎖・何点か」を得る。pairs は2組以上必要。"""
        req = build_request(field, pairs)
        req["mode"] = "analyze"
        self._send(req)
        return parse_analysis(self._recv())

    def battle(
        self,
        me: PlayerState,
        enemy: PlayerState,
        fire_score: int | None = None,
        enemy_unknown: bool = False,
        target_point: int | None = None,
    ) -> BattleMove:
        """相手の盤面を凝視した対戦思考(ama の ai::think: 潰し・受け・本線発火の判断を含む)で最善手を得る。

        fire_score は「今の組ぷよで打てる攻撃がこの点数以上なら発火」の目安(PBA に残した唯一の独自判断。
        None なら ama の判断そのまま)。enemy_unknown=True なら enemy は認識失敗時の仮の状態(記録用)。
        """
        req = {"mode": "battle", "self": me.to_json(), "enemy": enemy.to_json()}
        # おじゃま1個に必要な点(マージンタイム。None なら送らず、ブリッジの既定の 70点)
        if target_point is not None:
            req["target_point"] = int(target_point)
        if fire_score is not None:
            req["fire_score"] = int(fire_score)
        if enemy_unknown:
            req["enemy_unknown"] = True
        self._send(req)
        move = parse_battle(self._recv())
        move.request = json.dumps(req, ensure_ascii=False)
        return move

    def solo(
        self,
        field: PuyoField,
        pairs: list[str],
        history: list[str] | None = None,
    ) -> BattleMove:
        """一人用モード: 相手を考慮しない1P用ビームサーチ(連鎖構築)で最善手を得る。

        対戦思考(battle)は相手の状態を前提にしているため、相手がいない一人用では
        構築専用の build モードを使い、自分の盤面の凝視結果(analyze)だけ添える。
        戻り値は battle と同じ BattleMove(label は常に "build"、enemy_analysis は None)。
        history(第1手からこれまでに置いた組ぷよ)はエンジンには送らない(定跡を引く BookClient が使う)。
        """
        move = self.suggest(field, pairs)
        analysis = self.analyze(field, pairs) if len(pairs) >= 2 else None
        return BattleMove(
            x=move.x,
            rotation=move.rotation,
            eval=move.score,
            axis_cell=move.axis_cell,
            child_cell=move.child_cell,
            label="build",
            self_analysis=analysis,
            enemy_analysis=None,
            think_ms=move.think_ms,
            field_after=move.field_after,
            form=move.form,
            path=move.path,
        )

    def close(self) -> None:
        try:
            if self._proc.stdin is not None and self._proc.poll() is None:
                self._proc.stdin.write("quit\n")
                self._proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self._proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        # パイプを明示的に閉じる(閉じ忘れの ResourceWarning を防ぐ)
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                stream.close()

    def __enter__(self) -> "AmaClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
