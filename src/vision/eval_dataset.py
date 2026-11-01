"""画像認識の評価用データ(正解ラベル付きの画面画像)の読み書きと採点

閾値を手で直すたびに「別の色が悪くならなかったか」を確かめられるよう、保存済みの実機画像に
正解ラベルを付けて、認識結果との一致率と混同行列(どの正解をどれと読んだか)を出す。

ラベル(JSON)の形式:
  {
    "image": "013_turn12.png",           # 同じフォルダの images/ にある画面画像(キャプチャ範囲そのもの)
    "calibration": {...},                 # 撮影時のキャリブレーション(config/calibration.json の中身)
    "screen": "game",                     # 画面の種類。game=試合中 / result=勝敗の画面 / other=その他
    "verified": false,                    # 目視で確認済みか(確認済みだけを採点する)
    "note": "",
    "players": {
      "self":  {"board": [12段目..1段目 の6文字×12行], "next1": "RP", "next2": "BR", "pending": 1},
      "enemy": {...}
    }
  }
文字: R/Y/G/B/P = 赤/黄/緑/青/紫、# = おじゃま、. = 空、* = 判定しない(動いている途中など正解を決められない)。
ネクストは (子, 軸) の2文字。認識結果の ? (unknown) は常に誤り。
盤面は「画面に見えているもの」が正解(落下中・操作中の組ぷよも含む。着地予測の小さな玉は空)。
screen が game 以外の画像は盤面・ネクスト・予告を採点しない。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..capture.calibrate import CalibrationResult
from ..engine.puyo_field import COLS, ROWS
from .board_reader import read_board, read_next

LETTERS = {
    "red": "R", "yellow": "Y", "green": "G", "blue": "B", "purple": "P",
    "garbage": "#", "empty": ".", "unknown": "?",
}
DONT_CARE = "*"
SCREENS = ("game", "result", "other")
PLAYERS = ("self", "enemy")


def read_frame(image_bgr: np.ndarray, calibration: CalibrationResult) -> dict:
    """画面画像(キャプチャ範囲そのもの)を今の認識で読み、ラベルと同じ形の dict を返す。"""
    region = calibration.capture_region()
    origin = (region.left, region.top)
    out = {}
    for who, pc in (("self", calibration.self_player), ("enemy", calibration.enemy_player)):
        board = read_board(image_bgr, pc.geometry(), origin)
        nexts = [read_next(image_bgr, rect, origin) for rect in pc.next_rects]
        out[who] = {
            # row=0(13段目)は常に空なので持たない
            "board": ["".join(LETTERS.get(c, "?") for c in board.cells[row]) for row in range(1, ROWS)],
            "next1": "".join(LETTERS.get(c, "?") for c in nexts[0]),
            "next2": "".join(LETTERS.get(c, "?") for c in nexts[1]),
            "pending": board.pending_garbage,
        }
    return out


def make_label(image_name: str, calibration: CalibrationResult, reading: dict, screen: str = "game") -> dict:
    """今の認識結果を初期値にしたラベルを作る(確認前なので verified=False)。

    読めなかったマス(?)は正解を決められないので * にしておく(確認の時に正しい色へ直す)。"""
    if screen not in SCREENS:
        raise ValueError(f"screen は {SCREENS} のどれか: {screen}")
    reading = json.loads(json.dumps(reading).replace("?", DONT_CARE))
    return {
        "image": image_name,
        "calibration": calibration.to_dict(),
        "screen": screen,
        "verified": False,
        "note": "",
        "players": reading,
    }


def validate_label(label: dict) -> None:
    """ラベルの形式を確かめる(手で直した時の書き間違いを早く見つける)。"""
    if label.get("screen") not in SCREENS:
        raise ValueError(f"screen が不正: {label.get('screen')}")
    allowed = set(LETTERS.values()) - {"?"} | {DONT_CARE}
    for who in PLAYERS:
        p = label["players"][who]
        board = p["board"]
        if len(board) != ROWS - 1 or any(len(line) != COLS for line in board):
            raise ValueError(f"{who}: board は {COLS}文字×{ROWS - 1}行")
        for key in ("next1", "next2"):
            if len(p[key]) != 2:
                raise ValueError(f"{who}: {key} は2文字")
        bad = {ch for ch in "".join(board) + p["next1"] + p["next2"] if ch not in allowed}
        if bad:
            raise ValueError(f"{who}: 使えない文字 {sorted(bad)}")
        if p["pending"] != DONT_CARE and not isinstance(p["pending"], int):
            raise ValueError(f"{who}: pending は整数か \"*\"")


@dataclass
class CaseResult:
    name: str
    cells: Counter = field(default_factory=Counter)  # (正解, 読み) → 個数
    nexts: Counter = field(default_factory=Counter)
    pending_ok: int = 0
    pending_total: int = 0
    errors: list[str] = field(default_factory=list)


def _cell_name(who: str, row: int, col: int) -> str:
    return f"{who} {ROWS - 1 - row}段目{col + 1}列目"


def compare(name: str, label: dict, reading: dict) -> CaseResult:
    """ラベルと認識結果を比べる。screen が game 以外なら何も採点しない。"""
    res = CaseResult(name=name)
    if label["screen"] != "game":
        return res
    for who in PLAYERS:
        want, got = label["players"][who], reading[who]
        for row in range(ROWS - 1):
            for col in range(COLS):
                w, g = want["board"][row][col], got["board"][row][col]
                if w == DONT_CARE:
                    continue
                res.cells[(w, g)] += 1
                if w != g:
                    res.errors.append(f"{_cell_name(who, row, col)}: 正解{w} 読み{g}")
        for key in ("next1", "next2"):
            for i, part in enumerate(("子", "軸")):
                w, g = want[key][i], got[key][i]
                if w == DONT_CARE:
                    continue
                res.nexts[(w, g)] += 1
                if w != g:
                    res.errors.append(f"{who} {key}の{part}: 正解{w} 読み{g}")
        if want["pending"] != DONT_CARE:
            res.pending_total += 1
            if want["pending"] == got["pending"]:
                res.pending_ok += 1
            else:
                res.errors.append(f"{who} 予告: 正解{want['pending']} 読み{got['pending']}")
    return res


def _accuracy(c: Counter) -> tuple[int, int]:
    total = sum(c.values())
    return sum(n for (w, g), n in c.items() if w == g), total


def confusion_text(c: Counter) -> str:
    """混同行列(行=正解、列=読み)を文字で表す。"""
    order = [ch for ch in "RYGBP#.?" if any(ch in k for k in c)]
    lines = ["正解\\読み " + " ".join(f"{ch:>5}" for ch in order)]
    for w in order:
        if not any(k[0] == w for k in c):
            continue
        lines.append(f"{w:>9} " + " ".join(f"{c.get((w, g), 0):>5}" for g in order))
    return "\n".join(lines)


def summarize(results: list[CaseResult]) -> str:
    cells, nexts = Counter(), Counter()
    p_ok = p_total = 0
    for r in results:
        cells.update(r.cells)
        nexts.update(r.nexts)
        p_ok += r.pending_ok
        p_total += r.pending_total

    def pct(ok: int, total: int) -> str:
        return f"{ok}/{total} ({100.0 * ok / total:.2f}%)" if total else "0/0"

    c_ok, c_total = _accuracy(cells)
    n_ok, n_total = _accuracy(nexts)
    # 空マスが大半なので、ぷよのあるマス(正解が空以外)だけの一致率も出す
    filled = Counter({k: v for k, v in cells.items() if k[0] != "."})
    f_ok, f_total = _accuracy(filled)
    out = [
        f"画像 {len(results)}枚",
        f"盤面のマス: {pct(c_ok, c_total)}  うちぷよのあるマス: {pct(f_ok, f_total)}",
        f"ネクスト: {pct(n_ok, n_total)}",
        f"予告おじゃま: {pct(p_ok, p_total)}",
        "",
        "盤面の混同行列",
        confusion_text(cells),
        "",
        "ネクストの混同行列",
        confusion_text(nexts),
    ]
    return "\n".join(out)


def load_label(path: Path) -> dict:
    label = json.loads(path.read_text(encoding="utf-8"))
    validate_label(label)
    return label


def save_label(path: Path, label: dict) -> None:
    """indent 付きで書く(board は1段が1行になるので手で直しやすい)。"""
    validate_label(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(label, indent=2, ensure_ascii=False), encoding="utf-8")
