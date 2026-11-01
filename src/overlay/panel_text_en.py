"""情報パネルの英語表記(2026-10-25、利用者「情報パネルを英語表記に対応させたい」)

パネルの文言は app.py・ama_client.py が日本語で作り、ログにも同じ文言が残る。表記の切り替えは描画の直前に
ここで1行ずつ英語へ置き換える(作る側を変えずに済み、ログは日本語のまま)。
訳の無い行は日本語のまま出す(新しい文言を足した時はここにも足す)。
"""

from __future__ import annotations

import re

# 置き換えの無い固定の語句(行の一部にも現れるもの)
_PHRASES: dict[str, str] = {
    # ラベル(ama_client.LABEL_TEXT_JA)
    "本線発火": "Fire main chain",
    "潰し": "Harass",
    "受け": "Accept",
    "構築": "Build",
    "発火準備(数手先で発火)": "Prepare to fire (fires in a few moves)",
    # 行動(ama_client.action_text)
    "伸ばす(まだ返せない。着弾までに連鎖を伸ばす)": "Extend (can't counter yet; extend before the garbage lands)",
    "受ける(受けて組み続ける)": "Accept (take it and keep building)",
    "返す(撃つ)": "Counter (fire)",
    "撃つ(倒しきれる)": "Fire (it kills)",
    # 再計算の理由(app.py)
    "相手の消去開始を検出": "Opponent started clearing",
    "相手の連鎖が終了": "Opponent's chain ended",
    "予告おじゃまが増加": "Incoming garbage increased",
    "目安の連鎖に到達": "Target chain reached",
    # メモ(app.py)
    "開始前: 第1手の提案(第1〜3手から)": "Pre-start: first move (from pairs 1-3)",
    "開始前: 仮の提案(第3手が見えたら変わります)": "Pre-start: tentative (updates when pair 3 is visible)",
    "先読み(確認中)": "Precomputed (verifying)",
    "次の手を計算中": "Computing next move",
    "連鎖中: 次の手を待機": "Chaining: waiting for next move",
    "自動操作は先読みの手を実行済み": "Autoplay already sent the precomputed move",
    "自動操作は送った手を実行中": "Autoplay is executing the sent move",
    "相手の認識エラー: 相手を見ずに提案": "Opponent not recognized: suggesting without opponent",
    "対応が必要: 提案を変更": "Response needed: suggestion changed",
    # 開幕の形(ama_client.FORM_TEXT_JA)
    "フキゲンGTR": "Fukigen GTR",
    "(定型なし)": "(none)",
}

_NUM = r"([\d,]+)"

# 数値を含む行(行全体で一致させる)。置き換え先の {0} などは正規表現の組の順
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(rf"受ける\({_NUM}個は受けて組み続ける\)"), "Accept ({0} garbage; take it and keep building)"),
    (re.compile(rf"返しの準備 @{_NUM}"), "Prepare counter @{0}"),
    (re.compile(rf"撃つ準備 @{_NUM}\(倒しきれる\)"), "Prepare to fire @{0} (it kills)"),
    (re.compile(rf"この手で {_NUM}連鎖 {_NUM}点"), "This move: {0}-chain {1} pts"),
    (re.compile(rf"構想: {_NUM}連鎖 {_NUM}点 @{_NUM}"), "Plan: {0}-chain {1} pts @{2}"),
    (re.compile(rf"発火 {_NUM}連鎖 {_NUM}点"), "Fired: {0}-chain {1} pts"),
    (re.compile(rf"発火 {_NUM}連鎖\(点数は再現できず\)"), "Fired: {0}-chain (score unknown)"),
    (re.compile(rf"発火中 {_NUM}連鎖 {_NUM}点"), "Firing: {0}-chain {1} pts"),
    (re.compile(r"発火中\(段数は再現できず\)"), "Firing (chain unknown)"),
    (re.compile(rf"発火中 {_NUM}連鎖\(点数は再現できず\)"), "Firing: {0}-chain (score unknown)"),
    (re.compile(rf"点数を訂正: {_NUM}連鎖 {_NUM}点 → {_NUM}連鎖 {_NUM}点"), "Score corrected: {0}-chain {1} pts → {2}-chain {3} pts"),
    (re.compile(rf"点数を訂正: {_NUM}連鎖 {_NUM}点 → {_NUM}連鎖\(点数は再現できず\)"), "Score corrected: {0}-chain {1} pts → {2}-chain (score unknown)"),
    (re.compile(r"発火\(段数は再現できず\)"), "Fired (chain unknown)"),
    (re.compile(rf"{_NUM}連鎖 {_NUM}点"), "{0}-chain {1} pts"),
    (re.compile(rf"{_NUM}連鎖 {_NUM}点 @{_NUM}"), "{0}-chain {1} pts @{2}"),
    (re.compile(rf"(\s*)今撃つと {_NUM}連鎖 {_NUM}点"), "{0}Fire now: {1}-chain {2} pts"),
    (re.compile(rf"(\s*)潰し {_NUM}連鎖 {_NUM}点"), "{0}Harass: {1}-chain {2} pts"),
    (re.compile(rf"(\s*)組んでいる {_NUM}連鎖 {_NUM}点"), "{0}Built: {1}-chain {2} pts"),
    (re.compile(r"CPU (\S+)% / 観測 (\S+)fps"), "CPU {0}% / Vision {1}fps"),
]

# 「見出し: 中身」の形の行。中身は _PHRASES で置き換える
_PREFIXES: list[tuple[str, str]] = [
    ("提案: ", "Move: "),
    ("狙い: ", "Form: "),
    ("定型を離脱: ", "Left form: "),
    ("移行: ", "Switch form: "),
]


def _phrase(text: str) -> str:
    """固定の語句。「理由: 提案を更新」「対応が必要: 提案を変更 (理由)」のように組み合わさった形も置き換える。"""
    if text in _PHRASES:
        return _PHRASES[text]
    for pattern, template in _PATTERNS:
        m = pattern.fullmatch(text)
        if m:
            return template.format(*m.groups())
    m = re.fullmatch(r"(.+): 提案を更新(?: \((.+)\))?", text)
    if m:
        out = f"{_phrase(m.group(1))}: suggestion updated"
        return out + (f" ({_phrase(m.group(2))})" if m.group(2) else "")
    m = re.fullmatch(r"(.+?) \((.+)\)", text)
    if m and m.group(1) in _PHRASES:
        return f"{_PHRASES[m.group(1)]} ({_phrase(m.group(2))})"
    if " → " in text:
        return " → ".join(_phrase(part) for part in text.split(" → "))
    return text


def to_english(line: str) -> str:
    """パネルの1行を英語にする。訳の無い行はそのまま返す。"""
    for ja, en in _PREFIXES:
        if line.startswith(ja):
            return en + _phrase(line[len(ja):])
    return _phrase(line)
