"""実機ログ(debug_logs/debug_log_*.txt)の要約: 速さ・置き場所・認識の指標を1回で出す

改善の前後を同じ物差しで比べるためのもの(2026-10-21)。手作業で集計していた指標をまとめた。
  - 1手の間隔(出現から次の出現。連鎖・おじゃまの落下を挟む手は除く): 自分 / 相手
  - 全自動の操作の時間(操作全体・落下)と「操作後〜次の出現」
  - 着地した盤面が先読みと違った手番の割合と、その手がかり(置いた2マス: 提案どおり/ずれ、おじゃま)
  - 出現時点で送れなかった理由
  - 試合の途中の5色目・相手の認識エラー(試合の終わりの結果画面の時間は除く)
  - 観測の統計(観測の間隔・画面取得の待ち・処理時間)
  - 設置キャンセル(2026-10-21 仮): 狙った手と狙わなかった手の「操作後〜次の出現」と、置き場所が先読みとずれた割合
  - 自由落下の疑い(2026-10-21): 出現から送り始めるまでの遅れと、次の出現までに送れなかった手番(自由落下)とその理由

使い方:
  venv\\Scripts\\python tools\\log_summary.py                 # 最新のログ
  venv\\Scripts\\python tools\\log_summary.py debug_logs\\debug_log_20261020_222828.txt [...]
  venv\\Scripts\\python tools\\log_summary.py --all                # debug_logs の全部のログの自由落下をまとめて集計
"""

from __future__ import annotations

import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVENT = re.compile(r"\[\s*([\d.]+)s\] (self|enemy)\s+(\w+)\s+(.*)")
STAMP = re.compile(r"\[\s*([\d.]+)s\]")
RESULT_SCREEN_SEC = 9.0  # 試合のリセットの前のこの秒数は結果画面とみなす(負けの検出からリセットまで約7秒)
LATE_SEC = 0.3  # 出現から送り始めるまでがこれを超えた手番を「遅れ」とみなす


def turn_sends(lines: list[str]) -> list[dict]:
    """自分の手番ごとに、出現の時刻・送り始めた時刻・次の出現の時刻・送らなかった理由をまとめる(2026-10-21)。

    送り始めた時刻: 「先読みの手順を出現時点で送信」は出現の時刻、「出現のXms後で送信」はその値、「取り込み時点で送信」はその時刻。操作の記録「自動操作: turn=N 操作 (所要ms, ...)」は操作が終わった時に
    出るが時刻が付かないので、直前の時刻から所要時間を引いた値(出現より前なら出現の時刻)で、遅れは少なめに出る。
    次の出現までに送り始めていない手番は自由落下(free_fall=True)。試合の結果画面の時間と、次の出現が無い手番と、
    全自動でない間(支援モードで人間が操作する)の手番と、自分の連鎖をはさむ手番は除く。
    """
    t = 0.0
    game = 0
    turns: dict[tuple[int, int], dict] = {}
    order: list[tuple[int, int]] = []
    resets: list[float] = []
    auto = True  # 全自動の間だけ数える(支援モードでは人間が操作するので PBA は送らない)
    self_chains: list[float] = []  # 自分の連鎖が始まった時刻
    for line in lines:
        if "支援モード開始" in line:
            a = re.search(r"全自動=(True|False)", line)
            auto = a is None or a.group(1) == "True"
        m = STAMP.match(line)
        if m:
            t = float(m.group(1))
        e = EVENT.match(line)
        if e:
            if e.group(2) == "self" and e.group(3) == "chain_start":
                self_chains.append(t)
            if e.group(2) == "self" and e.group(3) == "spawn" and auto:
                n = re.search(r"turn=(\d+)", e.group(4))
                if n:
                    key = (game, int(n.group(1)))
                    if key not in turns:
                        turns[key] = {"game": game, "turn": key[1], "spawn": t, "send": None, "reasons": []}
                        order.append(key)
            continue
        if "新しい試合として" in line:
            resets.append(t)
            game += 1
            continue
        # 遅れが書いてある送信(先読みの手順を出現の320ms後で送信)はその値、出現時点の送信は出現の時刻、取り込み時点の送信はその時刻
        d = re.search(r"先読みの手順を(出現時点|出現の(\d+)ms後|取り込み時点)で送信\(turn=(\d+)", line)
        if d:
            rec = turns.get((game, int(d.group(3))))
            if rec is not None and rec["send"] is None:
                if d.group(1) == "出現時点":
                    rec["send"] = rec["spawn"]
                elif d.group(2) is not None:
                    rec["send"] = rec["spawn"] + int(d.group(2)) / 1000
                else:
                    rec["send"] = max(rec["spawn"], t)
            continue
        o = re.search(r"自動操作: turn=(\d+) [A-Z_ ]+ \((\d+)ms", line)
        if o:
            rec = turns.get((game, int(o.group(1))))
            if rec is not None and rec["send"] is None:
                rec["send"] = max(rec["spawn"], t - int(o.group(2)) / 1000)
            continue
        r = re.search(r"(出現時点では|取り込み時点でも|思考の手を)送りません\(turn=(\d+)\): (.*)", line)
        if r:
            rec = turns.get((game, int(r.group(2))))
            if rec is not None:
                reason = re.split(r"[\[(。]", r.group(3))[0].strip()[:30]
                if reason not in rec["reasons"]:
                    rec["reasons"].append(reason)
    out = []
    for key, nxt in zip(order + [None], order[1:] + [None]):
        if key is None:
            break
        rec = turns[key]
        # 一度もキーを送らなかった手番は、下の集計から外れる場合(試合の最後・結果画面の前・自分の連鎖をはさむ)も「送らなかった
        # 手番」として残す(2026-10-24: 出現位置で固定されて窒息した手番は試合の最後なので、自由落下の集計に出なかった)
        if rec["send"] is None and (
            nxt != (key[0], key[1] + 1)
            or any(0 <= r - rec["spawn"] <= RESULT_SCREEN_SEC for r in resets)
            or any(rec["spawn"] < c < turns[nxt]["spawn"] for c in self_chains)
        ):
            out.append({**rec, "unsent": True, "free_fall": False, "delay": None, "next_spawn": None})
            continue
        if nxt != (key[0], key[1] + 1):
            continue  # 次の出現が無い(試合の終わり・読み落とし)
        if any(0 <= r - rec["spawn"] <= RESULT_SCREEN_SEC for r in resets):
            continue
        # 自分の連鎖をはさむ手番は除く: 引き金を置いた瞬間を出現と数えていることがあり(盤面から設置を検出)、連鎖中は組ぷよが
        # 出ないので、遅れは連鎖の時間になる(実機 2026-10-11 11:29 turn=21: 6連鎖の間の8秒が遅れに見えた)
        if any(rec["spawn"] < c < turns[nxt]["spawn"] for c in self_chains):
            continue
        rec["next_spawn"] = turns[nxt]["spawn"]
        rec["free_fall"] = rec["send"] is None or rec["send"] >= rec["next_spawn"]
        rec["delay"] = None if rec["send"] is None else rec["send"] - rec["spawn"]
        out.append(rec)
    return out


def free_fall_report(recs: list[dict], label: str = "") -> list[str]:
    """turn_sends の結果を集計した行"""
    if not recs:
        return [f"[自由落下の疑い]{label}", "  (手番の記録なし)"]
    unsent = [r for r in recs if r.get("unsent")]
    recs = [r for r in recs if not r.get("unsent")]
    delays = [r["delay"] for r in recs if r["delay"] is not None and not r["free_fall"]]
    late = [r for r in recs if not r["free_fall"] and r["delay"] is not None and r["delay"] > LATE_SEC]
    falls = [r for r in recs if r["free_fall"]]
    def why(rs: list[dict]) -> list[tuple[str, int]]:
        c = Counter(r["reasons"][0] if r["reasons"] else "理由の記録なし" for r in rs)
        return c.most_common(8)
    out = [
        f"[自由落下の疑い]{label}",
        f"  手番 {len(recs)} / 出現から送り始めるまで: {_med(delays)}・最大 {max(delays):.2f}s" if delays else f"  手番 {len(recs)}",
        f"  遅れ(>{LATE_SEC}s)で送った手番 {len(late)}: {why(late)}",
        f"  自由落下(次の出現までに送れなかった)手番 {len(falls)}: {why(falls)}",
    ]
    for r in falls[:10]:
        out.append(f"    試合{r['game']} turn={r['turn']} 出現 {r['spawn']:.2f}s → 次の出現 {r['next_spawn']:.2f}s"
                   + (f"(送り始め {r['send']:.2f}s)" if r["send"] is not None else "(送らず)") + f" 理由: {'・'.join(r['reasons']) or '記録なし'}")
    # 試合の最後・自分の連鎖をはさむ手番で一度もキーを送らなかった手番(負けの直前の出現位置での固定など。試合の最初の手番は除く)
    unsent = [r for r in unsent if r["turn"] > 0]
    out.append(f"  試合の最後・連鎖をはさむ手番で送らなかった手番 {len(unsent)}: {why(unsent)}")
    for r in unsent[:10]:
        out.append(f"    試合{r['game']} turn={r['turn']} 出現 {r['spawn']:.2f}s 理由: {'・'.join(r['reasons']) or '記録なし'}")
    return out


def _med(values: list[float]) -> str:
    if not values:
        return "-"
    v = sorted(values)
    return f"中央 {st.median(v):.3f}s(n={len(v)}、10% {v[len(v) // 10]:.3f} / 90% {v[len(v) * 9 // 10]:.3f})"


def summarize(path: Path) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = lines[0] if lines else ""
    t = 0.0
    game = 0  # 試合の番号(手番の番号は試合ごとに0から数え直すので、(試合, 手番) で区別する)
    spawns: dict[str, list[tuple[float, int, bool]]] = {"self": [], "enemy": []}
    disturb: dict[str, list[float]] = {"self": [], "enemy": []}
    resets: list[float] = []
    ops: dict[tuple[int, int], tuple[float, float]] = {}
    cancels: set[tuple[int, int]] = set()  # 設置キャンセルを狙った手番
    cancel_lead: dict[tuple[int, int], str] = {}  # 設置キャンセルの試験の先行の段数(2026-10-23)
    step_base: dict[tuple[int, int], str] = {}  # 段差キャンセルの試験の基準のずらし方(「基準+1F」・「狙わない」、2026-10-25)
    sent_at_spawn: set[tuple[int, int]] = set()
    replans: dict[tuple[int, int], str] = {}  # 同じ手番で何度も出るので最後の1行
    not_sent: Counter = Counter()
    fifth: list[float] = []
    enemy_err: list[float] = []
    stats: list[str] = []
    timed: list[tuple[float, str]] = []
    for line in lines:
        m = STAMP.match(line)
        if m:
            t = float(m.group(1))
        timed.append((t, line))
        e = EVENT.match(line)
        if e:
            who, kind, rest = e.group(2), e.group(3), e.group(4)
            if kind == "spawn":
                n = re.search(r"turn=(\d+)", rest)
                if n:
                    spawns[who].append((t, (game, int(n.group(1))), "initial" in rest))
            if kind in ("chain_start", "chain_end", "garbage_rise"):
                disturb[who].append(t)
            continue
        if "新しい試合として" in line:
            resets.append(t)
            game += 1
        if "先読みの手順を出現時点で送信" in line:
            n = re.search(r"turn=(\d+)", line)
            if n:
                sent_at_spawn.add((game, int(n.group(1))))
        o = re.search(r"自動操作: turn=(\d+) [A-Z ]+ \((\d+)ms, 落下 (\d+)ms", line)
        if o:
            ops[(game, int(o.group(1)))] = (int(o.group(2)) / 1000, int(o.group(3)) / 1000)
            if "設置キャンセル" in line:
                cancels.add((game, int(o.group(1))))
                ld = re.search(r"先行(-?[\d.]+)段", line)
                cancel_lead[(game, int(o.group(1)))] = ld.group(1) if ld else "1"
            sb = re.search(r"段差キャンセル: \S+ 基準([+-]\d+)F", line)
            if sb:
                step_base[(game, int(o.group(1)))] = f"基準{sb.group(1)}F"
        ns = re.search(r"段差キャンセルの対象\(比べるために狙わない\): turn=(\d+)", line)
        if ns:
            step_base[(game, int(ns.group(1)))] = "狙わない"
        rp = re.search(r"着地した盤面が先読みと違う.*turn=(\d+)", line)
        if rp:
            replans[(game, int(rp.group(1)))] = line
        r = re.search(r"出現時点では送りません\(turn=\d+\): (.*)", line)
        if r:
            not_sent[re.split(r"[\[(。]", r.group(1))[0].strip()[:30]] += 1
        if "観測の統計" in line:
            stats.append(line.strip())

    def near_reset(x: float) -> bool:
        return any(0 <= r - x <= RESULT_SCREEN_SEC for r in resets)

    for x, line in timed:
        if "5色目" in line and ("思考スキップ" in line or "先読みスキップ" in line) and not near_reset(x):
            fifth.append(x)
        if "相手の認識エラー" in line and not near_reset(x):
            enemy_err.append(x)

    def intervals(who: str) -> list[tuple[tuple[int, int], float, float]]:
        out = []
        seq = spawns[who]
        for (t1, n1, _), (t2, n2, init2) in zip(seq, seq[1:]):
            if n2 != (n1[0], n1[1] + 1) or init2 or t2 - t1 > 3.0:
                continue
            if any(t1 < d < t2 for d in disturb["self"] + disturb["enemy"]):
                continue
            out.append((n1, t1, t2 - t1))
        return out

    self_iv = intervals("self")
    enemy_iv = intervals("enemy")
    op_rows = [(d, ops[n]) for n, _, d in self_iv if n in ops and n in sent_at_spawn]
    placed = Counter()
    garbage = 0
    for line in replans.values():
        p = re.search(r"置いた2マス: (\S+?)[ /\]]", line)
        placed[p.group(1) if p else "記録なし"] += 1
        g = re.search(r"うちおじゃま(\d+)", line)
        if g and int(g.group(1)) > 0:
            garbage += 1
    sent_total = len(ops)
    out = [
        f"=== {path.name}",
        head[:160],
        f"試合のリセット {len(resets)}回 / 最終時刻 {t:.0f}s",
        "",
        "[速さ](連鎖・おじゃまの落下を挟まない手)",
        f"  1手の間隔 自分: {_med([d for _, _, d in self_iv])}",
        f"  1手の間隔 相手: {_med([d for _, _, d in enemy_iv])}",
    ]
    if op_rows:
        out += [
            f"  出現時点で送った手の操作全体: {_med([o[0] for _, o in op_rows])}",
            f"  うち落下(下キー): {_med([o[1] for _, o in op_rows])}",
            f"  操作後〜次の出現: {_med([d - o[0] for d, o in op_rows])}",
        ]
    out += [
        "",
        "[全自動の置き場所]",
        f"  送った手 {sent_total} / 着地した盤面が先読みと違う {len(replans)}"
        + (f"({100 * len(replans) / sent_total:.0f}%)" if sent_total else ""),
        f"  その手がかり: 置いた2マス {dict(placed)} / おじゃまを含む {garbage}",
        f"  出現時点で送れなかった理由: {not_sent.most_common(6)}",
        "",
        "[認識](試合の終わりの結果画面の時間は除く)",
        f"  5色目で思考・先読みできなかった: {len(fifth)}回 {[round(x, 1) for x in fifth[:8]]}",
        f"  相手の認識エラー: {len(enemy_err)}回 {[round(x, 1) for x in enemy_err[:8]]}",
        "",
        "[観測の統計]",
    ]
    out += [f"  {s}" for s in stats[-5:]] or ["  (記録なし。2026-10-21 より前のログ)"]
    if cancels:
        # 設置キャンセル: 連鎖・おじゃまを挟まない手で、操作の後から次の出現までの時間を比べる(キャンセルが効けば短い)
        after = {n: d - ops[n][0] for n, _, d in self_iv if n in ops}
        with_c = [v for n, v in after.items() if n in cancels]
        without = [v for n, v in after.items() if n not in cancels]
        replan_c = sum(1 for n in cancels if n in replans)
        out += [
            "",
            "[設置キャンセル]",
            f"  狙った手 {len(cancels)} / 置き場所が先読みとずれた {replan_c}({100 * replan_c / len(cancels):.0f}%)",
            f"  操作後〜次の出現 狙った手: {_med(with_c)}",
            f"  操作後〜次の出現 狙わなかった手: {_med(without)}",
        ]
        # 先行の段数ごと(2026-10-23): 本当に置き場所がずれた手(置いた2マスがずれ)と、着地の後〜次の出現
        moved = {n for n, l in replans.items() if "置いた2マス: ずれ" in l}
        for lead in sorted(set(cancel_lead.values()), key=float, reverse=True):
            ns = [n for n in cancels if cancel_lead.get(n) == lead]
            bad = sum(1 for n in ns if n in moved)
            vs = [after[n] for n in ns if n in after]
            out.append(f"  先行{lead}段: 狙った手 {len(ns)} / 置いた2マスがずれた {bad} / 操作後〜次の出現 {_med(vs)}")
    if step_base:
        # 段差キャンセル(2026-10-25): 基準のずらし方ごとに、置いた2マスがずれた手と、出現から次の出現までの間隔(連鎖・おじゃまを
        # 挟まない手)。「狙わない」は同じ条件で普通に置いた手(比べる基準)
        iv = {n: d for n, _, d in self_iv}
        moved = {n for n, l in replans.items() if "置いた2マス: ずれ" in l}
        out += ["", "[段差キャンセル]"]
        for base in sorted(set(step_base.values()), key=lambda b: (b == "狙わない", 0 if b == "狙わない" else int(b[2:-1]))):
            ns = [n for n, b in step_base.items() if b == base]
            bad = sum(1 for n in ns if n in moved)
            out.append(f"  {base}: 手 {len(ns)} / 置いた2マスがずれた {bad} / 出現〜次の出現 {_med([iv[n] for n in ns if n in iv])}")
    out += [""] + free_fall_report(turn_sends(lines))
    return "\n".join(out)


def main() -> None:
    if "--all" in sys.argv[1:]:
        recs: list[dict] = []
        logs = sorted((ROOT / "debug_logs").glob("debug_log_*.txt"))
        for p in logs:
            for r in turn_sends(p.read_text(encoding="utf-8", errors="replace").splitlines()):
                r["log"] = p.name
                recs.append(r)
        print("\n".join(free_fall_report(recs, f"(debug_logs の {len(logs)} 本)")))
        falls = [r for r in recs if r["free_fall"]]
        if falls:
            print("  自由落下の手番(ログごと): " + str(Counter(r["log"] for r in falls).most_common()))
        return
    paths = [Path(a) for a in sys.argv[1:]]
    if not paths:
        logs = sorted((ROOT / "debug_logs").glob("debug_log_*.txt"))
        if not logs:
            print("debug_logs にログがありません")
            return
        paths = [logs[-1]]
    for p in paths:
        print(summarize(p))
        print()


if __name__ == "__main__":
    main()
