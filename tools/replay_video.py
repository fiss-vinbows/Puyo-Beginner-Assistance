"""録画動画を使った認識・状態追跡のリプレイ検証ツール

実機なしで、画面認識(board_reader)と状態追跡(tracker)を動画の全フレームに
適用し、手番の切り替わり・連鎖・予告おじゃまのイベントを時系列で表示する。
--think を付けると spawn ごとに ama(battle)を呼び、提案とその時間も表示する。

使い方:
  venv\\Scripts\\python tools\\replay_video.py input\\xxx.mp4 tests\\fixtures\\calibration_1080p.json [--step 2] [--think]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.capture.calibrate import load_calibration  # noqa: E402
from src.engine.ama_client import AmaClient  # noqa: E402
from src.engine.think_input import build_think_input  # noqa: E402
from src.engine.tracker import PlayerTracker, Snapshot  # noqa: E402
from src.vision.board_reader import cells_to_text, read_board, read_next  # noqa: E402
from src.vision.color_mapper import ColorMapper  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("calibration")
    ap.add_argument("--step", type=int, default=2, help="何フレームおきに処理するか(60fps動画なら2で30fps相当)")
    ap.add_argument("--think", action="store_true", help="spawn ごとに ama で思考する")
    ap.add_argument("--quiet", action="store_true", help="イベント以外を表示しない")
    args = ap.parse_args()

    cal = load_calibration(Path(args.calibration))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0

    trackers = {"self": PlayerTracker(), "enemy": PlayerTracker()}
    players = {"self": cal.self_player, "enemy": cal.enemy_player}
    mapper = ColorMapper()
    client = AmaClient() if args.think else None

    frame_index = 0
    recog_total = 0.0
    recog_n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            if frame_index % args.step:
                continue
            t = frame_index / fps

            t0 = time.perf_counter()
            snaps: dict[str, Snapshot] = {}
            for who, pc in players.items():
                board = read_board(frame, pc.geometry())
                n1 = read_next(frame, pc.next_rects[0])
                n2 = read_next(frame, pc.next_rects[1])
                snaps[who] = Snapshot(t=t, cells=board.cells, next1=n1, next2=n2, pending_garbage=board.pending_garbage, eyeless_cells=board.eyeless_cells)
            recog_total += time.perf_counter() - t0
            recog_n += 1

            # 本体(src/app.py)と同じく、自分・相手の両方を更新し終えてから思考する
            think = False
            for who, snap in snaps.items():
                for ev in trackers[who].update(snap):
                    print(f"[{t:7.2f}s] {who:5s} {ev.kind:11s} {ev.detail}")
                    if ev.kind == "turn_ready" and who == "self":
                        think = True
            if think and client is not None:
                mapper = _think(client, mapper, trackers, t)
    finally:
        if client is not None:
            client.close()

    print(f"\n認識時間: 平均 {recog_total / max(recog_n, 1) * 1000:.1f} ms/フレーム(両プレイヤー分)")
    for who, tr in trackers.items():
        print(f"{who}: 手番 {tr.turn} 回, 最終盤面:\n{cells_to_text(tr.settled)}")


def _think(client: AmaClient, mapper: ColorMapper, trackers: dict[str, PlayerTracker], t: float) -> ColorMapper:
    """本体と同じ入力(src/engine/think_input.py: 全消し・連鎖中の消去個数・ボーナス込み)で思考する。
    採用した色の割当を返す。"""
    me, en = trackers["self"], trackers["enemy"]
    if me.current_pair is None or me.next1 is None or me.next2 is None:
        return mapper
    try:
        inp = build_think_input(mapper, me, en, me.settled, [me.current_pair, me.next1, me.next2])
    except ValueError as e:
        print(f"           思考スキップ: {e}")
        return mapper
    if inp.enemy_error:
        print(f"           相手の認識エラー: {inp.enemy_error} → 相手を仮の状態にして思考")
    en_state = inp.en_state
    enemy_unknown = en_state is None
    if en_state is None:
        from src.engine.ama_client import PlayerState
        from src.engine.puyo_field import PuyoField

        en_state = PlayerState(PuyoField(), list(inp.my_state.pairs))
    move = client.battle(inp.my_state, en_state, enemy_unknown=enemy_unknown)
    print(
        f"           提案: x={move.x} {move.rotation} [{move.label}] eval={move.eval} fire={move.fire_reason} "
        f"自分:最大{move.self_analysis.max.count}連鎖{move.self_analysis.max.score}点 "
        f"相手:最大{move.enemy_analysis.max.count}連鎖{move.enemy_analysis.max.score}点/潰し{move.enemy_analysis.harass.count}連鎖 "
        f"({move.think_ms}ms)"
    )
    return inp.mapper


if __name__ == "__main__":
    main()
