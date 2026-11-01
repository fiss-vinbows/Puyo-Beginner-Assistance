"""観測ループを止めないための裏方のスレッド(画像保存・思考)。

観測ループ(src/app.py の _loop)が思考(ama ブリッジの呼び出し)・2手先の先読み・画像の PNG 保存を
同期で行っていたため、思考のたびに約0.4秒画面を見られなかった(実機 2026-09-27: 観測の空白の中央値 406ms)。
その間は着地や次の組ぷよの出現を検出できず、1手が相手(CPU)より約0.2秒遅かった。
重い処理をこのスレッドに渡し、観測ループは結果を待たずに画面を見続ける。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


class ImageSaver:
    """画像の保存を別スレッドで行う。保存が追いつかない時は観測を待たせず捨てる(捨てた数を数える)。"""

    def __init__(self, maxsize: int = 16) -> None:
        self._queue: queue.Queue[tuple[Path, np.ndarray, list[int]] | None] = queue.Queue(maxsize=maxsize)
        self.dropped = 0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def save(self, path: Path, frame: np.ndarray, params: list[int] | None = None) -> bool:
        """保存を依頼する。frame は呼び出し側で書き換えない配列を渡すこと。"""
        try:
            self._queue.put_nowait((path, frame, params or []))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def stop(self) -> None:
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, frame, params = item
            try:
                path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(path), frame, params)
            except Exception:  # noqa: BLE001 保存の失敗で思考・観測を止めない
                pass


class ThinkWorker:
    """思考(ama ブリッジの呼び出し)を1本の別スレッドで順に実行し、結果を観測ループへ返す。

    依頼は kind ごとに最新の1件だけを残す(古い入力の思考は結果が出ても捨てられるので、先に捨てる)。
    ブリッジは標準入出力で1本の通信なので、呼び出しはこのスレッドからだけ行う
    (観測ループ側で呼ぶ時は lock を取る)。
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()  # ブリッジの呼び出しを直列にする
        self._jobs: dict[str, tuple[object, Callable[[], object]]] = {}
        self._order: list[str] = []  # 優先順(先頭から実行)
        self._cond = threading.Condition()
        self._results: queue.Queue[tuple[str, object, object, BaseException | None]] = queue.Queue()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, kind: str, ctx: object, fn: Callable[[], object], urgent: bool = False) -> None:
        """kind の依頼を登録する(同じ kind の未実行の依頼は置き換える)。urgent なら他の kind より先に実行する。"""
        with self._cond:
            self._jobs[kind] = (ctx, fn)
            if kind in self._order:
                self._order.remove(kind)
            if urgent:
                self._order.insert(0, kind)
            else:
                self._order.append(kind)
            self._cond.notify()

    def cancel(self, kind: str) -> bool:
        """実行前の kind の依頼を取り消す。取り消せたら True(実行中・依頼なしなら False。実行中のものは結果が届く)。"""
        with self._cond:
            removed = self._jobs.pop(kind, None) is not None
            if kind in self._order:
                self._order.remove(kind)
            return removed

    def poll(self) -> list[tuple[str, object, object, BaseException | None]]:
        """終わった思考の結果 (kind, ctx, 結果, 例外) を取り出す(待たない)。"""
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def stop(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._running and not self._order:
                    self._cond.wait()
                if not self._running:
                    return
                kind = self._order.pop(0)
                ctx, fn = self._jobs.pop(kind)
            try:
                with self.lock:
                    result = fn()
                self._results.put((kind, ctx, result, None))
            except BaseException as exc:  # noqa: BLE001 例外も観測ループへ返して判断させる
                self._results.put((kind, ctx, None, exc))
