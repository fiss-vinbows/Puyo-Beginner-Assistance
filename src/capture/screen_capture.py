"""mssによる画面領域の高速キャプチャ"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def to_mss_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}


class ScreenCapture:
    """1つの画面領域を継続的にキャプチャするためのラッパー。

    mss.mss()はスレッドごとに1インスタンスが必要なため、
    このクラスのインスタンスは呼び出し元スレッド内でのみ使い回すこと。
    """

    def __init__(self) -> None:
        self._sct = mss.mss()

    def grab(self, region: CaptureRegion) -> np.ndarray:
        """指定領域をキャプチャし、RGB順のnumpy配列 (height, width, 3) で返す"""
        shot = self._sct.grab(region.to_mss_dict())
        # mssはBGRA順で返すため、RGBに変換する
        frame = np.array(shot)[:, :, :3][:, :, ::-1]
        return frame

    def grab_bgr(self, region: CaptureRegion) -> np.ndarray:
        """指定領域をキャプチャし、BGR順の連続した numpy 配列 (height, width, 3) で返す(OpenCV 用)。

        grab() で RGB にしてから BGR に戻すと、並べ替えと連続化のコピーで観測1回あたり約9ms かかっていた
        (1698×1040)。mss の BGRA をそのまま OpenCV で BGR にすると約4ms(2026-10-21、結果は同じ画素)。"""
        shot = self._sct.grab(region.to_mss_dict())
        bgra = np.frombuffer(shot.bgra, np.uint8).reshape(shot.height, shot.width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

    def close(self) -> None:
        self._sct.close()

    def __enter__(self) -> "ScreenCapture":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class PrefetchCapture:
    """画面の取得を別スレッドで1枚先に進めておく(取得と読み取りを並行させる)。使い方は ScreenCapture.grab_bgr と同じ。

    観測1回(約40ms)のうち画面の取得が約26msで、読み取り(約14ms)の間は取得が止まっていた(2026-10-21 の計測)。
    受け取られたらすぐ次の1枚の取得を始めるので、読み取りの間に次の画面が用意され、観測の間隔が取得の時間まで縮む。
    取得は受け取られた分だけ行う(画面を取り続けて CPU を使わない)。mss は使うスレッドで作る必要があるので、
    取得用のスレッドの中で ScreenCapture を作る(プロセスで最初の作成は約1.2秒かかるので、作ってから返す。
    観測ループの最初の1回に入ると「観測の空白」になった)。領域は最初の grab_bgr で決まり、以後は同じ領域を取る。
    grab は差し替えられる(テスト用)。"""

    def __init__(self, grab=None) -> None:  # noqa: ANN001
        self._grab = grab  # (region) -> BGR 画像。None なら取得用のスレッドで ScreenCapture.grab_bgr を使う
        self._region: CaptureRegion | None = None
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._error: BaseException | None = None
        self._want = False  # 次の1枚を取ってよい(前の1枚が受け取られた)
        self._stop = False
        self._ready = threading.Event()  # 取得用のスレッドの準備(ScreenCapture の作成)が済んだ
        self._thread = threading.Thread(target=self._run, name="PrefetchCapture", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10.0)

    def _run(self) -> None:
        cap = None
        try:
            grab = self._grab
            if grab is None:
                cap = ScreenCapture()
                grab = cap.grab_bgr
            self._ready.set()
            while True:
                with self._cond:
                    while not self._want and not self._stop:
                        self._cond.wait()
                    if self._stop:
                        return
                    self._want = False
                    region = self._region
                frame = grab(region)
                with self._cond:
                    self._frame = frame
                    self._cond.notify_all()
        except BaseException as exc:  # noqa: BLE001  受け取る側(観測ループ)で投げ直す
            with self._cond:
                self._error = exc
                self._cond.notify_all()
        finally:
            self._ready.set()
            if cap is not None:
                cap.close()

    def grab_bgr(self, region: CaptureRegion, timeout: float = 5.0) -> np.ndarray:
        """先に取ってある1枚を返し、次の1枚の取得を始めさせる(まだ無ければ取れるまで待つ)。"""
        with self._cond:
            if self._region is None:
                self._region = region
                self._want = True
                self._cond.notify_all()
            elif region != self._region:
                raise ValueError("PrefetchCapture は最初に指定した領域だけを取得します")
            if not self._cond.wait_for(lambda: self._frame is not None or self._error is not None, timeout):
                raise TimeoutError("画面の取得が返ってきません")
            if self._error is not None:
                raise self._error
            frame, self._frame = self._frame, None
            self._want = True
            self._cond.notify_all()
            return frame

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "PrefetchCapture":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
