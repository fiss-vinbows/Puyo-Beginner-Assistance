"""CPU 使用率と観測の FPS を測る(2026-10-24、利用者「ぷよぷよ画面枠の外で影響のないところに CPU 使用率と FPS を表示」)。

CPU は PC 全体の使用率(Windows の GetSystemTimes。追加のライブラリは使わない)。FPS は PBA が画面を取り込んで読む回数で、
ゲーム自体の FPS ではない(外からは測れない)。
"""

from __future__ import annotations

import ctypes
import sys
from collections import deque


class _FileTime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    def value(self) -> int:
        return (self.high << 32) | self.low


def _system_times() -> tuple[int, int, int] | None:
    """(待機, カーネル, ユーザー) の累計時間。カーネルには待機を含む。Windows 以外・失敗時は None"""
    if sys.platform != "win32":
        return None
    idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
    try:
        ok = ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    except (OSError, AttributeError):
        return None
    return (idle.value(), kernel.value(), user.value()) if ok else None


class CpuMeter:
    """前回の sample からの PC 全体の CPU 使用率(%)。初回と測れない時は None"""

    def __init__(self, read=_system_times) -> None:
        self._read = read
        self._last = read()

    def sample(self) -> float | None:
        now = self._read()
        last, self._last = self._last, now
        if now is None or last is None:
            return None
        idle = now[0] - last[0]
        total = (now[1] - last[1]) + (now[2] - last[2])
        if total <= 0:
            return None
        return max(0.0, min(100.0, (total - idle) * 100.0 / total))


class FpsMeter:
    """直近 window 秒の観測の回数から FPS を出す"""

    def __init__(self, window: float = 1.0) -> None:
        self.window = window
        self._ticks: deque[float] = deque()

    def tick(self, t: float) -> None:
        self._ticks.append(t)
        while self._ticks and t - self._ticks[0] > self.window:
            self._ticks.popleft()

    def fps(self) -> float:
        if len(self._ticks) < 2:
            return 0.0
        span = self._ticks[-1] - self._ticks[0]
        return (len(self._ticks) - 1) / span if span > 0 else 0.0


def perf_text(cpu: float | None, fps: float) -> str:
    """オーバーレイに出す1行"""
    return f"CPU {'--' if cpu is None else f'{cpu:.0f}'}% / 観測 {fps:.0f}fps"
