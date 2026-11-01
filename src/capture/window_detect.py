"""Steam(PC)版のウィンドウ検出(2026-10-28、プラグイン方式の段階2。利用者はウィンドウ表示で遊ぶ。ぷよぷよテトリス2・ぷよぷよeスポーツ)

ぷよぷよテトリス2 のウィンドウを探し、その描画領域(クライアント領域)の中の盤面・ネクスト枠の位置を
1920×1080 の画面での位置(録画で検証済みの tests/fixtures/calibration_1080p.json と同じ値)を縮めて求める。
16:9 でないウィンドウでは、画面は縦横比を保って中央に描かれる(上下か左右に黒帯)とみなす。
ウィンドウの画像から盤面の暗い矩形を見つけられた時は(盤面が空に近い時)、その位置に合わせ直す。
手動のキャリブレーション(Switch 対応プラグイン)は要らない。

座標は物理ピクセル(PyQt6 のプロセスは DPI を認識するので、Win32 の座標も物理ピクセルで返る)。
"""

from __future__ import annotations

import ctypes
import json
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..paths import app_root
from .calibrate import CalibrationResult, PlayerCalibration, Rect
from .geometry import Quad, detect_board_quads
from .screen_capture import CaptureRegion

REFERENCE_SIZE = (1920, 1080)
# 1920×1080 の画面での盤面(可視12段の4隅)とネクスト枠。一人用(とことん)も自分の盤面・枠は対戦と同じ位置
# (利用者の config/calibration.json と calibration_solo.json で確認)
REFERENCE_LAYOUT = CalibrationResult(
    self_player=PlayerCalibration(
        board_quad=((272.0, 159.0), (672.0, 159.0), (672.0, 882.0), (272.0, 882.0)),
        next_rects=[(720, 160, 60, 125), (775, 293, 46, 102)],
    ),
    enemy_player=PlayerCalibration(
        board_quad=((1248.0, 159.0), (1648.0, 159.0), (1648.0, 882.0), (1248.0, 882.0)),
        next_rects=[(1140, 160, 60, 125), (1099, 293, 46, 102)],
    ),
)
GAME_EXE_NAMES = ("puyopuyotetris2.exe",)
# PC 版はぷよぷよテトリス2 とぷよぷよeスポーツ(ウィンドウのタイトルは Puyo Puyo Champions のことがある。利用者 2026-10-28)
GAME_TITLE_KEYS = ("puyopuyotetris2", "ぷよぷよテトリス2", "puyopuyochampions", "ぷよぷよeスポーツ")
# メイン画面の「ゲームのウィンドウを検出」で一覧から選んだウィンドウ(実行ファイル名とタイトル)。次からはこれでも探す
# (2026-10-28 実機: 決め打ちの実行ファイル名・タイトルでは見つからなかった)
SAVED_WINDOW_PATH = app_root() / "config" / "game_window.json"
# 一覧に出すウィンドウの最小の大きさ(小さい道具のウィンドウを除く)
MIN_LIST_SIZE = (320, 180)
# 見つけた盤面の暗い矩形を採用する、計算した位置とのずれの上限(盤面の幅に対する割合)
DETECT_TOLERANCE = 0.08


@dataclass(frozen=True)
class GameWindow:
    hwnd: int
    title: str
    left: int  # クライアント領域の左上(画面の物理ピクセル)
    top: int
    width: int
    height: int

    def region(self) -> CaptureRegion:
        return CaptureRegion(left=self.left, top=self.top, width=self.width, height=self.height)


@dataclass(frozen=True)
class SavedWindow:
    """利用者が選んだゲームのウィンドウ。exe(小文字のファイル名)が合えばそれ、無ければ title(正規化して一致)で探す。"""

    exe: str = ""
    title: str = ""

    def matches(self, title: str, exe: str) -> bool:
        if self.exe:
            return exe == self.exe
        return bool(self.title) and normalize_title(title) == normalize_title(self.title)


def load_saved_window(path: Path = SAVED_WINDOW_PATH) -> SavedWindow | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        saved = SavedWindow(exe=str(data.get("exe", "")).lower(), title=str(data.get("title", "")))
    except (OSError, ValueError, AttributeError):
        return None
    return saved if saved.exe or saved.title else None


def save_saved_window(saved: SavedWindow, path: Path = SAVED_WINDOW_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"exe": saved.exe, "title": saved.title}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_game_window(title: str, exe: str, saved: SavedWindow | None = None) -> bool:
    """ゲームのウィンドウか(決め打ちのタイトル・実行ファイル名か、利用者が選んだもの)。"""
    return is_game_title(title) or exe in GAME_EXE_NAMES or (saved is not None and saved.matches(title, exe))


def normalize_title(title: str) -> str:
    """比べやすいタイトル(™・®・記号・空白を除き、全角を半角に、小文字に)。"""
    title = "".join(c for c in title if c not in "™®©")
    return "".join(c for c in unicodedata.normalize("NFKC", title).lower() if c.isalnum())


def is_game_title(title: str) -> bool:
    norm = normalize_title(title)
    return any(key in norm for key in GAME_TITLE_KEYS)


def content_rect(width: int, height: int) -> tuple[float, float, float]:
    """クライアント領域の中で画面が描かれる位置(左の余白, 上の余白, 1920×1080 に対する倍率)。"""
    rw, rh = REFERENCE_SIZE
    scale = min(width / rw, height / rh)
    return (width - rw * scale) / 2, (height - rh * scale) / 2, scale


def _board_box(quad: Quad) -> tuple[float, float, float, float]:
    (x0, y0), _, (x1, y1), _ = quad
    return x0, y0, x1 - x0, y1 - y0


def _player_on_board(ref: PlayerCalibration, board: Quad) -> PlayerCalibration:
    """基準の盤面とネクスト枠の位置関係を保ったまま、盤面を board に置いた時の枠。"""
    rx, ry, rw, rh = _board_box(ref.board_quad)
    bx, by, bw, bh = _board_box(board)
    rects: list[Rect] = []
    for x, y, w, h in ref.next_rects:
        rects.append((round(bx + (x - rx) * bw / rw), round(by + (y - ry) * bh / rh), round(w * bw / rw), round(h * bh / rh)))
    return PlayerCalibration(board_quad=board, next_rects=rects)


def _scaled_quad(quad: Quad, ox: float, oy: float, scale: float) -> Quad:
    return tuple((round(ox + x * scale), round(oy + y * scale)) for x, y in quad)  # type: ignore[return-value]


def layout_calibration(window: GameWindow, solo: bool = False) -> CalibrationResult:
    """ウィンドウの大きさから求めた盤面・ネクスト枠(画面の物理ピクセル)。一人用は相手を自分の複製にする(使わない)。"""
    ox, oy, scale = content_rect(window.width, window.height)
    ox += window.left
    oy += window.top
    players = [
        _player_on_board(ref, _scaled_quad(ref.board_quad, ox, oy, scale))
        for ref in (REFERENCE_LAYOUT.self_player, REFERENCE_LAYOUT.enemy_player)
    ]
    if solo:
        players[1] = players[0]
    return CalibrationResult(self_player=players[0], enemy_player=players[1])


def refine_with_detected(calibration: CalibrationResult, quads: list[Quad], solo: bool = False) -> CalibrationResult:
    """見つけた盤面の暗い矩形(画面の物理ピクセル)のうち、計算した位置に近いものに盤面を合わせ直す。
    近いものが無い側は計算した位置のまま。"""

    def refine(player: PlayerCalibration) -> PlayerCalibration:
        x, y, w, h = _board_box(player.board_quad)
        tol = w * DETECT_TOLERANCE
        for quad in quads:
            qx, qy, qw, qh = _board_box(quad)
            if abs(qx - x) <= tol and abs(qy - y) <= tol and abs(qw - w) <= tol and abs(qh - h) <= tol:
                return _player_on_board(player, quad)
        return player

    self_player = refine(calibration.self_player)
    enemy_player = self_player if solo else refine(calibration.enemy_player)
    if not solo and self_player is calibration.self_player and enemy_player is calibration.enemy_player and len(quads) == 2:
        # どちらも計算した位置に無いが、同じ大きさの盤面が2つ見えている: 画面の配置が違うゲーム(ぷよぷよeスポーツなど)
        # とみなし、見えた盤面に合わせる(ネクスト枠は盤面との位置関係をぷよぷよテトリス2 と同じとみなす)
        (_, _, w0, h0), (_, _, w1, h1) = _board_box(quads[0]), _board_box(quads[1])
        if abs(w0 - w1) <= w0 * DETECT_TOLERANCE and abs(h0 - h1) <= h0 * DETECT_TOLERANCE:
            self_player = _player_on_board(REFERENCE_LAYOUT.self_player, quads[0])
            enemy_player = _player_on_board(REFERENCE_LAYOUT.enemy_player, quads[1])
    return CalibrationResult(self_player=self_player, enemy_player=enemy_player)


def detect_calibration(
    window: GameWindow, solo: bool = False, grab: Callable[[CaptureRegion], np.ndarray] | None = None
) -> CalibrationResult:
    """ウィンドウの盤面・ネクスト枠。grab(領域 → BGR 画像)を渡すと、その画像の盤面の矩形で合わせ直す。"""
    calibration = layout_calibration(window, solo)
    if grab is None:
        return calibration
    region = window.region()
    try:
        image = grab(region)
    except Exception:  # 取得できなくても、計算した位置で動かせる
        return calibration
    min_height = max(50, round(calibration.self_player.geometry().bounding_box()[3] * 0.6))
    quads = [
        tuple((x + region.left, y + region.top) for x, y in q)  # type: ignore[misc]
        for q in detect_board_quads(image, min_height=min_height)
    ]
    return refine_with_detected(calibration, quads, solo)  # type: ignore[arg-type]


# ---- Win32(Windows だけ) ----

def _exe_name(pid: int) -> str:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
    kernel32.QueryFullProcessImageNameW.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_wchar_p,
                                                    ctypes.POINTER(ctypes.c_uint32)]
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = ctypes.c_uint32(len(buf))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return buf.value.replace("/", "\\").rsplit("\\", 1)[-1].lower()
    finally:
        kernel32.CloseHandle(handle)


# ウィンドウの大きさのプリセット(クライアント領域。ゲームは 16:9 で描く。2026-10-28、利用者「ピクセル単位の調整は困難」)
CLIENT_SIZE_PRESETS = ((1280, 720), (1600, 900), (1920, 1080), (2560, 1440))


def resized_window_rect(
    window: tuple[int, int, int, int], client: tuple[int, int, int, int], size: tuple[int, int],
    work_area: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """クライアント領域を size にする時のウィンドウの位置と大きさ(左, 上, 幅, 高さ)。枠・タイトルバーの分は今の差をそのまま足す。
    左上はそのままにし、作業領域(左, 上, 幅, 高さ)からはみ出す時は収まるよう左・上へずらす(入りきらなければ左上に合わせる)。"""
    wx, wy, ww, wh = window
    _, _, cw, ch = client
    new_w, new_h = size[0] + (ww - cw), size[1] + (wh - ch)
    x, y = wx, wy
    if work_area is not None:
        ax, ay, aw, ah = work_area
        x = max(ax, min(x, ax + aw - new_w))
        y = max(ay, min(y, ay + ah - new_h))
    return x, y, new_w, new_h


def resize_client(hwnd: int, size: tuple[int, int]) -> tuple[int, int] | None:
    """ウィンドウのクライアント領域を size(物理ピクセル)にする。結果のクライアント領域の大きさを返す(できなければ None)。
    ゲームが大きさを丸めることがあるので、1回ずれたらもう1回だけ合わせ直す。"""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    wintypes.UINT]
    user32.MonitorFromWindow.restype = wintypes.HANDLE
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    user32.GetMonitorInfoW.argtypes = [wintypes.HANDLE, ctypes.POINTER(MONITORINFO)]
    for _ in range(2):
        client = client_rect(hwnd)
        rect = wintypes.RECT()
        if client is None or not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        if client[2:] == size:
            return size
        work = None
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        monitor = user32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
        if monitor and user32.GetMonitorInfoW(monitor, ctypes.byref(info)):
            r = info.rcWork
            work = (r.left, r.top, r.right - r.left, r.bottom - r.top)
        window = (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        x, y, w, h = resized_window_rect(window, client, size, work)
        if not user32.SetWindowPos(hwnd, None, x, y, w, h, 0x0004 | 0x0010):  # SWP_NOZORDER | SWP_NOACTIVATE
            return None
    client = client_rect(hwnd)
    return None if client is None else client[2:]


def client_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """ウィンドウのクライアント領域(左, 上, 幅, 高さ。画面の物理ピクセル)。最小化中・閉じた時は None。"""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsIconic.argtypes = [wintypes.HWND]
    user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None
    return origin.x, origin.y, width, height


@dataclass(frozen=True)
class WindowEntry:
    """表示中のウィンドウ(一覧から選ぶ用)。"""

    window: GameWindow
    exe: str


def list_windows() -> list[WindowEntry]:
    """表示中でタイトルのある、ある程度大きいウィンドウ(画面の手前から順)。Windows 以外は空。"""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [enum_proc, wintypes.LPARAM]
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    found: list[WindowEntry] = []

    def visit(hwnd, _lparam):  # noqa: ANN001, ANN202
        # 例外を外に出すと EnumWindows が途中で止まるので、1つのウィンドウの失敗は飛ばして続ける
        try:
            if not hwnd or not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            rect = client_rect(hwnd)
            if rect is None or rect[2] < MIN_LIST_SIZE[0] or rect[3] < MIN_LIST_SIZE[1]:
                return True
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            try:
                exe = _exe_name(pid.value)
            except OSError:
                exe = ""
            found.append(WindowEntry(GameWindow(int(hwnd), buf.value, *rect), exe))
        except Exception:  # noqa: BLE001
            pass
        return True

    user32.EnumWindows(enum_proc(visit), 0)
    return found


def title_bar_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """ウィンドウのタイトルバー(左, 上, 幅, 高さ。画面の物理ピクセル)。枠なし・全画面・読めない時は None。"""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    client = client_rect(hwnd)
    if client is None:
        return None
    rect = wintypes.RECT()
    try:  # 見えている枠(Windows 10 以降は GetWindowRect に見えない縁が入るので DWM に聞く)
        dwmapi = ctypes.WinDLL("dwmapi")
        dwmapi.DwmGetWindowAttribute.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        ok = dwmapi.DwmGetWindowAttribute(hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)) == 0  # DWMWA_EXTENDED_FRAME_BOUNDS
    except OSError:
        ok = False
    if not ok:
        user32 = ctypes.WinDLL("user32")
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
    height = client[1] - rect.top
    if height < 16:
        return None
    return rect.left, rect.top, rect.right - rect.left, height


def find_game_window(saved: SavedWindow | None = None, windows: list[WindowEntry] | None = None) -> GameWindow | None:
    """表示中のぷよぷよテトリス2 のウィンドウ(実行ファイル名・タイトル、または利用者が選んだもの)。複数あれば一番大きいもの。
    Windows 以外・見つからない時は None。windows はテスト用(既定は list_windows())。"""
    if saved is None:
        saved = load_saved_window()
    entries = list_windows() if windows is None else windows
    found = [e.window for e in entries if is_game_window(e.window.title, e.exe, saved)]
    if not found:
        return None
    return max(found, key=lambda w: w.width * w.height)
