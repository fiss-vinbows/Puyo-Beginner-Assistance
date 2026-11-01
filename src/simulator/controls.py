"""シミュレーションモードの入力: キーボード・コントローラーの割り当て(変更可能)と押しっぱなしの連続入力。

TBA(../Tetris)の src/education/keybindings.py・gamepad.py と、window.py の割り当て画面を
ぷよぷよ用の操作に合わせて移したもの。

- キー名は Qt の表記("Left", "Z", "Space")。修飾キーは "Shift+R" のように前置する
- コントローラーは Windows 標準の winmm(joyGetPosEx)で読む(追加ライブラリ不要)。
  入力名は "Button1"〜"Button32"、十字キー "PovUp/PovDown/PovLeft/PovRight"、スティック "X-/X+/Y-/Y+"
- 割り当ては config/simulator_keys.json・config/simulator_pad.json に保存する
- 連続入力は OS のキーリピートに頼らず独自に行う(左右移動だけ。TBA で単押し2マスになった対策と同じ)
"""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets

from ..paths import app_root

KEYS_PATH = app_root() / "config" / "simulator_keys.json"
PAD_PATH = app_root() / "config" / "simulator_pad.json"

# 操作名 → 説明(設定画面・一覧表示に使う)
ACTIONS: dict[str, str] = {
    "move_left": "左へ移動",
    "move_right": "右へ移動",
    "rotate_ccw": "左回転",
    "rotate_cw": "右回転",
    "drop": "置く",
    "undo": "一手戻す",
    "reset_same": "同一配列でリセット",
    "reset_new": "別配列でリセット",
    "toggle_hints": "提示の表示/非表示",
    "screenshot": "画像を保存",
}

DEFAULT_BINDINGS: dict[str, str] = {
    "move_left": "Left",
    "move_right": "Right",
    "rotate_ccw": "Z",
    "rotate_cw": "X",
    "drop": "Down",
    "undo": "Backspace",
    "reset_same": "R",
    "reset_new": "Shift+R",
    "toggle_hints": "F3",
    "screenshot": "F12",
}

# 別配列リセットは誤操作防止のため初期状態では割り当てない(TBA と同じ)
DEFAULT_PAD_BINDINGS: dict[str, list[str]] = {
    "move_left": ["PovLeft", "X-"],
    "move_right": ["PovRight", "X+"],
    "rotate_ccw": ["Button1"],
    "rotate_cw": ["Button2"],
    "drop": ["PovDown", "Y+"],
    "undo": ["Button7"],
    "reset_same": ["Button8"],
    "reset_new": [],
    "toggle_hints": [],
    "screenshot": [],
}

REPEATABLE_ACTIONS = frozenset({"move_left", "move_right"})
REPEAT_DELAY_MS = 170  # 押しっぱなしで連続入力が始まるまで
REPEAT_INTERVAL_MS = 50  # 連続入力の間隔
POLL_MS = 16  # コントローラーの読み取りと連続入力の判定の間隔

_MODIFIERS = {
    "Shift": QtCore.Qt.KeyboardModifier.ShiftModifier,
    "Ctrl": QtCore.Qt.KeyboardModifier.ControlModifier,
    "Alt": QtCore.Qt.KeyboardModifier.AltModifier,
}


# ---- キーボード ----
def load_bindings(path: Path = KEYS_PATH) -> dict[str, str]:
    """保存済みの割り当てを読む。無い・壊れている・不足している項目は初期値で補う。"""
    bindings = dict(DEFAULT_BINDINGS)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for action in ACTIONS:
                value = data.get(action)
                if isinstance(value, str):
                    bindings[action] = value
    except (OSError, ValueError):
        pass
    return bindings


def save_bindings(bindings: dict[str, str], path: Path = KEYS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bindings, ensure_ascii=False, indent=2), encoding="utf-8")


def key_name(key: int, modifiers: QtCore.Qt.KeyboardModifier) -> str | None:
    """キーを "Shift+R" のような割り当て用の名前にする。修飾キー単独なら None。"""
    name = QtCore.Qt.Key(key).name.removeprefix("Key_")
    if name in ("Shift", "Control", "Alt", "Meta", "unknown"):
        return None
    return "".join(f"{label}+" for label, flag in _MODIFIERS.items() if modifiers & flag) + name


def key_event_name(event: QtGui.QKeyEvent) -> str | None:
    return key_name(event.key(), event.modifiers())


def action_for(bindings: dict[str, str], event_name: str) -> str | None:
    for action, name in bindings.items():
        if name and name == event_name:
            return action
    return None


def bindings_help(bindings: dict[str, str]) -> str:
    return " ".join(f"{text}:{bindings.get(action) or 'なし'}" for action, text in ACTIONS.items())


# ---- コントローラー ----
class _JOYINFOEX(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint32) for name in (
        "dwSize", "dwFlags", "dwXpos", "dwYpos", "dwZpos", "dwRpos", "dwUpos", "dwVpos",
        "dwButtons", "dwButtonNumber", "dwPOV", "dwReserved1", "dwReserved2",
    )]


_JOY_RETURNALL = 0xFF
_POV_CENTERED = 0xFFFF
_AXIS_THRESHOLD = 0.5  # 中心からの割合がこれを超えたら倒したとみなす


def inputs_from_raw(buttons: int, pov: int, x: int, y: int) -> set[str]:
    """winmm の生の値を入力名の集合にする(軸は0〜65535、中心32767)。"""
    active = {f"Button{i + 1}" for i in range(32) if buttons & (1 << i)}
    if pov != _POV_CENTERED:
        deg = pov / 100.0
        if deg >= 315 or deg <= 45:
            active.add("PovUp")
        if 45 <= deg <= 135:
            active.add("PovRight")
        if 135 <= deg <= 225:
            active.add("PovDown")
        if 225 <= deg <= 315:
            active.add("PovLeft")
    for name, value in (("X", x), ("Y", y)):
        offset = (value - 32767.5) / 32767.5
        if offset <= -_AXIS_THRESHOLD:
            active.add(f"{name}-")
        elif offset >= _AXIS_THRESHOLD:
            active.add(f"{name}+")
    return active


class Gamepad:
    """接続されている最初のコントローラーを読む。無ければ connected() が False。"""

    def __init__(self) -> None:
        self._winmm = ctypes.WinDLL("winmm") if sys.platform == "win32" else None
        self.device_id: int | None = None
        self.find()

    def _read(self, device_id: int) -> _JOYINFOEX | None:
        if self._winmm is None:
            return None
        info = _JOYINFOEX()
        info.dwSize = ctypes.sizeof(_JOYINFOEX)
        info.dwFlags = _JOY_RETURNALL
        if self._winmm.joyGetPosEx(device_id, ctypes.byref(info)) != 0:
            return None
        return info

    def find(self) -> bool:
        self.device_id = next((i for i in range(16) if self._read(i) is not None), None)
        return self.device_id is not None

    def connected(self) -> bool:
        return self.device_id is not None

    def poll(self) -> set[str]:
        """今押されている入力。切断されたら空集合(押しっぱなしが残らない)。"""
        if self.device_id is None:
            return set()
        info = self._read(self.device_id)
        if info is None:
            self.device_id = None
            return set()
        return inputs_from_raw(info.dwButtons, info.dwPOV, info.dwXpos, info.dwYpos)


def load_pad_bindings(path: Path = PAD_PATH) -> dict[str, list[str]]:
    bindings = {action: list(inputs) for action, inputs in DEFAULT_PAD_BINDINGS.items()}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for action in ACTIONS:
                value = data.get(action)
                if isinstance(value, list) and all(isinstance(v, str) for v in value):
                    bindings[action] = value
    except (OSError, ValueError):
        pass
    return bindings


def save_pad_bindings(bindings: dict[str, list[str]], path: Path = PAD_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bindings, ensure_ascii=False, indent=2), encoding="utf-8")


def actions_for_inputs(bindings: dict[str, list[str]], inputs: set[str]) -> set[str]:
    return {action for action, names in bindings.items() if inputs & set(names)}


# ---- 押しっぱなしの連続入力 ----
class RepeatTracker:
    """押された瞬間に1回、REPEATABLE_ACTIONS だけは delay_ms 後から interval_ms 間隔で繰り返す。"""

    def __init__(self) -> None:
        self.pressed: set[str] = set()
        self.repeat_at: dict[str, float] = {}

    def update(self, actions: set[str], now_ms: float, perform, delay_ms: int = REPEAT_DELAY_MS, interval_ms: int = REPEAT_INTERVAL_MS) -> None:  # noqa: ANN001
        for action in actions - self.pressed:
            perform(action)
            self.repeat_at[action] = now_ms + delay_ms
        for action in actions & self.pressed:
            if action in REPEATABLE_ACTIONS and now_ms >= self.repeat_at.get(action, now_ms):
                perform(action)
                self.repeat_at[action] = now_ms + interval_ms
        self.pressed = set(actions)

    def clear(self) -> None:
        self.pressed.clear()


# ---- 割り当て画面 ----
class KeyMappingDialog(QtWidgets.QDialog):
    """キーボードの割り当て画面。「設定」を押してからキーを押すと、その操作に割り当てる。

    1つのキーは1つの操作にだけ割り当てる。別の操作で使っているキーを押したら入れ替える。Esc で取り消す。
    """

    def __init__(self, bindings: dict[str, str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("キーボードの割り当て")
        self.bindings = dict(bindings)
        self.waiting_for: str | None = None
        self.labels: dict[str, QtWidgets.QLabel] = {}
        grid = QtWidgets.QGridLayout()
        for row, (action, text) in enumerate(ACTIONS.items()):
            grid.addWidget(QtWidgets.QLabel(text), row, 0)
            label = QtWidgets.QLabel()
            self.labels[action] = label
            grid.addWidget(label, row, 1)
            set_btn = QtWidgets.QPushButton("設定")
            set_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)  # キー入力をこの画面で受け取る
            set_btn.clicked.connect(lambda _c=False, a=action: self.start_waiting(a))
            grid.addWidget(set_btn, row, 2)
        self.message = QtWidgets.QLabel("「設定」を押してから、割り当てたいキーを押してください。")
        defaults_btn = QtWidgets.QPushButton("初期値に戻す")
        defaults_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        defaults_btn.clicked.connect(self._reset_defaults)
        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        for button in box.buttons():
            button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(self.message)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(defaults_btn)
        bottom.addStretch(1)
        bottom.addWidget(box)
        layout.addLayout(bottom)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._refresh()

    def _refresh(self) -> None:
        for action, label in self.labels.items():
            text = self.bindings.get(action, "") or "(なし)"
            if action == self.waiting_for:
                text = "▶ キーを押してください…"
            label.setText(text)

    def start_waiting(self, action: str) -> None:
        self.waiting_for = action
        self.setFocus()
        self._refresh()

    def _reset_defaults(self) -> None:
        self.bindings = dict(DEFAULT_BINDINGS)
        self.waiting_for = None
        self._refresh()

    def assign(self, name: str) -> None:
        """待機中の操作にキー名を割り当てる(重複したら入れ替える)。"""
        action = self.waiting_for
        if action is None:
            return
        note = ""
        other = action_for(self.bindings, name)
        if other is not None and other != action:
            self.bindings[other] = self.bindings.get(action, "")
            note = f"(「{ACTIONS[other]}」は {self.bindings[other] or 'なし'} に入れ替えました)"
        self.bindings[action] = name
        self.message.setText(f"{name} を「{ACTIONS[action]}」に割り当てました{note}")
        self.waiting_for = None
        self._refresh()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        if self.waiting_for is None:
            super().keyPressEvent(event)
            return
        if event.key() == QtCore.Qt.Key.Key_Escape:
            self.waiting_for = None
            self.message.setText("割り当てを取り消しました。")
            self._refresh()
            return
        name = key_event_name(event)
        if name is not None:  # 修飾キー単独は待ち続ける
            self.assign(name)


class PadMappingDialog(QtWidgets.QDialog):
    """コントローラーの割り当て画面。「設定」を押してからボタン(十字キー・スティックも可)を押す。

    1つの入力は1つの操作にだけ割り当てる(重複したら前の操作から外す)。
    入力は親ウィンドウが feed() で渡す。
    """

    def __init__(self, bindings: dict[str, list[str]], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("コントローラーの割り当て")
        self.bindings = {action: list(inputs) for action, inputs in bindings.items()}
        self.waiting_for: str | None = None
        self._previous: set[str] = set()
        self.labels: dict[str, QtWidgets.QLabel] = {}
        grid = QtWidgets.QGridLayout()
        for row, (action, text) in enumerate(ACTIONS.items()):
            grid.addWidget(QtWidgets.QLabel(text), row, 0)
            label = QtWidgets.QLabel()
            self.labels[action] = label
            grid.addWidget(label, row, 1)
            set_btn = QtWidgets.QPushButton("設定")
            set_btn.clicked.connect(lambda _c=False, a=action: self.start_waiting(a))
            clear_btn = QtWidgets.QPushButton("クリア")
            clear_btn.clicked.connect(lambda _c=False, a=action: self._clear(a))
            grid.addWidget(set_btn, row, 2)
            grid.addWidget(clear_btn, row, 3)
        self.message = QtWidgets.QLabel("「設定」を押してから、割り当てたいボタンを押してください。")
        defaults_btn = QtWidgets.QPushButton("初期値に戻す")
        defaults_btn.clicked.connect(self._reset_defaults)
        box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(self.message)
        bottom = QtWidgets.QHBoxLayout()
        bottom.addWidget(defaults_btn)
        bottom.addStretch(1)
        bottom.addWidget(box)
        layout.addLayout(bottom)
        self._refresh()

    def _refresh(self) -> None:
        for action, label in self.labels.items():
            names = self.bindings.get(action, [])
            text = ", ".join(names) if names else "(なし)"
            if action == self.waiting_for:
                text = "▶ ボタンを押してください…"
            label.setText(text)

    def start_waiting(self, action: str) -> None:
        self.waiting_for = action
        self._refresh()

    def _clear(self, action: str) -> None:
        self.bindings[action] = []
        self._refresh()

    def _reset_defaults(self) -> None:
        self.bindings = {action: list(inputs) for action, inputs in DEFAULT_PAD_BINDINGS.items()}
        self.waiting_for = None
        self._refresh()

    def feed(self, inputs: set[str]) -> None:
        """コントローラーの今の入力。待機中なら新しく押された入力を割り当てる。"""
        pressed = inputs - self._previous
        self._previous = set(inputs)
        if self.waiting_for is None or not pressed:
            return
        name = sorted(pressed)[0]
        note = ""
        for action, names in self.bindings.items():
            if name in names and action != self.waiting_for:
                names.remove(name)
                note = f"(「{ACTIONS[action]}」から外しました)"
        self.bindings[self.waiting_for] = [name]
        self.message.setText(f"{name} を「{ACTIONS[self.waiting_for]}」に割り当てました{note}")
        self.waiting_for = None
        self._refresh()
