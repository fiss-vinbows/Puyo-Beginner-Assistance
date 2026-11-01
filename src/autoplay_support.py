"""全自動の操作の仕組みを読む窓口(2026-11-01、本体の公開に合わせて全自動を本体から外した)。

キー・仮想パッドの送信、操作手順のキー入力への変換、キー割り当て・切り替えの画面、ネームプレートの確認は、
非公開のプラグイン(plugins/full_auto/pba_autoplay、リポジトリ PT-plugins)にある。本体はここから読み、
プラグインが無ければ AVAILABLE=False で、何もしない代わりの部品を返す(全自動の項目は画面に出ない)。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .paths import app_root

AUTOPLAY_DIR = app_root() / "plugins" / "full_auto"


def attack_involved(self_pending: int, incoming_carry: int, solo: bool, enemy_chain: bool, enemy_pending: int, engine: str) -> bool:
    """相手の攻撃が絡む局面か(全自動で先読みの手順を出現時点で送らない条件)。思考が受け・潰しに手を変えることが
    あり、送ったキーは取り消せないため(実機 2026-09-27 12:01)。

    自分への予告・着弾見込み・相手の連鎖中は常に絡む。相手への予告(自分が送ったおじゃまがまだ落ちていない)は
    ama では判断が変わりうるので絡むとするが、自作AI(yamaneko)の判断は相手への予告を使わないので絡まないとする
    (実機 2026-10-10 14:08: 連鎖を撃った後は相手に予告が残り、先読みを送れず設置が約0.5秒遅れた)。"""
    if self_pending or incoming_carry:
        return True
    if solo:
        return False
    return bool(enemy_chain or (enemy_pending and engine != "yamaneko"))


try:
    if AUTOPLAY_DIR.is_dir() and str(AUTOPLAY_DIR) not in sys.path:
        sys.path.append(str(AUTOPLAY_DIR))  # 後ろに足す(本体の同名のモジュールを隠さない)
    from pba_autoplay.autoplay import (  # noqa: F401
        AutoPlayer,
        KeyMap,
        cancel_rows,
        early_drop_seconds,
        rotate_cancel,
        step_cancel,
        wall_prehold_action,
    )
    from pba_autoplay.autoplay_toggle import KeyToggle, PadToggleWatcher  # noqa: F401
    from pba_autoplay.autoplay_toggle import load_bindings as load_toggle_bindings  # noqa: F401
    from pba_autoplay.autoplay_toggle import save_bindings as save_toggle_bindings  # noqa: F401
    from pba_autoplay.autoplay_toggle_dialog import AutoplayToggleDialog  # noqa: F401
    from pba_autoplay.keymap_editor import AutoplayKeyDialog  # noqa: F401
    from pba_autoplay.nameplate import is_player2, search_box  # noqa: F401
    from pba_autoplay.pad_keyboard import (  # noqa: F401
        CONFIG_NAME,
        PRESET_TEXT,
        KeyboardHook,
        KeyToPad,
        PadKeyboardDialog,
        load_side_configs,
        save_side_configs,
    )
    from pba_autoplay.pad_sender import PAD_BUTTON_TEXT, PadSender, shared_pad  # noqa: F401

    AVAILABLE = True
except ImportError:
    AVAILABLE = False

    # ---- プラグインが無い時の代わり。全自動は常にオフで、画面の項目も出ないので送る処理には届かない ----

    @dataclass
    class KeyMap:  # type: ignore[no-redef]
        device: str = "keyboard"
        place_cancel: str = "off"
        step_cancel: bool = False
        step_cancel_max_height: int = 0
        prehold: bool = False

        @classmethod
        def load(cls, path) -> "KeyMap":  # noqa: ANN001
            return cls()

        def resolved(self) -> "KeyMap":
            return self

    class _NoSender:
        failures = 0
        last_error = 0
        error = ""
        timed = ()

    class AutoPlayer:  # type: ignore[no-redef]
        def __init__(self, keymap: KeyMap, sender=None, log=None) -> None:  # noqa: ANN001
            self.keymap = keymap
            self.sender = _NoSender()
            self.enabled = False
            self.committed_turn = None
            self.resume_after_turn = None
            self.prehold_provider = None

        def play(self, *args, **kwargs) -> bool:  # noqa: ANN002, ANN003
            return False

        def reset(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def hold_down_for_start(self) -> None:
            pass

    class PadSender:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            raise RuntimeError("全自動のプラグインがありません")

    def wall_prehold_action(*args, **kwargs):  # type: ignore[no-redef]  # noqa: ANN002, ANN003, ANN201
        return None

    def early_drop_seconds(*args, **kwargs) -> float:  # type: ignore[no-redef]  # noqa: ANN002, ANN003
        return 0.0

    def rotate_cancel(*args, **kwargs):  # type: ignore[no-redef]  # noqa: ANN002, ANN003, ANN201
        return None

    def cancel_rows(*args, **kwargs):  # type: ignore[no-redef]  # noqa: ANN002, ANN003, ANN201
        return None

    def step_cancel(*args, **kwargs):  # type: ignore[no-redef]  # noqa: ANN002, ANN003, ANN201
        return None
