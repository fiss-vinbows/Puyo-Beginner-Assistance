"""plugins/ フォルダのプラグインを読み込み、有効・無効を管理する(2026-10-28)。

各プラグインは plugins/<名前>/plugin.py に PLUGIN = Plugin(...) を書く。有効・無効は PBA の「プラグイン」画面で切り替え、
config/plugins.json に無効にしたものだけを保存する(新しく置いたプラグインは有効で始まる)。読み込めないプラグインは理由を残して飛ばす。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .paths import app_root
from .plugin_api import Language, Plugin

PLUGIN_DIR = app_root() / "plugins"
CONFIG_PATH = app_root() / "config" / "plugins.json"


def discover(plugin_dir: Path = PLUGIN_DIR) -> tuple[list[Plugin], dict[str, str]]:
    """プラグインを読み込む。(読み込めたプラグイン, {フォルダ名: 読み込めなかった理由})。"""
    plugins: list[Plugin] = []
    errors: dict[str, str] = {}
    if not plugin_dir.is_dir():
        return plugins, errors
    for folder in sorted(p for p in plugin_dir.iterdir() if p.is_dir() and (p / "plugin.py").exists()):
        name = f"pba_plugin_{folder.name}"
        try:
            spec = importlib.util.spec_from_file_location(name, folder / "plugin.py", submodule_search_locations=[str(folder)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            # 1つのフォルダに複数のプラグインを置く時は PLUGINS = [...](全自動モードとローカル対戦。2026-11-01)
            found = getattr(module, "PLUGINS", None) or [getattr(module, "PLUGIN")]
            for plugin in found:
                if not isinstance(plugin, Plugin):
                    raise TypeError("PLUGIN が Plugin ではありません")
            plugins.extend(found)
        except Exception as exc:  # 1つのプラグインの不具合で PBA を止めない
            errors[folder.name] = f"{type(exc).__name__}: {exc}"
    return plugins, errors


def load_disabled(path: Path = CONFIG_PATH) -> set[str]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    return {str(x) for x in data.get("disabled", [])}


def save_disabled(disabled: set[str], path: Path = CONFIG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class PluginManager:
    plugins: list[Plugin] = field(default_factory=list)
    disabled: set[str] = field(default_factory=set)
    errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, plugin_dir: Path = PLUGIN_DIR, config: Path = CONFIG_PATH) -> "PluginManager":
        plugins, errors = discover(plugin_dir)
        return cls(plugins, load_disabled(config), errors)

    @property
    def enabled(self) -> list[Plugin]:
        return [p for p in self.plugins if p.id not in self.disabled]

    def autoplay_sides(self) -> set[str]:
        """全自動で動かしてよい側(有効なプラグインが許可した側の和)。"""
        return {s for p in self.enabled for s in p.autoplay_sides}

    def guarded_autoplay_sides(self) -> set[str]:
        """ネームプレートの確認(2P・Player 2)が要る側。条件なしで許可するプラグインが無い側だけ。"""
        free = {s for p in self.enabled if not p.nameplate_guard for s in p.autoplay_sides}
        return self.autoplay_sides() - free

    def manual_calibration(self) -> bool:
        return any(p.manual_calibration for p in self.enabled)

    def debug_log(self) -> bool:
        return any(p.debug_log for p in self.enabled)

    def window_resize(self) -> bool:
        return any(p.window_resize for p in self.enabled)

    def languages(self) -> dict[str, Language]:
        out: dict[str, Language] = {}
        for p in self.enabled:
            out.update(p.languages)
        return out
