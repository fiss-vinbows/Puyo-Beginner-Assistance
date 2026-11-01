"""PBA.exe をビルドして dist/PBA/ に配布物一式を揃え、dist/PBA-<版>.zip にまとめる。

使い方: ./venv/Scripts/python.exe tools/build_exe.py [版(例 0.2.0)]
前提: venv に pyinstaller が入っていること、ama のブリッジがビルド済みであること(tools/build_ama.bat)。
TBA(../Tetris)の tools/build_exe.py を手本にした(2026-11-01)。
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist" / "PBA"
SITE_PACKAGES = ROOT / "venv" / "Lib" / "site-packages"
AMA_BIN = ROOT / "external" / "ama-main" / "bin" / "bridge"

# exe に入れるパッケージと、そのライセンス文の入った dist-info の名前(先頭一致)。ライセンスの一覧は THIRD_PARTY_NOTICES.md
BUNDLED_PACKAGES = ["pyqt6-", "pyqt6_qt6-", "pyqt6_sip-", "numpy-", "opencv_python-", "mss-"]


def folder_plugins(folder: Path) -> list:
    """プラグインのフォルダの plugin.py を読み、入っているプラグイン(PLUGINS か PLUGIN)を返す。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.plugin_api import Plugin

    name = f"pba_build_plugin_{folder.name}"
    spec = importlib.util.spec_from_file_location(name, folder / "plugin.py", submodule_search_locations=[str(folder)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    found = getattr(module, "PLUGINS", None) or [getattr(module, "PLUGIN")]
    if not all(isinstance(p, Plugin) for p in found):
        raise TypeError(f"{folder.name}: PLUGIN が Plugin ではありません")
    return found


def copy_distributable_plugins(source: Path, target: Path) -> list[str]:
    """配布してよいプラグインのフォルダだけを写す。フォルダの中に1つでも配布しないプラグイン(distributable=False)が
    あればフォルダごと写さない(全自動のフォルダには、配布しない全自動モードと、同じ仕組みを使うローカル対戦が入っている)。"""
    if target.exists():
        shutil.rmtree(target)
    copied = []
    for folder in sorted(p for p in source.iterdir() if p.is_dir() and (p / "plugin.py").exists()):
        plugins = folder_plugins(folder)
        if all(p.distributable for p in plugins):
            shutil.copytree(folder, target / folder.name, ignore=shutil.ignore_patterns("__pycache__"))
            copied += [p.id for p in plugins]
    print(f"配布するプラグイン: {copied or 'なし'}")
    return copied


def copy_licenses(target: Path) -> None:
    """同梱するもののライセンス文を licenses/ に写す(MIT・BSD・Apache は許諾文の同梱、GPL・LGPL は全文の同梱が条件)。"""
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    shutil.copy2(ROOT / "external" / "ama-main" / "LICENSE", target / "ama-LICENSE.txt")
    # PyInstaller が同梱する Python 本体(PSF License)
    python_license = Path(sys.base_prefix) / "LICENSE.txt"
    if not python_license.exists():
        sys.exit(f"Python のライセンス文が見つかりません: {python_license}")
    shutil.copy2(python_license, target / f"python-{sys.version_info.major}.{sys.version_info.minor}-LICENSE.txt")
    for prefix in BUNDLED_PACKAGES:
        infos = [d for d in SITE_PACKAGES.iterdir() if d.name.lower().startswith(prefix) and d.name.endswith(".dist-info")]
        if not infos:
            sys.exit(f"ライセンス文が見つかりません: {prefix}")
        info = infos[0]
        name = info.name[: -len(".dist-info")]
        files = [f for f in info.rglob("*") if f.is_file() and any(k in f.name.upper() for k in ("LICEN", "COPYING", "NOTICE"))]
        if not files:
            sys.exit(f"ライセンス文が見つかりません: {info}")
        for f in files:
            # numpy は同梱の部品ごとに同じ名前(LICENSE.md など)があるので、dist-info からの階層を名前に残す
            rel = f.relative_to(info).as_posix().removeprefix("licenses/").replace("/", "_")
            shutil.copy2(f, target / f"{name}-{rel}")


def make_zip(version: str) -> Path:
    archive = shutil.make_archive(str(ROOT / "dist" / f"PBA-{version}"), "zip", root_dir=DIST.parent, base_dir=DIST.name)
    return Path(archive)


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "dev"
    bridge = AMA_BIN / "ama-bridge.exe"
    if not bridge.exists():
        sys.exit(f"ama のブリッジがビルドされていません(tools/build_ama.bat): {bridge}")
    pyinstaller = ROOT / "venv" / "Scripts" / "pyinstaller.exe"
    subprocess.run([str(pyinstaller), "--noconfirm", "--clean", str(ROOT / "PBA.spec")], check=True, cwd=ROOT)

    # 利用者が触るファイル・外部の実行ファイルは exe と同じフォルダに置く(src/paths.py の app_root() 参照)
    shutil.copy2(bridge, DIST / "ama-bridge.exe")
    shutil.copy2(AMA_BIN / "config.json", DIST / "config.json")  # ama の評価の重み(ブリッジが同じフォルダから読む)
    (DIST / "config").mkdir(exist_ok=True)
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT / name, DIST / name)
    copy_distributable_plugins(ROOT / "plugins", DIST / "plugins")
    copy_licenses(DIST / "licenses")
    print(f"配布物: {DIST}")
    print(f"zip: {make_zip(version)}")


if __name__ == "__main__":
    main()
