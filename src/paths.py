"""実行ファイル(PyInstaller)として配布する場合と、ソースから起動する場合のパスの違いを吸収する。

・app_root(): 利用者が触るファイル(設定 config/、デバッグログ、録画、
  ama ブリッジの実行ファイル)を置く場所。
    - 実行ファイル形式: PBA.exe と同じフォルダ
    - ソースから起動:   リポジトリの直下
・resource_root(): 同梱する読み取り専用データの場所。
    - 実行ファイル形式: PyInstaller が展開した一時フォルダ(sys._MEIPASS)
    - ソースから起動:   リポジトリの直下
"""

from __future__ import annotations

import sys
from pathlib import Path

APP_NAME = "PBA"
APP_TITLE = "PBA (Puyo Beginner Assistance)"

_SOURCE_ROOT = Path(__file__).resolve().parent.parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _SOURCE_ROOT


def resource_root() -> Path:
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_root()))
    return _SOURCE_ROOT
