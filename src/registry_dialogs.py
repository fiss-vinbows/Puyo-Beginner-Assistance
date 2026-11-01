"""思考エンジン・定跡ファイルの登録画面(2026-10-26、利用者「PBA の中で登録ボタンを設けて選ぶ方式」)"""

from __future__ import annotations

import shlex
from pathlib import Path

from PyQt6 import QtWidgets

from .engine.registry import BookEntry, EngineEntry, Registry
from .simulator.opening_book import OpeningBook


def _args_text(args: list[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


class EngineRegistryDialog(QtWidgets.QDialog):
    """思考エンジンの登録。実行ファイルを選び、名前と起動引数を付ける。PBA と同じ標準入出力 JSON で話すエンジンに限る。"""

    def __init__(self, registry: Registry, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("思考エンジンの登録")
        self.engines = [EngineEntry(e.name, e.exe, list(e.args)) for e in registry.engines]
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("PBA と同じ形(ama のブリッジの標準入出力 JSON)で話す思考エンジンを登録できます。\n"
                                          "登録したエンジンは、メイン画面の「思考エンジン」の一覧に並びます。"))
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("追加…")
        add.clicked.connect(self._add)
        row.addWidget(add)
        remove = QtWidgets.QPushButton("削除")
        remove.clicked.connect(self._remove)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for e in self.engines:
            self.list.addItem(f"{e.name}  —  {e.exe}" + (f"  {_args_text(e.args)}" if e.args else ""))

    def add_engine(self, name: str, exe: str, args_text: str = "") -> str | None:
        """登録を足す。登録できなければ理由を返す。"""
        name = name.strip()
        if not name:
            return "名前を入れてください"
        if name in ("YamaNEKO", "ama", "支援なし") or any(e.name == name for e in self.engines):
            return f"「{name}」はもう使われている名前です"
        if not Path(exe).exists():
            return f"実行ファイルが見つかりません: {exe}"
        try:
            args = shlex.split(args_text)
        except ValueError as exc:
            return f"起動引数を読めません: {exc}"
        self.engines.append(EngineEntry(name, exe, args))
        self._refresh()
        return None

    def _add(self) -> None:
        exe, _ = QtWidgets.QFileDialog.getOpenFileName(self, "思考エンジンの実行ファイル", "", "実行ファイル (*.exe);;すべてのファイル (*)")
        if not exe:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "名前", "一覧に出す名前:", text=Path(exe).stem)
        if not ok:
            return
        args, ok = QtWidgets.QInputDialog.getText(self, "起動引数", "起動引数(無ければ空のまま):")
        if not ok:
            return
        error = self.add_engine(name, exe, args)
        if error:
            QtWidgets.QMessageBox.warning(self, "登録できません", error)

    def _remove(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.engines):
            del self.engines[row]
            self._refresh()


class BookRegistryDialog(QtWidgets.QDialog):
    """定跡ファイルの登録。今の YamaNEKO の書き出し(opening_tree.exe --export)と同じ形式のファイルを選ぶ。"""

    def __init__(self, registry: Registry, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("定跡ファイルの登録")
        self.books = [BookEntry(b.name, b.path) for b in registry.books]
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("YamaNEKO の定跡の書き出し(opening_tree.exe --export)と同じ形式の JSON を登録できます。\n"
                                          "組み込みの「標準の定跡」はいつでも選べます。"))
        self.list = QtWidgets.QListWidget()
        layout.addWidget(self.list)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("追加…")
        add.clicked.connect(self._add)
        row.addWidget(add)
        remove = QtWidgets.QPushButton("削除")
        remove.clicked.connect(self._remove)
        row.addWidget(remove)
        row.addStretch(1)
        layout.addLayout(row)
        box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for b in self.books:
            self.list.addItem(f"{b.name}  —  {b.path}")

    def add_book(self, name: str, path: str) -> str | None:
        """登録を足す。読めない・形式が違うファイルは登録しない(理由を返す)。"""
        name = name.strip()
        if not name:
            return "名前を入れてください"
        if name == "標準の定跡" or any(b.name == name for b in self.books):
            return f"「{name}」はもう使われている名前です"
        try:
            book = OpeningBook.load(Path(path))
        except (ValueError, KeyError, TypeError) as exc:
            return f"定跡ファイルの形式が違います: {exc}"
        if book is None:
            return f"ファイルが見つかりません: {path}"
        self.books.append(BookEntry(name, path))
        self._refresh()
        return None

    def _add(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "定跡ファイル", "", "定跡 (*.json);;すべてのファイル (*)")
        if not path:
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "名前", "一覧に出す名前:", text=Path(path).stem)
        if not ok:
            return
        error = self.add_book(name, path)
        if error:
            QtWidgets.QMessageBox.warning(self, "登録できません", error)

    def _remove(self) -> None:
        row = self.list.currentRow()
        if 0 <= row < len(self.books):
            del self.books[row]
            self._refresh()
