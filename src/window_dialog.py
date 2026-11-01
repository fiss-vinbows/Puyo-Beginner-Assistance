"""ゲームのウィンドウを一覧から選ぶ画面(2026-10-28、利用者「PC 版はウィンドウ検出ボタンを設けて、再起動したときにも検出できるように」)

決め打ちの実行ファイル名・タイトルで見つからなかった時に、表示中のウィンドウを並べて選んでもらう。選んだウィンドウの
実行ファイル名(読めなければタイトル)を config/game_window.json に残し、ゲームを起動し直しても同じウィンドウを探せるようにする。
"""

from __future__ import annotations

from PyQt6 import QtWidgets

from .capture.window_detect import SavedWindow, WindowEntry


class WindowPickDialog(QtWidgets.QDialog):
    def __init__(self, entries: list[WindowEntry], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ゲームのウィンドウを選ぶ")
        self.entries = entries
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(
            "ぷよぷよテトリス2 のウィンドウが自動では見つかりませんでした。\n"
            "ゲームをウィンドウ表示で起動し、下の一覧からゲームのウィンドウを選んでください。\n"
            "選んだウィンドウは覚えておき、ゲームを起動し直しても探します。"
        ))
        self.list = QtWidgets.QListWidget()
        for e in entries:
            w = e.window
            self.list.addItem(f"{w.title}  [{e.exe or '実行ファイル名を読めません'}]  {w.width}×{w.height}")
        if entries:
            self.list.setCurrentRow(0)
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        layout.addWidget(self.list, 1)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(640, 360)

    def selected(self) -> SavedWindow | None:
        """選んだウィンドウ(覚える形)。実行ファイル名が読めればそれで、読めなければタイトルで探す。"""
        row = self.list.currentRow()
        if row < 0 or row >= len(self.entries):
            return None
        e = self.entries[row]
        return SavedWindow(exe=e.exe, title="" if e.exe else e.window.title)
