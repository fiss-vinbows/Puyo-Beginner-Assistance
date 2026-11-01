"""トグルスイッチ(2026-10-25、利用者「チェックボックスはすべてトグルスイッチに」)

QCheckBox を継承して見た目だけ変える(isChecked・toggled・setChecked はそのまま使える)。
無効(setEnabled(False))の時は灰色に暗転する。
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets

TRACK_W = 34
TRACK_H = 18
GAP = 8
ON_COLOR = QtGui.QColor(52, 152, 219)
OFF_COLOR = QtGui.QColor(170, 170, 170)
DISABLED_COLOR = QtGui.QColor(215, 215, 215)


class ToggleSwitch(QtWidgets.QCheckBox):
    def sizeHint(self) -> QtCore.QSize:  # noqa: N802
        fm = self.fontMetrics()
        text_w = fm.horizontalAdvance(self.text()) + GAP if self.text() else 0
        return QtCore.QSize(TRACK_W + text_w + 2, max(TRACK_H, fm.height()) + 4)

    def minimumSizeHint(self) -> QtCore.QSize:  # noqa: N802
        return self.sizeHint()

    def hitButton(self, pos: QtCore.QPoint) -> bool:  # noqa: N802
        return self.rect().contains(pos)  # 文字の上を押しても切り替える

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        top = (self.height() - TRACK_H) / 2
        track = QtCore.QRectF(1, top, TRACK_W, TRACK_H)
        if not self.isEnabled():
            color = DISABLED_COLOR
        else:
            color = ON_COLOR if self.isChecked() else OFF_COLOR
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawRoundedRect(track, TRACK_H / 2, TRACK_H / 2)
        d = TRACK_H - 4
        x = track.right() - d - 2 if self.isChecked() else track.left() + 2
        p.setBrush(QtGui.QColor(255, 255, 255) if self.isEnabled() else QtGui.QColor(245, 245, 245))
        p.drawEllipse(QtCore.QRectF(x, top + 2, d, d))
        if self.text():
            pal = self.palette()
            group = QtGui.QPalette.ColorGroup.Active if self.isEnabled() else QtGui.QPalette.ColorGroup.Disabled
            p.setPen(pal.color(group, QtGui.QPalette.ColorRole.WindowText))
            text_rect = QtCore.QRectF(TRACK_W + GAP, 0, self.width() - TRACK_W - GAP, self.height())
            p.drawText(text_rect, int(QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft), self.text())
        p.end()
