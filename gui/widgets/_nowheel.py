from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):  # pragma: no cover
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):  # pragma: no cover
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):  # pragma: no cover
        event.ignore()

