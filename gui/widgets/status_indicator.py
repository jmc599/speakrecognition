from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget


_DOT_STYLE = {
    "ok": "background:#22c55e;",
    "warn": "background:#eab308;",
    "error": "background:#ef4444;",
    "off": "background:#94a3b8;",
}


class StatusIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._dot = QLabel()
        self._dot.setFixedSize(10, 10)
        self._dot.setStyleSheet(
            "border-radius:5px;" + _DOT_STYLE["off"]
        )
        self._label = QLabel("")
        self._label.setStyleSheet("color:#cbd5e1;font-size:9pt;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._dot)
        layout.addWidget(self._label, 1)

    def set_state(self, state: str, text: str = ""):
        style = _DOT_STYLE.get(state, _DOT_STYLE["off"])
        self._dot.setStyleSheet("border-radius:5px;" + style)
        self._label.setText(text)
