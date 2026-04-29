from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QWidget

from gui.widgets.status_indicator import StatusIndicator


class SidebarButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(44)


class SidebarNav(QWidget):
    page_selected = pyqtSignal(int)

    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setFixedWidth(180)
        self.setObjectName("SidebarNav")

        self._buttons = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 16, 8, 12)
        layout.setSpacing(4)

        for i, label in enumerate(labels):
            btn = SidebarButton(label)
            btn.clicked.connect(lambda checked, idx=i: self._on_clicked(idx))
            self._buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch(1)

        self.backend_indicator = StatusIndicator()
        self.profile_indicator = StatusIndicator()
        self.threshold_indicator = StatusIndicator()

        layout.addWidget(self.backend_indicator)
        layout.addWidget(self.profile_indicator)
        layout.addWidget(self.threshold_indicator)

    def _on_clicked(self, index):
        self.select(index)
        self.page_selected.emit(index)

    def select(self, index):
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)
