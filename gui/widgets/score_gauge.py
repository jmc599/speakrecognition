from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class _GaugeBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._score = None
        self._threshold = None
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_values(self, score, threshold):
        self._score = score
        self._threshold = threshold
        self.update()

    def clear(self):
        self._score = None
        self._threshold = None
        self.update()

    def paintEvent(self, event):
        if self._score is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()

        bar_h = 24
        bar_y = (h - bar_h) // 2 + 4
        bar_rect = QRectF(0, bar_y, w, bar_h)
        radius = bar_h / 2

        # gradient: red → yellow → green
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor("#ef4444"))
        grad.setColorAt(0.35, QColor("#f59e0b"))
        grad.setColorAt(0.65, QColor("#22c55e"))
        grad.setColorAt(1.0, QColor("#16a34a"))

        # background track
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#e2e8f0"))
        p.drawRoundedRect(bar_rect, radius, radius)

        # filled portion
        score_clamped = max(0.0, min(1.0, self._score))
        fill_w = score_clamped * w
        if fill_w > 0:
            fill_rect = QRectF(0, bar_y, fill_w, bar_h)
            p.setBrush(grad)
            p.setClipRect(fill_rect)
            p.drawRoundedRect(bar_rect, radius, radius)
            p.setClipping(False)

        # threshold marker
        if self._threshold is not None:
            th_clamped = max(0.0, min(1.0, self._threshold))
            th_x = th_clamped * w
            pen = QPen(QColor("#1e293b"), 2.5)
            pen.setStyle(Qt.DashLine)
            p.setPen(pen)
            p.drawLine(int(th_x), bar_y - 6, int(th_x), bar_y + bar_h + 6)

            # threshold label
            p.setPen(QColor("#475569"))
            font = QFont("Microsoft YaHei UI", 8)
            p.setFont(font)
            label = f"{self._threshold:.3f}"
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label)
            tx = th_x - tw / 2
            tx = max(0, min(tx, w - tw))
            p.drawText(int(tx), bar_y + bar_h + 18, label)

        p.end()


class ScoreGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._score_label = QLabel("-")
        self._score_label.setAlignment(Qt.AlignCenter)
        self._score_label.setStyleSheet(
            "font-size:28pt;font-weight:700;color:#0f172a;"
        )

        self._gauge = _GaugeBar()

        self._decision_label = QLabel("")
        self._decision_label.setAlignment(Qt.AlignCenter)
        self._decision_label.setFixedHeight(44)
        self._decision_label.setStyleSheet(
            "font-size:14pt;font-weight:700;color:#64748b;"
            "background:#f1f5f9;border-radius:12px;padding:4px 24px;"
        )
        self._decision_label.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(6)
        layout.addWidget(self._score_label)
        layout.addWidget(self._gauge)
        layout.addSpacing(8)
        layout.addWidget(self._decision_label, 0, Qt.AlignCenter)

    def set_result(self, score, threshold, decision):
        self._score_label.setText(f"{score:.6f}")
        self._gauge.set_values(score, threshold)

        normalized = str(decision).strip().lower()
        is_accept = normalized in {
            "accept", "same speaker", "same speaker / pass", "通过",
        }
        if is_accept:
            self._decision_label.setText("通 过")
            self._decision_label.setStyleSheet(
                "font-size:14pt;font-weight:700;color:#067647;"
                "background:#ecfdf3;border-radius:12px;padding:4px 24px;"
            )
        else:
            self._decision_label.setText("拒 绝")
            self._decision_label.setStyleSheet(
                "font-size:14pt;font-weight:700;color:#b42318;"
                "background:#fef3f2;border-radius:12px;padding:4px 24px;"
            )
        self._decision_label.show()

    def clear(self):
        self._score_label.setText("-")
        self._gauge.clear()
        self._decision_label.hide()
