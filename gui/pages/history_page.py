from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.widgets import NoWheelComboBox


class HistoryPage(QWidget):
    COLUMNS = [
        ("ID", "id"),
        ("时间", "created_at"),
        ("分数", "score"),
        ("阈值", "threshold"),
        ("结果", "decision"),
        ("音频", "audio_path"),
        ("备注", "notes"),
    ]

    def __init__(self):
        super().__init__()
        self._all_entries = []

        title = QLabel("验证历史")
        title.setObjectName("PageTitle")
        hint = QLabel(
            "记录每次验证的分数、阈值、结论和补充说明，"
            "便于复盘不同输入设备与后端模式的表现差异。"
        )
        hint.setWordWrap(True)
        hint.setObjectName("PageHint")

        # -- filters --
        self._decision_filter = NoWheelComboBox()
        self._decision_filter.addItem("全部结论", "")
        self._decision_filter.addItem("通过 (accept)", "accept")
        self._decision_filter.addItem("拒绝 (reject)", "reject")
        self._decision_filter.currentIndexChanged.connect(self._apply_filters)

        self._stats_label = QLabel("")
        self._stats_label.setObjectName("MutedValue")

        filter_row = QHBoxLayout()
        filter_row.setSpacing(10)
        filter_row.addWidget(QLabel("筛选:"))
        filter_row.addWidget(self._decision_filter)
        filter_row.addStretch(1)
        filter_row.addWidget(self._stats_label)

        # -- table --
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([label for label, _ in self.COLUMNS])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setColumnHidden(0, True)
        self.table.setSortingEnabled(False)

        # -- actions --
        self.refresh_button = QPushButton("刷新")
        self.export_button = QPushButton("导出 CSV")
        self.delete_button = QPushButton("删除选中")
        self.clear_button = QPushButton("清空历史")

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.export_button)
        actions.addWidget(self.delete_button)
        actions.addWidget(self.clear_button)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addLayout(filter_row)
        layout.addLayout(actions)
        layout.addWidget(self.table)

    def set_entries(self, entries):
        self._all_entries = list(entries)
        self._apply_filters()

    def _apply_filters(self):
        decision_filter = self._decision_filter.currentData()
        filtered = self._all_entries
        if decision_filter:
            filtered = [e for e in filtered if str(e.get("decision", "")).strip().lower() == decision_filter]

        total = len(self._all_entries)
        accept_count = sum(1 for e in self._all_entries if str(e.get("decision", "")).strip().lower() == "accept")
        reject_count = total - accept_count
        rate = f"{accept_count / total * 100:.0f}%" if total > 0 else "-"
        self._stats_label.setText(f"共 {total} 条 | 通过 {accept_count} | 拒绝 {reject_count} | 通过率 {rate}")

        self._render_table(filtered)

    def _render_table(self, entries):
        self.table.setRowCount(len(entries))
        for row_index, entry in enumerate(entries):
            for col_index, (_, key) in enumerate(self.COLUMNS):
                value = entry.get(key, "")
                if key in {"score", "threshold"} and value != "":
                    value = f"{float(value):.6f}"
                item = QTableWidgetItem(str(value))
                if col_index == 0:
                    item.setData(Qt.UserRole, entry.get("id"))
                if key == "decision":
                    normalized = str(value).strip().lower()
                    if normalized in {"accept", "same speaker", "通过"}:
                        item.setForeground(Qt.darkGreen)
                    else:
                        item.setForeground(Qt.red)
                self.table.setItem(row_index, col_index, item)
        self.table.resizeColumnsToContents()
        if self.table.columnCount() >= 7:
            self.table.horizontalHeader().setStretchLastSection(True)

    def selected_entry_ids(self):
        ids = []
        for index in self.table.selectionModel().selectedRows():
            item = self.table.item(index.row(), 0)
            if item is not None:
                ids.append(int(item.text()))
        return ids
