from PyQt5.QtGui import QFont


def app_font():
    return QFont("Microsoft YaHei UI", 10)


def build_stylesheet():
    return """
    QMainWindow, QWidget {
        background: #eef2f7;
        color: #18212b;
        font-family: "Microsoft YaHei UI", "Segoe UI";
        font-size: 10pt;
    }

    /* ---- Sidebar ---- */
    QWidget#SidebarNav {
        background: #1f2937;
        border: none;
    }

    SidebarButton {
        background: transparent;
        color: #94a3b8;
        border: none;
        border-radius: 10px;
        padding: 8px 14px;
        text-align: left;
        font-weight: 600;
        font-size: 10pt;
    }
    SidebarButton:hover {
        background: #334155;
        color: #e2e8f0;
    }
    SidebarButton:checked {
        background: #0a84ff;
        color: #ffffff;
    }

    /* ---- Tabs (removed, kept for compat) ---- */
    QTabWidget::pane {
        border: 1px solid #d6dde8;
        background: #f7f9fc;
        border-radius: 18px;
        top: -2px;
    }
    QTabBar::tab {
        background: #dde5ef;
        color: #51606f;
        padding: 11px 18px;
        margin-right: 6px;
        border-top-left-radius: 14px;
        border-top-right-radius: 14px;
        min-width: 100px;
        font-weight: 600;
    }
    QTabBar::tab:selected {
        background: #0a84ff;
        color: white;
    }
    QTabBar::tab:hover:!selected {
        background: #e8eef6;
        color: #1d3550;
    }

    /* ---- Group boxes ---- */
    QGroupBox {
        background: #fbfdff;
        border: 1px solid #d8e0ea;
        border-radius: 18px;
        margin-top: 18px;
        padding: 26px 18px 18px 18px;
        font-weight: 600;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        left: 16px;
        padding: 0 8px;
        color: #23527c;
        background: #fbfdff;
    }

    /* ---- Labels ---- */
    QLabel#PageTitle {
        font-size: 18pt;
        font-weight: 700;
        color: #101828;
        padding: 4px 0 0 0;
    }
    QLabel#PageHint {
        color: #66758a;
        font-size: 10pt;
        padding-bottom: 8px;
    }
    QLabel#SectionNote {
        color: #23527c;
        background: #f2f8ff;
        border: 1px solid #cfe2ff;
        border-radius: 12px;
        padding: 10px 12px;
    }
    QLabel#MetricValue {
        color: #0f3d66;
        font-weight: 700;
        font-size: 11pt;
        background: #eef6ff;
        border: 1px solid #d5e6ff;
        border-radius: 12px;
        padding: 8px 12px;
        min-height: 20px;
    }
    QLabel#MutedValue {
        color: #66758a;
        background: #f5f7fa;
        border: 1px solid #e0e6ed;
        border-radius: 12px;
        padding: 8px 12px;
        min-height: 20px;
    }

    /* ---- Inputs ---- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
        background: #ffffff;
        border: 1px solid #cfd8e3;
        border-radius: 12px;
        padding: 6px 12px;
        min-height: 38px;
        selection-background-color: #0a84ff;
        selection-color: white;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
        border: 1px solid #0a84ff;
        background: #ffffff;
    }
    QComboBox::drop-down {
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 32px;
        border-left: 1px solid #d6dde8;
        background: #f6f9fc;
        border-top-right-radius: 12px;
        border-bottom-right-radius: 12px;
    }
    QComboBox::down-arrow {
        width: 10px;
        height: 10px;
    }
    QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
        width: 22px;
        border: none;
        background: transparent;
    }

    /* ---- Tables & Lists ---- */
    QTableWidget, QListWidget {
        background: #ffffff;
        border: 1px solid #d8e0ea;
        border-radius: 14px;
        padding: 6px;
        selection-background-color: #0a84ff;
        selection-color: white;
    }
    QTableWidget:focus, QListWidget:focus {
        border: 1px solid #0a84ff;
    }
    QTableWidget {
        gridline-color: #edf2f7;
        alternate-background-color: #f8fbff;
    }
    QHeaderView::section {
        background: #eef3f8;
        color: #334155;
        border: none;
        border-right: 1px solid #dde5ef;
        padding: 8px 10px;
        font-weight: 700;
    }

    /* ---- Buttons ---- */
    QPushButton {
        background: #0a84ff;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 18px;
        font-weight: 600;
        min-height: 38px;
    }
    QPushButton:hover {
        background: #0077ed;
    }
    QPushButton:pressed {
        background: #0066cc;
    }
    QPushButton:disabled {
        background: #d6dde6;
        color: #7b8794;
    }

    /* ---- Progress bar ---- */
    QProgressBar {
        border: 1px solid #d9e1ea;
        border-radius: 10px;
        background: #edf2f7;
        text-align: center;
        min-height: 16px;
    }
    QProgressBar::chunk {
        background: #64d2ff;
        border-radius: 9px;
    }

    /* ---- Status bar ---- */
    QStatusBar {
        background: #1f2937;
        color: #f8fafc;
        border-top: 1px solid #111827;
    }

    /* ---- Scroll ---- */
    QScrollArea {
        border: none;
        background: transparent;
    }
    QScrollBar:vertical {
        background: #eef2f7;
        width: 12px;
        margin: 10px 4px 10px 0;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical {
        background: #c5d1df;
        min-height: 40px;
        border-radius: 6px;
    }
    QScrollBar::handle:vertical:hover {
        background: #9fb4ca;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
        background: none;
        border: none;
        height: 0px;
        width: 0px;
    }

    /* ---- Audio quality card ---- */
    QWidget#AudioQualityCard {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
    }

    /* ---- Stacked widget (no border) ---- */
    QStackedWidget {
        background: transparent;
        border: none;
    }
    """
