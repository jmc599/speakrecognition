import sys

from PyQt5.QtWidgets import QApplication

from gui.main_window import SpeakerIdentityTerminalWindow
from gui.styles import app_font, build_stylesheet


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("声纹身份验证终端")
    app.setFont(app_font())
    app.setStyleSheet(build_stylesheet())
    window = SpeakerIdentityTerminalWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
