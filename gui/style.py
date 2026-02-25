DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", Arial;
    font-size: 13px;
}
QTabWidget::pane { border: 1px solid #45475a; border-radius: 6px; }
QTabBar::tab {
    background: #313244; color: #cdd6f4;
    padding: 8px 18px; border-radius: 6px 6px 0 0;
    min-width: 120px;
}
QTabBar::tab:selected { background: #89b4fa; color: #1e1e2e; font-weight: bold; }
QPushButton {
    background-color: #89b4fa; color: #1e1e2e;
    border-radius: 6px; padding: 7px 14px;
    font-weight: bold;
}
QPushButton:hover    { background-color: #b4befe; }
QPushButton:pressed  { background-color: #74c7ec; }
QPushButton:disabled { background-color: #45475a; color: #6c7086; }
QPushButton#danger   { background-color: #f38ba8; }
QPushButton#danger:hover { background-color: #fab387; }
QLineEdit, QComboBox, QTextEdit, QSpinBox, QDoubleSpinBox {
    background-color: #313244; border: 1px solid #45475a;
    border-radius: 5px; padding: 5px;
    color: #cdd6f4;
}
QProgressBar {
    border: 1px solid #45475a; border-radius: 5px;
    text-align: center; background: #313244; color: #cdd6f4;
}
QProgressBar::chunk { background: #89b4fa; border-radius: 5px; }
QLabel#title {
    font-size: 16px; font-weight: bold; color: #89b4fa;
}
QGroupBox {
    border: 1px solid #45475a; border-radius: 6px;
    margin-top: 10px; padding: 10px;
    color: #a6e3a1;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; }
"""
