import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QLabel
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from gui.style          import DARK_STYLE
from gui.annotator_panel import AnnotatorPanel
from gui.trainer_panel   import TrainerPanel
from gui.inference_panel import InferencePanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MyCustomOCR — זיהוי כתב יד עברי")
        self.setMinimumSize(1050, 720)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        tabs = QTabWidget()
        tabs.addTab(AnnotatorPanel(), "📸  תיוג נתונים")
        tabs.addTab(TrainerPanel(),   "🧠  אימון מודל")
        tabs.addTab(InferencePanel(), "🔍  זיהוי כתב")
        self.setCentralWidget(tabs)

        self.statusBar().showMessage(
            "MyCustomOCR v2.0  |  מוכן  |  GPU: " +
            _gpu_info()
        )


def _gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "לא זוהה GPU — פועל על CPU"
    except Exception:
        return "CPU"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    app.setFont(QFont("Segoe UI", 11))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
