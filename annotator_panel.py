from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QLabel, QComboBox,
    QMessageBox, QGroupBox, QSizePolicy
)
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt
import pandas as pd, os

CSV_PATH = "data/labels.csv"


class AnnotatorPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.setAcceptDrops(True)
        self.image_path = ""

        if not os.path.exists(CSV_PATH):
            pd.DataFrame(
                columns=["image_path","text","type","confidence","source_page"]
            ).to_csv(CSV_PATH, index=False)

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("📸 מנהל תיוג נתונים")
        title.setObjectName("title")
        root.addWidget(title)

        # ── תמונה ─────────────────────────────────────────
        grp_img = QGroupBox("תמונה לתיוג — גרור/שחרר או לחץ לבחור")
        v = QVBoxLayout(grp_img)
        self.viewer = QLabel("גרור לכאן תמונה...")
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setMinimumHeight(140)
        self.viewer.setStyleSheet("border: 2px dashed #89b4fa; border-radius:8px;")
        v.addWidget(self.viewer)
        btn_open = QPushButton("📂 בחר קובץ תמונה")
        btn_open.clicked.connect(self.open_image)
        v.addWidget(btn_open)
        root.addWidget(grp_img)

        # ── טקסט ──────────────────────────────────────────
        grp_txt = QGroupBox("תיוג")
        v2 = QVBoxLayout(grp_txt)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("הקלד כאן את הטקסט בתמונה (עברית / אנגלית)...")
        self.text_input.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        v2.addWidget(QLabel("טקסט:"))
        v2.addWidget(self.text_input)

        h = QHBoxLayout()
        h.addWidget(QLabel("סוג:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["word","line","sentence"])
        h.addWidget(self.type_combo)
        h.addStretch()
        v2.addLayout(h)
        root.addWidget(grp_txt)

        # ── כפתורים ───────────────────────────────────────
        row = QHBoxLayout()
        btn_save = QPushButton("💾 שמור תיוג ל-CSV")
        btn_save.clicked.connect(self.save_label)
        btn_stats = QPushButton("📊 סטטיסטיקות")
        btn_stats.clicked.connect(self.show_stats)
        row.addWidget(btn_save)
        row.addWidget(btn_stats)
        root.addLayout(row)

        self.status = QLabel("")
        root.addWidget(self.status)
        root.addStretch()

    # ── Drag & Drop ───────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            self._load_image(urls[0].toLocalFile())

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "בחר תמונה", "data/processed",
            "תמונות (*.png *.jpg *.bmp *.tiff)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path):
        self.image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self.viewer.setPixmap(
                pix.scaled(500, 160, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )

    def save_label(self):
        if not self.image_path or not self.text_input.text().strip():
            QMessageBox.warning(self, "שגיאה", "יש לבחור תמונה ולהקליד טקסט.")
            return
        row = {
            "image_path":  self.image_path,
            "text":        self.text_input.text().strip(),
            "type":        self.type_combo.currentText(),
            "confidence":  1.0,
            "source_page": os.path.basename(self.image_path),
        }
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        count = len(df)
        self.status.setText(f"✅ נשמר! סהכ: {count} רשומות")
        self.text_input.clear()
        self.viewer.clear()
        self.viewer.setText("גרור לכאן תמונה...")
        self.image_path = ""

    def show_stats(self):
        try:
            df = pd.read_csv(CSV_PATH)
            msg = (
                f"📊 סטטיסטיקות\n\n"
                f"סה\"כ רשומות: {len(df)}\n"
                f"מילים:  {len(df[df.type=='word'])}\n"
                f"שורות:  {len(df[df.type=='line'])}\n"
                f"משפטים: {len(df[df.type=='sentence'])}\n\n"
                f"{'✅ מספיק לאימון ראשוני' if len(df)>=200 else '⚠️ מומלץ לפחות 200 דוגמאות'}"
            )
            QMessageBox.information(self, "סטטיסטיקות", msg)
        except Exception as e:
            QMessageBox.warning(self, "שגיאה", str(e))
