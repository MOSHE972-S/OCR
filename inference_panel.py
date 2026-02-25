import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QFileDialog, QComboBox, QLabel,
    QGroupBox, QMessageBox
)
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import Qt

from src.page_segmenter import SegmentWorker


class InferencePanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.setAcceptDrops(True)
        self.engine    = None
        self.segmenter = None
        self.seg_worker = None
        self.image_path = ""
        self._try_load_engine()

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("🔍 מנוע זיהוי כתב יד")
        title.setObjectName("title")
        root.addWidget(title)

        if not self.engine:
            warn = QLabel("⚠️  מודל ONNX לא נמצא — יש לאמן ולייצא במסך האימון")
            warn.setStyleSheet("color:#f38ba8; font-weight:bold;")
            root.addWidget(warn)

        # ── תמונה ─────────────────────────────────────────
        grp_img = QGroupBox("תמונה לזיהוי — גרור/שחרר או לחץ לבחור")
        v = QVBoxLayout(grp_img)
        self.viewer = QLabel("גרור לכאן תמונה...")
        self.viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viewer.setMinimumHeight(130)
        self.viewer.setStyleSheet("border: 2px dashed #a6e3a1; border-radius:8px;")
        v.addWidget(self.viewer)
        btn_load = QPushButton("📂 בחר תמונה")
        btn_load.clicked.connect(self.load_image)
        v.addWidget(btn_load)
        root.addWidget(grp_img)

        # ── מצב + כפתורים ──────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(QLabel("מצב:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["דף שלם","מילה בודדת"])
        row.addWidget(self.mode_combo)
        self.btn_run = QPushButton("🔍 בצע זיהוי")
        self.btn_run.clicked.connect(self.run_inference)
        self.btn_stop = QPushButton("⏹ עצור")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop_inference)
        self.btn_stop.setEnabled(False)
        row.addWidget(self.btn_run)
        row.addWidget(self.btn_stop)
        root.addLayout(row)

        # ── תוצאות ─────────────────────────────────────────
        self.result_text = QTextEdit()
        self.result_text.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.result_text.setPlaceholderText("תוצאות הזיהוי יופיעו כאן...")
        root.addWidget(self.result_text)

        btn_export = QPushButton("💾 ייצא לטקסט (TXT)")
        btn_export.clicked.connect(self.export_txt)
        root.addWidget(btn_export)

    # ── drag & drop ──────────────────────────────────────
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e: QDropEvent):
        urls = e.mimeData().urls()
        if urls:
            self._load(urls[0].toLocalFile())

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "בחר תמונה", "", "תמונות (*.png *.jpg *.bmp *.tiff *.pdf)"
        )
        if path:
            self._load(path)

    def _load(self, path):
        self.image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self.viewer.setPixmap(
                pix.scaled(500, 150, Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )

    def _try_load_engine(self):
        onnx = "weights/best_model_int8.onnx"
        if not os.path.exists(onnx):
            onnx = "weights/crnn_fp32.onnx"
        if os.path.exists(onnx):
            try:
                from src.inference import InferenceEngine
                from src.page_segmenter import PageSegmenter
                self.engine    = InferenceEngine(onnx)
                self.segmenter = PageSegmenter(self.engine)
            except Exception:
                pass

    def run_inference(self):
        if not self.engine:
            QMessageBox.critical(self, "שגיאה",
                "מודל לא נמצא. יש לאמן ולייצא ל-ONNX קודם.")
            return
        if not self.image_path:
            QMessageBox.warning(self, "שגיאה", "יש לבחור תמונה קודם.")
            return

        self.result_text.clear()
        self.result_text.append("⏳ מעבד...")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        if self.mode_combo.currentText() == "דף שלם":
            self.seg_worker = SegmentWorker(self.segmenter, self.image_path)
            self.seg_worker.line_done.connect(
                lambda l: self.result_text.append(l)
            )
            self.seg_worker.finished_ok.connect(self._on_seg_done)
            self.seg_worker.error.connect(
                lambda e: self.result_text.append(f"❌ {e}")
            )
            self.seg_worker.start()
        else:
            from src.inference import InferenceEngine
            result = self.engine.recognize(self.image_path)
            self.result_text.setPlainText(result)
            self._on_seg_done()

    def stop_inference(self):
        if self.seg_worker:
            self.seg_worker.stop()
        self.btn_stop.setEnabled(False)

    def _on_seg_done(self):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def export_txt(self):
        text = self.result_text.toPlainText()
        path, _ = QFileDialog.getSaveFileName(
            self, "שמור", "", "קובץ טקסט (*.txt)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)

