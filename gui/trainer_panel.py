import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QProgressBar, QGroupBox,
    QSpinBox, QComboBox
)
from PyQt6.QtCore import Qt

from src.train import TrainWorker
from src.export_onnx import ExportWorker


class TrainerPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.worker = None

        root = QVBoxLayout(self)
        root.setSpacing(10)

        title = QLabel("🧠 מנוע אימון מודל")
        title.setObjectName("title")
        root.addWidget(title)

        # ── הגדרות ──────────────────────────────────────────
        grp = QGroupBox("הגדרות אימון")
        h = QHBoxLayout(grp)

        h.addWidget(QLabel("מספר Epochs:"))
        self.epoch_spin = QSpinBox()
        self.epoch_spin.setRange(5, 500)
        self.epoch_spin.setValue(50)
        h.addWidget(self.epoch_spin)

        h.addWidget(QLabel("סוג קלט:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(["word","line","sentence"])
        h.addWidget(self.type_combo)

        self.resume_btn = QPushButton("▶ המשך מנקודת עצירה")
        self.resume_btn.setCheckable(True)
        self.resume_btn.setChecked(True)
        h.addWidget(self.resume_btn)
        h.addStretch()
        root.addWidget(grp)

        # ── פרוגרס ──────────────────────────────────────────
        grp2 = QGroupBox("התקדמות")
        v2 = QVBoxLayout(grp2)
        self.epoch_label = QLabel("Epoch: —")
        self.batch_bar   = QProgressBar()
        self.batch_bar.setFormat("Batch: %v / %m")
        self.epoch_bar   = QProgressBar()
        self.epoch_bar.setFormat("Epoch: %v / %m")
        v2.addWidget(self.epoch_label)
        v2.addWidget(QLabel("Batches בעיבוד:"))
        v2.addWidget(self.batch_bar)
        v2.addWidget(QLabel("Epochs שהושלמו:"))
        v2.addWidget(self.epoch_bar)
        root.addWidget(grp2)

        # ── כפתורים ──────────────────────────────────────────
        row = QHBoxLayout()
        self.btn_train = QPushButton("🚀 התחל אימון")
        self.btn_train.clicked.connect(self.start_train)
        self.btn_stop = QPushButton("⏹ עצור")
        self.btn_stop.setObjectName("danger")
        self.btn_stop.clicked.connect(self.stop_train)
        self.btn_stop.setEnabled(False)
        self.btn_export = QPushButton("📦 ייצא ל-ONNX")
        self.btn_export.clicked.connect(self.start_export)
        row.addWidget(self.btn_train)
        row.addWidget(self.btn_stop)
        row.addWidget(self.btn_export)
        root.addLayout(row)

        # ── לוג ──────────────────────────────────────────────
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        root.addWidget(self.log)

    def _log(self, msg):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum()
        )

    def start_train(self):
        csv = "data/labels.csv"
        if not os.path.exists(csv):
            self._log("❌ לא נמצא data/labels.csv — תייג נתונים קודם.")
            return
        epochs = self.epoch_spin.value()
        resume = self.resume_btn.isChecked()
        itype  = self.type_combo.currentText()

        self.epoch_bar.setMaximum(epochs)
        self.epoch_bar.setValue(0)
        self.batch_bar.setValue(0)

        self.worker = TrainWorker(csv, epochs, resume, itype)
        self.worker.epoch_done.connect(self._on_epoch)
        self.worker.batch_done.connect(self._on_batch)
        self.worker.log_msg.connect(self._log)
        self.worker.finished_ok.connect(self._on_done)
        self.worker.error.connect(self._on_error)

        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker.start()
        self._log(f"🚀 מתחיל {epochs} epochs ({itype}) | Resume={resume}")

    def stop_train(self):
        if self.worker:
            self.worker.stop()
        self._log("⏹ נשלחה פקודת עצירה...")

    def _on_epoch(self, ep, total, loss):
        self.epoch_label.setText(
            f"Epoch {ep}/{total}  |  Loss: {loss:.4f}"
        )
        self.epoch_bar.setValue(ep)
        self._log(f"  📈 Epoch {ep}/{total} — Loss: {loss:.4f}")

    def _on_batch(self, b, total):
        self.batch_bar.setMaximum(total)
        self.batch_bar.setValue(b)

    def _on_done(self):
        self._log("✅ האימון הסתיים! המודל נשמר ב-weights/best_model.pth")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _on_error(self, msg):
        self._log(f"❌ שגיאה: {msg}")
        self.btn_train.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def start_export(self):
        self._log("📦 מייצא ל-ONNX...")
        self.exp_worker = ExportWorker()
        self.exp_worker.log_msg.connect(self._log)
        self.exp_worker.finished_ok.connect(
            lambda: self._log("✅ הייצוא הושלם!")
        )
        self.exp_worker.error.connect(
            lambda e: self._log(f"❌ {e}")
        )
        self.exp_worker.start()
