import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


class PageSegmenter:
    def __init__(self, ocr_engine):
        self.ocr = ocr_engine

    def _regions(self, proj, min_gap=5):
        out, in_t, start = [], False, 0
        for i, v in enumerate(proj):
            if v > 0 and not in_t:
                in_t, start = True, i
            elif v == 0 and in_t:
                if i - start > min_gap:
                    out.append((start, i))
                in_t = False
        if in_t:
            out.append((start, len(proj)))
        return out

    def segment_page(self, path, stop_flag=None):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ["❌ לא ניתן לקרוא את הקובץ"]
        _, binary = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        lines_text = []
        for y1, y2 in self._regions(np.sum(binary, axis=1), min_gap=8):
            if stop_flag and stop_flag():
                lines_text.append("⏹ הופסק על ידי המשתמש")
                break
            line_img = binary[y1:y2, :]
            words = [
                self.ocr.recognize(line_img[:, x1:x2])
                for x1, x2 in reversed(
                    self._regions(np.sum(line_img, axis=0), min_gap=5)
                )
            ]
            lines_text.append(" ".join(w for w in words if w))
        return lines_text


class SegmentWorker(QThread):
    line_done   = pyqtSignal(str)
    finished_ok = pyqtSignal()
    error       = pyqtSignal(str)

    def __init__(self, segmenter, image_path, parent=None):
        super().__init__(parent)
        self.segmenter  = segmenter
        self.image_path = image_path
        self._stop      = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            lines = self.segmenter.segment_page(
                self.image_path, stop_flag=lambda: self._stop
            )
            for line in lines:
                self.line_done.emit(line)
            self.finished_ok.emit()
        except Exception as e:
            self.error.emit(str(e))
