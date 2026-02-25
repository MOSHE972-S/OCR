import os, torch
from PyQt6.QtCore import QThread, pyqtSignal
from src.model import MyCRNN
from src.dataset import NUM_CLASSES

try:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    HAS_QUANT = True
except ImportError:
    HAS_QUANT = False


class ExportWorker(QThread):
    log_msg     = pyqtSignal(str)
    finished_ok = pyqtSignal()
    error       = pyqtSignal(str)

    def run(self):
        try:
            self._export()
        except Exception as e:
            self.error.emit(str(e))

    def _export(self):
        pth = "weights/best_model.pth"
        if not os.path.exists(pth):
            self.error.emit("❌ יש לאמן ולשמור מודל קודם.")
            return

        model = MyCRNN(NUM_CLASSES)
        model.load_state_dict(torch.load(pth, map_location="cpu"))
        model.eval()

        dummy  = torch.randn(1, 1, 32, 128)
        fp32   = "weights/crnn_fp32.onnx"

        torch.onnx.export(
            model, dummy, fp32,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch", 3: "width"},
                          "output": {0: "batch", 1: "seq_len"}},
            opset_version=17
        )
        self.log_msg.emit("✅ יוצא ל-ONNX (FP32)")

        if HAS_QUANT:
            q_path = "weights/best_model_int8.onnx"
            quantize_dynamic(fp32, q_path, weight_type=QuantType.QUInt8)
            os.remove(fp32)
            self.log_msg.emit("✅ כווץ ל-INT8 — " + q_path)
        else:
            self.log_msg.emit("⚠️  quantization לא זמין — נשמר כ-FP32")

        self.finished_ok.emit()
