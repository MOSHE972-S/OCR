import onnxruntime as ort
import numpy as np
import cv2
import re
from src.dataset import VOCAB


class InferenceEngine:
    def __init__(self, onnx_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, img):
        h, w = img.shape
        new_w = max(1, int(w * 32.0 / h))
        img = cv2.resize(img, (new_w, 32))
        img = img.astype(np.float32) / 255.0
        return img[np.newaxis, np.newaxis, :, :]

    def recognize(self, image):
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return ""
        tensor = self.preprocess(image)
        probs  = self.session.run(None, {self.input_name: tensor})[0][0]
        return self._greedy_decode(probs)

    def _greedy_decode(self, probs):
        ids = np.argmax(probs, axis=-1)
        result, prev = [], -1
        for idx in ids:
            if idx != prev and idx != 0:
                result.append(VOCAB[idx])
            prev = idx
        return "".join(result)
